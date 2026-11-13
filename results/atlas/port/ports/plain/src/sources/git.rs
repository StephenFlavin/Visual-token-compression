//! Git source: runs `git log -p` / `git diff` and yields fragments.

use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Command, Stdio};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use super::gitdiff::{LineOp, Parser};
use super::{clean_path, is_archive, CommitInfo, File, Fragment, RemoteInfo, YieldFn};
use crate::cmd::scm::Platform;
use crate::config::Config;
use crate::logging;
use crate::regexp::Regexp;

/// GitCmd helps to work with Git's output.
pub struct GitCmd {
    child: Child,
    cmd_string: String,
    repo_path: String,
    stderr_thread: Option<std::thread::JoinHandle<bool>>,
    stderr_done: Arc<Mutex<bool>>,
}

fn quoted_opt_pattern() -> &'static Regexp {
    static P: std::sync::OnceLock<Regexp> = std::sync::OnceLock::new();
    P.get_or_init(|| Regexp::must_compile(r#"^(?:"[^"]+"|'[^']+')$"#))
}

fn cmd_to_string(args: &[String]) -> String {
    let git = which_git();
    let mut s = git;
    for a in args {
        s.push(' ');
        s.push_str(a);
    }
    s
}

fn which_git() -> String {
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            let p = std::path::Path::new(if dir.is_empty() { "." } else { dir }).join("git");
            if p.is_file() {
                return p.to_string_lossy().into_owned();
            }
        }
    }
    "git".to_string()
}

fn spawn(args: Vec<String>) -> Result<GitCmd, String> {
    let cmd_string = cmd_to_string(&args);
    logging::debug().msg(&format!("executing: {}", cmd_string));
    let mut child = Command::new("git")
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("exec: \"git\": {}", e))?;
    let stderr = child.stderr.take().unwrap();
    let done = Arc::new(Mutex::new(false));
    let handle = std::thread::spawn(move || listen_for_std_err(stderr));
    Ok(GitCmd { child, cmd_string, repo_path: String::new(), stderr_thread: Some(handle), stderr_done: done })
}

impl GitCmd {
    /// NewGitLogCmdContext
    pub fn new_log(source: &str, log_opts: &str) -> Result<GitCmd, String> {
        let source_clean = clean_path(source);
        let args: Vec<String> = if !log_opts.is_empty() {
            let mut args: Vec<String> = vec!["-C".into(), source_clean.clone(), "log".into(), "-p".into(), "-U0".into()];
            let user_args: Vec<String> = log_opts.split(' ').map(|s| s.to_string()).collect();
            let quoted: Vec<String> = user_args.iter().filter(|e| quoted_opt_pattern().is_match(e)).cloned().collect();
            if !quoted.is_empty() {
                logging::warn().msg(&format!(
                    "the following `--log-opts` values may not work as expected: [{}]\n\tsee https://github.com/gitleaks/gitleaks/issues/1153 for more information",
                    quoted.join(" ")
                ));
            }
            args.extend(user_args);
            args
        } else {
            vec![
                "-C".into(),
                source_clean.clone(),
                "log".into(),
                "-p".into(),
                "-U0".into(),
                "--full-history".into(),
                "--all".into(),
                "--diff-filter=tuxdb".into(),
            ]
        };
        let mut c = spawn(args)?;
        c.repo_path = source_clean;
        Ok(c)
    }

    /// NewGitDiffCmdContext
    pub fn new_diff(source: &str, staged: bool) -> Result<GitCmd, String> {
        let source_clean = clean_path(source);
        let mut args: Vec<String> = vec!["-C".into(), source_clean.clone(), "diff".into(), "-U0".into(), "--no-ext-diff".into()];
        if staged {
            args.push("--staged".into());
        }
        args.push(".".into());
        let mut c = spawn(args)?;
        c.repo_path = source_clean;
        Ok(c)
    }

    pub fn wait(&mut self) -> Result<(), String> {
        match self.child.wait() {
            Ok(st) if st.success() => Ok(()),
            Ok(st) => Err(format!("exit status {}", st.code().unwrap_or(-1))),
            Err(e) => Err(e.to_string()),
        }
    }

    /// NewBlobReader returns the content of a blob within the repo
    pub fn blob(&self, commit: &str, path: &str) -> Result<Vec<u8>, String> {
        let out = Command::new("git")
            .args(["-C", &self.repo_path, "cat-file", "blob", &format!("{}:{}", commit, path)])
            .stderr(Stdio::null())
            .output()
            .map_err(|e| format!("failed to start git command: {}", e))?;
        Ok(out.stdout)
    }
}

/// listenForStdErr listens for stderr output from git and logs it.
/// Returns true if an error was encountered.
fn listen_for_std_err<R: Read>(stderr: R) -> bool {
    let mut err_encountered = false;
    let reader = BufReader::new(stderr);
    for line in reader.split(b'\n') {
        let line = match line {
            Ok(l) => String::from_utf8_lossy(&l).into_owned(),
            Err(_) => break,
        };
        let line = line.strip_suffix('\r').unwrap_or(&line).to_string();
        if line.contains("exhaustive rename detection was skipped")
            || line.contains("inexact rename detection was skipped")
            || line.contains("you may want to set your diff.renameLimit")
            || line.contains("See \"git help gc\" for manual housekeeping")
            || line.contains("Auto packing the repository in background for optimum performance")
        {
            logging::warn().msg(&line);
        } else {
            logging::error().msg(&format!("[git] {}", line));
            err_encountered = true;
        }
    }
    err_encountered
}

/// Git is a source for yielding fragments from a git repo
pub struct Git<'a> {
    pub cmd: GitCmd,
    pub config: &'a Config,
    pub remote: Rc<RemoteInfo>,
    pub max_archive_depth: i64,
}

fn format_rfc3339_utc(t: &Option<chrono::DateTime<chrono::FixedOffset>>) -> String {
    match t {
        Some(t) => t.with_timezone(&chrono::Utc).format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        None => "0001-01-01T00:00:00Z".to_string(),
    }
}

impl<'a> Git<'a> {
    pub fn fragments(&mut self, yield_: &mut YieldFn) -> Result<(), String> {
        let res = self.fragments_inner(yield_);
        if let Err(e) = self.cmd.wait() {
            logging::debug().err(e).str("cmd", &self.cmd.cmd_string).msg("command aborted");
        }
        res
    }

    fn fragments_inner(&mut self, yield_: &mut YieldFn) -> Result<(), String> {
        let stdout = self.cmd.child.stdout.take().unwrap();
        let mut parser = Parser::new(BufReader::with_capacity(64 * 1024, stdout));
        while let Some(file) = parser.next_file() {
            if super::ctx_done() {
                let _ = self.cmd.child.kill();
                return Err(super::CTX_ERR.to_string());
            }
            if file.is_delete {
                continue;
            }
            let mut yield_as_archive = false;
            if file.is_binary {
                if !is_archive(&file.new_name) {
                    continue;
                }
                yield_as_archive = true;
            }
            let mut commit_sha = String::new();
            let mut commit_info: Option<Rc<CommitInfo>> = None;
            if let Some(ph) = &file.patch_header {
                commit_sha = ph.sha.clone();
                for a in &self.config.allowlists {
                    let (ok, c) = a.commit_allowed(&ph.sha);
                    if ok {
                        logging::trace().str("allowed-commit", &c).msg("skipping commit: global allowlist");
                        continue;
                    }
                }
                let mut ci = CommitInfo {
                    author_email: String::new(),
                    author_name: String::new(),
                    date: format_rfc3339_utc(&ph.author_date),
                    message: ph.message(),
                    remote: self.remote.clone(),
                    sha: commit_sha.clone(),
                };
                if let Some(a) = &ph.author {
                    ci.author_name = a.name.clone();
                    ci.author_email = a.email.clone();
                }
                commit_info = Some(Rc::new(ci));
            }

            if yield_as_archive {
                let blob = match self.cmd.blob(&commit_sha, &file.new_name) {
                    Ok(b) => b,
                    Err(e) => {
                        logging::error().err(e).msg("could not read archive blob");
                        continue;
                    }
                };
                let f = File {
                    content: Box::new(std::io::Cursor::new(blob)),
                    path: file.new_name.clone(),
                    symlink: String::new(),
                    buffer_size: 100 * 1000,
                    config: Some(self.config),
                    outer_paths: Vec::new(),
                    max_archive_depth: self.max_archive_depth,
                    archive_depth: 0,
                };
                let cs = commit_sha.clone();
                let ci = commit_info.clone();
                let mut enrich = |mut fragment: Fragment, err: Option<String>| -> Result<(), String> {
                    fragment.commit_sha = cs.clone();
                    fragment.commit_info = ci.clone();
                    yield_(fragment, err)
                };
                f.fragments(&mut enrich)?;
                continue;
            }

            for tf in &file.text_fragments {
                let fragment = Fragment {
                    commit_sha: commit_sha.clone(),
                    file_path: file.new_name.clone(),
                    raw: tf.raw(LineOp::Add),
                    start_line: tf.new_position,
                    commit_info: commit_info.clone(),
                    ..Default::default()
                };
                yield_(fragment, None)?;
            }
        }

        // stderr
        if let Some(h) = self.cmd.stderr_thread.take() {
            let err = h.join().unwrap_or(false);
            *self.cmd.stderr_done.lock().unwrap() = true;
            if err {
                return yield_(Fragment::default(), Some("stderr is not empty".to_string()));
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ remote info

/// NewRemoteInfoContext builds a new RemoteInfo for generating finding links
pub fn new_remote_info(platform: Platform, source: &str) -> RemoteInfo {
    if platform == Platform::None {
        return RemoteInfo { platform, url: String::new() };
    }
    let mut platform = platform;
    let remote = match get_remote_url(source) {
        Ok(u) => Some(u),
        Err(e) => {
            if e.contains("No remote configured") {
                logging::debug().msg("skipping finding links: repository has no configured remote.");
                platform = Platform::None;
            } else {
                logging::error().err(&e).msg("skipping finding links: unable to parse remote URL");
            }
            None
        }
    };
    if let Some(u) = &remote {
        if platform == Platform::Unknown {
            platform = platform_from_host(&u.hostname());
            if platform == Platform::Unknown {
                logging::info()
                    .str("host", &u.hostname())
                    .msg("Unknown SCM platform. Use --platform to include links in findings.");
            } else {
                logging::debug()
                    .str("host", &u.hostname())
                    .str("platform", platform.as_str())
                    .msg("SCM platform parsed from host");
            }
        }
    }
    RemoteInfo { platform, url: remote.map(|u| u.to_string()).unwrap_or_default() }
}

struct Url {
    scheme: String,
    host: String,
    path: String,
    query: Option<String>,
    fragment: Option<String>,
    opaque: Option<String>,
}

impl Url {
    fn parse(s: &str) -> Url {
        let mut rest = s.to_string();
        let mut fragment = None;
        if let Some(i) = rest.find('#') {
            fragment = Some(rest[i + 1..].to_string());
            rest.truncate(i);
        }
        let mut query = None;
        if let Some(i) = rest.find('?') {
            query = Some(rest[i + 1..].to_string());
            rest.truncate(i);
        }
        let mut scheme = String::new();
        if let Some(i) = rest.find(':') {
            let cand = &rest[..i];
            if !cand.is_empty()
                && cand.chars().next().unwrap().is_ascii_alphabetic()
                && cand.chars().all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
            {
                scheme = cand.to_lowercase();
                rest = rest[i + 1..].to_string();
            }
        }
        let mut host = String::new();
        let mut opaque = None;
        if let Some(r) = rest.strip_prefix("//") {
            let end = r.find('/').unwrap_or(r.len());
            let authority = &r[..end];
            host = match authority.rfind('@') {
                Some(i) => authority[i + 1..].to_string(),
                None => authority.to_string(),
            };
            rest = r[end..].to_string();
        } else if !scheme.is_empty() && !rest.starts_with('/') {
            opaque = Some(rest.clone());
            rest = String::new();
        }
        Url { scheme, host, path: rest, query, fragment, opaque }
    }

    fn hostname(&self) -> String {
        let h = &self.host;
        if h.starts_with('[') {
            if let Some(i) = h.find(']') {
                return h[1..i].to_string();
            }
        }
        match h.rfind(':') {
            Some(i) => h[..i].to_string(),
            None => h.clone(),
        }
    }
}

impl std::fmt::Display for Url {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        if !self.scheme.is_empty() {
            s.push_str(&self.scheme);
            s.push(':');
        }
        if let Some(o) = &self.opaque {
            s.push_str(o);
        } else {
            if !self.scheme.is_empty() || !self.host.is_empty() {
                if !self.host.is_empty() || self.path.starts_with('/') || !self.scheme.is_empty() {
                    s.push_str("//");
                }
                s.push_str(&self.host);
            }
            if !self.path.is_empty() && !self.path.starts_with('/') && !self.host.is_empty() {
                s.push('/');
            }
            s.push_str(&self.path);
        }
        if let Some(q) = &self.query {
            s.push('?');
            s.push_str(q);
        }
        if let Some(fr) = &self.fragment {
            s.push('#');
            s.push_str(fr);
        }
        write!(f, "{}", s)
    }
}

fn ssh_url_pat() -> &'static Regexp {
    static P: std::sync::OnceLock<Regexp> = std::sync::OnceLock::new();
    P.get_or_init(|| Regexp::must_compile(r"^git@([a-zA-Z0-9.-]+):(?:\d{1,5}/)?([\w/.-]+?)(?:\.git)?$"))
}

fn get_remote_url(source: &str) -> Result<Url, String> {
    let mut cmd = Command::new("git");
    cmd.args(["ls-remote", "--quiet", "--get-url"]);
    if source != "." {
        cmd.current_dir(source);
    }
    let out = cmd.stdin(Stdio::null()).output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        let code = out.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(format!("command failed ({}): exit status {}, stderr: {}", code, code, stderr));
    }
    let mut remote = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if let Some(m) = ssh_url_pat().find_submatch(&remote) {
        remote = format!("https://{}/{}", m[1], m[2]);
    }
    let remote = remote.strip_suffix(".git").unwrap_or(&remote).to_string();
    Ok(Url::parse(&remote))
}

fn platform_from_host(host: &str) -> Platform {
    match host.to_lowercase().as_str() {
        "github.com" => Platform::GitHub,
        "gitlab.com" => Platform::GitLab,
        "dev.azure.com" | "visualstudio.com" => Platform::AzureDevOps,
        "gitea.com" | "code.forgejo.org" | "codeberg.org" => Platform::Gitea,
        "bitbucket.org" => Platform::Bitbucket,
        _ => Platform::Unknown,
    }
}
