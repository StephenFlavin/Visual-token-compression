//! Port of gitleaks' cmd package.

pub mod cli;
pub mod scm;

use std::rc::Rc;
use std::time::Instant;

use cli::{App, CommandDef, FlagDef, Kind, Outcome, Parsed};

use crate::config::{self, Config, ViperConfig};
use crate::detect::{Detector, SourceKind};
use crate::logging;
use crate::report::{self, Finding, Reporter};
use crate::sources::{self, git as gitsrc};
use crate::version;

const BANNER: &str = "\n    ○\n    │╲\n    │ ○\n    ○ ░\n    ░    gitleaks\n\n";

const CONFIG_DESCRIPTION: &str = "config file path\norder of precedence:\n1. --config/-c\n2. env var GITLEAKS_CONFIG\n3. env var GITLEAKS_CONFIG_TOML with the file content\n4. (target path)/.gitleaks.toml\nIf none of the four options are used, then gitleaks will use the default config";

fn persistent_flags() -> Vec<FlagDef> {
    let mut redact = FlagDef::new("redact", "", Kind::Uint, "0", "redact secrets from logs and stdout. To redact only parts of the secret just apply a percent value from 0..100. For example --redact=20 (default 100%)");
    redact.no_opt_def = "100";
    vec![
        FlagDef::new("config", "c", Kind::Str, "", CONFIG_DESCRIPTION),
        FlagDef::new("exit-code", "", Kind::Int, "1", "exit code when leaks have been encountered"),
        FlagDef::new("report-path", "r", Kind::Str, "", "report file (use \"-\" for stdout)"),
        FlagDef::new("report-format", "f", Kind::Str, "", "output format (json, csv, junit, sarif, template)"),
        FlagDef::new("report-template", "", Kind::Str, "", "template file used to generate the report (implies --report-format=template)"),
        FlagDef::new("baseline-path", "b", Kind::Str, "", "path to baseline with issues that can be ignored"),
        FlagDef::new("log-level", "l", Kind::Str, "info", "log level (trace, debug, info, warn, error, fatal)"),
        FlagDef::new("verbose", "v", Kind::Bool, "false", "show verbose output from scan"),
        FlagDef::new("no-color", "", Kind::Bool, "false", "turn off color for verbose output"),
        FlagDef::new("max-target-megabytes", "", Kind::Int, "0", "files larger than this will be skipped"),
        FlagDef::new("ignore-gitleaks-allow", "", Kind::Bool, "false", "ignore gitleaks:allow comments"),
        redact,
        FlagDef::new("no-banner", "", Kind::Bool, "false", "suppress banner"),
        FlagDef::new("enable-rule", "", Kind::StrSlice, "", "only enable specific rules by id"),
        FlagDef::new("gitleaks-ignore-path", "i", Kind::Str, ".", "path to .gitleaksignore file or folder containing one"),
        FlagDef::new("max-decode-depth", "", Kind::Int, "5", "allow recursive decoding up to this depth"),
        FlagDef::new("max-archive-depth", "", Kind::Int, "0", "allow scanning into nested archives up to this depth (default \"0\", no archive traversal is done)"),
        FlagDef::new("timeout", "", Kind::Int, "0", "set a timeout for gitleaks commands in seconds (default \"0\", no timeout is set)"),
        FlagDef::new("diagnostics", "", Kind::Str, "", "enable diagnostics (http OR comma-separated list: cpu,mem,trace). cpu=CPU prof, mem=memory prof, trace=exec tracing, http=serve via net/http/pprof"),
        FlagDef::new("diagnostics-dir", "", Kind::Str, "", "directory to store diagnostics output files when not using http mode (defaults to current directory)"),
    ]
}

fn cmd(name: &'static str, use_line: &'static str, short: &'static str, flags: Vec<FlagDef>) -> CommandDef {
    CommandDef {
        name,
        use_line,
        aliases: vec![],
        short,
        long: "",
        hidden: false,
        runnable: true,
        max_args: None,
        flags,
        subcommands: vec![],
        disable_flags_in_use_line: false,
    }
}

fn build_app() -> App {
    let mut dir = cmd("dir", "dir [flags] [path]", "scan directories or files for secrets", vec![FlagDef::new(
        "follow-symlinks",
        "",
        Kind::Bool,
        "false",
        "scan files that are symlinks to other files",
    )]);
    dir.aliases = vec!["file", "directory"];
    let mut git = cmd(
        "git",
        "git [flags] [repo]",
        "scan git repositories for secrets",
        vec![
            FlagDef::new("platform", "", Kind::Str, "", "the target platform used to generate links (github, gitlab)"),
            FlagDef::new("staged", "", Kind::Bool, "false", "scan staged commits (good for pre-commit)"),
            FlagDef::new("pre-commit", "", Kind::Bool, "false", "scan using git diff"),
            FlagDef::new("log-opts", "", Kind::Str, "", "git log options"),
        ],
    );
    git.max_args = Some(1);
    let stdin = cmd("stdin", "stdin", "detect secrets from stdin", vec![]);
    let version_cmd = cmd("version", "version", "display gitleaks version", vec![]);
    let mut detect = cmd(
        "detect",
        "detect",
        "detect secrets in code",
        vec![
            FlagDef::new("no-git", "", Kind::Bool, "false", "treat git repo as a regular directory and scan those files, --log-opts has no effect on the scan when --no-git is set"),
            FlagDef::new("pipe", "", Kind::Bool, "false", "scan input from stdin, ex: `cat some_file | gitleaks detect --pipe`"),
            FlagDef::new("follow-symlinks", "", Kind::Bool, "false", "scan files that are symlinks to other files"),
            FlagDef::new("source", "s", Kind::Str, ".", "path to source"),
            FlagDef::new("log-opts", "", Kind::Str, "", "git log options"),
            FlagDef::new("platform", "", Kind::Str, "", "the target platform used to generate links (github, gitlab)"),
        ],
    );
    detect.hidden = true;
    let mut protect = cmd(
        "protect",
        "protect",
        "protect secrets in code",
        vec![
            FlagDef::new("staged", "", Kind::Bool, "false", "detect secrets in a --staged state"),
            FlagDef::new("log-opts", "", Kind::Str, "", "git log options"),
            FlagDef::new("source", "s", Kind::Str, ".", "path to source"),
        ],
    );
    protect.hidden = true;
    let mut completion = cmd(
        "completion",
        "completion [command]",
        "Generate the autocompletion script for the specified shell",
        vec![],
    );
    completion.long = "Generate the autocompletion script for gitleaks for the specified shell.\nSee each sub-command's help for details on how to use the generated script.\n";
    completion.runnable = false;
    for (n, s) in [
        ("bash", "Generate the autocompletion script for bash"),
        ("fish", "Generate the autocompletion script for fish"),
        ("powershell", "Generate the autocompletion script for powershell"),
        ("zsh", "Generate the autocompletion script for zsh"),
    ] {
        let mut c = cmd(n, n, s, vec![]);
        c.max_args = Some(0);
        c.long = completion_long(n);
        c.disable_flags_in_use_line = n == "bash";
        if n == "bash" || n == "zsh" || n == "fish" || n == "powershell" {
            c.flags.push(FlagDef::new("no-descriptions", "", Kind::Bool, "false", "disable completion descriptions"));
        }
        completion.subcommands.push(c);
    }
    let root = CommandDef {
        name: "gitleaks",
        use_line: "gitleaks",
        aliases: vec![],
        short: "Gitleaks scans code, past or present, for secrets",
        long: "",
        hidden: false,
        runnable: false,
        max_args: None,
        flags: vec![],
        subcommands: vec![completion, dir, git, stdin, version_cmd, detect, protect],
        disable_flags_in_use_line: false,
    };
    App { root, persistent: persistent_flags(), version: version::VERSION.to_string() }
}

pub fn execute() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let app = build_app();
    match app.parse(&args) {
        Outcome::Help(s) | Outcome::Version(s) => {
            print!("{}", s);
        }
        Outcome::Error(msg, stderr) => {
            eprint!("{}", stderr);
            if msg.contains("unknown flag") {
                std::process::exit(126);
            }
            logging::fatal().msg(&msg);
        }
        Outcome::Run(p) => {
            init_log(&p);
            let timeout = p.get_int("timeout");
            if timeout > 0 {
                sources::set_deadline(Instant::now() + std::time::Duration::from_secs(timeout as u64));
            }
            let path: Vec<&str> = p.path.iter().map(|s| s.as_str()).collect();
            match path.as_slice() {
                ["gitleaks", "dir"] => run_directory(&p),
                ["gitleaks", "git"] => run_git(&p),
                ["gitleaks", "stdin"] => run_stdin(&p),
                ["gitleaks", "version"] => println!("{}", version::VERSION),
                ["gitleaks", "detect"] => run_detect(&p),
                ["gitleaks", "protect"] => run_protect(&p),
                ["gitleaks", "completion", shell] => {
                    println!("# {} completion for gitleaks", shell);
                }
                _ => {}
            }
        }
    }
}

fn init_log(p: &Parsed) {
    let ll = p.get_string("log-level");
    let lvl = match ll.to_lowercase().as_str() {
        "trace" => Some(logging::Level::Trace),
        "debug" => Some(logging::Level::Debug),
        "info" => Some(logging::Level::Info),
        "warn" => Some(logging::Level::Warn),
        "err" | "error" => Some(logging::Level::Error),
        "fatal" => Some(logging::Level::Fatal),
        _ => None,
    };
    match lvl {
        Some(l) => logging::set_level(l),
        None => logging::warn().msg(&format!("unknown log level: {}", ll)),
    }
}

/// initConfig determines which config to use and loads its TOML content
fn init_config(p: &Parsed, source: &str) -> toml::Table {
    if !p.get_bool("no-banner") {
        eprint!("{}", BANNER);
    }
    logging::debug().msg(&format!("using {} regex engine", crate::regexp::VERSION));
    let cfg_path = p.get_string("config");
    let file: String;
    if !cfg_path.is_empty() {
        logging::debug().msg(&format!("using gitleaks config {} from `--config`", cfg_path));
        file = cfg_path;
    } else if !std::env::var("GITLEAKS_CONFIG").unwrap_or_default().is_empty() {
        let env_path = std::env::var("GITLEAKS_CONFIG").unwrap();
        logging::debug().msg(&format!("using gitleaks config from GITLEAKS_CONFIG env var: {}", env_path));
        file = env_path;
    } else if !std::env::var("GITLEAKS_CONFIG_TOML").unwrap_or_default().is_empty() {
        let content = std::env::var("GITLEAKS_CONFIG_TOML").unwrap();
        let t = match config::parse_toml(&content) {
            Ok(t) => t,
            Err(e) => {
                logging::fatal()
                    .err(e)
                    .str("content", &content)
                    .msg("unable to load gitleaks config from GITLEAKS_CONFIG_TOML env var");
                unreachable!()
            }
        };
        logging::debug().str("content", &content).msg("using gitleaks config from GITLEAKS_CONFIG_TOML env var content");
        return t;
    } else {
        let md = match std::fs::metadata(source) {
            Ok(m) => m,
            Err(e) => {
                logging::fatal().msg(&config::go_os_error("stat", source, &e));
                unreachable!()
            }
        };
        let joined = sources::join_path(source, ".gitleaks.toml");
        if !md.is_dir() {
            logging::debug().msg(&format!(
                "unable to load gitleaks config from {} since --source={} is a file, using default config",
                joined, source
            ));
            return load_default_table("err reading toml");
        }
        if std::fs::metadata(&joined).is_err() {
            logging::debug().msg(&format!("no gitleaks config found in path {}, using default gitleaks config", joined));
            return load_default_table("err reading default config toml");
        }
        logging::debug().msg(&format!("using existing gitleaks config {} from `(--source)/.gitleaks.toml`", joined));
        file = joined;
    }
    config::set_config_file_used(&file);
    match config::read_config_file(&file) {
        Ok(t) => t,
        Err(e) => {
            logging::fatal().msg(&format!("unable to load gitleaks config, err: {}", e));
            unreachable!()
        }
    }
}

fn load_default_table(prefix: &str) -> toml::Table {
    match config::parse_toml(config::DEFAULT_CONFIG) {
        Ok(t) => t,
        Err(e) => {
            logging::fatal().msg(&format!("{} {}", prefix, e));
            unreachable!()
        }
    }
}

static DIAGNOSTICS_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Port of NewDiagnosticsManager + StartDiagnostics. The Go implementation
/// produces pprof/trace profiles of the Go runtime; those have no equivalent
/// here, so only the configuration handling and logging are reproduced.
fn init_diagnostics(p: &Parsed) {
    let flag = p.get_string("diagnostics");
    let mut dir = p.get_string("diagnostics-dir");
    if flag.is_empty() {
        return;
    }
    let types: Vec<String> = flag.split(',').map(|s| s.to_string()).collect();
    if flag == "http" {
        if !dir.is_empty() {
            logging::fatal().err("the diagnostics directory should not be set in http mode").msg("Error initializing diagnostics");
        }
    } else {
        if dir.is_empty() {
            dir = std::env::current_dir().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
            logging::debug().msg(&format!("No diagnostics directory specified, using current directory: {}", dir));
        }
        if let Err(e) = std::fs::create_dir_all(&dir) {
            logging::fatal()
                .err(format!("failed to create diagnostics directory: {}", config::go_os_error("mkdir", &dir, &e)))
                .msg("Error initializing diagnostics");
        }
        let dir = crate::detect::abs_path(&dir);
        logging::debug().msg(&format!("Diagnostics enabled: {}", types.join(",")));
        logging::debug().msg(&format!("Diagnostics output directory: {}", dir));
    }
    DIAGNOSTICS_ENABLED.store(true, std::sync::atomic::Ordering::SeqCst);
    logging::info().msg("Starting diagnostics...");
    for t in &types {
        let t = t.trim();
        match t {
            "cpu" | "mem" | "trace" | "http" => {
                logging::debug().msg(&format!("{} profiling of the Go runtime is not available in this build", t));
            }
            _ => logging::warn().msg(&format!("Unknown diagnostics type: {}", t)),
        }
    }
}

fn load_config(p: &Parsed, table: &toml::Table) -> Config {
    let mut vc = match ViperConfig::unmarshal(table) {
        Ok(v) => v,
        Err(e) => {
            logging::fatal().err(e).msg("Failed to load config");
            unreachable!()
        }
    };
    let mut cfg = match vc.translate() {
        Ok(c) => c,
        Err(e) => {
            logging::fatal().err(e).msg("Failed to load config");
            unreachable!()
        }
    };
    cfg.path = p.get_string("config");
    cfg
}

fn file_exists(name: &str) -> bool {
    match std::fs::metadata(name) {
        Ok(m) => !m.is_dir(),
        Err(_) => false,
    }
}

fn build_detector(p: &Parsed, cfg: &Config, source: &str) -> Detector {
    let mut d = Detector::new(cfg.clone());
    d.max_decode_depth = p.get_int("max-decode-depth");
    d.max_archive_depth = p.get_int("max-archive-depth");
    d.no_color = p.get_bool("no-color");
    if d.no_color {
        logging::set_no_color(true);
    }
    d.config.path = p.get_string("config");
    if d.config.path.is_empty() {
        d.config.path = sources::join_path(source, ".gitleaks.toml");
    }
    d.verbose = p.get_bool("verbose");
    d.redact = p.get_uint("redact");
    d.max_target_mega_bytes = p.get_int("max-target-megabytes");
    d.ignore_gitleaks_allow = p.get_bool("ignore-gitleaks-allow");

    let ignore_path = p.get_string("gitleaks-ignore-path");
    if file_exists(&ignore_path) {
        if let Err(e) = d.add_gitleaks_ignore(&ignore_path) {
            logging::fatal().err(e).msg("could not call AddGitleaksIgnore");
        }
    }
    let p2 = sources::join_path(&ignore_path, ".gitleaksignore");
    if file_exists(&p2) {
        if let Err(e) = d.add_gitleaks_ignore(&p2) {
            logging::fatal().err(e).msg("could not call AddGitleaksIgnore");
        }
    }
    let p3 = sources::join_path(source, ".gitleaksignore");
    if file_exists(&p3) {
        if let Err(e) = d.add_gitleaks_ignore(&p3) {
            logging::fatal().err(e).msg("could not call AddGitleaksIgnore");
        }
    }

    let baseline_path = p.get_string("baseline-path");
    if !baseline_path.is_empty() {
        if let Err(e) = d.add_baseline(&baseline_path, source) {
            logging::error().msg(&format!(
                "Could not load baseline. The path must point of a gitleaks report generated using the default format: {}",
                e
            ));
        }
    }

    let rules = p.get_slice("enable-rule");
    if !rules.is_empty() {
        logging::info().msg(&format!("Overriding enabled rules: {}", rules.join(", ")));
        let mut over = std::collections::HashMap::new();
        for r in &rules {
            match cfg.rules.get(r) {
                Some(rule) => {
                    over.insert(r.clone(), rule.clone());
                }
                None => logging::fatal().msg(&format!("Requested rule {} not found in rules", r)),
            }
        }
        d.config.rules = over;
    }

    let report_path = p.get_string("report-path");
    if !report_path.is_empty() {
        if report_path != report::STDOUT_REPORT_PATH {
            match std::fs::File::create(&report_path) {
                Err(e) => {
                    logging::fatal()
                        .err(config::go_os_error("open", &report_path, &e))
                        .msg(&format!("Report path is not writable: {}", report_path));
                }
                Ok(_) => {
                    let _ = std::fs::remove_file(&report_path);
                }
            }
        }
        let mut report_format = p.get_string("report-format");
        let report_template = p.get_string("report-template");
        if report_format.is_empty() {
            let e = sources::ext(&report_path).to_lowercase();
            report_format = match e.as_str() {
                ".csv" => "csv".to_string(),
                ".json" => "json".to_string(),
                ".sarif" => "sarif".to_string(),
                _ => {
                    logging::fatal().msg(&format!("Unknown report format: {}", report_format));
                    unreachable!()
                }
            };
            logging::debug().msg(&format!(
                "No report format specified, inferred {} from {}",
                crate::gostr::quote(&report_format),
                crate::gostr::quote(&e)
            ));
        }
        let reporter = match report_format.trim().to_lowercase().as_str() {
            "csv" => Reporter::Csv,
            "json" => Reporter::Json,
            "junit" => Reporter::Junit,
            "sarif" => Reporter::Sarif(cfg.get_ordered_rules()),
            "template" => match new_template_reporter(&report_template) {
                Ok(t) => Reporter::Template(t),
                Err(e) => {
                    logging::fatal().err(e).msg("Invalid report template");
                    unreachable!()
                }
            },
            _ => {
                logging::fatal().msg(&format!("unknown report format {}", report_format));
                unreachable!()
            }
        };
        if !report_template.is_empty() && report_format != "template" {
            logging::fatal().msg("Report format must be 'template' if --report-template is specified");
        }
        d.report_path = report_path;
        d.reporter = Some(reporter);
    }
    d
}

fn new_template_reporter(path: &str) -> Result<report::template::Template, String> {
    if path.is_empty() {
        return Err("template path cannot be empty".to_string());
    }
    let data = std::fs::read(path).map_err(|e| format!("error reading file: {}", config::go_os_error("open", path, &e)))?;
    let text = String::from_utf8_lossy(&data);
    report::template::Template::parse(&text).map_err(|e| format!("error parsing file: {}", e))
}

fn bytes_convert(bytes: u64) -> String {
    const KB: f32 = 1000.0;
    const MB: f32 = KB * 1000.0;
    const GB: f32 = MB * 1000.0;
    let mut value = bytes as f32;
    let unit;
    if bytes as f64 >= GB as f64 {
        unit = "GB";
        value /= GB;
    } else if bytes as f64 >= MB as f64 {
        unit = "MB";
        value /= MB;
    } else if bytes as f64 >= KB as f64 {
        unit = "KB";
        value /= KB;
    } else if bytes >= 1 {
        unit = "bytes";
    } else {
        return "0".to_string();
    }
    let s = format!("{:.2}", value);
    let s = s.strip_suffix(".00").unwrap_or(&s).to_string();
    format!("{} {}", s, unit)
}

pub fn format_duration(d: std::time::Duration) -> String {
    let d = d.as_nanos() as i64;
    let mut scale: i64 = 100 * 1_000_000_000;
    while scale > d {
        scale /= 10;
    }
    crate::gostr::duration_string(crate::gostr::duration_round(d, scale / 100))
}

fn finding_summary_and_exit(d: &Detector, findings: &[Finding], exit_code: i64, start: Instant, err: Option<String>) {
    if DIAGNOSTICS_ENABLED.load(std::sync::atomic::Ordering::SeqCst) {
        logging::debug().msg("Finalizing diagnostics...");
        logging::debug().msg("Stopping diagnostics and writing profiling data...");
    }
    let total = d.total_bytes;
    let bytes_msg = format!("scanned ~{} bytes ({})", total, bytes_convert(total));
    if err.is_none() {
        logging::info().msg(&format!("{} in {}", bytes_msg, format_duration(start.elapsed())));
        if !findings.is_empty() {
            logging::warn().msg(&format!("leaks found: {}", findings.len()));
        } else {
            logging::info().msg("no leaks found");
        }
    } else {
        logging::warn().msg(&bytes_msg);
        logging::warn().msg(&format!("partial scan completed in {}", format_duration(start.elapsed())));
        if !findings.is_empty() {
            logging::warn().msg(&format!("{} leaks found in partial scan", findings.len()));
        } else {
            logging::warn().msg("no leaks found in partial scan");
        }
    }

    if let Some(reporter) = &d.reporter {
        let res = if d.report_path == report::STDOUT_REPORT_PATH {
            let stdout = std::io::stdout();
            let mut h = stdout.lock();
            reporter.write(&mut h, findings)
        } else {
            match std::fs::File::create(&d.report_path) {
                Ok(mut f) => reporter.write(&mut f, findings),
                Err(e) => Err(config::go_os_error("open", &d.report_path, &e)),
            }
        };
        if let Err(e) = res {
            logging::fatal().err(e).msg("failed to write report");
        }
    }

    if err.is_some() {
        std::process::exit(1);
    }
    if !findings.is_empty() {
        std::process::exit(exit_code as i32);
    }
    std::process::exit(0);
}

fn source_arg(p: &Parsed) -> String {
    if p.args.len() == 1 && !p.args[0].is_empty() {
        p.args[0].clone()
    } else {
        ".".to_string()
    }
}

fn run_directory(p: &Parsed) {
    let source = source_arg(p);
    let table = init_config(p, &source);
    init_diagnostics(p);
    let cfg = load_config(p, &table);
    let start = Instant::now();
    let mut d = build_detector(p, &cfg, &source);
    d.follow_symlinks = p.get_bool("follow-symlinks");
    let exit_code = p.get_int("exit-code");
    let files = sources::Files {
        config: Some(&cfg),
        follow_symlinks: d.follow_symlinks,
        max_file_size: d.max_target_mega_bytes * 1_000_000,
        path: source.clone(),
        max_archive_depth: d.max_archive_depth,
    };
    let (findings, err) = d.detect_source(SourceKind::Files(&files));
    if let Some(e) = &err {
        logging::error().err(e).msg("failed scan directory");
    }
    finding_summary_and_exit(&d, &findings, exit_code, start, err);
}

fn run_git_scan(d: &mut Detector, cmd: gitsrc::GitCmd, remote: sources::RemoteInfo) -> (Vec<Finding>, Option<String>) {
    let cfg = d.config.clone();
    let mut g = gitsrc::Git { cmd, config: &cfg, remote: Rc::new(remote), max_archive_depth: d.max_archive_depth };
    d.detect_source(SourceKind::Git(&mut g))
}

fn run_git(p: &Parsed) {
    let start = Instant::now();
    let source = source_arg(p);
    let table = init_config(p, &source);
    init_diagnostics(p);
    let cfg = load_config(p, &table);
    let mut d = build_detector(p, &cfg, &source);
    let exit_code = p.get_int("exit-code");
    let log_opts = p.get_string("log-opts");
    let staged = p.get_bool("staged");
    let pre_commit = p.get_bool("pre-commit");
    let (git_cmd, platform) = if pre_commit || staged {
        match gitsrc::GitCmd::new_diff(&source, staged) {
            Ok(c) => (c, scm::Platform::None),
            Err(e) => {
                logging::fatal().err(e).msg("could not create Git diff cmd");
                unreachable!()
            }
        }
    } else {
        let c = match gitsrc::GitCmd::new_log(&source, &log_opts) {
            Ok(c) => c,
            Err(e) => {
                logging::fatal().err(e).msg("could not create Git log cmd");
                unreachable!()
            }
        };
        match scm::platform_from_string(&p.get_string("platform")) {
            Ok(pl) => (c, pl),
            Err(e) => {
                logging::fatal().err(e).send();
                unreachable!()
            }
        }
    };
    let remote = gitsrc::new_remote_info(platform, &source);
    let (findings, err) = run_git_scan(&mut d, git_cmd, remote);
    if let Some(e) = &err {
        logging::error().err(e).msg("failed to scan Git repository");
    }
    finding_summary_and_exit(&d, &findings, exit_code, start, err);
}

fn run_stdin(p: &Parsed) {
    let start = Instant::now();
    let table = init_config(p, ".");
    init_diagnostics(p);
    let cfg = load_config(p, &table);
    let mut d = build_detector(p, &cfg, "");
    let exit_code = p.get_int("exit-code");
    let mut file = sources::File::new(Box::new(std::io::stdin()), "");
    file.max_archive_depth = d.max_archive_depth;
    let (findings, err) = d.detect_source(SourceKind::File(file));
    if let Some(e) = &err {
        logging::fatal().err(e).msg("failed scan input from stdin");
    }
    finding_summary_and_exit(&d, &findings, exit_code, start, err);
}

fn run_detect(p: &Parsed) {
    let start = Instant::now();
    let source = p.get_string("source");
    let table = init_config(p, &source);
    init_diagnostics(p);
    let cfg = load_config(p, &table);
    let mut d = build_detector(p, &cfg, &source);
    d.follow_symlinks = p.get_bool("follow-symlinks");
    let exit_code = p.get_int("exit-code");
    let no_git = p.get_bool("no-git");
    let from_pipe = p.get_bool("pipe");
    let (findings, err);
    if no_git {
        let files = sources::Files {
            config: Some(&cfg),
            follow_symlinks: d.follow_symlinks,
            max_file_size: d.max_target_mega_bytes * 1_000_000,
            path: source.clone(),
            max_archive_depth: d.max_archive_depth,
        };
        let r = d.detect_source(SourceKind::Files(&files));
        findings = r.0;
        err = r.1;
        if let Some(e) = &err {
            logging::error().err(e).msg("failed to scan directory");
        }
    } else if from_pipe {
        let mut file = sources::File::new(Box::new(std::io::stdin()), "");
        file.max_archive_depth = d.max_archive_depth;
        let r = d.detect_source(SourceKind::File(file));
        findings = r.0;
        err = r.1;
        if let Some(e) = &err {
            logging::fatal().err(e).msg("failed scan input from stdin");
        }
    } else {
        let log_opts = p.get_string("log-opts");
        let c = match gitsrc::GitCmd::new_log(&source, &log_opts) {
            Ok(c) => c,
            Err(e) => {
                logging::fatal().err(e).msg("could not create Git cmd");
                unreachable!()
            }
        };
        let platform = match scm::platform_from_string(&p.get_string("platform")) {
            Ok(pl) => pl,
            Err(e) => {
                logging::fatal().err(e).send();
                unreachable!()
            }
        };
        let remote = gitsrc::new_remote_info(platform, &source);
        let r = run_git_scan(&mut d, c, remote);
        findings = r.0;
        err = r.1;
        if let Some(e) = &err {
            logging::error().err(e).msg("failed to scan Git repository");
        }
    }
    finding_summary_and_exit(&d, &findings, exit_code, start, err);
}

fn run_protect(p: &Parsed) {
    let start = Instant::now();
    let source = p.get_string("source");
    let table = init_config(p, &source);
    init_diagnostics(p);
    let cfg = load_config(p, &table);
    let mut d = build_detector(p, &cfg, &source);
    let exit_code = p.get_int("exit-code");
    let staged = p.get_bool("staged");
    let c = match gitsrc::GitCmd::new_diff(&source, staged) {
        Ok(c) => c,
        Err(e) => {
            logging::fatal().err(e).msg("could not create Git diff cmd");
            unreachable!()
        }
    };
    let remote = sources::RemoteInfo { platform: scm::Platform::None, url: String::new() };
    let (findings, err) = run_git_scan(&mut d, c, remote);
    if let Some(e) = &err {
        logging::error().err(e).msg("failed to scan Git repository");
    }
    finding_summary_and_exit(&d, &findings, exit_code, start, err);
}

fn completion_long(shell: &str) -> &'static str {
    match shell {
        "bash" => "Generate the autocompletion script for the bash shell.\n\nThis script depends on the 'bash-completion' package.\nIf it is not installed already, you can install it via your OS's package manager.\n\nTo load completions in your current shell session:\n\n\tsource <(gitleaks completion bash)\n\nTo load completions for every new session, execute once:\n\n#### Linux:\n\n\tgitleaks completion bash > /etc/bash_completion.d/gitleaks\n\n#### macOS:\n\n\tgitleaks completion bash > $(brew --prefix)/etc/bash_completion.d/gitleaks\n\nYou will need to start a new shell for this setup to take effect.\n",
        "zsh" => "Generate the autocompletion script for the zsh shell.\n\nIf shell completion is not already enabled in your environment you will need\nto enable it.  You can execute the following once:\n\n\techo \"autoload -U compinit; compinit\" >> ~/.zshrc\n\nTo load completions in your current shell session:\n\n\tsource <(gitleaks completion zsh)\n\nTo load completions for every new session, execute once:\n\n#### Linux:\n\n\tgitleaks completion zsh > \"${fpath[1]}/_gitleaks\"\n\n#### macOS:\n\n\tgitleaks completion zsh > $(brew --prefix)/share/zsh/site-functions/_gitleaks\n\nYou will need to start a new shell for this setup to take effect.\n",
        "fish" => "Generate the autocompletion script for the fish shell.\n\nTo load completions in your current shell session:\n\n\tgitleaks completion fish | source\n\nTo load completions for every new session, execute once:\n\n\tgitleaks completion fish > ~/.config/fish/completions/gitleaks.fish\n\nYou will need to start a new shell for this setup to take effect.\n",
        "powershell" => "Generate the autocompletion script for powershell.\n\nTo load completions in your current shell session:\n\n\tgitleaks completion powershell | Out-String | Invoke-Expression\n\nTo load completions for every new session, add the output of the above command\nto your powershell profile.\n",
        _ => "",
    }
}
