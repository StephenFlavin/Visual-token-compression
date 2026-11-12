use std::collections::HashMap;
use std::io::Write;

use super::style;
use crate::cmd::scm::Platform;
use crate::gostr;
use crate::logging;
use crate::report::Finding;
use crate::sources::{ext, RemoteInfo, INNER_PATH_SEPARATOR};

pub fn create_scm_link(remote: &RemoteInfo, finding: &Finding) -> String {
    if remote.platform == Platform::Unknown || remote.platform == Platform::None || finding.commit.is_empty() {
        return String::new();
    }
    let (file_path, has_inner) = match finding.file.find(INNER_PATH_SEPARATOR) {
        Some(i) => (&finding.file[..i], true),
        None => (finding.file.as_str(), false),
    };
    // strings.NewReplacer(" ", "%20", "%", "%25")
    let mut fp = String::new();
    for c in file_path.chars() {
        match c {
            ' ' => fp.push_str("%20"),
            '%' => fp.push_str("%25"),
            c => fp.push(c),
        }
    }
    let file_path = fp;
    let (sl, el) = (finding.start_line, finding.end_line);
    match remote.platform {
        Platform::GitHub => {
            let mut link = format!("{}/blob/{}/{}", remote.url, finding.commit, file_path);
            if has_inner {
                return link;
            }
            let e = ext(&file_path).to_lowercase();
            if e == ".ipynb" || e == ".md" {
                link.push_str("?plain=1");
            }
            if sl != 0 {
                link.push_str(&format!("#L{}", sl));
            }
            if el != sl {
                link.push_str(&format!("-L{}", el));
            }
            link
        }
        Platform::GitLab => {
            let mut link = format!("{}/blob/{}/{}", remote.url, finding.commit, file_path);
            if has_inner {
                return link;
            }
            if sl != 0 {
                link.push_str(&format!("#L{}", sl));
            }
            if el != sl {
                link.push_str(&format!("-{}", el));
            }
            link
        }
        Platform::AzureDevOps => {
            let mut link = format!("{}/commit/{}?path=/{}", remote.url, finding.commit, file_path);
            if has_inner {
                return link;
            }
            if sl != 0 {
                link.push_str(&format!("&line={}", sl));
            }
            if el != sl {
                link.push_str(&format!("&lineEnd={}", el));
            }
            link.push_str("&lineStartColumn=1&lineEndColumn=10000000&type=2&lineStyle=plain&_a=files");
            link
        }
        Platform::Gitea => {
            let mut link = format!("{}/src/commit/{}/{}", remote.url, finding.commit, file_path);
            if has_inner {
                return link;
            }
            let e = ext(&file_path).to_lowercase();
            if e == ".ipynb" || e == ".md" {
                link.push_str("?display=source");
            }
            if sl != 0 {
                link.push_str(&format!("#L{}", sl));
            }
            if el != sl {
                link.push_str(&format!("-L{}", el));
            }
            link
        }
        Platform::Bitbucket => {
            let mut link = format!("{}/src/{}/{}", remote.url, finding.commit, file_path);
            if has_inner {
                return link;
            }
            if sl != 0 {
                link.push_str(&format!("#lines-{}", sl));
            }
            if el != sl {
                link.push_str(&format!(":{}", el));
            }
            link
        }
        _ => String::new(),
    }
}

/// shannonEntropy calculates the entropy of data
pub fn shannon_entropy(data: &str) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<char, usize> = HashMap::new();
    let mut order: Vec<char> = Vec::new();
    for c in data.chars() {
        let e = counts.entry(c).or_insert(0);
        if *e == 0 {
            order.push(c);
        }
        *e += 1;
    }
    let inv = 1.0 / data.len() as f64;
    let mut entropy = 0.0f64;
    for c in order {
        let freq = counts[&c] as f64 * inv;
        entropy -= freq * freq.log2();
    }
    entropy
}

/// filter will dedupe and redact findings
pub fn filter(findings: Vec<Finding>, redact: u64) -> Vec<Finding> {
    let mut ret = Vec::new();
    for f in findings.iter() {
        let mut include = true;
        if f.rule_id.to_lowercase().contains("generic") {
            for fp in findings.iter() {
                if f.start_line == fp.start_line
                    && f.commit == fp.commit
                    && f.rule_id != fp.rule_id
                    && fp.secret.contains(f.secret.as_str())
                    && !fp.rule_id.to_lowercase().contains("generic")
                {
                    let ev = logging::trace();
                    if ev.is_enabled() {
                        let gm = if f.secret.is_empty() { f.match_.clone() } else { f.match_.replace(&f.secret, "REDACTED") };
                        let bm = if fp.secret.is_empty() { fp.match_.clone() } else { fp.match_.replace(&fp.secret, "REDACTED") };
                        ev.msg(&format!(
                            "skipping {} finding ({}), {} rule takes precedence ({})",
                            f.rule_id, gm, fp.rule_id, bm
                        ));
                    }
                    include = false;
                    break;
                }
            }
        }
        let mut f = f.clone();
        if redact > 0 {
            f.redact(redact);
        }
        if include {
            ret.push(f);
        }
    }
    ret
}

pub fn print_finding(f: &Finding, no_color: bool) {
    let mut f = f.clone();
    f.line = gostr::trim_space(&f.line).to_string();
    f.secret = gostr::trim_space(&f.secret).to_string();
    f.match_ = gostr::trim_space(&f.match_).to_string();

    let is_file_match = f.match_.starts_with("file detected:");
    let mut skip_color = no_color;
    let mut finding = String::new();
    let mut secret_styled = String::new();
    const YELLOW: (u8, u8, u8) = (0xf5, 0xd4, 0x45);
    const ORANGE: (u8, u8, u8) = (0xf0, 0x5c, 0x07);

    if !is_file_match {
        let mut match_in_line = f.line.find(&f.match_).map(|i| i as i64).unwrap_or(-1);
        let secret_in_match = f.match_.find(&f.secret).unwrap_or(0);
        skip_color = false;
        if match_in_line == -1 || no_color {
            skip_color = true;
            match_in_line = 0;
        }
        let mil = match_in_line as usize;
        let mut start = gostr::bslice(&f.line, 0, mil);
        if mil > 20 {
            start = format!("...{}", gostr::bslice(&f.line, mil - 20, mil));
        }
        let match_beginning = style::render(&gostr::bslice(&f.match_, 0, secret_in_match), false, false, Some(YELLOW));
        secret_styled = style::render(&f.secret, true, true, Some(ORANGE));
        let match_end = style::render(
            &gostr::bslice(&f.match_, secret_in_match + f.secret.len(), f.match_.len()),
            false,
            false,
            Some(YELLOW),
        );
        let mut line_end_idx = mil + f.match_.len();
        if f.line.len() as i64 - 1 <= line_end_idx as i64 {
            line_end_idx = f.line.len();
        }
        let mut line_end = gostr::bslice(&f.line, line_end_idx, f.line.len());
        if f.secret.len() > 100 {
            secret_styled = style::render(&format!("{}...", gostr::bslice(&f.secret, 0, 100)), true, true, Some(ORANGE));
        }
        if line_end.len() > 20 {
            line_end = format!("{}...", gostr::bslice(&line_end, 0, 20));
        }
        let start_trimmed = start.trim_start_matches(' ');
        let start_trimmed = start_trimmed.strip_prefix('\n').unwrap_or(start_trimmed);
        finding = format!("{}{}{}{}{}\n", start_trimmed, match_beginning, secret_styled, match_end, line_end);
    }

    let mut out = String::new();
    if skip_color || is_file_match {
        out.push_str(&format!("{:<12} {}\n", "Finding:", f.match_));
        out.push_str(&format!("{:<12} {}\n", "Secret:", f.secret));
    } else {
        out.push_str(&format!("{:<12} {}", "Finding:", finding));
        out.push_str(&format!("{:<12} {}\n", "Secret:", secret_styled));
    }
    out.push_str(&format!("{:<12} {}\n", "RuleID:", f.rule_id));
    out.push_str(&format!("{:<12} {:.6}\n", "Entropy:", f.entropy));

    if f.file.is_empty() {
        f.print_required_findings(&mut out);
        out.push('\n');
        emit(&out);
        return;
    }
    if !f.tags.is_empty() {
        out.push_str(&format!("{:<12} [{}]\n", "Tags:", f.tags.join(" ")));
    }
    out.push_str(&format!("{:<12} {}\n", "File:", f.file));
    out.push_str(&format!("{:<12} {}\n", "Line:", f.start_line));
    if f.commit.is_empty() {
        out.push_str(&format!("{:<12} {}\n", "Fingerprint:", f.fingerprint));
        f.print_required_findings(&mut out);
        out.push('\n');
        emit(&out);
        return;
    }
    out.push_str(&format!("{:<12} {}\n", "Commit:", f.commit));
    out.push_str(&format!("{:<12} {}\n", "Author:", f.author));
    out.push_str(&format!("{:<12} {}\n", "Email:", f.email));
    out.push_str(&format!("{:<12} {}\n", "Date:", f.date));
    out.push_str(&format!("{:<12} {}\n", "Fingerprint:", f.fingerprint));
    if !f.link.is_empty() {
        out.push_str(&format!("{:<12} {}\n", "Link:", f.link));
    }
    f.print_required_findings(&mut out);
    out.push('\n');
    emit(&out);
}

fn emit(s: &str) {
    let stdout = std::io::stdout();
    let mut h = stdout.lock();
    let _ = h.write_all(s.as_bytes());
    let _ = h.flush();
}

pub struct Location {
    pub start_line: i64,
    pub end_line: i64,
    pub start_column: i64,
    pub end_column: i64,
    pub start_line_index: i64,
    pub end_line_index: i64,
}

pub fn location(newline_indices: &[(usize, usize)], raw: &str, match_index: (i64, i64)) -> Location {
    let mut prev_new_line: i64 = 0;
    let mut loc = Location { start_line: 0, end_line: 0, start_column: 0, end_column: 0, start_line_index: 0, end_line_index: 0 };
    let mut line_set = false;
    let mut last_line_num: i64 = 0;
    let (start, end) = match_index;
    let default = [(raw.len(), raw.len() + 1)];
    let indices: &[(usize, usize)] = if newline_indices.is_empty() { &default } else { newline_indices };
    for (line_num, pair) in indices.iter().enumerate() {
        let line_num = line_num as i64;
        last_line_num = line_num;
        let nl = pair.0 as i64;
        if prev_new_line <= start && start < nl {
            line_set = true;
            loc.start_line = line_num;
            loc.end_line = line_num;
            loc.start_column = (start - prev_new_line) + 1;
            loc.start_line_index = prev_new_line;
            loc.end_line_index = nl;
        }
        if prev_new_line < end && end <= nl {
            loc.end_line = line_num;
            loc.end_column = end - prev_new_line;
            loc.end_line_index = nl;
        }
        prev_new_line = pair.0 as i64;
    }
    if !line_set {
        loc.start_column = (start - prev_new_line) + 1;
        loc.end_column = end - prev_new_line;
        loc.start_line = last_line_num + 1;
        loc.end_line = last_line_num + 1;
        let b = raw.as_bytes();
        let mut i: i64 = 0;
        while ((end + i) as usize) < b.len() {
            let c = b[(end + i) as usize];
            if c == b'\n' || c == b'\r' {
                break;
            }
            i += 1;
        }
        loc.end_line_index = end + i;
    }
    loc
}
