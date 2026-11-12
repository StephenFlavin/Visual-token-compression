//! Port of gitleaks' detect package.

pub mod codec;
pub mod style;
pub mod utils;

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use aho_corasick::AhoCorasick;

use crate::config::{AllowlistMatchCondition, Allowlist, Config, Required, Rule};
use crate::logging;
use crate::report::{Finding, Reporter, RequiredFinding};
use crate::sources::{self, Fragment};
use codec::{Decoder, EncodedSegment};

const GITLEAKS_ALLOW_SIGNATURE: &str = "gitleaks:allow";

pub struct Detector {
    pub config: Config,
    pub redact: u64,
    pub verbose: bool,
    pub max_decode_depth: i64,
    pub max_archive_depth: i64,
    pub max_target_mega_bytes: i64,
    pub follow_symlinks: bool,
    pub no_color: bool,
    pub ignore_gitleaks_allow: bool,
    commit_map: HashSet<String>,
    findings: Vec<Finding>,
    prefilter: Option<AhoCorasick>,
    prefilter_words: Vec<String>,
    baseline: Option<Vec<Finding>>,
    baseline_path: String,
    gitleaks_ignore: HashSet<String>,
    pub report_path: String,
    pub reporter: Option<Reporter>,
    pub total_bytes: u64,
}

pub enum SourceKind<'a, 'b> {
    Files(&'b sources::Files<'a>),
    File(sources::File<'a>),
    Git(&'b mut sources::git::Git<'a>),
}

impl Detector {
    pub fn new(cfg: Config) -> Detector {
        let words: Vec<String> = cfg.keywords.iter().cloned().collect();
        let prefilter = if words.is_empty() { None } else { AhoCorasick::new(&words).ok() };
        Detector {
            config: cfg,
            redact: 0,
            verbose: false,
            max_decode_depth: 0,
            max_archive_depth: 0,
            max_target_mega_bytes: 0,
            follow_symlinks: false,
            no_color: false,
            ignore_gitleaks_allow: false,
            commit_map: HashSet::new(),
            findings: Vec::new(),
            prefilter,
            prefilter_words: words,
            baseline: None,
            baseline_path: String::new(),
            gitleaks_ignore: HashSet::new(),
            report_path: String::new(),
            reporter: None,
            total_bytes: 0,
        }
    }

    pub fn add_gitleaks_ignore(&mut self, path: &str) -> Result<(), String> {
        logging::debug().str("path", path).msg("found .gitleaksignore file");
        let data = std::fs::read(path).map_err(|e| crate::config::go_os_error("open", path, &e))?;
        let content = String::from_utf8_lossy(&data);
        for raw_line in content.split('\n') {
            let line = crate::gostr::trim_space(raw_line);
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut s: Vec<String> = line.split(':').map(|x| x.to_string()).collect();
            match s.len() {
                3 => s[0] = s[0].replace('\\', "/"),
                4 => s[1] = s[1].replace('\\', "/"),
                _ => logging::warn().str("fingerprint", line).msg("Invalid .gitleaksignore entry"),
            }
            self.gitleaks_ignore.insert(s.join(":"));
        }
        Ok(())
    }

    pub fn add_baseline(&mut self, baseline_path: &str, source: &str) -> Result<(), String> {
        let mut bp = baseline_path.to_string();
        if !baseline_path.is_empty() {
            let abs_source = abs_path(source);
            let abs_baseline = abs_path(baseline_path);
            let rel = rel_path(&abs_source, &abs_baseline)?;
            let baseline = load_baseline(baseline_path)?;
            self.baseline = Some(baseline);
            bp = rel;
        }
        self.baseline_path = bp;
        Ok(())
    }

    /// DetectSource scans the given source and returns a list of findings
    pub fn detect_source(&mut self, source: SourceKind) -> (Vec<Finding>, Option<String>) {
        let is_git = matches!(source, SourceKind::Git(_));
        let res = {
            let mut cb = |fragment: Fragment, err: Option<String>| -> Result<(), String> {
                let mut ctx = logging::with();
                if !fragment.file_path.is_empty() {
                    ctx = ctx.str("path", &fragment.file_path);
                }
                if fragment.commit_sha.len() > 6 {
                    ctx = ctx.str("commit", &fragment.commit_sha[..7]);
                    self.commit_map.insert(fragment.commit_sha.clone());
                } else if !fragment.commit_sha.is_empty() {
                    ctx = ctx.str("commit", &fragment.commit_sha);
                    self.commit_map.insert(fragment.commit_sha.clone());
                    ctx.warn().msg("commit SHAs should be >= 7 characters long");
                }
                if let Some(e) = err {
                    ctx.error().err(e).send();
                    return Ok(());
                }
                if fragment.raw.is_empty() && fragment.file_path.is_empty() {
                    ctx.trace().msg("skipping empty fragment");
                    return Ok(());
                }
                let start = std::time::Instant::now();
                for finding in self.detect(&fragment) {
                    self.add_finding(finding);
                }
                if logging::get_level() <= logging::Level::Debug && start.elapsed().as_secs() >= 5 {
                    ctx.debug().msg("Taking longer than 5s to inspect fragment");
                }
                Ok(())
            };
            match source {
                SourceKind::Files(f) => f.fragments(&mut cb),
                SourceKind::File(f) => f.fragments(&mut cb),
                SourceKind::Git(g) => g.fragments(&mut cb),
            }
        };
        if is_git {
            logging::info().msg(&format!("{} commits scanned.", self.commit_map.len()));
            logging::debug().msg("Note: this number might be smaller than expected due to commits with no additions");
        }
        (self.findings.clone(), res.err())
    }

    /// DetectString scans the given string
    pub fn detect_string(&mut self, content: &str) -> Vec<Finding> {
        self.detect(&Fragment { raw: content.to_string(), ..Default::default() })
    }

    /// Detect scans the given fragment and returns a list of findings
    pub fn detect(&mut self, fragment: &Fragment) -> Vec<Finding> {
        match fragment.bytes_len {
            None => self.total_bytes += fragment.raw.len() as u64,
            Some(n) => self.total_bytes += n as u64,
        }
        let mut findings: Vec<Finding> = Vec::new();
        let logger = {
            let mut l = logging::with().str("path", &fragment.file_path);
            if !fragment.commit_sha.is_empty() {
                l = l.str("commit", &fragment.commit_sha);
            }
            l
        };

        if !fragment.file_path.is_empty()
            && (fragment.file_path == self.config.path
                || (!self.baseline_path.is_empty() && fragment.file_path == self.baseline_path))
        {
            logging::trace().msg("skipping file: matches config or baseline path");
            return findings;
        }
        if check_commit_or_path_allowed(&logger, fragment, &self.config.allowlists, "skipping file: global allowlist") {
            return findings;
        }

        let mut current_raw = fragment.raw.clone();
        let mut encoded_segments: Vec<Rc<EncodedSegment>> = Vec::new();
        let mut current_decode_depth = 0;
        let mut decoder = Decoder::new();
        let rule_ids = self.config.rule_ids_in_order();

        loop {
            if sources::ctx_done() {
                break;
            }
            let mut keywords: HashSet<String> = HashSet::new();
            let normalized = current_raw.to_lowercase();
            if let Some(pf) = &self.prefilter {
                for m in pf.find_overlapping_iter(&normalized) {
                    keywords.insert(self.prefilter_words[m.pattern().as_usize()].clone());
                }
            }
            for id in &rule_ids {
                let rule = match self.config.rules.get(id) {
                    Some(r) => r,
                    None => continue,
                };
                if rule.keywords.is_empty() {
                    findings.extend(self.detect_rule(fragment, &current_raw, rule, &encoded_segments));
                    continue;
                }
                for k in &rule.keywords {
                    if keywords.contains(&k.to_lowercase()) {
                        findings.extend(self.detect_rule(fragment, &current_raw, rule, &encoded_segments));
                        break;
                    }
                }
            }
            current_decode_depth += 1;
            if current_decode_depth > self.max_decode_depth {
                break;
            }
            let (nr, ns) = decoder.decode(&current_raw, &encoded_segments);
            current_raw = nr;
            encoded_segments = ns;
            if encoded_segments.is_empty() {
                break;
            }
        }
        utils::filter(findings, self.redact)
    }

    fn detect_rule(&self, fragment: &Fragment, current_raw: &str, r: &Rule, encoded_segments: &[Rc<EncodedSegment>]) -> Vec<Finding> {
        let mut findings: Vec<Finding> = Vec::new();
        let logger = {
            let mut l = logging::with().str("rule-id", &r.rule_id).str("path", &fragment.file_path);
            if !fragment.commit_sha.is_empty() {
                l = l.str("commit", &fragment.commit_sha);
            }
            l
        };

        if r.skip_report && !fragment.inherited_from_finding {
            return findings;
        }
        let rule_allowlists: Vec<Rc<Allowlist>> = Vec::new();
        let _ = rule_allowlists;
        if check_commit_or_path_allowed(&logger, fragment, &r.allowlists, "skipping file: rule allowlist") {
            return findings;
        }

        if let Some(path) = &r.path {
            let path_match = path.is_match(&fragment.file_path)
                || (!fragment.windows_file_path.is_empty() && path.is_match(&fragment.windows_file_path));
            if r.regex.is_none() && encoded_segments.is_empty() {
                if path_match {
                    let mut finding = Finding {
                        commit: fragment.commit_sha.clone(),
                        rule_id: r.rule_id.clone(),
                        description: r.description.clone(),
                        file: fragment.file_path.clone(),
                        symlink_file: fragment.symlink_file.clone(),
                        match_: format!("file detected: {}", fragment.file_path),
                        tags: r.tags.clone(),
                        ..Default::default()
                    };
                    if let Some(ci) = &fragment.commit_info {
                        finding.author = ci.author_name.clone();
                        finding.date = ci.date.clone();
                        finding.email = ci.author_email.clone();
                        finding.link = utils::create_scm_link(&ci.remote, &finding);
                        finding.message = ci.message.clone();
                    }
                    findings.push(finding);
                    return findings;
                }
            } else if !path_match {
                return findings;
            }
        }

        let regex = match &r.regex {
            Some(re) => re,
            None => return findings,
        };

        if self.max_target_mega_bytes > 0 {
            let raw_length = (current_raw.len() / 1_000_000) as i64;
            if raw_length > self.max_target_mega_bytes {
                logger
                    .debug()
                    .int("size", raw_length)
                    .int("max-size", self.max_target_mega_bytes)
                    .msg("skipping fragment: size");
                return findings;
            }
        }

        let matches = regex.find_all_index(current_raw);
        if matches.is_empty() {
            return findings;
        }

        let newline_indices: Vec<(usize, usize)> =
            fragment.raw.match_indices('\n').map(|(i, _)| (i, i + 1)).collect();

        for (ms, me) in matches {
            let secret_full = crate::gostr::bslice(current_raw, ms, me);
            let secret = secret_full.trim_matches('\n').to_string();
            let mut match_index = (ms as i64, me as i64);
            let mut meta_tags: Vec<String> = Vec::new();
            let mut current_line = String::new();

            if !encoded_segments.is_empty() {
                let segments = codec::segments_with_decoded_overlap(encoded_segments, match_index.0, match_index.1);
                if segments.is_empty() {
                    continue;
                }
                match_index = codec::adjust_match_index(&segments, match_index);
                meta_tags.extend(codec::tags(&segments));
                current_line = codec::current_line(&segments, current_raw);
            } else {
                match_index.1 = match_index.0 + secret.len() as i64;
            }

            let mut loc = utils::location(&newline_indices, &fragment.raw, match_index);
            if match_index.1 > loc.end_line_index {
                loc.end_line_index = match_index.1;
            }

            let mut tags = r.tags.clone();
            tags.extend(meta_tags);
            let mut finding = Finding {
                commit: fragment.commit_sha.clone(),
                rule_id: r.rule_id.clone(),
                description: r.description.clone(),
                start_line: fragment.start_line + loc.start_line,
                end_line: fragment.start_line + loc.end_line,
                start_column: loc.start_column,
                end_column: loc.end_column,
                line: crate::gostr::bslice(&fragment.raw, loc.start_line_index.max(0) as usize, loc.end_line_index.max(0) as usize),
                match_: secret.clone(),
                secret: secret.clone(),
                file: fragment.file_path.clone(),
                symlink_file: fragment.symlink_file.clone(),
                tags,
                ..Default::default()
            };
            if let Some(ci) = &fragment.commit_info {
                finding.author = ci.author_name.clone();
                finding.date = ci.date.clone();
                finding.email = ci.author_email.clone();
                finding.link = utils::create_scm_link(&ci.remote, &finding);
                finding.message = ci.message.clone();
            }
            if !self.ignore_gitleaks_allow && finding.line.contains(GITLEAKS_ALLOW_SIGNATURE) {
                logger.trace().str("finding", &finding.secret).msg("skipping finding: 'gitleaks:allow' signature");
                continue;
            }
            if current_line.is_empty() {
                current_line = finding.line.clone();
            }

            if let Some(groups) = regex.find_submatch(&finding.secret) {
                if groups.len() >= 2 {
                    if r.secret_group > 0 {
                        if groups.len() as i64 <= r.secret_group {
                            continue;
                        }
                        finding.secret = groups[r.secret_group as usize].clone();
                    } else {
                        for s in &groups[1..] {
                            if !s.is_empty() {
                                finding.secret = s.clone();
                                break;
                            }
                        }
                    }
                }
            }

            let entropy = utils::shannon_entropy(&finding.secret);
            finding.entropy = entropy as f32;
            if r.entropy != 0.0 && entropy <= r.entropy {
                logger
                    .trace()
                    .str("finding", &finding.secret)
                    .float32("entropy", finding.entropy)
                    .msg("skipping finding: low entropy");
                continue;
            }

            if check_finding_allowed(&logger, &finding, fragment, &current_line, &self.config.allowlists, "skipping finding: global allowlist") {
                continue;
            }
            if check_finding_allowed(&logger, &finding, fragment, &current_line, &r.allowlists, "skipping finding: rule allowlist") {
                continue;
            }
            findings.push(finding);
        }

        if fragment.inherited_from_finding || r.required_rules.is_empty() {
            return findings;
        }
        self.process_required_rules(fragment, current_raw, r, encoded_segments, findings, &logger)
    }

    fn process_required_rules(
        &self,
        fragment: &Fragment,
        current_raw: &str,
        r: &Rule,
        encoded_segments: &[Rc<EncodedSegment>],
        primary: Vec<Finding>,
        logger: &logging::Ctx,
    ) -> Vec<Finding> {
        if primary.is_empty() {
            logger.debug().msg("no primary findings to process for required rules");
            return primary;
        }
        let mut all_required: HashMap<String, Vec<Finding>> = HashMap::new();
        for rr in &r.required_rules {
            let rule = match self.config.rules.get(&rr.rule_id) {
                Some(x) => x,
                None => {
                    logger.error().str("rule-id", &rr.rule_id).msg("required rule not found in config");
                    continue;
                }
            };
            let mut inherited = fragment.clone();
            inherited.inherited_from_finding = true;
            let found = self.detect_rule(&inherited, current_raw, rule, encoded_segments);
            logger
                .debug()
                .str("rule-id", &rr.rule_id)
                .int("findings", found.len() as i64)
                .msg("collected required rule findings");
            all_required.insert(rr.rule_id.clone(), found);
        }

        let mut final_findings = Vec::new();
        for pf in &primary {
            let mut required: Vec<RequiredFinding> = Vec::new();
            for rr in &r.required_rules {
                let found = match all_required.get(&rr.rule_id) {
                    Some(f) => f,
                    None => continue,
                };
                for rf in found {
                    if within_proximity(pf, rf, rr) {
                        required.push(RequiredFinding {
                            rule_id: rf.rule_id.clone(),
                            start_line: rf.start_line,
                            end_line: rf.end_line,
                            start_column: rf.start_column,
                            end_column: rf.end_column,
                            line: rf.line.clone(),
                            match_: rf.match_.clone(),
                            secret: rf.secret.clone(),
                        });
                    }
                }
            }
            if !required.is_empty() && has_all_required_rules(&required, &r.required_rules) {
                let mut nf = pf.clone();
                let count = required.len();
                nf.add_required_findings(required);
                final_findings.push(nf);
                logger
                    .debug()
                    .str("primary-rule", &r.rule_id)
                    .int("primary-line", pf.start_line)
                    .int("auxiliary-count", count as i64)
                    .msg("multi-part rule satisfied");
            }
        }
        final_findings
    }

    /// AddFinding synchronously adds a finding to the findings slice
    pub fn add_finding(&mut self, mut finding: Finding) {
        let global_fp = format!("{}:{}:{}", finding.file, finding.rule_id, finding.start_line);
        if !finding.commit.is_empty() {
            finding.fingerprint = format!("{}:{}:{}:{}", finding.commit, finding.file, finding.rule_id, finding.start_line);
        } else {
            finding.fingerprint = global_fp.clone();
        }
        let logger = logging::with().str("finding", &finding.secret);
        if self.gitleaks_ignore.contains(&global_fp) {
            logger.debug().str("fingerprint", &global_fp).msg("skipping finding: global fingerprint");
            return;
        } else if !finding.commit.is_empty() && self.gitleaks_ignore.contains(&finding.fingerprint) {
            logger.debug().str("fingerprint", &finding.fingerprint).msg("skipping finding: fingerprint");
            return;
        }
        if let Some(b) = &self.baseline {
            if !is_new(&finding, self.redact, b) {
                logger.debug().str("fingerprint", &finding.fingerprint).msg("skipping finding: baseline");
                return;
            }
        }
        if self.verbose {
            utils::print_finding(&finding, self.no_color);
        }
        self.findings.push(finding);
    }

    pub fn findings(&self) -> &[Finding] {
        &self.findings
    }
}

fn has_all_required_rules(aux: &[RequiredFinding], required: &[Required]) -> bool {
    let found: HashSet<&str> = aux.iter().map(|a| a.rule_id.as_str()).collect();
    required.iter().all(|r| found.contains(r.rule_id.as_str()))
}

fn within_proximity(primary: &Finding, required: &Finding, rr: &Required) -> bool {
    if rr.within_lines.is_none() && rr.within_columns.is_none() {
        return true;
    }
    if let Some(wl) = rr.within_lines {
        if (primary.start_line - required.start_line).abs() > wl {
            return false;
        }
    }
    if let Some(wc) = rr.within_columns {
        if (primary.start_column - required.start_column).abs() > wc {
            return false;
        }
    }
    true
}

/// checkCommitOrPathAllowed evaluates |fragment| against all provided |allowlists|.
fn check_commit_or_path_allowed(logger: &logging::Ctx, fragment: &Fragment, allowlists: &[std::sync::Arc<Allowlist>], msg: &str) -> bool {
    if fragment.file_path.is_empty() && fragment.commit_sha.is_empty() {
        return false;
    }
    for a in allowlists {
        let (commit_allowed, _) = a.commit_allowed(&fragment.commit_sha);
        let path_allowed = a.path_allowed(&fragment.file_path)
            || (!fragment.windows_file_path.is_empty() && a.path_allowed(&fragment.windows_file_path));
        let is_allowed;
        if a.match_condition == AllowlistMatchCondition::And {
            let mut checks = Vec::new();
            if !a.commits.is_empty() {
                checks.push(commit_allowed);
            }
            if !a.paths.is_empty() {
                checks.push(path_allowed);
            }
            if !a.regexes.is_empty() || !a.stop_words.is_empty() {
                continue;
            }
            is_allowed = checks.iter().all(|c| *c);
        } else {
            is_allowed = commit_allowed || path_allowed;
        }
        if is_allowed {
            let mut ev = logger.trace().str("condition", a.match_condition.as_str());
            if commit_allowed {
                ev = ev.bool("allowed-commit", commit_allowed);
            }
            if path_allowed {
                ev = ev.bool("allowed-path", path_allowed);
            }
            ev.msg(msg);
            return true;
        }
    }
    false
}

/// checkFindingAllowed evaluates |finding| against all provided |allowlists|.
fn check_finding_allowed(
    logger: &logging::Ctx,
    finding: &Finding,
    fragment: &Fragment,
    current_line: &str,
    allowlists: &[std::sync::Arc<Allowlist>],
    msg: &str,
) -> bool {
    for a in allowlists {
        let target = match a.regex_target.as_str() {
            "match" => &finding.match_,
            "line" => current_line,
            _ => &finding.secret,
        };
        let regex_allowed = a.regex_allowed(target);
        let (contains_stopword, word) = a.contains_stop_word(&finding.secret);
        let mut commit_allowed = false;
        let mut commit = String::new();
        let mut path_allowed = false;
        let is_allowed;
        if a.match_condition == AllowlistMatchCondition::And {
            let mut checks = Vec::new();
            if !a.commits.is_empty() {
                let (c, s) = a.commit_allowed(&fragment.commit_sha);
                commit_allowed = c;
                commit = s;
                checks.push(commit_allowed);
            }
            if !a.paths.is_empty() {
                path_allowed = a.path_allowed(&fragment.file_path)
                    || (!fragment.windows_file_path.is_empty() && a.path_allowed(&fragment.windows_file_path));
                checks.push(path_allowed);
            }
            if !a.regexes.is_empty() {
                checks.push(regex_allowed);
            }
            if !a.stop_words.is_empty() {
                checks.push(contains_stopword);
            }
            is_allowed = checks.iter().all(|c| *c);
        } else {
            is_allowed = regex_allowed || contains_stopword;
        }
        if is_allowed {
            let mut ev = logger.trace().str("finding", &finding.secret).str("condition", a.match_condition.as_str());
            if commit_allowed {
                ev = ev.str("allowed-commit", &commit);
            }
            if path_allowed {
                ev = ev.bool("allowed-path", path_allowed);
            }
            if regex_allowed {
                ev = ev.bool("allowed-regex", regex_allowed);
            }
            if contains_stopword {
                ev = ev.str("allowed-stopword", &word);
            }
            ev.msg(msg);
            return true;
        }
    }
    false
}

// ------------------------------------------------------------------ baseline

pub fn is_new(finding: &Finding, redact: u64, baseline: &[Finding]) -> bool {
    for b in baseline {
        if finding.rule_id == b.rule_id
            && finding.description == b.description
            && finding.start_line == b.start_line
            && finding.end_line == b.end_line
            && finding.start_column == b.start_column
            && finding.end_column == b.end_column
            && (redact > 0 || (finding.match_ == b.match_ && finding.secret == b.secret))
            && finding.file == b.file
            && finding.commit == b.commit
            && finding.author == b.author
            && finding.email == b.email
            && finding.date == b.date
            && finding.message == b.message
            && finding.entropy == b.entropy
        {
            return false;
        }
    }
    true
}

pub fn load_baseline(path: &str) -> Result<Vec<Finding>, String> {
    let data = std::fs::read(path).map_err(|_| format!("could not open {}", path))?;
    let v: serde_json::Value =
        serde_json::from_slice(&data).map_err(|_| format!("the format of the file {} is not supported", path))?;
    match v {
        serde_json::Value::Array(a) => {
            let mut out = Vec::new();
            for x in a {
                match Finding::from_json(&x) {
                    Some(f) => out.push(f),
                    None => {
                        if !x.is_null() {
                            return Err(format!("the format of the file {} is not supported", path));
                        }
                        out.push(Finding::default());
                    }
                }
            }
            Ok(out)
        }
        serde_json::Value::Null => Ok(Vec::new()),
        _ => Err(format!("the format of the file {} is not supported", path)),
    }
}

pub fn abs_path(p: &str) -> String {
    if p.starts_with('/') {
        return sources::clean_path(p);
    }
    let cwd = std::env::current_dir().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
    sources::join_path(&cwd, p)
}

/// filepath.Rel for absolute, cleaned paths
pub fn rel_path(base: &str, targ: &str) -> Result<String, String> {
    if base == targ {
        return Ok(".".to_string());
    }
    let b: Vec<&str> = if base == "/" { vec![] } else { base.trim_start_matches('/').split('/').collect() };
    let t: Vec<&str> = if targ == "/" { vec![] } else { targ.trim_start_matches('/').split('/').collect() };
    let mut i = 0;
    while i < b.len() && i < t.len() && b[i] == t[i] {
        i += 1;
    }
    let mut parts: Vec<String> = Vec::new();
    for _ in i..b.len() {
        parts.push("..".to_string());
    }
    for x in &t[i..] {
        parts.push(x.to_string());
    }
    Ok(parts.join("/"))
}
