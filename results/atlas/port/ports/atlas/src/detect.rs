// Port of the detect package: the Detector that applies rules to fragments.

use crate::codec::{self, Decoder, EncodedSegment};
use crate::config::{Allowlist, Config, MatchCondition, Required, Rule};
use crate::logging;
use crate::report::{Finding, RequiredFinding, Reporter};
use crate::sources::{Fragment, RemoteInfo, INNER_PATH_SEPARATOR};
use crate::util;
use aho_corasick::AhoCorasick;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

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

impl Detector {
    pub fn new(cfg: Config) -> Detector {
        let words: Vec<String> = cfg.keywords.iter().cloned().collect();
        let prefilter = if words.is_empty() {
            None
        } else {
            AhoCorasick::builder().build(&words).ok()
        };
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
            findings: vec![],
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
        let content = std::fs::read(path).map_err(|e| format!("open {}: {}", path, crate::io_err_str(&e)))?;
        let content = String::from_utf8_lossy(&content);
        for raw_line in content.split('\n') {
            let raw_line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
            let line = util::trim_space(raw_line);
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
            let abs_source = util::abs(source);
            let abs_baseline = util::abs(baseline_path);
            let relative = util::rel(&abs_source, &abs_baseline);
            let baseline = load_baseline(baseline_path)?;
            self.baseline = Some(baseline);
            bp = relative;
        }
        self.baseline_path = bp;
        Ok(())
    }

    pub fn add_commit(&mut self, c: &str) {
        self.commit_map.insert(c.to_string());
    }

    pub fn commits_scanned(&self) -> usize {
        self.commit_map.len()
    }

    pub fn findings(&self) -> &Vec<Finding> {
        &self.findings
    }

    /// The callback used by sources (DetectSource body).
    pub fn handle_fragment(&mut self, fragment: Fragment, err: Option<String>) {
        if fragment.commit_sha.len() > 6 {
            let sha = fragment.commit_sha.clone();
            self.add_commit(&sha);
        } else if !fragment.commit_sha.is_empty() {
            let sha = fragment.commit_sha.clone();
            self.add_commit(&sha);
            let mut e = logging::warn();
            if !fragment.file_path.is_empty() {
                e = e.str("path", &fragment.file_path);
            }
            e.str("commit", &fragment.commit_sha).msg("commit SHAs should be >= 7 characters long");
        }
        if let Some(err) = err {
            let mut e = logging::error();
            if !fragment.file_path.is_empty() {
                e = e.str("path", &fragment.file_path);
            }
            if fragment.commit_sha.len() > 6 {
                e = e.str("commit", &fragment.commit_sha[..7]);
            } else if !fragment.commit_sha.is_empty() {
                e = e.str("commit", &fragment.commit_sha);
            }
            e.err(&err).send();
            return;
        }
        if fragment.raw.is_empty() && fragment.file_path.is_empty() {
            return;
        }
        let findings = self.detect(&fragment);
        for f in findings {
            self.add_finding(f);
        }
    }

    pub fn detect(&mut self, fragment: &Fragment) -> Vec<Finding> {
        match fragment.bytes_len {
            None => self.total_bytes += fragment.raw.len() as u64,
            Some(n) => self.total_bytes += n as u64,
        }
        let mut findings: Vec<Finding> = vec![];
        if !fragment.file_path.is_empty()
            && (fragment.file_path == self.config.path
                || (!self.baseline_path.is_empty() && fragment.file_path == self.baseline_path))
        {
            logging::trace().msg("skipping file: matches config or baseline path");
            return findings;
        }
        if check_commit_or_path_allowed(fragment, &self.config.allowlists) {
            logging::trace().str("path", &fragment.file_path).msg("skipping file: global allowlist");
            return findings;
        }

        let mut current_raw = fragment.raw.clone();
        let mut encoded_segments: Vec<Rc<EncodedSegment>> = vec![];
        let mut current_decode_depth = 0i64;
        let mut decoder = Decoder::new();
        let rule_ids: Vec<String> = self.config.rules.keys().cloned().collect();

        loop {
            let normalized = current_raw.to_lowercase();
            let mut keywords: HashSet<&str> = HashSet::new();
            if let Some(pf) = &self.prefilter {
                for m in pf.find_overlapping_iter(&normalized) {
                    keywords.insert(self.prefilter_words[m.pattern().as_usize()].as_str());
                }
            }
            for id in &rule_ids {
                let rule = &self.config.rules[id];
                if rule.keywords.is_empty() {
                    let r = self.detect_rule(fragment, &current_raw, rule, &encoded_segments);
                    findings.extend(r);
                    continue;
                }
                for k in &rule.keywords {
                    if keywords.contains(k.to_lowercase().as_str()) {
                        let r = self.detect_rule(fragment, &current_raw, rule, &encoded_segments);
                        findings.extend(r);
                        break;
                    }
                }
            }
            current_decode_depth += 1;
            if current_decode_depth > self.max_decode_depth {
                break;
            }
            let (nr, segs) = decoder.decode(&current_raw, &encoded_segments);
            current_raw = nr;
            encoded_segments = segs;
            if encoded_segments.is_empty() {
                break;
            }
        }
        filter(findings, self.redact)
    }

    fn detect_rule(&self, fragment: &Fragment, current_raw: &str, r: &Rule, encoded_segments: &[Rc<EncodedSegment>]) -> Vec<Finding> {
        let mut findings = vec![];
        if r.skip_report && !fragment.inherited_from_finding {
            return findings;
        }
        if check_commit_or_path_allowed(fragment, &r.allowlists) {
            logging::trace()
                .str("rule-id", &r.rule_id)
                .str("path", &fragment.file_path)
                .msg("skipping file: rule allowlist");
            return findings;
        }
        let path_match = |p: &crate::goregex::GoRegex| {
            p.is_match(&fragment.file_path)
                || (!fragment.windows_file_path.is_empty() && p.is_match(&fragment.windows_file_path))
        };
        if let Some(p) = &r.path {
            if r.regex.is_none() && encoded_segments.is_empty() {
                if path_match(p) {
                    let mut f = Finding {
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
                        f.author = ci.author_name.clone();
                        f.date = ci.date.clone();
                        f.email = ci.author_email.clone();
                        f.link = create_scm_link(&ci.remote, &f);
                        f.message = ci.message.clone();
                    }
                    findings.push(f);
                    return findings;
                }
            } else if !path_match(p) {
                return findings;
            }
        }
        let regex = match &r.regex {
            None => return findings,
            Some(re) => re,
        };
        if self.max_target_mega_bytes > 0 {
            let raw_len = (current_raw.len() / 1_000_000) as i64;
            if raw_len > self.max_target_mega_bytes {
                logging::debug()
                    .int("size", raw_len)
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
            fragment.raw.bytes().enumerate().filter(|(_, b)| *b == b'\n').map(|(i, _)| (i, i + 1)).collect();

        for mi in matches {
            let secret = current_raw[mi.0..mi.1].trim_matches('\n').to_string();
            let mut match_index = mi;
            let mut meta_tags: Vec<String> = vec![];
            let mut current_line = String::new();
            if !encoded_segments.is_empty() {
                let segments = codec::segments_with_decoded_overlap(encoded_segments, mi.0, mi.1);
                if segments.is_empty() {
                    continue;
                }
                match_index = codec::adjust_match_index(&segments, match_index);
                meta_tags.extend(codec::tags(&segments));
                current_line = codec::current_line(&segments, current_raw);
            } else {
                match_index.1 = match_index.0 + secret.len();
            }
            let mut loc = location(&newline_indices, &fragment.raw, match_index);
            if match_index.1 > loc.end_line_index {
                loc.end_line_index = match_index.1;
            }
            let mut tags = r.tags.clone();
            tags.extend(meta_tags);
            let mut f = Finding {
                commit: fragment.commit_sha.clone(),
                rule_id: r.rule_id.clone(),
                description: r.description.clone(),
                start_line: fragment.start_line + loc.start_line,
                end_line: fragment.start_line + loc.end_line,
                start_column: loc.start_column,
                end_column: loc.end_column,
                line: util::bslice(&fragment.raw, loc.start_line_index, loc.end_line_index),
                match_: secret.clone(),
                secret: secret.clone(),
                file: fragment.file_path.clone(),
                symlink_file: fragment.symlink_file.clone(),
                tags,
                ..Default::default()
            };
            if let Some(ci) = &fragment.commit_info {
                f.author = ci.author_name.clone();
                f.date = ci.date.clone();
                f.email = ci.author_email.clone();
                f.link = create_scm_link(&ci.remote, &f);
                f.message = ci.message.clone();
            }
            if !self.ignore_gitleaks_allow && f.line.contains(GITLEAKS_ALLOW_SIGNATURE) {
                logging::trace().str("finding", &f.secret).msg("skipping finding: 'gitleaks:allow' signature");
                continue;
            }
            if current_line.is_empty() {
                current_line = f.line.clone();
            }
            if let Some(groups) = regex.find_submatch(&f.secret) {
                if groups.len() >= 2 {
                    if r.secret_group > 0 {
                        if groups.len() as i64 <= r.secret_group {
                            continue;
                        }
                        f.secret = groups[r.secret_group as usize].clone();
                    } else {
                        for g in &groups[1..] {
                            if !g.is_empty() {
                                f.secret = g.clone();
                                break;
                            }
                        }
                    }
                }
            }
            let entropy = shannon_entropy(&f.secret);
            f.entropy = entropy as f32;
            if r.entropy != 0.0 && entropy <= r.entropy {
                logging::trace()
                    .str("finding", &f.secret)
                    .float("entropy", f.entropy as f64)
                    .msg("skipping finding: low entropy");
                continue;
            }
            if check_finding_allowed(&f, fragment, &current_line, &self.config.allowlists) {
                logging::trace().str("finding", &f.secret).msg("skipping finding: global allowlist");
                continue;
            }
            if check_finding_allowed(&f, fragment, &current_line, &r.allowlists) {
                logging::trace().str("finding", &f.secret).msg("skipping finding: rule allowlist");
                continue;
            }
            findings.push(f);
        }

        if fragment.inherited_from_finding || r.required_rules.is_empty() {
            return findings;
        }
        self.process_required_rules(fragment, current_raw, r, encoded_segments, findings)
    }

    fn process_required_rules(
        &self,
        fragment: &Fragment,
        current_raw: &str,
        r: &Rule,
        encoded_segments: &[Rc<EncodedSegment>],
        primary: Vec<Finding>,
    ) -> Vec<Finding> {
        if primary.is_empty() {
            logging::debug().msg("no primary findings to process for required rules");
            return primary;
        }
        let mut all_required: HashMap<String, Vec<Finding>> = HashMap::new();
        for req in &r.required_rules {
            let rule = match self.config.rules.get(&req.rule_id) {
                Some(x) => x,
                None => {
                    logging::error().str("rule-id", &req.rule_id).msg("required rule not found in config");
                    continue;
                }
            };
            let mut inherited = fragment.clone();
            inherited.inherited_from_finding = true;
            let found = self.detect_rule(&inherited, current_raw, rule, encoded_segments);
            all_required.insert(req.rule_id.clone(), found);
        }
        let mut final_findings = vec![];
        for pf in primary {
            let mut required: Vec<RequiredFinding> = vec![];
            for req in &r.required_rules {
                let found = match all_required.get(&req.rule_id) {
                    Some(f) => f,
                    None => continue,
                };
                for rf in found {
                    if within_proximity(&pf, rf, req) {
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
                nf.required_findings.extend(required);
                final_findings.push(nf);
            }
        }
        final_findings
    }

    pub fn add_finding(&mut self, mut f: Finding) {
        let global_fp = format!("{}:{}:{}", f.file, f.rule_id, f.start_line);
        if !f.commit.is_empty() {
            f.fingerprint = format!("{}:{}:{}:{}", f.commit, f.file, f.rule_id, f.start_line);
        } else {
            f.fingerprint = global_fp.clone();
        }
        if self.gitleaks_ignore.contains(&global_fp) {
            logging::debug().str("finding", &f.secret).str("fingerprint", &global_fp).msg("skipping finding: global fingerprint");
            return;
        } else if !f.commit.is_empty() && self.gitleaks_ignore.contains(&f.fingerprint) {
            logging::debug().str("finding", &f.secret).str("fingerprint", &f.fingerprint).msg("skipping finding: fingerprint");
            return;
        }
        if let Some(b) = &self.baseline {
            if !is_new(&f, self.redact, b) {
                logging::debug().str("finding", &f.secret).str("fingerprint", &f.fingerprint).msg("skipping finding: baseline");
                return;
            }
        }
        if self.verbose {
            print_finding(&f, self.no_color);
        }
        self.findings.push(f);
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

fn all_true(v: &[bool]) -> bool {
    v.iter().all(|b| *b)
}

pub fn check_commit_or_path_allowed(fragment: &Fragment, allowlists: &[Arc<Allowlist>]) -> bool {
    if fragment.file_path.is_empty() && fragment.commit_sha.is_empty() {
        return false;
    }
    for a in allowlists {
        let (commit_allowed, _) = a.commit_allowed(&fragment.commit_sha);
        let path_allowed = a.path_allowed(&fragment.file_path)
            || (!fragment.windows_file_path.is_empty() && a.path_allowed(&fragment.windows_file_path));
        let is_allowed;
        if a.match_condition == MatchCondition::And {
            let mut checks = vec![];
            if !a.commits.is_empty() {
                checks.push(commit_allowed);
            }
            if !a.paths.is_empty() {
                checks.push(path_allowed);
            }
            if !a.regexes.is_empty() || !a.stop_words.is_empty() {
                continue;
            }
            is_allowed = all_true(&checks);
        } else {
            is_allowed = commit_allowed || path_allowed;
        }
        if is_allowed {
            return true;
        }
    }
    false
}

pub fn check_finding_allowed(f: &Finding, fragment: &Fragment, current_line: &str, allowlists: &[Arc<Allowlist>]) -> bool {
    for a in allowlists {
        let target = match a.regex_target.as_str() {
            "match" => f.match_.as_str(),
            "line" => current_line,
            _ => f.secret.as_str(),
        };
        let regex_allowed = a.regex_allowed(target);
        let (contains_stopword, _) = a.contains_stop_word(&f.secret);
        let is_allowed;
        if a.match_condition == MatchCondition::And {
            let mut checks = vec![];
            if !a.commits.is_empty() {
                checks.push(a.commit_allowed(&fragment.commit_sha).0);
            }
            if !a.paths.is_empty() {
                checks.push(
                    a.path_allowed(&fragment.file_path)
                        || (!fragment.windows_file_path.is_empty() && a.path_allowed(&fragment.windows_file_path)),
                );
            }
            if !a.regexes.is_empty() {
                checks.push(regex_allowed);
            }
            if !a.stop_words.is_empty() {
                checks.push(contains_stopword);
            }
            is_allowed = all_true(&checks);
        } else {
            is_allowed = regex_allowed || contains_stopword;
        }
        if is_allowed {
            return true;
        }
    }
    false
}

#[derive(Default, Debug)]
pub struct Location {
    pub start_line: i64,
    pub end_line: i64,
    pub start_column: i64,
    pub end_column: i64,
    pub start_line_index: usize,
    pub end_line_index: usize,
}

pub fn location(newline_indices: &[(usize, usize)], raw: &str, match_index: (usize, usize)) -> Location {
    let mut prev_new_line: usize = 0;
    let mut loc = Location::default();
    let mut line_set = false;
    let mut line_num_last: i64 = 0;
    let (start, end) = match_index;
    let default = vec![(raw.len(), raw.len() + 1)];
    let indices: &[(usize, usize)] = if newline_indices.is_empty() { &default } else { newline_indices };
    for (line_num, pair) in indices.iter().enumerate() {
        line_num_last = line_num as i64;
        let nl = pair.0;
        if prev_new_line <= start && start < nl {
            line_set = true;
            loc.start_line = line_num as i64;
            loc.end_line = line_num as i64;
            loc.start_column = (start as i64 - prev_new_line as i64) + 1;
            loc.start_line_index = prev_new_line;
            loc.end_line_index = nl;
        }
        if prev_new_line < end && end <= nl {
            loc.end_line = line_num as i64;
            loc.end_column = end as i64 - prev_new_line as i64;
            loc.end_line_index = nl;
        }
        prev_new_line = pair.0;
    }
    if !line_set {
        loc.start_column = (start as i64 - prev_new_line as i64) + 1;
        loc.end_column = end as i64 - prev_new_line as i64;
        loc.start_line = line_num_last + 1;
        loc.end_line = line_num_last + 1;
        let b = raw.as_bytes();
        let mut i = 0;
        while end + i < b.len() {
            if b[end + i] == b'\n' || b[end + i] == b'\r' {
                break;
            }
            i += 1;
        }
        loc.end_line_index = end + i;
    }
    loc
}

pub fn shannon_entropy(data: &str) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<char, usize> = HashMap::new();
    for c in data.chars() {
        *counts.entry(c).or_insert(0) += 1;
    }
    let inv = 1.0 / data.len() as f64;
    let mut entropy = 0.0;
    // deterministic summation order is irrelevant for practical purposes, but sort for stability
    let mut v: Vec<(char, usize)> = counts.into_iter().collect();
    v.sort();
    for (_, count) in v {
        let freq = count as f64 * inv;
        entropy -= freq * freq.log2();
    }
    entropy
}

pub fn filter(findings: Vec<Finding>, redact: u64) -> Vec<Finding> {
    let mut ret = vec![];
    for f in findings.iter() {
        let mut include = true;
        if f.rule_id.to_lowercase().contains("generic") {
            for fp in findings.iter() {
                if f.start_line == fp.start_line
                    && f.commit == fp.commit
                    && f.rule_id != fp.rule_id
                    && fp.secret.contains(&f.secret)
                    && !fp.rule_id.to_lowercase().contains("generic")
                {
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

pub fn is_new(f: &Finding, redact: u64, baseline: &[Finding]) -> bool {
    for b in baseline {
        if f.rule_id == b.rule_id
            && f.description == b.description
            && f.start_line == b.start_line
            && f.end_line == b.end_line
            && f.start_column == b.start_column
            && f.end_column == b.end_column
            && (redact > 0 || (f.match_ == b.match_ && f.secret == b.secret))
            && f.file == b.file
            && f.commit == b.commit
            && f.author == b.author
            && f.email == b.email
            && f.date == b.date
            && f.message == b.message
            && f.entropy == b.entropy
        {
            return false;
        }
    }
    true
}

pub fn load_baseline(path: &str) -> Result<Vec<Finding>, String> {
    let bytes = std::fs::read(path).map_err(|_| format!("could not open {}", path))?;
    let unsupported = || format!("the format of the file {} is not supported", path);
    let v = crate::json::parse(&String::from_utf8_lossy(&bytes)).map_err(|_| unsupported())?;
    use crate::json::JValue;
    let arr = match v {
        JValue::Arr(a) => a,
        JValue::Null => vec![],
        _ => return Err(unsupported()),
    };
    let mut out = vec![];
    for item in arr {
        if let JValue::Null = item {
            out.push(Finding::default());
            continue;
        }
        if !matches!(item, JValue::Obj(_)) {
            return Err(unsupported());
        }
        let s = |k: &str| -> Result<String, String> {
            match item.get(k) {
                None | Some(JValue::Null) => Ok(String::new()),
                Some(JValue::Str(s)) => Ok(s.clone()),
                _ => Err(unsupported()),
            }
        };
        let n = |k: &str| -> Result<i64, String> {
            match item.get(k) {
                None | Some(JValue::Null) => Ok(0),
                Some(JValue::Num(x)) => x.parse::<i64>().map_err(|_| unsupported()),
                _ => Err(unsupported()),
            }
        };
        let entropy = match item.get("Entropy") {
            None | Some(JValue::Null) => 0.0f32,
            Some(JValue::Num(x)) => x.parse::<f32>().map_err(|_| unsupported())?,
            _ => return Err(unsupported()),
        };
        let tags = match item.get("Tags") {
            Some(JValue::Arr(a)) => a
                .iter()
                .map(|t| if let JValue::Str(s) = t { s.clone() } else { String::new() })
                .collect(),
            _ => vec![],
        };
        out.push(Finding {
            rule_id: s("RuleID")?,
            description: s("Description")?,
            start_line: n("StartLine")?,
            end_line: n("EndLine")?,
            start_column: n("StartColumn")?,
            end_column: n("EndColumn")?,
            match_: s("Match")?,
            secret: s("Secret")?,
            file: s("File")?,
            symlink_file: s("SymlinkFile")?,
            commit: s("Commit")?,
            link: s("Link")?,
            entropy,
            author: s("Author")?,
            email: s("Email")?,
            date: s("Date")?,
            message: s("Message")?,
            tags,
            fingerprint: s("Fingerprint")?,
            ..Default::default()
        });
    }
    Ok(out)
}

pub fn create_scm_link(remote: &RemoteInfo, f: &Finding) -> String {
    use crate::sources::Platform;
    if remote.platform == Platform::Unknown || remote.platform == Platform::None || f.commit.is_empty() {
        return String::new();
    }
    let (fp, has_inner) = match f.file.find(INNER_PATH_SEPARATOR) {
        Some(i) => (&f.file[..i], true),
        None => (f.file.as_str(), false),
    };
    let file_path = fp.replace(' ', "%20").replace('%', "%25");
    // strings.NewReplacer applies replacements in a single pass
    let file_path = {
        let mut out = String::new();
        for c in fp.chars() {
            match c {
                ' ' => out.push_str("%20"),
                '%' => out.push_str("%25"),
                c => out.push(c),
            }
        }
        let _ = file_path;
        out
    };
    let e = util::ext(&file_path).to_lowercase();
    match remote.platform {
        Platform::GitHub => {
            let mut link = format!("{}/blob/{}/{}", remote.url, f.commit, file_path);
            if has_inner {
                return link;
            }
            if e == ".ipynb" || e == ".md" {
                link.push_str("?plain=1");
            }
            if f.start_line != 0 {
                link.push_str(&format!("#L{}", f.start_line));
            }
            if f.end_line != f.start_line {
                link.push_str(&format!("-L{}", f.end_line));
            }
            link
        }
        Platform::GitLab => {
            let mut link = format!("{}/blob/{}/{}", remote.url, f.commit, file_path);
            if has_inner {
                return link;
            }
            if f.start_line != 0 {
                link.push_str(&format!("#L{}", f.start_line));
            }
            if f.end_line != f.start_line {
                link.push_str(&format!("-{}", f.end_line));
            }
            link
        }
        Platform::AzureDevOps => {
            let mut link = format!("{}/commit/{}?path=/{}", remote.url, f.commit, file_path);
            if has_inner {
                return link;
            }
            if f.start_line != 0 {
                link.push_str(&format!("&line={}", f.start_line));
            }
            if f.end_line != f.start_line {
                link.push_str(&format!("&lineEnd={}", f.end_line));
            }
            link.push_str("&lineStartColumn=1&lineEndColumn=10000000&type=2&lineStyle=plain&_a=files");
            link
        }
        Platform::Gitea => {
            let mut link = format!("{}/src/commit/{}/{}", remote.url, f.commit, file_path);
            if has_inner {
                return link;
            }
            if e == ".ipynb" || e == ".md" {
                link.push_str("?display=source");
            }
            if f.start_line != 0 {
                link.push_str(&format!("#L{}", f.start_line));
            }
            if f.end_line != f.start_line {
                link.push_str(&format!("-L{}", f.end_line));
            }
            link
        }
        Platform::Bitbucket => {
            let mut link = format!("{}/src/{}/{}", remote.url, f.commit, file_path);
            if has_inner {
                return link;
            }
            if f.start_line != 0 {
                link.push_str(&format!("#lines-{}", f.start_line));
            }
            if f.end_line != f.start_line {
                link.push_str(&format!(":{}", f.end_line));
            }
            link
        }
        _ => String::new(),
    }
}

fn styled_secret(s: &str) -> String {
    // lipgloss renders each line separately and pads lines to a common width
    let lines: Vec<&str> = s.split('\n').collect();
    let width = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    lines
        .iter()
        .map(|l| {
            let pad = if lines.len() > 1 { width - l.chars().count() } else { 0 };
            format!("\x1b[1;3;m{}\x1b[0m{}", l, " ".repeat(pad))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn print_finding(f: &Finding, no_color: bool) {
    let mut f = f.clone();
    f.line = util::trim_space(&f.line).to_string();
    f.secret = util::trim_space(&f.secret).to_string();
    f.match_ = util::trim_space(&f.match_).to_string();
    let is_file_match = f.match_.starts_with("file detected:");
    let mut skip_color = no_color;
    let mut finding = String::new();
    let mut secret = String::new();
    let mut out = String::new();
    if !is_file_match {
        let mut match_in_line = f.line.find(&f.match_).map(|x| x as i64).unwrap_or(-1);
        let secret_in_match = f.match_.find(&f.secret).unwrap_or(0);
        skip_color = false;
        if match_in_line == -1 || no_color {
            skip_color = true;
            match_in_line = 0;
        }
        let mil = match_in_line as usize;
        let mut start = util::bslice(&f.line, 0, mil);
        if mil > 20 {
            start = format!("...{}", util::bslice(&f.line, mil - 20, mil));
        }
        let match_beginning = util::bslice(&f.match_, 0, secret_in_match);
        secret = styled_secret(&f.secret);
        let match_end = util::bslice(&f.match_, secret_in_match + f.secret.len(), f.match_.len());
        let mut line_end_idx = mil + f.match_.len();
        if f.line.len() as i64 - 1 <= line_end_idx as i64 {
            line_end_idx = f.line.len();
        }
        let mut line_end = util::bslice(&f.line, line_end_idx, f.line.len());
        if f.secret.len() > 100 {
            secret = styled_secret(&(util::bslice(&f.secret, 0, 100) + "..."));
        }
        if line_end.len() > 20 {
            line_end = util::bslice(&line_end, 0, 20) + "...";
        }
        let st = start.trim_start_matches(' ');
        let st = st.strip_prefix('\n').unwrap_or(st);
        finding = format!("{}{}{}{}{}\n", st, match_beginning, secret, match_end, line_end);
    }
    if skip_color || is_file_match {
        out.push_str(&format!("{:<12} {}\n", "Finding:", f.match_));
        out.push_str(&format!("{:<12} {}\n", "Secret:", f.secret));
    } else {
        out.push_str(&format!("{:<12} {}", "Finding:", finding));
        out.push_str(&format!("{:<12} {}\n", "Secret:", secret));
    }
    out.push_str(&format!("{:<12} {}\n", "RuleID:", f.rule_id));
    out.push_str(&format!("{:<12} {:.6}\n", "Entropy:", f.entropy as f64));
    use std::io::Write;
    let flush = |s: &str| {
        let stdout = std::io::stdout();
        let mut h = stdout.lock();
        let _ = h.write_all(s.as_bytes());
        let _ = h.flush();
    };
    if f.file.is_empty() {
        flush(&out);
        f.print_required_findings();
        flush("\n");
        return;
    }
    if !f.tags.is_empty() {
        out.push_str(&format!("{:<12} [{}]\n", "Tags:", f.tags.join(" ")));
    }
    out.push_str(&format!("{:<12} {}\n", "File:", f.file));
    out.push_str(&format!("{:<12} {}\n", "Line:", f.start_line));
    if f.commit.is_empty() {
        out.push_str(&format!("{:<12} {}\n", "Fingerprint:", f.fingerprint));
        flush(&out);
        f.print_required_findings();
        flush("\n");
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
    flush(&out);
    f.print_required_findings();
    flush("\n");
}
