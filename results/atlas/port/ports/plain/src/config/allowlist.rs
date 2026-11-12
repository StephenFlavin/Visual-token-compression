use std::collections::HashSet;

use aho_corasick::AhoCorasick;

use crate::regexp::Regexp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllowlistMatchCondition {
    Or,
    And,
}

impl AllowlistMatchCondition {
    pub fn as_str(&self) -> &'static str {
        match self {
            AllowlistMatchCondition::Or => "OR",
            AllowlistMatchCondition::And => "AND",
        }
    }
}

/// Allowlist allows a rule to be ignored for specific regexes, paths, and/or commits
#[derive(Clone, Debug)]
pub struct Allowlist {
    pub description: String,
    pub match_condition: AllowlistMatchCondition,
    pub commits: Vec<String>,
    pub paths: Vec<Regexp>,
    /// Can be `match` or `line`; empty means the secret.
    pub regex_target: String,
    pub regexes: Vec<Regexp>,
    pub stop_words: Vec<String>,

    pub(crate) validated: bool,
    pub(crate) commit_map: Option<HashSet<String>>,
    pub(crate) regex_pat: Option<Regexp>,
    pub(crate) path_pat: Option<Regexp>,
    pub(crate) stopword_trie: Option<AhoCorasick>,
}

impl Default for Allowlist {
    fn default() -> Self {
        Allowlist {
            description: String::new(),
            match_condition: AllowlistMatchCondition::Or,
            commits: Vec::new(),
            paths: Vec::new(),
            regex_target: String::new(),
            regexes: Vec::new(),
            stop_words: Vec::new(),
            validated: false,
            commit_map: None,
            regex_pat: None,
            path_pat: None,
            stopword_trie: None,
        }
    }
}

impl Allowlist {
    pub fn validate(&mut self) -> Result<(), String> {
        if self.validated {
            return Ok(());
        }
        if self.commits.is_empty() && self.paths.is_empty() && self.regexes.is_empty() && self.stop_words.is_empty() {
            return Err("must contain at least one check for: commits, paths, regexes, or stopwords".to_string());
        }

        // Deduplicate commits and stopwords.
        if !self.commits.is_empty() {
            let mut seen = HashSet::new();
            let mut uniq = Vec::new();
            for c in &self.commits {
                let n = c.trim().to_lowercase();
                if seen.insert(n.clone()) {
                    uniq.push(n);
                }
            }
            self.commits = uniq;
            self.commit_map = Some(seen);
        }
        if !self.stop_words.is_empty() {
            let mut seen = HashSet::new();
            let mut uniq = Vec::new();
            for w in &self.stop_words {
                let n = w.to_lowercase();
                if seen.insert(n.clone()) {
                    uniq.push(n);
                }
            }
            self.stop_words = uniq;
            self.stopword_trie = AhoCorasick::new(&self.stop_words).ok();
        }

        if !self.paths.is_empty() {
            self.path_pat = Some(super::utils::join_regex_or(&self.paths));
        }
        if !self.regexes.is_empty() {
            self.regex_pat = Some(super::utils::join_regex_or(&self.regexes));
        }

        self.validated = true;
        Ok(())
    }

    /// CommitAllowed returns true if the commit is allowed to be ignored.
    pub fn commit_allowed(&self, c: &str) -> (bool, String) {
        if c.is_empty() {
            return (false, String::new());
        }
        if let Some(m) = &self.commit_map {
            if m.contains(&c.to_lowercase()) {
                return (true, String::new());
            }
        } else if !self.commits.is_empty() {
            for commit in &self.commits {
                if commit == c {
                    return (true, c.to_string());
                }
            }
        }
        (false, String::new())
    }

    /// PathAllowed returns true if the path is allowed to be ignored.
    pub fn path_allowed(&self, path: &str) -> bool {
        if path.is_empty() {
            return false;
        }
        if let Some(p) = &self.path_pat {
            return p.is_match(path);
        } else if !self.paths.is_empty() {
            return super::utils::any_regex_match(path, &self.paths);
        }
        false
    }

    /// RegexAllowed returns true if the regex is allowed to be ignored.
    pub fn regex_allowed(&self, secret: &str) -> bool {
        if secret.is_empty() {
            return false;
        }
        if let Some(p) = &self.regex_pat {
            return p.is_match(secret);
        } else if !self.regexes.is_empty() {
            return super::utils::any_regex_match(secret, &self.regexes);
        }
        false
    }

    pub fn contains_stop_word(&self, s: &str) -> (bool, String) {
        if s.is_empty() {
            return (false, String::new());
        }
        let s = s.to_lowercase();
        if let Some(t) = &self.stopword_trie {
            if let Some(m) = t.find(&s) {
                return (true, s[m.start()..m.end()].to_string());
            }
        } else if !self.stop_words.is_empty() {
            for w in &self.stop_words {
                if s.contains(w.as_str()) {
                    return (true, w.clone());
                }
            }
        }
        (false, String::new())
    }
}
