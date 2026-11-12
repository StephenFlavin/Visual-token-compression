use std::sync::Arc;

use super::allowlist::Allowlist;
use crate::regexp::Regexp;

/// Rules contain information that define details on how to detect secrets
#[derive(Clone, Debug, Default)]
pub struct Rule {
    pub rule_id: String,
    pub description: String,
    pub entropy: f64,
    pub secret_group: i64,
    pub regex: Option<Regexp>,
    pub path: Option<Regexp>,
    pub tags: Vec<String>,
    pub keywords: Vec<String>,
    pub allowlists: Vec<Arc<Allowlist>>,
    pub required_rules: Vec<Required>,
    pub skip_report: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Required {
    pub rule_id: String,
    pub within_lines: Option<i64>,
    pub within_columns: Option<i64>,
}

impl Rule {
    /// Validate guards against common misconfigurations.
    pub fn validate(&self) -> Result<(), String> {
        if self.rule_id.trim().is_empty() {
            let mut sb = String::new();
            if !self.description.is_empty() {
                sb.push_str(&format!(", description: {}", self.description));
            }
            if let Some(r) = &self.regex {
                sb.push_str(&format!(", regex: {}", r.as_str()));
            }
            if let Some(p) = &self.path {
                sb.push_str(&format!(", path: {}", p.as_str()));
            }
            return Err(format!("rule |id| is missing or empty{}", sb));
        }

        if self.regex.is_none() && self.path.is_none() {
            return Err(format!("{}: both |regex| and |path| are empty, this rule will have no effect", self.rule_id));
        }

        if let Some(r) = &self.regex {
            if self.secret_group > r.num_subexp() as i64 {
                return Err(format!(
                    "{}: invalid regex secret group {}, max regex secret group {}",
                    self.rule_id,
                    self.secret_group,
                    r.num_subexp()
                ));
            }
        }

        for a in &self.allowlists {
            let mut a2 = (**a).clone();
            if let Err(e) = a2.validate() {
                return Err(format!("{}: {}", self.rule_id, e));
            }
        }
        Ok(())
    }
}
