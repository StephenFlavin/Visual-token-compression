//! Port of gitleaks' report package.

pub mod template;

use std::io::Write;

use crate::config::Rule;
use crate::gojson::{encode_string, JVal};
use crate::gostr;

pub const CWE: &str = "CWE-798";
pub const CWE_DESCRIPTION: &str = "Use of Hard-coded Credentials";
pub const STDOUT_REPORT_PATH: &str = "-";
const VERSION: &str = "v8.0.0";
const DRIVER: &str = "gitleaks";

#[derive(Clone, Debug, Default)]
pub struct RequiredFinding {
    pub rule_id: String,
    pub start_line: i64,
    pub end_line: i64,
    pub start_column: i64,
    pub end_column: i64,
    pub line: String,
    pub match_: String,
    pub secret: String,
}

/// Finding contains a whole bunch of information about a secret finding.
#[derive(Clone, Debug, Default)]
pub struct Finding {
    pub rule_id: String,
    pub description: String,
    pub start_line: i64,
    pub end_line: i64,
    pub start_column: i64,
    pub end_column: i64,
    pub line: String,
    pub match_: String,
    pub secret: String,
    pub file: String,
    pub symlink_file: String,
    pub commit: String,
    pub link: String,
    pub entropy: f32,
    pub author: String,
    pub email: String,
    pub date: String,
    pub message: String,
    pub tags: Vec<String>,
    pub fingerprint: String,
    pub required_findings: Vec<RequiredFinding>,
}

impl Finding {
    pub fn add_required_findings(&mut self, afs: Vec<RequiredFinding>) {
        self.required_findings.extend(afs);
    }

    /// Redact removes sensitive information from a finding.
    pub fn redact(&mut self, percent: u64) {
        let mut secret = mask_secret(&self.secret, percent);
        if percent >= 100 {
            secret = "REDACTED".to_string();
        }
        if !self.secret.is_empty() {
            self.line = self.line.replace(&self.secret, &secret);
            self.match_ = self.match_.replace(&self.secret, &secret);
        } else {
            self.line = go_replace_all_empty(&self.line, &secret);
            self.match_ = go_replace_all_empty(&self.match_, &secret);
        }
        self.secret = secret;
    }

    pub fn to_json(&self) -> JVal {
        let mut o: Vec<(String, JVal)> = vec![
            ("RuleID".into(), JVal::s(&self.rule_id)),
            ("Description".into(), JVal::s(&self.description)),
            ("StartLine".into(), JVal::i(self.start_line)),
            ("EndLine".into(), JVal::i(self.end_line)),
            ("StartColumn".into(), JVal::i(self.start_column)),
            ("EndColumn".into(), JVal::i(self.end_column)),
            ("Match".into(), JVal::s(&self.match_)),
            ("Secret".into(), JVal::s(&self.secret)),
            ("File".into(), JVal::s(&self.file)),
            ("SymlinkFile".into(), JVal::s(&self.symlink_file)),
            ("Commit".into(), JVal::s(&self.commit)),
        ];
        if !self.link.is_empty() {
            o.push(("Link".into(), JVal::s(&self.link)));
        }
        o.push(("Entropy".into(), JVal::Num(gostr::format_float32_json(self.entropy))));
        o.push(("Author".into(), JVal::s(&self.author)));
        o.push(("Email".into(), JVal::s(&self.email)));
        o.push(("Date".into(), JVal::s(&self.date)));
        o.push(("Message".into(), JVal::s(&self.message)));
        o.push(("Tags".into(), JVal::strs(&self.tags)));
        o.push(("Fingerprint".into(), JVal::s(&self.fingerprint)));
        JVal::Obj(o)
    }

    /// Parse a finding from a JSON value (used for baselines)
    pub fn from_json(v: &serde_json::Value) -> Option<Finding> {
        let o = v.as_object()?;
        let get = |k: &str| -> Option<&serde_json::Value> {
            o.get(k).or_else(|| o.iter().find(|(kk, _)| kk.eq_ignore_ascii_case(k)).map(|(_, v)| v))
        };
        let s = |k: &str| -> String { get(k).and_then(|v| v.as_str()).unwrap_or("").to_string() };
        let i = |k: &str| -> i64 { get(k).and_then(|v| v.as_i64()).unwrap_or(0) };
        Some(Finding {
            rule_id: s("RuleID"),
            description: s("Description"),
            start_line: i("StartLine"),
            end_line: i("EndLine"),
            start_column: i("StartColumn"),
            end_column: i("EndColumn"),
            line: String::new(),
            match_: s("Match"),
            secret: s("Secret"),
            file: s("File"),
            symlink_file: s("SymlinkFile"),
            commit: s("Commit"),
            link: s("Link"),
            entropy: get("Entropy").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            author: s("Author"),
            email: s("Email"),
            date: s("Date"),
            message: s("Message"),
            tags: get("Tags")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default(),
            fingerprint: s("Fingerprint"),
            required_findings: Vec::new(),
        })
    }

    pub fn print_required_findings(&self, out: &mut String) {
        if self.required_findings.is_empty() {
            return;
        }
        out.push_str(&format!("{:<12} ", "Required:"));
        for (i, aux) in self.required_findings.iter().enumerate() {
            let mut s = gostr::trim_space(&aux.secret).to_string();
            if s.len() > 40 {
                s = format!("{}...", gostr::bslice(&s, 0, 37));
            }
            let styled = crate::detect::style::render(&s, false, false, Some((0xbf, 0x94, 0x78)));
            if i == 0 {
                out.push_str(&format!("{}:{}:{}\n", aux.rule_id, aux.start_line, styled));
            } else {
                out.push_str(&format!("{:<12} {}:{}:{}\n", "", aux.rule_id, aux.start_line, styled));
            }
        }
    }
}

fn go_replace_all_empty(s: &str, new: &str) -> String {
    // strings.ReplaceAll with empty old inserts `new` around every rune
    let mut out = String::new();
    out.push_str(new);
    for c in s.chars() {
        out.push(c);
        out.push_str(new);
    }
    out
}

pub fn mask_secret(secret: &str, percent: u64) -> String {
    let percent = percent.min(100);
    let len = secret.len() as f64;
    if len <= 0.0 {
        return secret.to_string();
    }
    let prc = (100 - percent) as f64;
    let lth = round_to_even(len * prc / 100.0) as usize;
    format!("{}...", gostr::bslice(secret, 0, lth))
}

fn round_to_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        let t = x.trunc();
        if (t as i64) % 2 == 0 {
            t
        } else {
            t + x.signum()
        }
    } else {
        r
    }
}

pub enum Reporter {
    Json,
    Csv,
    Junit,
    Sarif(Vec<Rule>),
    Template(template::Template),
}

impl Reporter {
    pub fn write<W: Write>(&self, w: &mut W, findings: &[Finding]) -> Result<(), String> {
        let out = match self {
            Reporter::Json => write_json(findings),
            Reporter::Csv => write_csv(findings),
            Reporter::Junit => write_junit(findings),
            Reporter::Sarif(rules) => write_sarif(rules, findings),
            Reporter::Template(t) => t.execute(findings)?,
        };
        w.write_all(out.as_bytes()).map_err(|e| e.to_string())?;
        w.flush().map_err(|e| e.to_string())
    }
}

pub fn write_json(findings: &[Finding]) -> String {
    let v = JVal::Arr(findings.iter().map(|f| f.to_json()).collect());
    let mut s = v.indent("", " ");
    s.push('\n');
    s
}

// ------------------------------------------------------------------- CSV

fn csv_field(field: &str) -> String {
    let needs = if field.is_empty() {
        false
    } else if field == "\\." {
        true
    } else {
        field.bytes().any(|c| c == b'\n' || c == b'\r' || c == b'"' || c == b',')
            || field.chars().next().map(|c| c.is_whitespace()).unwrap_or(false)
    };
    if !needs {
        return field.to_string();
    }
    let mut out = String::from("\"");
    for c in field.chars() {
        if c == '"' {
            out.push_str("\"\"");
        } else {
            out.push(c);
        }
    }
    out.push('"');
    out
}

fn csv_row(fields: &[String]) -> String {
    let mut s = fields.iter().map(|f| csv_field(f)).collect::<Vec<_>>().join(",");
    s.push('\n');
    s
}

pub fn write_csv(findings: &[Finding]) -> String {
    if findings.is_empty() {
        return String::new();
    }
    let mut columns: Vec<String> = [
        "RuleID", "Commit", "File", "SymlinkFile", "Secret", "Match", "StartLine", "EndLine", "StartColumn",
        "EndColumn", "Author", "Message", "Date", "Email", "Fingerprint", "Tags",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let with_link = !findings[0].link.is_empty();
    if with_link {
        columns.push("Link".to_string());
    }
    let mut out = csv_row(&columns);
    for f in findings {
        let mut row = vec![
            f.rule_id.clone(),
            f.commit.clone(),
            f.file.clone(),
            f.symlink_file.clone(),
            f.secret.clone(),
            f.match_.clone(),
            f.start_line.to_string(),
            f.end_line.to_string(),
            f.start_column.to_string(),
            f.end_column.to_string(),
            f.author.clone(),
            f.message.clone(),
            f.date.clone(),
            f.email.clone(),
            f.fingerprint.clone(),
            f.tags.join(" "),
        ];
        if with_link {
            row.push(f.link.clone());
        }
        out.push_str(&csv_row(&row));
    }
    out
}

// ------------------------------------------------------------------- JUnit

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("&#34;"),
            '\'' => out.push_str("&#39;"),
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\t' => out.push_str("&#x9;"),
            '\n' => out.push_str("&#xA;"),
            '\r' => out.push_str("&#xD;"),
            c => {
                let u = c as u32;
                let valid = u == 0x09
                    || u == 0x0A
                    || u == 0x0D
                    || (0x20..=0xD7FF).contains(&u)
                    || (0xE000..=0xFFFD).contains(&u)
                    || (0x10000..=0x10FFFF).contains(&u);
                if valid {
                    out.push(c)
                } else {
                    out.push('\u{FFFD}')
                }
            }
        }
    }
    out
}

fn junit_message(f: &Finding) -> String {
    if f.commit.is_empty() {
        return format!("{} has detected a secret in file {}, line {}.", f.rule_id, f.file, f.start_line);
    }
    format!(
        "{} has detected a secret in file {}, line {}, at commit {}.",
        f.rule_id, f.file, f.start_line, f.commit
    )
}

pub fn write_junit(findings: &[Finding]) -> String {
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<testsuites>\n");
    let n = findings.len();
    out.push_str(&format!(
        "\t<testsuite failures=\"{}\" name=\"gitleaks\" tests=\"{}\" time=\"\">",
        n, n
    ));
    if findings.is_empty() {
        out.push_str("</testsuite>\n");
    } else {
        for f in findings {
            let msg = junit_message(f);
            out.push_str(&format!(
                "\n\t\t<testcase classname=\"{}\" file=\"{}\" name=\"{}\" time=\"\">",
                xml_escape(&f.description),
                xml_escape(&f.file),
                xml_escape(&msg)
            ));
            let data = f.to_json().indent("", "\t");
            out.push_str(&format!(
                "\n\t\t\t<failure message=\"{}\" type=\"{}\">{}</failure>",
                xml_escape(&msg),
                xml_escape(&f.description),
                xml_escape(&data)
            ));
            out.push_str("\n\t\t</testcase>");
        }
        out.push_str("\n\t</testsuite>\n");
    }
    out.push_str("</testsuites>");
    out
}

// ------------------------------------------------------------------- SARIF

pub fn write_sarif(rules: &[Rule], findings: &[Finding]) -> String {
    let rules_j: Vec<JVal> = rules
        .iter()
        .map(|r| {
            JVal::Obj(vec![
                ("id".into(), JVal::s(&r.rule_id)),
                ("shortDescription".into(), JVal::Obj(vec![("text".into(), JVal::s(&r.description))])),
            ])
        })
        .collect();
    let results: Vec<JVal> = findings
        .iter()
        .map(|f| {
            let text = if f.commit.is_empty() {
                format!("{} has detected secret for file {}.", f.rule_id, f.file)
            } else {
                format!("{} has detected secret for file {} at commit {}.", f.rule_id, f.file, f.commit)
            };
            let uri = if !f.symlink_file.is_empty() { &f.symlink_file } else { &f.file };
            JVal::Obj(vec![
                ("message".into(), JVal::Obj(vec![("text".into(), JVal::s(&text))])),
                ("ruleId".into(), JVal::s(&f.rule_id)),
                (
                    "locations".into(),
                    JVal::Arr(vec![JVal::Obj(vec![(
                        "physicalLocation".into(),
                        JVal::Obj(vec![
                            ("artifactLocation".into(), JVal::Obj(vec![("uri".into(), JVal::s(uri))])),
                            (
                                "region".into(),
                                JVal::Obj(vec![
                                    ("startLine".into(), JVal::i(f.start_line)),
                                    ("startColumn".into(), JVal::i(f.start_column)),
                                    ("endLine".into(), JVal::i(f.end_line)),
                                    ("endColumn".into(), JVal::i(f.end_column)),
                                    ("snippet".into(), JVal::Obj(vec![("text".into(), JVal::s(&f.secret))])),
                                ]),
                            ),
                        ]),
                    )])]),
                ),
                (
                    "partialFingerprints".into(),
                    JVal::Obj(vec![
                        ("commitSha".into(), JVal::s(&f.commit)),
                        ("email".into(), JVal::s(&f.email)),
                        ("author".into(), JVal::s(&f.author)),
                        ("date".into(), JVal::s(&f.date)),
                        ("commitMessage".into(), JVal::s(&f.message)),
                    ]),
                ),
                ("properties".into(), JVal::Obj(vec![("tags".into(), JVal::strs(&f.tags))])),
            ])
        })
        .collect();
    let sarif = JVal::Obj(vec![
        ("$schema".into(), JVal::s("https://json.schemastore.org/sarif-2.1.0.json")),
        ("version".into(), JVal::s("2.1.0")),
        (
            "runs".into(),
            JVal::Arr(vec![JVal::Obj(vec![
                (
                    "tool".into(),
                    JVal::Obj(vec![(
                        "driver".into(),
                        JVal::Obj(vec![
                            ("name".into(), JVal::s(DRIVER)),
                            ("semanticVersion".into(), JVal::s(VERSION)),
                            ("informationUri".into(), JVal::s("https://github.com/gitleaks/gitleaks")),
                            ("rules".into(), JVal::Arr(rules_j)),
                        ]),
                    )]),
                ),
                ("results".into(), JVal::Arr(results)),
            ])]),
        ),
    ]);
    let mut s = sarif.indent("", " ");
    s.push('\n');
    let _ = encode_string;
    s
}
