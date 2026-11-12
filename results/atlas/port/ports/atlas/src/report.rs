// Port of the report package: Finding plus json, csv, junit, sarif and template reporters.

use crate::config::Rule;
use crate::json::{json_escape_str, JsonWriter};
use crate::util::format_json_f32;
use std::io::Write;

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
    pub fn redact(&mut self, percent: u64) {
        let mut secret = mask_secret(&self.secret, percent);
        if percent >= 100 {
            secret = "REDACTED".to_string();
        }
        if !self.secret.is_empty() {
            self.line = self.line.replace(&self.secret, &secret);
            self.match_ = self.match_.replace(&self.secret, &secret);
        } else {
            self.line = go_replace_empty(&self.line, &secret);
            self.match_ = go_replace_empty(&self.match_, &secret);
        }
        self.secret = secret;
    }

    pub fn print_required_findings(&self) {
        if self.required_findings.is_empty() {
            return;
        }
        let mut out = String::new();
        out.push_str(&format!("{:<12} ", "Required:"));
        for (i, aux) in self.required_findings.iter().enumerate() {
            let mut s = crate::util::trim_space(&aux.secret).to_string();
            if s.len() > 40 {
                s = crate::util::bslice(&s, 0, 37) + "...";
            }
            if i == 0 {
                out.push_str(&format!("{}:{}:{}\n", aux.rule_id, aux.start_line, s));
            } else {
                out.push_str(&format!("{:<12} {}:{}:{}\n", "", aux.rule_id, aux.start_line, s));
            }
        }
        print!("{}", out);
    }
}

/// strings.ReplaceAll with an empty old string inserts `new` around every rune.
fn go_replace_empty(s: &str, new: &str) -> String {
    let mut out = String::new();
    out.push_str(new);
    for c in s.chars() {
        out.push(c);
        out.push_str(new);
    }
    out
}

fn mask_secret(secret: &str, percent: u64) -> String {
    let percent = percent.min(100);
    let len = secret.len() as f64;
    if len <= 0.0 {
        return secret.to_string();
    }
    let prc = (100 - percent) as f64;
    let lth = round_to_even(len * prc / 100.0) as usize;
    crate::util::bslice(secret, 0, lth) + "..."
}

fn round_to_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        let t = x.trunc();
        if t % 2.0 == 0.0 {
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
    Template(crate::template::Template),
}

impl Reporter {
    pub fn write(&self, w: &mut dyn Write, findings: &[Finding]) -> std::io::Result<()> {
        match self {
            Reporter::Json => w.write_all(json_report(findings).as_bytes()),
            Reporter::Csv => w.write_all(csv_report(findings).as_bytes()),
            Reporter::Junit => w.write_all(junit_report(findings).as_bytes()),
            Reporter::Sarif(rules) => w.write_all(sarif_report(rules, findings).as_bytes()),
            Reporter::Template(t) => {
                let s = t.execute(findings).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                w.write_all(s.as_bytes())
            }
        }
    }
}

pub fn write_finding_json(w: &mut JsonWriter, f: &Finding) {
    w.begin_obj();
    w.key("RuleID");
    w.string(&f.rule_id);
    w.key("Description");
    w.string(&f.description);
    w.key("StartLine");
    w.raw(&f.start_line.to_string());
    w.key("EndLine");
    w.raw(&f.end_line.to_string());
    w.key("StartColumn");
    w.raw(&f.start_column.to_string());
    w.key("EndColumn");
    w.raw(&f.end_column.to_string());
    w.key("Match");
    w.string(&f.match_);
    w.key("Secret");
    w.string(&f.secret);
    w.key("File");
    w.string(&f.file);
    w.key("SymlinkFile");
    w.string(&f.symlink_file);
    w.key("Commit");
    w.string(&f.commit);
    if !f.link.is_empty() {
        w.key("Link");
        w.string(&f.link);
    }
    w.key("Entropy");
    w.raw(&format_json_f32(f.entropy));
    w.key("Author");
    w.string(&f.author);
    w.key("Email");
    w.string(&f.email);
    w.key("Date");
    w.string(&f.date);
    w.key("Message");
    w.string(&f.message);
    w.key("Tags");
    w.string_list(Some(&f.tags));
    w.key("Fingerprint");
    w.string(&f.fingerprint);
    w.end_obj();
}

pub fn json_report(findings: &[Finding]) -> String {
    let mut w = JsonWriter::new(" ");
    w.begin_arr();
    for f in findings {
        w.elem();
        write_finding_json(&mut w, f);
    }
    w.end_arr();
    let mut s = w.finish();
    s.push('\n');
    s
}

fn csv_field(f: &str) -> String {
    let needs = !f.is_empty()
        && (f == "\\."
            || f.contains(',')
            || f.contains('"')
            || f.contains('\r')
            || f.contains('\n')
            || f.chars().next().map(|c| c.is_whitespace()).unwrap_or(false));
    if !needs {
        return f.to_string();
    }
    let mut out = String::from("\"");
    for c in f.chars() {
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

pub fn csv_report(findings: &[Finding]) -> String {
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
    let has_link = !findings[0].link.is_empty();
    if has_link {
        columns.push("Link".into());
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
        if has_link {
            row.push(f.link.clone());
        }
        out.push_str(&csv_row(&row));
    }
    out
}

/// encoding/xml EscapeText semantics
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
            c if is_xml_char(c) => out.push(c),
            _ => out.push('\u{FFFD}'),
        }
    }
    out
}

fn is_xml_char(c: char) -> bool {
    let v = c as u32;
    v == 0x09
        || v == 0x0A
        || v == 0x0D
        || (0x20..=0xD7FF).contains(&v)
        || (0xE000..=0xFFFD).contains(&v)
        || (0x10000..=0x10FFFF).contains(&v)
}

fn junit_message(f: &Finding) -> String {
    if f.commit.is_empty() {
        format!("{} has detected a secret in file {}, line {}.", f.rule_id, f.file, f.start_line)
    } else {
        format!(
            "{} has detected a secret in file {}, line {}, at commit {}.",
            f.rule_id, f.file, f.start_line, f.commit
        )
    }
}

pub fn junit_report(findings: &[Finding]) -> String {
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<testsuites>\n");
    let n = findings.len();
    out.push_str(&format!(
        "\t<testsuite failures=\"{}\" name=\"gitleaks\" tests=\"{}\" time=\"\">",
        n, n
    ));
    for f in findings {
        let msg = xml_escape(&junit_message(f));
        let mut jw = JsonWriter::new("\t");
        write_finding_json(&mut jw, f);
        let data = jw.finish();
        out.push_str(&format!(
            "\n\t\t<testcase classname=\"{}\" file=\"{}\" name=\"{}\" time=\"\">",
            xml_escape(&f.description),
            xml_escape(&f.file),
            msg
        ));
        out.push_str(&format!(
            "\n\t\t\t<failure message=\"{}\" type=\"{}\">{}</failure>",
            msg,
            xml_escape(&f.description),
            xml_escape(&data)
        ));
        out.push_str("\n\t\t</testcase>");
    }
    if n > 0 {
        out.push_str("\n\t</testsuite>");
    } else {
        out.push_str("</testsuite>");
    }
    out.push_str("\n</testsuites>");
    out
}

pub fn sarif_report(rules: &[Rule], findings: &[Finding]) -> String {
    let mut w = JsonWriter::new(" ");
    w.begin_obj();
    w.key("$schema");
    w.string("https://json.schemastore.org/sarif-2.1.0.json");
    w.key("version");
    w.string("2.1.0");
    w.key("runs");
    w.begin_arr();
    w.elem();
    w.begin_obj();
    w.key("tool");
    w.begin_obj();
    w.key("driver");
    w.begin_obj();
    w.key("name");
    w.string(DRIVER);
    w.key("semanticVersion");
    w.string(VERSION);
    w.key("informationUri");
    w.string("https://github.com/gitleaks/gitleaks");
    w.key("rules");
    w.begin_arr();
    for r in rules {
        w.elem();
        w.begin_obj();
        w.key("id");
        w.string(&r.rule_id);
        w.key("shortDescription");
        w.begin_obj();
        w.key("text");
        w.string(&r.description);
        w.end_obj();
        w.end_obj();
    }
    w.end_arr();
    w.end_obj();
    w.end_obj();
    w.key("results");
    w.begin_arr();
    for f in findings {
        w.elem();
        w.begin_obj();
        w.key("message");
        w.begin_obj();
        w.key("text");
        let text = if f.commit.is_empty() {
            format!("{} has detected secret for file {}.", f.rule_id, f.file)
        } else {
            format!("{} has detected secret for file {} at commit {}.", f.rule_id, f.file, f.commit)
        };
        w.string(&text);
        w.end_obj();
        w.key("ruleId");
        w.string(&f.rule_id);
        w.key("locations");
        w.begin_arr();
        w.elem();
        w.begin_obj();
        w.key("physicalLocation");
        w.begin_obj();
        w.key("artifactLocation");
        w.begin_obj();
        w.key("uri");
        w.string(if f.symlink_file.is_empty() { &f.file } else { &f.symlink_file });
        w.end_obj();
        w.key("region");
        w.begin_obj();
        w.key("startLine");
        w.raw(&f.start_line.to_string());
        w.key("startColumn");
        w.raw(&f.start_column.to_string());
        w.key("endLine");
        w.raw(&f.end_line.to_string());
        w.key("endColumn");
        w.raw(&f.end_column.to_string());
        w.key("snippet");
        w.begin_obj();
        w.key("text");
        w.string(&f.secret);
        w.end_obj();
        w.end_obj();
        w.end_obj();
        w.end_obj();
        w.end_arr();
        w.key("partialFingerprints");
        w.begin_obj();
        w.key("commitSha");
        w.string(&f.commit);
        w.key("email");
        w.string(&f.email);
        w.key("author");
        w.string(&f.author);
        w.key("date");
        w.string(&f.date);
        w.key("commitMessage");
        w.string(&f.message);
        w.end_obj();
        w.key("properties");
        w.begin_obj();
        w.key("tags");
        w.string_list(Some(&f.tags));
        w.end_obj();
        w.end_obj();
    }
    w.end_arr();
    w.end_obj();
    w.end_arr();
    w.end_obj();
    let mut s = w.finish();
    s.push('\n');
    s
}

#[allow(dead_code)]
pub fn quote_json(s: &str) -> String {
    json_escape_str(s)
}
