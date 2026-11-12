//! Configuration loading. This is a port of gitleaks' config package along
//! with the small portion of viper/mapstructure behaviour it relies on
//! (case-insensitive keys, weakly typed decoding).

pub mod allowlist;
pub mod rule;
pub mod utils;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use toml::{Table, Value};

pub use allowlist::{Allowlist, AllowlistMatchCondition};
pub use rule::{Required, Rule};

use crate::logging;
use crate::regexp::Regexp;
use crate::version;

pub const DEFAULT_CONFIG: &str = include_str!("../gitleaks.toml");

static EXTEND_DEPTH: AtomicUsize = AtomicUsize::new(0);
const MAX_EXTEND_DEPTH: usize = 2;

/// Viper-like global state: which file was used for the configuration.
static CONFIG_FILE_USED: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

pub fn set_config_file_used(p: &str) {
    *CONFIG_FILE_USED.lock().unwrap() = p.to_string();
}

pub fn config_file_used() -> String {
    CONFIG_FILE_USED.lock().unwrap().clone()
}

/// Convert an io error to the message Go would print (e.g. "open x: no such file or directory")
pub fn go_os_error(op: &str, path: &str, e: &std::io::Error) -> String {
    format!("{} {}: {}", op, path, go_errno_text(e))
}

pub fn go_errno_text(e: &std::io::Error) -> String {
    if let Some(code) = e.raw_os_error() {
        let s = match code {
            libc::ENOENT => "no such file or directory",
            libc::EACCES => "permission denied",
            libc::EISDIR => "is a directory",
            libc::ENOTDIR => "not a directory",
            libc::EEXIST => "file exists",
            libc::ELOOP => "too many levels of symbolic links",
            libc::EPERM => "operation not permitted",
            libc::ENAMETOOLONG => "file name too long",
            libc::EROFS => "read-only file system",
            _ => "",
        };
        if !s.is_empty() {
            return s.to_string();
        }
    }
    let msg = e.to_string();
    let msg = match msg.find(" (os error") {
        Some(p) => msg[..p].to_string(),
        None => msg,
    };
    let mut c = msg.chars();
    match c.next() {
        Some(f) => f.to_lowercase().collect::<String>() + c.as_str(),
        None => msg,
    }
}

/// Parse TOML content the way viper does (returning a viper-style error).
pub fn parse_toml(content: &str) -> Result<Table, String> {
    content.parse::<Table>().map_err(|e| {
        let msg = e.message().to_string();
        format!("While parsing config: toml: {}", msg)
    })
}

/// viper.ReadInConfig for an explicit config file
pub fn read_config_file(path: &str) -> Result<Table, String> {
    let data = std::fs::read(path).map_err(|e| go_os_error("open", path, &e))?;
    let content = String::from_utf8_lossy(&data);
    parse_toml(&content)
}

// ---------------------------------------------------------------------------
// Weakly typed, case-insensitive decoding helpers (mapstructure semantics)

fn lookup<'a>(t: &'a Table, key: &str) -> Option<&'a Value> {
    if let Some(v) = t.get(key) {
        return Some(v);
    }
    for (k, v) in t.iter() {
        if k.eq_ignore_ascii_case(key) {
            return Some(v);
        }
    }
    None
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::String(_) => "string",
        Value::Integer(_) => "int64",
        Value::Float(_) => "float64",
        Value::Boolean(_) => "bool",
        Value::Datetime(_) => "time.Time",
        Value::Array(_) => "[]interface {}",
        Value::Table(_) => "map[string]interface {}",
    }
}

fn decode_err(name: &str, expected: &str, v: &Value) -> String {
    format!(
        "1 error(s) decoding:\n\n* '{}' expected type '{}', got unconvertible type '{}', value: '{}'",
        name,
        expected,
        type_name(v),
        format!("{:?}", v)
    )
}

fn as_string(v: &Value, name: &str) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Integer(i) => Ok(i.to_string()),
        Value::Float(f) => Ok(crate::gostr::format_float_v(*f, false)),
        Value::Boolean(b) => Ok(if *b { "1".to_string() } else { "0".to_string() }),
        _ => Err(decode_err(name, "string", v)),
    }
}

fn as_int(v: &Value, name: &str) -> Result<i64, String> {
    match v {
        Value::Integer(i) => Ok(*i),
        Value::Float(f) => Ok(*f as i64),
        Value::Boolean(b) => Ok(if *b { 1 } else { 0 }),
        Value::String(s) => {
            if s.is_empty() {
                return Ok(0);
            }
            parse_go_int(s).ok_or_else(|| {
                format!(
                    "1 error(s) decoding:\n\n* cannot parse '{}' as int: strconv.ParseInt: parsing {}: invalid syntax",
                    name,
                    crate::gostr::quote(s)
                )
            })
        }
        _ => Err(decode_err(name, "int", v)),
    }
}

fn parse_go_int(s: &str) -> Option<i64> {
    let (neg, rest) = if let Some(r) = s.strip_prefix('-') {
        (true, r)
    } else if let Some(r) = s.strip_prefix('+') {
        (false, r)
    } else {
        (false, s)
    };
    let rest = rest.replace('_', "");
    let v = if let Some(h) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        i64::from_str_radix(h, 16).ok()?
    } else if let Some(b) = rest.strip_prefix("0b").or_else(|| rest.strip_prefix("0B")) {
        i64::from_str_radix(b, 2).ok()?
    } else if let Some(o) = rest.strip_prefix("0o").or_else(|| rest.strip_prefix("0O")) {
        i64::from_str_radix(o, 8).ok()?
    } else if rest.len() > 1 && rest.starts_with('0') {
        i64::from_str_radix(&rest[1..], 8).ok()?
    } else {
        rest.parse::<i64>().ok()?
    };
    Some(if neg { -v } else { v })
}

fn as_float(v: &Value, name: &str) -> Result<f64, String> {
    match v {
        Value::Integer(i) => Ok(*i as f64),
        Value::Float(f) => Ok(*f),
        Value::Boolean(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => {
            if s.is_empty() {
                return Ok(0.0);
            }
            s.parse::<f64>().map_err(|_| {
                format!(
                    "1 error(s) decoding:\n\n* cannot parse '{}' as float: strconv.ParseFloat: parsing {}: invalid syntax",
                    name,
                    crate::gostr::quote(s)
                )
            })
        }
        _ => Err(decode_err(name, "float64", v)),
    }
}

fn as_bool(v: &Value, name: &str) -> Result<bool, String> {
    match v {
        Value::Boolean(b) => Ok(*b),
        Value::Integer(i) => Ok(*i != 0),
        Value::Float(f) => Ok(*f != 0.0),
        Value::String(s) => match s.as_str() {
            "" => Ok(false),
            "1" | "t" | "T" | "TRUE" | "true" | "True" => Ok(true),
            "0" | "f" | "F" | "FALSE" | "false" | "False" => Ok(false),
            _ => Err(format!(
                "1 error(s) decoding:\n\n* cannot parse '{}' as bool: strconv.ParseBool: parsing {}: invalid syntax",
                name,
                crate::gostr::quote(s)
            )),
        },
        _ => Err(decode_err(name, "bool", v)),
    }
}

fn as_strings(v: &Value, name: &str) -> Result<Vec<String>, String> {
    match v {
        Value::Array(a) => {
            let mut out = Vec::new();
            for (i, x) in a.iter().enumerate() {
                out.push(as_string(x, &format!("{}[{}]", name, i))?);
            }
            Ok(out)
        }
        Value::String(s) => {
            if s.is_empty() {
                Ok(Vec::new())
            } else {
                Ok(s.split(',').map(|x| x.to_string()).collect())
            }
        }
        other => Ok(vec![as_string(other, name)?]),
    }
}

fn as_tables<'a>(v: &'a Value, name: &str) -> Result<Vec<&'a Table>, String> {
    match v {
        Value::Array(a) => {
            let mut out = Vec::new();
            for (i, x) in a.iter().enumerate() {
                match x {
                    Value::Table(t) => out.push(t),
                    other => return Err(decode_err(&format!("{}[{}]", name, i), "struct", other)),
                }
            }
            Ok(out)
        }
        Value::Table(t) => Ok(vec![t]),
        other => Err(decode_err(name, "slice", other)),
    }
}

fn as_table<'a>(v: &'a Value, name: &str) -> Result<&'a Table, String> {
    match v {
        Value::Table(t) => Ok(t),
        other => Err(decode_err(name, "struct", other)),
    }
}

fn opt_string(t: &Table, key: &str, name: &str) -> Result<String, String> {
    match lookup(t, key) {
        Some(v) => as_string(v, &format!("{}{}", name, key)),
        None => Ok(String::new()),
    }
}

fn opt_strings(t: &Table, key: &str, name: &str) -> Result<Option<Vec<String>>, String> {
    match lookup(t, key) {
        Some(v) => Ok(Some(as_strings(v, &format!("{}{}", name, key))?)),
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct ViperRuleAllowlist {
    pub description: String,
    pub condition: String,
    pub commits: Vec<String>,
    pub paths: Vec<String>,
    pub regex_target: String,
    pub regexes: Vec<String>,
    pub stop_words: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ViperGlobalAllowlist {
    pub target_rules: Vec<String>,
    pub allowlist: ViperRuleAllowlist,
}

#[derive(Clone, Debug, Default)]
pub struct ViperRequired {
    pub id: String,
    pub within_lines: Option<i64>,
    pub within_columns: Option<i64>,
}

#[derive(Clone, Debug, Default)]
pub struct ViperRule {
    pub id: String,
    pub description: String,
    pub path: String,
    pub regex: String,
    pub secret_group: i64,
    pub entropy: f64,
    pub keywords: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub allow_list: Option<ViperRuleAllowlist>,
    pub allowlists: Vec<ViperRuleAllowlist>,
    pub required: Vec<ViperRequired>,
    pub skip_report: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Extend {
    pub path: String,
    pub url: String,
    pub use_default: bool,
    pub disabled_rules: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ViperConfig {
    pub title: String,
    pub description: String,
    pub extend: Extend,
    pub rules: Vec<ViperRule>,
    pub allow_list: Option<ViperGlobalAllowlist>,
    pub allowlists: Vec<ViperGlobalAllowlist>,
    pub min_version: String,
    pub config_path: String,
}

fn decode_rule_allowlist(t: &Table, name: &str) -> Result<ViperRuleAllowlist, String> {
    Ok(ViperRuleAllowlist {
        description: opt_string(t, "description", name)?,
        condition: opt_string(t, "condition", name)?,
        commits: opt_strings(t, "commits", name)?.unwrap_or_default(),
        paths: opt_strings(t, "paths", name)?.unwrap_or_default(),
        regex_target: opt_string(t, "regexTarget", name)?,
        regexes: opt_strings(t, "regexes", name)?.unwrap_or_default(),
        stop_words: opt_strings(t, "stopWords", name)?.unwrap_or_default(),
    })
}

fn decode_global_allowlist(t: &Table, name: &str) -> Result<ViperGlobalAllowlist, String> {
    Ok(ViperGlobalAllowlist {
        target_rules: opt_strings(t, "targetRules", name)?.unwrap_or_default(),
        allowlist: decode_rule_allowlist(t, name)?,
    })
}

impl ViperConfig {
    /// viper.Unmarshal(&vc)
    pub fn unmarshal(t: &Table) -> Result<ViperConfig, String> {
        let mut vc = ViperConfig {
            title: opt_string(t, "title", "")?,
            description: opt_string(t, "description", "")?,
            min_version: opt_string(t, "minVersion", "")?,
            ..Default::default()
        };
        if let Some(v) = lookup(t, "extend") {
            let et = as_table(v, "extend")?;
            vc.extend = Extend {
                path: opt_string(et, "path", "extend.")?,
                url: opt_string(et, "url", "extend.")?,
                use_default: match lookup(et, "useDefault") {
                    Some(v) => as_bool(v, "extend.usedefault")?,
                    None => false,
                },
                disabled_rules: opt_strings(et, "disabledRules", "extend.")?.unwrap_or_default(),
            };
        }
        if let Some(v) = lookup(t, "rules") {
            for (i, rt) in as_tables(v, "rules")?.into_iter().enumerate() {
                let name = format!("rules[{}].", i);
                let mut r = ViperRule {
                    id: opt_string(rt, "id", &name)?,
                    description: opt_string(rt, "description", &name)?,
                    path: opt_string(rt, "path", &name)?,
                    regex: opt_string(rt, "regex", &name)?,
                    keywords: opt_strings(rt, "keywords", &name)?,
                    tags: opt_strings(rt, "tags", &name)?,
                    ..Default::default()
                };
                if let Some(v) = lookup(rt, "secretGroup") {
                    r.secret_group = as_int(v, &format!("{}secretgroup", name))?;
                }
                if let Some(v) = lookup(rt, "entropy") {
                    r.entropy = as_float(v, &format!("{}entropy", name))?;
                }
                if let Some(v) = lookup(rt, "skipReport") {
                    r.skip_report = as_bool(v, &format!("{}skipreport", name))?;
                }
                if let Some(v) = lookup(rt, "allowList") {
                    let at = as_table(v, &format!("{}allowlist", name))?;
                    r.allow_list = Some(decode_rule_allowlist(at, &format!("{}allowlist.", name))?);
                }
                if let Some(v) = lookup(rt, "allowlists") {
                    for (j, at) in as_tables(v, &format!("{}allowlists", name))?.into_iter().enumerate() {
                        r.allowlists.push(decode_rule_allowlist(at, &format!("{}allowlists[{}].", name, j))?);
                    }
                }
                if let Some(v) = lookup(rt, "required") {
                    for (j, qt) in as_tables(v, &format!("{}required", name))?.into_iter().enumerate() {
                        let qn = format!("{}required[{}].", name, j);
                        let mut q = ViperRequired { id: opt_string(qt, "id", &qn)?, ..Default::default() };
                        if let Some(v) = lookup(qt, "withinLines") {
                            q.within_lines = Some(as_int(v, &format!("{}withinLines", qn))?);
                        }
                        if let Some(v) = lookup(qt, "withinColumns") {
                            q.within_columns = Some(as_int(v, &format!("{}withinColumns", qn))?);
                        }
                        r.required.push(q);
                    }
                }
                vc.rules.push(r);
            }
        }
        if let Some(v) = lookup(t, "allowList") {
            let at = as_table(v, "allowlist")?;
            vc.allow_list = Some(decode_global_allowlist(at, "allowlist.")?);
        }
        if let Some(v) = lookup(t, "allowlists") {
            for (i, at) in as_tables(v, "allowlists")?.into_iter().enumerate() {
                vc.allowlists.push(decode_global_allowlist(at, &format!("allowlists[{}].", i))?);
            }
        }
        Ok(vc)
    }

    pub fn translate(&mut self) -> Result<Config, String> {
        let mut keywords: BTreeSet<String> = BTreeSet::new();
        let mut ordered_rules: Vec<String> = Vec::new();
        let mut rules_map: HashMap<String, Rule> = HashMap::new();
        let mut rule_allowlists: Vec<(String, Vec<Arc<Allowlist>>)> = Vec::new();

        for vr in self.rules.iter_mut() {
            let path_pat = if !vr.path.is_empty() { Some(Regexp::must_compile(&vr.path)) } else { None };
            let regex_pat = if !vr.regex.is_empty() { Some(Regexp::must_compile(&vr.regex)) } else { None };
            let kws = match &vr.keywords {
                None => Vec::new(),
                Some(k) => {
                    let mut out = Vec::new();
                    for x in k {
                        let kw = x.to_lowercase();
                        keywords.insert(kw.clone());
                        out.push(kw);
                    }
                    out
                }
            };
            let tags = vr.tags.clone().unwrap_or_default();
            let mut cr = Rule {
                rule_id: vr.id.clone(),
                description: vr.description.clone(),
                regex: regex_pat,
                secret_group: vr.secret_group,
                entropy: vr.entropy,
                path: path_pat,
                keywords: kws,
                tags,
                skip_report: vr.skip_report,
                ..Default::default()
            };

            if let Some(al) = &vr.allow_list {
                if !vr.allowlists.is_empty() {
                    return Err(format!(
                        "{}: [rules.allowlist] is deprecated, it cannot be used alongside [[rules.allowlist]]",
                        cr.rule_id
                    ));
                }
                vr.allowlists.push(al.clone());
            }
            for a in &vr.allowlists {
                match parse_allowlist(a) {
                    Ok(al) => cr.allowlists.push(Arc::new(al)),
                    Err(e) => return Err(format!("{}: [[rules.allowlists]] {}", cr.rule_id, e)),
                }
            }

            for r in &vr.required {
                if r.id.is_empty() {
                    return Err(format!("{}: [[rules.required]] rule ID is empty", cr.rule_id));
                }
                cr.required_rules.push(Required {
                    rule_id: r.id.clone(),
                    within_lines: r.within_lines,
                    within_columns: r.within_columns,
                });
            }

            ordered_rules.push(cr.rule_id.clone());
            rules_map.insert(cr.rule_id.clone(), cr);
        }

        // ensure the required rules actually exist
        for id in &ordered_rules {
            if let Some(r) = rules_map.get(id) {
                for rr in &r.required_rules {
                    if !rules_map.contains_key(&rr.rule_id) {
                        return Err(format!(
                            "{}: [[rules.required]] rule ID '{}' does not exist",
                            r.rule_id, rr.rule_id
                        ));
                    }
                }
            }
        }

        let mut c = Config {
            title: self.title.clone(),
            description: self.description.clone(),
            extend: self.extend.clone(),
            rules: rules_map,
            keywords,
            ordered_rules,
            min_version: self.min_version.clone(),
            ..Default::default()
        };

        if EXTEND_DEPTH.load(Ordering::SeqCst) > 0 {
            c.path = self.config_path.clone();
        } else {
            c.path = config_file_used();
        }

        validate_min_version(&c.min_version, &c.path)?;

        if let Some(al) = &self.allow_list {
            if !self.allowlists.is_empty() {
                return Err("[allowlist] is deprecated, it cannot be used alongside [[allowlists]]".to_string());
            }
            self.allowlists.push(al.clone());
        }
        for a in &self.allowlists {
            let allowlist = match parse_allowlist(&a.allowlist) {
                Ok(al) => Arc::new(al),
                Err(e) => return Err(format!("[[allowlists]] {}", e)),
            };
            if !a.target_rules.is_empty() {
                for rid in &a.target_rules {
                    if let Some(entry) = rule_allowlists.iter_mut().find(|(k, _)| k == rid) {
                        entry.1.push(allowlist.clone());
                    } else {
                        rule_allowlists.push((rid.clone(), vec![allowlist.clone()]));
                    }
                }
            } else {
                c.allowlists.push(allowlist);
            }
        }

        let current_extend_depth = EXTEND_DEPTH.load(Ordering::SeqCst);
        if MAX_EXTEND_DEPTH != current_extend_depth {
            if !c.extend.path.is_empty() && c.extend.use_default {
                return Err("unable to load config due to extend.path and extend.useDefault being set".to_string());
            }
            if c.extend.use_default {
                c.extend_default()?;
            } else if !c.extend.path.is_empty() {
                c.extend_path()?;
            }
        }

        if current_extend_depth == 0 {
            let ids = c.rule_ids_in_order();
            for id in &ids {
                if let Some(rule) = c.rules.get(id) {
                    rule.validate()?;
                }
            }
            for (rule_id, allowlists) in rule_allowlists {
                match c.rules.get_mut(&rule_id) {
                    Some(rule) => rule.allowlists.extend(allowlists),
                    None => {
                        return Err(format!("[[allowlists]] target rule ID '{}' does not exist", rule_id));
                    }
                }
            }
        }

        Ok(c)
    }
}

fn validate_min_version(min_ver: &str, config_path: &str) -> Result<(), String> {
    if min_ver.is_empty() {
        logging::debug()
            .str("config path", config_path)
            .msg("no minVersion specified in config... consider adding minVersion to ensure compatibility.");
        return Ok(());
    }
    if version::VERSION == version::DEFAULT_MSG {
        logging::debug().str("required", min_ver).msg("dev build, skipping config version check.");
        return Ok(());
    }
    let min = match parse_semver(min_ver) {
        Some(v) => v,
        None => return Err(format!("invalid minVersion '{}': Malformed version: {}", min_ver, min_ver)),
    };
    let cur = match parse_semver(version::VERSION) {
        Some(v) => v,
        None => {
            return Err(format!(
                "unable to parse current version: Malformed version: {}",
                version::VERSION
            ))
        }
    };
    if cur < min {
        logging::warn()
            .str("required", min_ver)
            .str("current", version::VERSION)
            .str("config path", config_path)
            .msg("config requires a newer Gitleaks version...");
    }
    Ok(())
}

fn parse_semver(s: &str) -> Option<(u64, u64, u64)> {
    let s = s.strip_prefix('v').unwrap_or(s);
    let core = s.split(|c| c == '-' || c == '+').next()?;
    let parts: Vec<&str> = core.split('.').collect();
    if parts.is_empty() || parts.len() > 3 {
        return None;
    }
    let mut nums = [0u64; 3];
    for (i, p) in parts.iter().enumerate() {
        nums[i] = p.parse().ok()?;
    }
    Some((nums[0], nums[1], nums[2]))
}

fn parse_allowlist(a: &ViperRuleAllowlist) -> Result<Allowlist, String> {
    let match_condition = match a.condition.to_uppercase().as_str() {
        "AND" | "&&" => AllowlistMatchCondition::And,
        "" | "OR" | "||" => AllowlistMatchCondition::Or,
        _ => {
            return Err(format!(
                "unknown allowlist |condition| '{}' (expected 'and', 'or')",
                a.condition
            ))
        }
    };

    let mut regex_target = a.regex_target.clone();
    if !regex_target.is_empty() {
        match regex_target.as_str() {
            "secret" => regex_target = String::new(),
            "match" | "line" => {}
            _ => {
                return Err(format!(
                    "unknown allowlist |regexTarget| '{}' (expected 'match', 'line')",
                    regex_target
                ))
            }
        }
    }
    let regexes: Vec<Regexp> = a.regexes.iter().map(|r| Regexp::must_compile(r)).collect();
    let paths: Vec<Regexp> = a.paths.iter().map(|r| Regexp::must_compile(r)).collect();

    let mut allowlist = Allowlist {
        description: a.description.clone(),
        match_condition,
        commits: a.commits.clone(),
        paths,
        regex_target,
        regexes,
        stop_words: a.stop_words.clone(),
        ..Default::default()
    };
    allowlist.validate()?;
    Ok(allowlist)
}

/// Config is a configuration struct that contains rules and an allowlist if present.
#[derive(Clone, Debug, Default)]
pub struct Config {
    pub title: String,
    pub extend: Extend,
    pub path: String,
    pub description: String,
    pub rules: HashMap<String, Rule>,
    pub keywords: BTreeSet<String>,
    pub ordered_rules: Vec<String>,
    pub allowlists: Vec<Arc<Allowlist>>,
    pub min_version: String,
}

impl Config {
    pub fn get_ordered_rules(&self) -> Vec<Rule> {
        let mut out = Vec::new();
        for id in &self.ordered_rules {
            if let Some(r) = self.rules.get(id) {
                out.push(r.clone());
            }
        }
        out
    }

    /// Deterministic iteration order over the rules map: ordered rules first,
    /// then any remaining rules sorted by ID.
    pub fn rule_ids_in_order(&self) -> Vec<String> {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut out = Vec::new();
        for id in &self.ordered_rules {
            if self.rules.contains_key(id) && seen.insert(id.as_str()) {
                out.push(id.clone());
            }
        }
        let mut rest: Vec<String> = self.rules.keys().filter(|k| !seen.contains(k.as_str())).cloned().collect();
        rest.sort();
        out.extend(rest);
        out
    }

    fn extend_default(&mut self) -> Result<(), String> {
        EXTEND_DEPTH.fetch_add(1, Ordering::SeqCst);
        let table = parse_toml(DEFAULT_CONFIG)
            .map_err(|e| format!("failed to load extended default config, err: {}", e))?;
        let mut dvc = ViperConfig::unmarshal(&table)
            .map_err(|e| format!("failed to load extended default config, err: {}", e))?;
        let cfg = dvc
            .translate()
            .map_err(|e| format!("failed to load extended default config, err: {}", e))?;
        logging::debug().msg("extending config with default config");
        self.extend_with(cfg);
        Ok(())
    }

    fn extend_path(&mut self) -> Result<(), String> {
        EXTEND_DEPTH.fetch_add(1, Ordering::SeqCst);
        let path = self.extend.path.clone();
        set_config_file_used(&path);
        let table = read_config_file(&path).map_err(|e| format!("failed to load extended config, err: {}", e))?;
        let mut evc =
            ViperConfig::unmarshal(&table).map_err(|e| format!("failed to load extended config, err: {}", e))?;
        evc.config_path = path.clone();
        logging::debug().msg(&format!("extending config with {}", path));
        let cfg = evc.translate().map_err(|e| format!("failed to load extended config, err: {}", e))?;
        self.extend_with(cfg);
        Ok(())
    }

    fn extend_with(&mut self, extension: Config) {
        let config_name = if !self.extend.path.is_empty() { self.extend.path.clone() } else { "default".to_string() };
        let mut disabled: HashSet<String> = HashSet::new();
        for id in &self.extend.disabled_rules {
            if !extension.rules.contains_key(id) {
                logging::warn()
                    .str("rule-id", id)
                    .str("config", &config_name)
                    .msg("Disabled rule doesn't exist in extended config.");
            }
            disabled.insert(id.clone());
        }

        let ext_ids = extension.rule_ids_in_order();
        let mut ext_rules = extension.rules;
        for rule_id in ext_ids {
            let mut base_rule = match ext_rules.remove(&rule_id) {
                Some(r) => r,
                None => continue,
            };
            if disabled.contains(&rule_id) {
                logging::debug()
                    .str("rule-id", &rule_id)
                    .str("config", &config_name)
                    .msg("Ignoring rule from extended config.");
                continue;
            }
            match self.rules.get(&rule_id) {
                None => {
                    for k in &base_rule.keywords {
                        self.keywords.insert(k.clone());
                    }
                    self.rules.insert(rule_id.clone(), base_rule);
                    self.ordered_rules.push(rule_id.clone());
                }
                Some(current) => {
                    if !current.description.is_empty() {
                        base_rule.description = current.description.clone();
                    }
                    if current.entropy != 0.0 {
                        base_rule.entropy = current.entropy;
                    }
                    if current.secret_group != 0 {
                        base_rule.secret_group = current.secret_group;
                    }
                    if current.regex.is_some() {
                        base_rule.regex = current.regex.clone();
                    }
                    if current.path.is_some() {
                        base_rule.path = current.path.clone();
                    }
                    base_rule.tags.extend(current.tags.iter().cloned());
                    base_rule.keywords.extend(current.keywords.iter().cloned());
                    base_rule.allowlists.extend(current.allowlists.iter().cloned());
                    for k in &base_rule.keywords {
                        self.keywords.insert(k.clone());
                    }
                    self.rules.insert(rule_id.clone(), base_rule);
                }
            }
        }

        self.allowlists.extend(extension.allowlists);
        self.ordered_rules.sort();
    }
}

/// Load a config from the default embedded TOML
pub fn load_default() -> Result<Config, String> {
    let t = parse_toml(DEFAULT_CONFIG)?;
    let mut vc = ViperConfig::unmarshal(&t)?;
    vc.translate()
}
