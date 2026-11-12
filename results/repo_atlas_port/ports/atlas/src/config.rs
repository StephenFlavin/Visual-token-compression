// Port of the config package: rules, allowlists, viper-style loading and extension.

use crate::goregex::{must_compile, GoRegex};
use crate::logging;
use crate::toml::Value;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub const DEFAULT_CONFIG: &str = include_str!("gitleaks.toml");

static EXTEND_DEPTH: AtomicUsize = AtomicUsize::new(0);
const MAX_EXTEND_DEPTH: usize = 2;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MatchCondition {
    Or,
    And,
}

impl MatchCondition {
    pub fn as_str(&self) -> &'static str {
        match self {
            MatchCondition::Or => "OR",
            MatchCondition::And => "AND",
        }
    }
}

#[derive(Debug)]
pub struct Allowlist {
    pub description: String,
    pub match_condition: MatchCondition,
    pub commits: Vec<String>,
    pub paths: Vec<GoRegex>,
    pub regex_target: String,
    pub regexes: Vec<GoRegex>,
    pub stop_words: Vec<String>,
    commit_map: Option<HashSet<String>>,
}

impl Allowlist {
    pub fn validate(&mut self) -> Result<(), String> {
        if self.commits.is_empty() && self.paths.is_empty() && self.regexes.is_empty() && self.stop_words.is_empty() {
            return Err("must contain at least one check for: commits, paths, regexes, or stopwords".into());
        }
        if !self.commits.is_empty() {
            let mut set = HashSet::new();
            let mut list = vec![];
            for c in &self.commits {
                let k = c.to_lowercase().trim().to_string();
                if set.insert(k.clone()) {
                    list.push(k);
                }
            }
            self.commits = list;
            self.commit_map = Some(set);
        }
        if !self.stop_words.is_empty() {
            let mut set = HashSet::new();
            let mut list = vec![];
            for w in &self.stop_words {
                let k = w.to_lowercase();
                if set.insert(k.clone()) {
                    list.push(k);
                }
            }
            self.stop_words = list;
        }
        Ok(())
    }

    pub fn commit_allowed(&self, c: &str) -> (bool, String) {
        if c.is_empty() {
            return (false, String::new());
        }
        if let Some(m) = &self.commit_map {
            if m.contains(&c.to_lowercase()) {
                return (true, String::new());
            }
        } else if self.commits.iter().any(|x| x == c) {
            return (true, c.to_string());
        }
        (false, String::new())
    }

    pub fn path_allowed(&self, path: &str) -> bool {
        if path.is_empty() {
            return false;
        }
        self.paths.iter().any(|r| r.is_match(path))
    }

    pub fn regex_allowed(&self, secret: &str) -> bool {
        if secret.is_empty() {
            return false;
        }
        self.regexes.iter().any(|r| r.is_match(secret))
    }

    pub fn contains_stop_word(&self, s: &str) -> (bool, String) {
        if s.is_empty() {
            return (false, String::new());
        }
        let s = s.to_lowercase();
        // emulate leftmost (first) match of the stopword trie
        let mut best: Option<(usize, usize, &String)> = None;
        for w in &self.stop_words {
            if let Some(pos) = s.find(w.as_str()) {
                let end = pos + w.len();
                match best {
                    None => best = Some((end, pos, w)),
                    Some((be, bp, _)) => {
                        if end < be || (end == be && pos > bp) {
                            best = Some((end, pos, w))
                        }
                    }
                }
            }
        }
        match best {
            Some((_, _, w)) => (true, w.clone()),
            None => (false, String::new()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Required {
    pub rule_id: String,
    pub within_lines: Option<i64>,
    pub within_columns: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct Rule {
    pub rule_id: String,
    pub description: String,
    pub entropy: f64,
    pub secret_group: i64,
    pub regex: Option<GoRegex>,
    pub path: Option<GoRegex>,
    pub tags: Vec<String>,
    pub keywords: Vec<String>,
    pub allowlists: Vec<Arc<Allowlist>>,
    pub required_rules: Vec<Required>,
    pub skip_report: bool,
}

impl Rule {
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
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct Extend {
    pub path: String,
    pub url: String,
    pub use_default: bool,
    pub disabled_rules: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Config {
    pub title: String,
    pub extend: Extend,
    pub path: String,
    pub description: String,
    pub rules: BTreeMap<String, Rule>,
    pub keywords: BTreeSet<String>,
    pub ordered_rules: Vec<String>,
    pub allowlists: Vec<Arc<Allowlist>>,
    pub min_version: String,
}

impl Config {
    pub fn get_ordered_rules(&self) -> Vec<Rule> {
        self.ordered_rules.iter().filter_map(|id| self.rules.get(id).cloned()).collect()
    }
}

// ---- weakly typed value helpers (mapstructure with WeaklyTypedInput) ----

fn v_str(v: Option<&Value>) -> String {
    match v {
        Some(Value::Str(s)) => s.clone(),
        Some(Value::Int(i)) => i.to_string(),
        Some(Value::Float(f)) => crate::util::format_float_g(*f),
        Some(Value::Bool(b)) => if *b { "1".into() } else { "0".into() },
        _ => String::new(),
    }
}

fn v_int(v: Option<&Value>) -> Result<i64, String> {
    match v {
        None => Ok(0),
        Some(Value::Int(i)) => Ok(*i),
        Some(Value::Float(f)) => Ok(*f as i64),
        Some(Value::Bool(b)) => Ok(*b as i64),
        Some(Value::Str(s)) => {
            if s.is_empty() {
                Ok(0)
            } else {
                s.parse::<i64>().map_err(|e| format!("cannot parse '{}' as int: {}", s, e))
            }
        }
        _ => Err("expected an integer".into()),
    }
}

fn v_float(v: Option<&Value>) -> Result<f64, String> {
    match v {
        None => Ok(0.0),
        Some(Value::Int(i)) => Ok(*i as f64),
        Some(Value::Float(f)) => Ok(*f),
        Some(Value::Bool(b)) => Ok(if *b { 1.0 } else { 0.0 }),
        Some(Value::Str(s)) => {
            if s.is_empty() {
                Ok(0.0)
            } else {
                s.parse::<f64>().map_err(|e| format!("cannot parse '{}' as float: {}", s, e))
            }
        }
        _ => Err("expected a float".into()),
    }
}

fn v_bool(v: Option<&Value>) -> bool {
    match v {
        Some(Value::Bool(b)) => *b,
        Some(Value::Int(i)) => *i != 0,
        Some(Value::Str(s)) => matches!(s.as_str(), "1" | "t" | "T" | "true" | "TRUE" | "True"),
        _ => false,
    }
}

fn v_list(v: Option<&Value>) -> Option<Vec<String>> {
    match v {
        None => None,
        Some(Value::Array(a)) => Some(a.iter().map(|x| v_str(Some(x))).collect()),
        Some(Value::Str(s)) => {
            if s.is_empty() {
                Some(vec![])
            } else {
                Some(s.split(',').map(|x| x.to_string()).collect())
            }
        }
        Some(x) => Some(vec![v_str(Some(x))]),
    }
}

fn v_tables(v: Option<&Value>) -> Vec<&Value> {
    match v {
        Some(Value::Array(a)) => a.iter().filter(|x| matches!(x, Value::Table(_))).collect(),
        Some(t @ Value::Table(_)) => vec![t],
        _ => vec![],
    }
}

struct ViperAllowlist {
    description: String,
    condition: String,
    commits: Vec<String>,
    paths: Vec<String>,
    regex_target: String,
    regexes: Vec<String>,
    stop_words: Vec<String>,
    target_rules: Vec<String>,
}

fn read_allowlist(v: &Value) -> ViperAllowlist {
    ViperAllowlist {
        description: v_str(v.get("description")),
        condition: v_str(v.get("condition")),
        commits: v_list(v.get("commits")).unwrap_or_default(),
        paths: v_list(v.get("paths")).unwrap_or_default(),
        regex_target: v_str(v.get("regexTarget")),
        regexes: v_list(v.get("regexes")).unwrap_or_default(),
        stop_words: v_list(v.get("stopwords")).unwrap_or_default(),
        target_rules: v_list(v.get("targetRules")).unwrap_or_default(),
    }
}

fn parse_allowlist(a: &ViperAllowlist) -> Result<Allowlist, String> {
    let mc = match a.condition.to_uppercase().as_str() {
        "AND" | "&&" => MatchCondition::And,
        "" | "OR" | "||" => MatchCondition::Or,
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
    let regexes = a.regexes.iter().map(|r| must_compile(r)).collect();
    let paths = a.paths.iter().map(|r| must_compile(r)).collect();
    let mut al = Allowlist {
        description: a.description.clone(),
        match_condition: mc,
        commits: a.commits.clone(),
        paths,
        regex_target,
        regexes,
        stop_words: a.stop_words.clone(),
        commit_map: None,
    };
    al.validate()?;
    Ok(al)
}

/// Equivalent of ViperConfig.Translate.
pub fn translate(root: &Value, config_path: &str) -> Result<Config, String> {
    let mut keywords: BTreeSet<String> = BTreeSet::new();
    let mut ordered_rules: Vec<String> = vec![];
    let mut rules_map: BTreeMap<String, Rule> = BTreeMap::new();
    let mut rule_allowlists: Vec<(String, Arc<Allowlist>)> = vec![];

    for vr in v_tables(root.get("rules")) {
        let id = v_str(vr.get("id"));
        let path_s = v_str(vr.get("path"));
        let regex_s = v_str(vr.get("regex"));
        let path = if !path_s.is_empty() { Some(must_compile(&path_s)) } else { None };
        let regex = if !regex_s.is_empty() { Some(must_compile(&regex_s)) } else { None };
        let mut kws = vec![];
        if let Some(list) = v_list(vr.get("keywords")) {
            for k in list {
                let kw = k.to_lowercase();
                keywords.insert(kw.clone());
                kws.push(kw);
            }
        }
        let tags = v_list(vr.get("tags")).unwrap_or_default();
        let mut cr = Rule {
            rule_id: id.clone(),
            description: v_str(vr.get("description")),
            regex,
            secret_group: v_int(vr.get("secretGroup"))?,
            entropy: v_float(vr.get("entropy"))?,
            path,
            keywords: kws,
            tags,
            allowlists: vec![],
            required_rules: vec![],
            skip_report: v_bool(vr.get("skipReport")),
        };
        let old = vr.get("allowlist");
        let mut lists: Vec<&Value> = v_tables(vr.get("allowlists"));
        if let Some(o) = old {
            if let Value::Table(_) = o {
                if !lists.is_empty() {
                    return Err(format!(
                        "{}: [rules.allowlist] is deprecated, it cannot be used alongside [[rules.allowlist]]",
                        cr.rule_id
                    ));
                }
                lists.push(o);
            }
        }
        for a in lists {
            let al = parse_allowlist(&read_allowlist(a))
                .map_err(|e| format!("{}: [[rules.allowlists]] {}", cr.rule_id, e))?;
            cr.allowlists.push(Arc::new(al));
        }
        for r in v_tables(vr.get("required")) {
            let rid = v_str(r.get("id"));
            if rid.is_empty() {
                return Err(format!("{}: [[rules.required]] rule ID is empty", cr.rule_id));
            }
            let wl = match r.get("withinLines") {
                Some(v) => Some(v_int(Some(v))?),
                None => None,
            };
            let wc = match r.get("withinColumns") {
                Some(v) => Some(v_int(Some(v))?),
                None => None,
            };
            cr.required_rules.push(Required { rule_id: rid, within_lines: wl, within_columns: wc });
        }
        ordered_rules.push(cr.rule_id.clone());
        rules_map.insert(cr.rule_id.clone(), cr);
    }

    for r in rules_map.values() {
        for rr in &r.required_rules {
            if !rules_map.contains_key(&rr.rule_id) {
                return Err(format!(
                    "{}: [[rules.required]] rule ID '{}' does not exist",
                    r.rule_id, rr.rule_id
                ));
            }
        }
    }

    let ext = root.get("extend");
    let extend = Extend {
        path: v_str(ext.and_then(|e| e.get("path"))),
        url: v_str(ext.and_then(|e| e.get("url"))),
        use_default: v_bool(ext.and_then(|e| e.get("useDefault"))),
        disabled_rules: v_list(ext.and_then(|e| e.get("disabledRules"))).unwrap_or_default(),
    };

    let mut c = Config {
        title: v_str(root.get("title")),
        description: v_str(root.get("description")),
        extend,
        rules: rules_map,
        keywords,
        ordered_rules,
        min_version: v_str(root.get("minVersion")),
        path: config_path.to_string(),
        allowlists: vec![],
    };

    validate_min_version(&c.min_version, &c.path);

    let old = root.get("allowlist");
    let mut lists = v_tables(root.get("allowlists"));
    if let Some(o @ Value::Table(_)) = old {
        if !lists.is_empty() {
            return Err("[allowlist] is deprecated, it cannot be used alongside [[allowlists]]".into());
        }
        lists.push(o);
    }
    for a in lists {
        let va = read_allowlist(a);
        let al = Arc::new(parse_allowlist(&va).map_err(|e| format!("[[allowlists]] {}", e))?);
        if !va.target_rules.is_empty() {
            for rid in &va.target_rules {
                rule_allowlists.push((rid.clone(), al.clone()));
            }
        } else {
            c.allowlists.push(al);
        }
    }

    let current_depth = EXTEND_DEPTH.load(Ordering::SeqCst);
    if current_depth != MAX_EXTEND_DEPTH {
        if !c.extend.path.is_empty() && c.extend.use_default {
            return Err("unable to load config due to extend.path and extend.useDefault being set".into());
        }
        if c.extend.use_default {
            extend_default(&mut c)?;
        } else if !c.extend.path.is_empty() {
            extend_path(&mut c)?;
        }
    }

    if current_depth == 0 {
        for r in c.rules.values() {
            r.validate()?;
        }
        // group by rule id preserving order
        let mut seen: Vec<String> = vec![];
        for (rid, _) in &rule_allowlists {
            if !seen.contains(rid) {
                seen.push(rid.clone());
            }
        }
        for rid in seen {
            match c.rules.get_mut(&rid) {
                None => return Err(format!("[[allowlists]] target rule ID '{}' does not exist", rid)),
                Some(rule) => {
                    for (r, al) in &rule_allowlists {
                        if *r == rid {
                            rule.allowlists.push(al.clone());
                        }
                    }
                }
            }
        }
    }
    Ok(c)
}

fn validate_min_version(min_ver: &str, config_path: &str) {
    if min_ver.is_empty() {
        logging::debug()
            .str("config path", config_path)
            .msg("no minVersion specified in config... consider adding minVersion to ensure compatibility.");
        return;
    }
    if crate::VERSION == crate::DEFAULT_VERSION_MSG {
        logging::debug().str("required", min_ver).msg("dev build, skipping config version check.");
    }
}

fn extend_default(c: &mut Config) -> Result<(), String> {
    EXTEND_DEPTH.fetch_add(1, Ordering::SeqCst);
    let v = crate::toml::parse(DEFAULT_CONFIG)
        .map_err(|e| format!("failed to load extended default config, err: {}", e))?;
    let cfg = translate(&v, "").map_err(|e| format!("failed to load extended default config, err: {}", e))?;
    logging::debug().msg("extending config with default config");
    extend(c, cfg);
    Ok(())
}

fn extend_path(c: &mut Config) -> Result<(), String> {
    EXTEND_DEPTH.fetch_add(1, Ordering::SeqCst);
    let p = c.extend.path.clone();
    let content = std::fs::read_to_string(&p).map_err(|e| {
        format!("failed to load extended config, err: open {}: {}", p, crate::io_err_str(&e))
    })?;
    let v = crate::toml::parse(&content)
        .map_err(|e| format!("failed to load extended config, err: While parsing config: {}", e))?;
    logging::debug().msg(&format!("extending config with {}", p));
    let cfg = translate(&v, &p).map_err(|e| format!("failed to load extended config, err: {}", e))?;
    extend(c, cfg);
    Ok(())
}

fn extend(c: &mut Config, ext: Config) {
    let config_name = if !c.extend.path.is_empty() { c.extend.path.clone() } else { "default".to_string() };
    let mut disabled: HashSet<String> = HashSet::new();
    for id in &c.extend.disabled_rules {
        if !ext.rules.contains_key(id) {
            logging::warn()
                .str("rule-id", id)
                .str("config", &config_name)
                .msg("Disabled rule doesn't exist in extended config.");
        }
        disabled.insert(id.clone());
    }
    for (rule_id, base_rule) in ext.rules.into_iter() {
        if disabled.contains(&rule_id) {
            logging::debug()
                .str("rule-id", &rule_id)
                .str("config", &config_name)
                .msg("Ignoring rule from extended config.");
            continue;
        }
        match c.rules.get(&rule_id).cloned() {
            None => {
                for k in &base_rule.keywords {
                    c.keywords.insert(k.clone());
                }
                c.rules.insert(rule_id.clone(), base_rule);
                c.ordered_rules.push(rule_id);
            }
            Some(cur) => {
                let mut b = base_rule;
                if !cur.description.is_empty() {
                    b.description = cur.description.clone();
                }
                if cur.entropy != 0.0 {
                    b.entropy = cur.entropy;
                }
                if cur.secret_group != 0 {
                    b.secret_group = cur.secret_group;
                }
                if cur.regex.is_some() {
                    b.regex = cur.regex.clone();
                }
                if cur.path.is_some() {
                    b.path = cur.path.clone();
                }
                b.tags.extend(cur.tags.iter().cloned());
                b.keywords.extend(cur.keywords.iter().cloned());
                b.allowlists.extend(cur.allowlists.iter().cloned());
                for k in &b.keywords {
                    c.keywords.insert(k.clone());
                }
                c.rules.insert(rule_id, b);
            }
        }
    }
    c.allowlists.extend(ext.allowlists.into_iter());
    c.ordered_rules.sort();
}
