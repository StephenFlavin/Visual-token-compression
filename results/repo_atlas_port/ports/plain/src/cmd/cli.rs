//! A small re-implementation of the cobra/pflag behaviour gitleaks relies on:
//! command lookup, flag parsing (with identical error messages) and help output.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Bool,
    Int,
    Uint,
    Str,
    StrSlice,
}

#[derive(Clone, Debug)]
pub struct FlagDef {
    pub name: &'static str,
    pub short: &'static str,
    pub kind: Kind,
    pub default: &'static str,
    pub usage: &'static str,
    pub no_opt_def: &'static str,
}

impl FlagDef {
    pub const fn new(name: &'static str, short: &'static str, kind: Kind, default: &'static str, usage: &'static str) -> Self {
        FlagDef { name, short, kind, default, usage, no_opt_def: "" }
    }
    fn no_opt(&self) -> &str {
        if self.kind == Kind::Bool {
            "true"
        } else {
            self.no_opt_def
        }
    }
    fn def_value(&self) -> String {
        match self.kind {
            Kind::StrSlice => format!("[{}]", self.default),
            _ => self.default.to_string(),
        }
    }
    fn is_zero_default(&self) -> bool {
        match self.kind {
            Kind::Bool => self.default == "false",
            Kind::Int | Kind::Uint => self.default == "0",
            Kind::Str => self.default.is_empty(),
            Kind::StrSlice => self.default.is_empty(),
        }
    }
    fn display(&self) -> String {
        if !self.short.is_empty() {
            format!("-{}, --{}", self.short, self.name)
        } else {
            format!("--{}", self.name)
        }
    }
}

pub struct CommandDef {
    pub name: &'static str,
    pub use_line: &'static str,
    pub aliases: Vec<&'static str>,
    pub short: &'static str,
    pub long: &'static str,
    pub hidden: bool,
    pub runnable: bool,
    pub max_args: Option<usize>,
    pub flags: Vec<FlagDef>,
    pub subcommands: Vec<CommandDef>,
    pub disable_flags_in_use_line: bool,
}

#[derive(Clone, Debug)]
pub enum FlagVal {
    Bool(bool),
    Int(i64),
    Uint(u64),
    Str(String),
    Slice(Vec<String>),
}

pub struct Parsed {
    pub path: Vec<String>, // command names from root
    pub values: HashMap<String, FlagVal>,
    pub changed: HashMap<String, bool>,
    pub args: Vec<String>,
}

impl Parsed {
    pub fn get_bool(&self, n: &str) -> bool {
        matches!(self.values.get(n), Some(FlagVal::Bool(true)))
    }
    pub fn get_int(&self, n: &str) -> i64 {
        match self.values.get(n) {
            Some(FlagVal::Int(i)) => *i,
            _ => 0,
        }
    }
    pub fn get_uint(&self, n: &str) -> u64 {
        match self.values.get(n) {
            Some(FlagVal::Uint(i)) => *i,
            _ => 0,
        }
    }
    pub fn get_string(&self, n: &str) -> String {
        match self.values.get(n) {
            Some(FlagVal::Str(s)) => s.clone(),
            _ => String::new(),
        }
    }
    pub fn get_slice(&self, n: &str) -> Vec<String> {
        match self.values.get(n) {
            Some(FlagVal::Slice(s)) => s.clone(),
            _ => Vec::new(),
        }
    }
}

pub enum Outcome {
    Run(Parsed),
    /// help text to print to stdout, exit 0
    Help(String),
    /// version text
    Version(String),
    /// error: (message, text to print to stderr before failing)
    Error(String, String),
}

fn parse_value(f: &FlagDef, v: &str) -> Result<FlagVal, String> {
    let flag_name = f.display();
    let wrap = |e: String| format!("invalid argument {} for {} flag: {}", crate::gostr::quote(v), crate::gostr::quote(&flag_name), e);
    match f.kind {
        Kind::Bool => match v {
            "1" | "t" | "T" | "TRUE" | "true" | "True" => Ok(FlagVal::Bool(true)),
            "0" | "f" | "F" | "FALSE" | "false" | "False" => Ok(FlagVal::Bool(false)),
            _ => Err(wrap(format!("strconv.ParseBool: parsing {}: invalid syntax", crate::gostr::quote(v)))),
        },
        Kind::Int => parse_go_int(v, true)
            .map(|x| FlagVal::Int(x as i64))
            .map_err(|e| wrap(format!("strconv.ParseInt: parsing {}: {}", crate::gostr::quote(v), e))),
        Kind::Uint => parse_go_int(v, false)
            .map(|x| FlagVal::Uint(x as u64))
            .map_err(|e| wrap(format!("strconv.ParseUint: parsing {}: {}", crate::gostr::quote(v), e))),
        Kind::Str => Ok(FlagVal::Str(v.to_string())),
        Kind::StrSlice => {
            if v.is_empty() {
                Ok(FlagVal::Slice(Vec::new()))
            } else {
                Ok(FlagVal::Slice(parse_csv_line(v)))
            }
        }
    }
}

fn parse_csv_line(v: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    let c: Vec<char> = v.chars().collect();
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if in_q {
            if ch == '"' {
                if i + 1 < c.len() && c[i + 1] == '"' {
                    cur.push('"');
                    i += 1;
                } else {
                    in_q = false;
                }
            } else {
                cur.push(ch);
            }
        } else if ch == '"' && cur.is_empty() {
            in_q = true;
        } else if ch == ',' {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(ch);
        }
        i += 1;
    }
    out.push(cur);
    out
}

/// strconv.ParseInt/ParseUint with base 0
fn parse_go_int(s: &str, signed: bool) -> Result<i128, String> {
    let syntax = || "invalid syntax".to_string();
    if s.is_empty() {
        return Err(syntax());
    }
    let (neg, body) = if let Some(r) = s.strip_prefix('-') {
        if !signed {
            return Err(syntax());
        }
        (true, r)
    } else if let Some(r) = s.strip_prefix('+') {
        if !signed {
            return Err(syntax());
        }
        (false, r)
    } else {
        (false, s)
    };
    let (base, digits) = if let Some(r) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        (16, r)
    } else if let Some(r) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
        (2, r)
    } else if let Some(r) = body.strip_prefix("0o").or_else(|| body.strip_prefix("0O")) {
        (8, r)
    } else if body.len() > 1 && body.starts_with('0') {
        (8, &body[1..])
    } else {
        (10, body)
    };
    let digits = digits.replace('_', "");
    if digits.is_empty() {
        return Err(syntax());
    }
    let v = u128::from_str_radix(&digits, base).map_err(|_| syntax())?;
    let max: u128 = if signed { if neg { 1u128 << 63 } else { (1u128 << 63) - 1 } } else { u64::MAX as u128 };
    if v > max {
        return Err("value out of range".to_string());
    }
    Ok(if neg { -(v as i128) } else { v as i128 })
}

fn rpad(s: &str, n: usize) -> String {
    format!("{:<width$}", s, width = n)
}

fn unquote_usage(f: &FlagDef) -> (String, String) {
    let usage = f.usage;
    if let Some(start) = usage.find('`') {
        if let Some(len) = usage[start + 1..].find('`') {
            let end = start + 1 + len;
            let name = usage[start + 1..end].to_string();
            let u = format!("{}{}{}", &usage[..start], name, &usage[end + 1..]);
            return (name, u);
        }
    }
    let name = match f.kind {
        Kind::Bool => "",
        Kind::Int => "int",
        Kind::Uint => "uint",
        Kind::Str => "string",
        Kind::StrSlice => "strings",
    };
    (name.to_string(), usage.to_string())
}

pub fn flag_usages(flags: &[&FlagDef]) -> String {
    let mut sorted: Vec<&FlagDef> = flags.to_vec();
    sorted.sort_by(|a, b| a.name.cmp(b.name));
    let mut lines = Vec::new();
    let mut maxlen = 0;
    for f in sorted {
        let mut line = if !f.short.is_empty() {
            format!("  -{}, --{}", f.short, f.name)
        } else {
            format!("      --{}", f.name)
        };
        let (varname, usage) = unquote_usage(f);
        if !varname.is_empty() {
            line.push(' ');
            line.push_str(&varname);
        }
        if !f.no_opt_def.is_empty() {
            match f.kind {
                Kind::Str => line.push_str(&format!("[=\"{}\"]", f.no_opt_def)),
                Kind::Bool => {
                    if f.no_opt_def != "true" {
                        line.push_str(&format!("[={}]", f.no_opt_def))
                    }
                }
                _ => line.push_str(&format!("[={}]", f.no_opt_def)),
            }
        }
        line.push('\x00');
        if line.len() > maxlen {
            maxlen = line.len();
        }
        line.push_str(&usage);
        if !f.is_zero_default() {
            if f.kind == Kind::Str {
                line.push_str(&format!(" (default {})", crate::gostr::quote(&f.def_value())));
            } else {
                line.push_str(&format!(" (default {})", f.def_value()));
            }
        }
        lines.push(line);
    }
    let mut out = String::new();
    for line in lines {
        let sidx = line.find('\x00').unwrap();
        let spacing = " ".repeat(maxlen - sidx);
        let usage = line[sidx + 1..].replace('\n', &format!("\n{}", " ".repeat(maxlen + 2)));
        out.push_str(&format!("{} {} {}\n", &line[..sidx], spacing, usage));
    }
    out
}

fn trim_trailing_ws(s: &str) -> String {
    s.trim_end_matches(|c: char| c.is_whitespace()).to_string()
}

pub struct App {
    pub root: CommandDef,
    pub persistent: Vec<FlagDef>,
    pub version: String,
}

fn help_flag(name: &str) -> FlagDef {
    let usage: &'static str = Box::leak(format!("help for {}", name).into_boxed_str());
    FlagDef::new("help", "h", Kind::Bool, "false", usage)
}

impl App {
    fn command_path(&self, path: &[&CommandDef]) -> String {
        path.iter().map(|c| c.name).collect::<Vec<_>>().join(" ")
    }

    fn local_flags(&self, path: &[&CommandDef]) -> Vec<FlagDef> {
        let cmd = path.last().unwrap();
        let mut v: Vec<FlagDef> = cmd.flags.clone();
        v.push(help_flag(cmd.name));
        if path.len() == 1 {
            v.extend(self.persistent.iter().cloned());
            if !self.version.is_empty() {
                let usage: &'static str = Box::leak(format!("version for {}", cmd.name).into_boxed_str());
                v.push(FlagDef::new("version", "", Kind::Bool, "false", usage));
            }
        }
        v
    }

    fn all_flags(&self, path: &[&CommandDef]) -> Vec<FlagDef> {
        let mut v = self.local_flags(path);
        if path.len() > 1 {
            v.extend(self.persistent.iter().cloned());
        }
        v
    }

    pub fn usage_string(&self, path: &[&CommandDef]) -> String {
        self.usage_string_opt(path, true)
    }

    pub fn usage_string_opt(&self, path: &[&CommandDef], lazy_flags: bool) -> String {
        let cmd = path.last().unwrap();
        let cp = self.command_path(path);
        let mut s = String::from("Usage:");
        let has_sub = cmd.subcommands.iter().any(|c| !c.hidden) || (path.len() == 1);
        if cmd.runnable {
            let mut use_line = if path.len() > 1 {
                format!("{} {}", self.command_path(&path[..path.len() - 1]), cmd.use_line)
            } else {
                cmd.use_line.to_string()
            };
            if !cmd.disable_flags_in_use_line && !use_line.contains("[flags]") {
                use_line.push_str(" [flags]");
            }
            s.push_str(&format!("\n  {}", use_line));
        }
        if has_sub {
            s.push_str(&format!("\n  {} [command]", cp));
        }
        if !cmd.aliases.is_empty() {
            let mut na = vec![cmd.name];
            na.extend(cmd.aliases.iter());
            s.push_str(&format!("\n\nAliases:\n  {}", na.join(", ")));
        }
        if has_sub {
            s.push_str("\n\nAvailable Commands:");
            let mut subs: Vec<(&str, &str, bool)> = cmd.subcommands.iter().map(|c| (c.name, c.short, c.hidden)).collect();
            if path.len() == 1 {
                subs.push(("help", "Help about any command", false));
            }
            subs.sort_by(|a, b| a.0.cmp(b.0));
            let maxlen = subs.iter().map(|c| c.0.len()).max().unwrap_or(0);
            let pad = if 11 > maxlen { 11 } else { maxlen };
            for (n, short, hidden) in subs {
                if hidden {
                    continue;
                }
                s.push_str(&format!("\n  {} {}", rpad(n, pad), short));
            }
        }
        let local = self.local_flags(path);
        let lrefs: Vec<&FlagDef> = local.iter().filter(|f| lazy_flags || (f.name != "help" && f.name != "version")).collect();
        s.push_str("\n\nFlags:\n");
        s.push_str(&trim_trailing_ws(&flag_usages(&lrefs)));
        if path.len() > 1 {
            let prefs: Vec<&FlagDef> = self.persistent.iter().collect();
            s.push_str("\n\nGlobal Flags:\n");
            s.push_str(&trim_trailing_ws(&flag_usages(&prefs)));
        }
        if has_sub {
            s.push_str(&format!("\n\nUse \"{} [command] --help\" for more information about a command.", cp));
        }
        s.push('\n');
        s
    }

    pub fn help_string(&self, path: &[&CommandDef]) -> String {
        let cmd = path.last().unwrap();
        let mut s = String::new();
        let text = if !cmd.long.is_empty() { cmd.long } else { cmd.short };
        if !text.is_empty() {
            s.push_str(&trim_trailing_ws(text));
            s.push_str("\n\n");
        }
        s.push_str(&self.usage_string(path));
        s
    }

    fn find_sub<'a>(cmd: &'a CommandDef, name: &str) -> Option<&'a CommandDef> {
        cmd.subcommands.iter().find(|c| c.name == name || c.aliases.contains(&name))
    }

    fn strip_flags(&self, args: &[String], flags: &[FlagDef]) -> Vec<String> {
        let mut cmds = Vec::new();
        let mut i = 0;
        // the help and version flags are added lazily by cobra, so they are unknown here
        let lazy = |f: &FlagDef| f.name == "help" || f.name == "version";
        let has_no_opt_long = |n: &str| flags.iter().any(|f| !lazy(f) && f.name == n && !f.no_opt().is_empty());
        let has_no_opt_short = |n: &str| flags.iter().any(|f| !lazy(f) && f.short == n && !f.no_opt().is_empty());
        while i < args.len() {
            let s = &args[i];
            i += 1;
            if s == "--" {
                break;
            }
            if (s.starts_with("--") && !s.contains('=') && !has_no_opt_long(&s[2..]))
                || (s.starts_with('-') && !s.starts_with("--") && !s.contains('=') && s.len() == 2 && !has_no_opt_short(&s[1..]))
            {
                if args.len() - i <= 0 {
                    break;
                }
                // Note: cobra breaks the loop if only one arg remains
                if args.len() - i <= 1 {
                    break;
                }
                i += 1;
                continue;
            }
            if !s.is_empty() && !s.starts_with('-') {
                cmds.push(s.clone());
            }
        }
        cmds
    }

    fn suggestions(&self, typed: &str) -> String {
        let mut out = Vec::new();
        let mut subs: Vec<&str> = self.root.subcommands.iter().filter(|c| !c.hidden).map(|c| c.name).collect();
        subs.push("help");
        subs.sort();
        for name in subs {
            let ld = levenshtein(&typed.to_lowercase(), &name.to_lowercase());
            if ld <= 2 || name.to_lowercase().starts_with(&typed.to_lowercase()) {
                out.push(name);
            }
        }
        if out.is_empty() {
            return String::new();
        }
        let mut s = String::from("\n\nDid you mean this?\n");
        for x in out {
            s.push_str(&format!("\t{}\n", x));
        }
        s
    }

    pub fn parse(&self, args: &[String]) -> Outcome {
        // locate the command
        let mut path: Vec<&CommandDef> = vec![&self.root];
        let mut rest: Vec<String> = args.to_vec();
        loop {
            let flags = self.all_flags(&path);
            let stripped = self.strip_flags(&rest, &flags);
            if stripped.is_empty() {
                break;
            }
            let next = &stripped[0];
            let cur = *path.last().unwrap();
            // the implicit help command
            if path.len() == 1 && next == "help" {
                let topic: Vec<String> = stripped[1..].to_vec();
                let mut hp: Vec<&CommandDef> = vec![&self.root];
                for t in &topic {
                    match Self::find_sub(hp.last().unwrap(), t) {
                        Some(c) => hp.push(c),
                        None => {
                            return Outcome::Help(format!(
                                "Unknown help topic [`{}`]\n{}",
                                topic.join(" "),
                                self.usage_string_opt(&[&self.root], false)
                            ))
                        }
                    }
                }
                return Outcome::Help(self.help_string(&hp));
            }
            match Self::find_sub(cur, next) {
                Some(c) => {
                    path.push(c);
                    if let Some(pos) = rest.iter().position(|a| a == next) {
                        rest.remove(pos);
                    }
                }
                None => {
                    if path.len() == 1 {
                        let msg = format!("unknown command {} for {}{}", crate::gostr::quote(next), crate::gostr::quote("gitleaks"), self.suggestions(next));
                        let stderr = format!("Error: {}\nRun 'gitleaks --help' for usage.\n", msg);
                        return Outcome::Error(msg, stderr);
                    }
                    break;
                }
            }
        }

        let flags = self.all_flags(&path);
        let mut values: HashMap<String, FlagVal> = HashMap::new();
        let mut changed: HashMap<String, bool> = HashMap::new();
        for f in &flags {
            let v = match f.kind {
                Kind::StrSlice => FlagVal::Slice(Vec::new()),
                _ => parse_value(f, f.default).unwrap_or(FlagVal::Str(String::new())),
            };
            values.insert(f.name.to_string(), v);
        }
        let mut positional = Vec::new();
        let flag_err = |msg: String| -> Outcome {
            let stderr = format!("Error: {}\n{}\n", msg, self.usage_string(&path));
            Outcome::Error(msg, stderr)
        };
        let mut set = |f: &FlagDef, v: &str, values: &mut HashMap<String, FlagVal>, changed: &mut HashMap<String, bool>| -> Result<(), String> {
            let nv = parse_value(f, v)?;
            if let (Kind::StrSlice, FlagVal::Slice(new)) = (f.kind, &nv) {
                if changed.contains_key(f.name) {
                    if let Some(FlagVal::Slice(old)) = values.get_mut(f.name) {
                        old.extend(new.iter().cloned());
                        return Ok(());
                    }
                }
            }
            values.insert(f.name.to_string(), nv);
            changed.insert(f.name.to_string(), true);
            Ok(())
        };
        let mut i = 0;
        while i < rest.len() {
            let a = rest[i].clone();
            i += 1;
            if a == "--" {
                positional.extend(rest[i..].iter().cloned());
                break;
            }
            if let Some(body) = a.strip_prefix("--") {
                if body.is_empty() || body.starts_with('-') || body.starts_with('=') {
                    return flag_err(format!("bad flag syntax: {}", a));
                }
                let (name, val) = match body.find('=') {
                    Some(p) => (&body[..p], Some(body[p + 1..].to_string())),
                    None => (body, None),
                };
                let f = match flags.iter().find(|f| f.name == name) {
                    Some(f) => f,
                    None => return flag_err(format!("unknown flag: --{}", name)),
                };
                let v = match val {
                    Some(v) => v,
                    None => {
                        if !f.no_opt().is_empty() {
                            f.no_opt().to_string()
                        } else if i < rest.len() {
                            i += 1;
                            rest[i - 1].clone()
                        } else {
                            return flag_err(format!("flag needs an argument: --{}", name));
                        }
                    }
                };
                if let Err(e) = set(f, &v, &mut values, &mut changed) {
                    return flag_err(e);
                }
            } else if a.starts_with('-') && a.len() > 1 {
                let shorts: Vec<char> = a[1..].chars().collect();
                let mut j = 0;
                while j < shorts.len() {
                    let c = shorts[j].to_string();
                    let f = match flags.iter().find(|f| f.short == c) {
                        Some(f) => f,
                        None => {
                            return flag_err(format!("unknown shorthand flag: '{}' in {}", c, a));
                        }
                    };
                    let rem: String = shorts[j + 1..].iter().collect();
                    let v;
                    if let Some(r) = rem.strip_prefix('=') {
                        v = r.to_string();
                        j = shorts.len();
                    } else if !f.no_opt().is_empty() {
                        v = f.no_opt().to_string();
                        j += 1;
                    } else if !rem.is_empty() {
                        v = rem;
                        j = shorts.len();
                    } else if i < rest.len() {
                        v = rest[i].clone();
                        i += 1;
                        j = shorts.len();
                    } else {
                        return flag_err(format!("flag needs an argument: '{}' in -{}", c, c));
                    }
                    if let Err(e) = set(f, &v, &mut values, &mut changed) {
                        return flag_err(e);
                    }
                }
            } else {
                positional.push(a);
            }
        }

        let parsed = Parsed { path: path.iter().map(|c| c.name.to_string()).collect(), values, changed, args: positional };
        if parsed.get_bool("help") {
            return Outcome::Help(self.help_string(&path));
        }
        if path.len() == 1 && parsed.get_bool("version") {
            return Outcome::Version(format!("gitleaks version {}\n", self.version));
        }
        let cmd = path.last().unwrap();
        if !cmd.runnable {
            if path.len() == 1 && !parsed.args.is_empty() {
                let msg = format!("unknown command {} for {}", crate::gostr::quote(&parsed.args[0]), crate::gostr::quote("gitleaks"));
                return Outcome::Error(msg.clone(), format!("Error: {}\nRun 'gitleaks --help' for usage.\n", msg));
            }
            return Outcome::Help(self.help_string(&path));
        }
        if let Some(max) = cmd.max_args {
            if parsed.args.len() > max {
                return flag_err(format!("accepts at most {} arg(s), received {}", max, parsed.args.len()));
            }
        }
        Outcome::Run(parsed)
    }
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[b.len()]
}
