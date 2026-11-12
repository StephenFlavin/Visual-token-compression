#![allow(dead_code)]
// gitleaks: Rust port of the Go implementation (cmd package lives here).

mod codec;
mod config;
mod detect;
mod gitdiff;
mod goregex;
mod inflate;
mod json;
mod logging;
mod lzma;
mod zstd;
mod report;
mod sources;
mod template;
mod toml;
mod util;

use detect::Detector;
use logging::Level;
use report::{Finding, Reporter};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;
use std::time::Instant;

pub const DEFAULT_VERSION_MSG: &str = "version is set by build process";
pub const VERSION: &str = DEFAULT_VERSION_MSG;

const BANNER: &str = "\n    ○\n    │╲\n    │ ○\n    ○ ░\n    ░    gitleaks\n\n";

pub fn io_err_str(e: &std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::NotFound => "no such file or directory".into(),
        std::io::ErrorKind::PermissionDenied => "permission denied".into(),
        _ => {
            let s = e.to_string();
            match s.find(" (os error") {
                Some(i) => s[..i].to_lowercase(),
                None => s,
            }
        }
    }
}

// ------------------------------------------------------------------ flags

#[derive(Clone, Copy, PartialEq)]
enum FT {
    Str,
    Int,
    Uint,
    Bool,
    Slice,
}

struct FlagDef {
    name: &'static str,
    short: Option<char>,
    ft: FT,
    default: &'static str,
    no_opt: Option<&'static str>,
}

const fn fd(name: &'static str, short: Option<char>, ft: FT, default: &'static str) -> FlagDef {
    FlagDef { name, short, ft, default, no_opt: None }
}

fn global_flags() -> Vec<FlagDef> {
    vec![
        fd("config", Some('c'), FT::Str, ""),
        fd("exit-code", None, FT::Int, "1"),
        fd("report-path", Some('r'), FT::Str, ""),
        fd("report-format", Some('f'), FT::Str, ""),
        fd("report-template", None, FT::Str, ""),
        fd("baseline-path", Some('b'), FT::Str, ""),
        fd("log-level", Some('l'), FT::Str, "info"),
        fd("verbose", Some('v'), FT::Bool, "false"),
        fd("no-color", None, FT::Bool, "false"),
        fd("max-target-megabytes", None, FT::Int, "0"),
        fd("ignore-gitleaks-allow", None, FT::Bool, "false"),
        FlagDef { name: "redact", short: None, ft: FT::Uint, default: "0", no_opt: Some("100") },
        fd("no-banner", None, FT::Bool, "false"),
        fd("enable-rule", None, FT::Slice, ""),
        fd("gitleaks-ignore-path", Some('i'), FT::Str, "."),
        fd("max-decode-depth", None, FT::Int, "5"),
        fd("max-archive-depth", None, FT::Int, "0"),
        fd("timeout", None, FT::Int, "0"),
        fd("diagnostics", None, FT::Str, ""),
        fd("diagnostics-dir", None, FT::Str, ""),
        fd("help", Some('h'), FT::Bool, "false"),
    ]
}

fn command_flags(cmd: &str) -> Vec<FlagDef> {
    match cmd {
        "git" => vec![
            fd("platform", None, FT::Str, ""),
            fd("staged", None, FT::Bool, "false"),
            fd("pre-commit", None, FT::Bool, "false"),
            fd("log-opts", None, FT::Str, ""),
        ],
        "dir" => vec![fd("follow-symlinks", None, FT::Bool, "false")],
        "detect" => vec![
            fd("no-git", None, FT::Bool, "false"),
            fd("pipe", None, FT::Bool, "false"),
            fd("follow-symlinks", None, FT::Bool, "false"),
            fd("source", Some('s'), FT::Str, "."),
            fd("log-opts", None, FT::Str, ""),
            fd("platform", None, FT::Str, ""),
        ],
        "protect" => vec![
            fd("staged", None, FT::Bool, "false"),
            fd("log-opts", None, FT::Str, ""),
            fd("source", Some('s'), FT::Str, "."),
        ],
        "" => vec![fd("version", None, FT::Bool, "false")],
        _ => vec![],
    }
}

struct Flags {
    values: HashMap<String, String>,
    slices: HashMap<String, Vec<String>>,
}

impl Flags {
    fn get_str(&self, n: &str) -> String {
        self.values.get(n).cloned().unwrap_or_default()
    }
    fn get_bool(&self, n: &str) -> bool {
        self.values.get(n).map(|v| v == "true").unwrap_or(false)
    }
    fn get_int(&self, n: &str) -> i64 {
        self.values.get(n).and_then(|v| v.parse().ok()).unwrap_or(0)
    }
    fn get_slice(&self, n: &str) -> Vec<String> {
        self.slices.get(n).cloned().unwrap_or_default()
    }
}

fn parse_bool(s: &str) -> Option<bool> {
    match s {
        "1" | "t" | "T" | "true" | "TRUE" | "True" => Some(true),
        "0" | "f" | "F" | "false" | "FALSE" | "False" => Some(false),
        _ => None,
    }
}

fn parse_go_int(s: &str) -> Option<i64> {
    let (neg, body) = if let Some(b) = s.strip_prefix('-') { (true, b) } else { (false, s.strip_prefix('+').unwrap_or(s)) };
    let body = body.replace('_', "");
    let v = if let Some(h) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        i64::from_str_radix(h, 16).ok()?
    } else if let Some(o) = body.strip_prefix("0o").or_else(|| body.strip_prefix("0O")) {
        i64::from_str_radix(o, 8).ok()?
    } else if let Some(b) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
        i64::from_str_radix(b, 2).ok()?
    } else if body.len() > 1 && body.starts_with('0') {
        i64::from_str_radix(&body[1..], 8).ok()?
    } else {
        body.parse::<i64>().ok()?
    };
    Some(if neg { -v } else { v })
}

fn set_flag(flags: &mut Flags, def: &FlagDef, val: &str, display: &str) -> Result<(), String> {
    let tname = match def.ft {
        FT::Int => "int",
        FT::Uint => "uint",
        FT::Bool => "bool",
        _ => "string",
    };
    let bad = |e: &str| format!("invalid argument {} for \"{}\" flag: {}", util::go_quote(val), display, e);
    match def.ft {
        FT::Str => {
            flags.values.insert(def.name.to_string(), val.to_string());
        }
        FT::Int => {
            let v = parse_go_int(val).ok_or_else(|| bad(&format!("strconv.ParseInt: parsing {}: invalid syntax", util::go_quote(val))))?;
            flags.values.insert(def.name.to_string(), v.to_string());
        }
        FT::Uint => {
            let v = parse_go_int(val)
                .filter(|v| *v >= 0 && !val.starts_with('-'))
                .ok_or_else(|| bad(&format!("strconv.ParseUint: parsing {}: invalid syntax", util::go_quote(val))))?;
            flags.values.insert(def.name.to_string(), v.to_string());
        }
        FT::Bool => {
            let v = parse_bool(val).ok_or_else(|| bad(&format!("strconv.ParseBool: parsing {}: invalid syntax", util::go_quote(val))))?;
            flags.values.insert(def.name.to_string(), v.to_string());
        }
        FT::Slice => {
            let entry = flags.slices.entry(def.name.to_string()).or_default();
            if !flags.values.contains_key(&format!("__set_{}", def.name)) {
                entry.clear();
            }
            if !val.is_empty() {
                // csv parsing
                for p in val.split(',') {
                    entry.push(p.trim_matches('"').to_string());
                }
            }
            flags.values.insert(format!("__set_{}", def.name), "1".into());
        }
    }
    let _ = tname;
    Ok(())
}

fn usage_error(cmd: &str, msg: &str) -> ! {
    eprintln!("Error: {}", msg);
    let help = help_text(cmd);
    // cobra prints the usage section (help without the short description)
    let usage = match help.find("Usage:") {
        Some(i) => &help[i..],
        None => help,
    };
    eprintln!("{}", usage);
    if msg.contains("unknown flag") {
        std::process::exit(126);
    }
    logging::fatal().msg(msg);
    std::process::exit(1);
}

fn help_text(cmd: &str) -> &'static str {
    match cmd {
        "git" => include_str!("help/git.txt"),
        "dir" => include_str!("help/dir.txt"),
        "stdin" => include_str!("help/stdin.txt"),
        "version" => include_str!("help/version.txt"),
        "detect" => include_str!("help/detect.txt"),
        "protect" => include_str!("help/protect.txt"),
        "completion" => include_str!("help/completion.txt"),
        _ => include_str!("help/root.txt"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // find the subcommand: first non-flag argument (cobra semantics, root flags may precede it)
    let gdefs = global_flags();
    let mut cmd = String::new();
    let mut cmd_pos = None;
    {
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            if a == "--" {
                break;
            }
            if a.starts_with("--") {
                let name = a[2..].split('=').next().unwrap_or("");
                if !a.contains('=') {
                    if let Some(d) = gdefs.iter().find(|d| d.name == name) {
                        if d.ft != FT::Bool && d.no_opt.is_none() {
                            i += 1;
                        }
                    }
                }
            } else if a.starts_with('-') && a.len() > 1 {
                let c = a[1..].chars().next().unwrap();
                if a.chars().count() == 2 {
                    if let Some(d) = gdefs.iter().find(|d| d.short == Some(c)) {
                        if d.ft != FT::Bool {
                            i += 1;
                        }
                    }
                }
            } else {
                cmd = a.clone();
                cmd_pos = Some(i);
                break;
            }
            i += 1;
        }
    }
    let canonical = match cmd.as_str() {
        "file" | "directory" => "dir".to_string(),
        c => c.to_string(),
    };
    let known = ["git", "dir", "stdin", "version", "detect", "protect", "help", "completion", ""];
    if !known.contains(&canonical.as_str()) {
        eprintln!("Error: unknown command {} for \"gitleaks\"", util::go_quote(&cmd));
        eprintln!("Run 'gitleaks --help' for usage.");
        logging::fatal().msg(&format!("unknown command {} for \"gitleaks\"", util::go_quote(&cmd)));
    }
    if canonical == "help" {
        let target = cmd_pos.and_then(|p| args.get(p + 1)).cloned().unwrap_or_default();
        let t = match target.as_str() {
            "file" | "directory" => "dir",
            t => t,
        };
        print!("{}", help_text(t));
        return;
    }
    let mut rest: Vec<String> = args.clone();
    if let Some(p) = cmd_pos {
        rest.remove(p);
    }
    let mut defs = global_flags();
    defs.extend(command_flags(&canonical));
    let mut flags = Flags { values: HashMap::new(), slices: HashMap::new() };
    for d in &defs {
        if d.ft != FT::Slice {
            flags.values.insert(d.name.to_string(), d.default.to_string());
        }
    }
    let mut positional: Vec<String> = vec![];
    let mut i = 0;
    while i < rest.len() {
        let a = rest[i].clone();
        if a == "--" {
            positional.extend(rest[i + 1..].iter().cloned());
            break;
        }
        if let Some(body) = a.strip_prefix("--") {
            let (name, val) = match body.find('=') {
                Some(p) => (&body[..p], Some(body[p + 1..].to_string())),
                None => (body, None),
            };
            let def = match defs.iter().find(|d| d.name == name) {
                Some(d) => d,
                None => usage_error(&canonical, &format!("unknown flag: --{}", name)),
            };
            let display = format!("--{}", def.name);
            let v = match val {
                Some(v) => v,
                None => {
                    if def.ft == FT::Bool {
                        "true".to_string()
                    } else if let Some(n) = def.no_opt {
                        n.to_string()
                    } else if i + 1 < rest.len() {
                        i += 1;
                        rest[i].clone()
                    } else {
                        usage_error(&canonical, &format!("flag needs an argument: --{}", name));
                    }
                }
            };
            if let Err(e) = set_flag(&mut flags, def, &v, &display) {
                usage_error(&canonical, &e);
            }
        } else if a.starts_with('-') && a.len() > 1 {
            let chars: Vec<char> = a[1..].chars().collect();
            let mut j = 0;
            while j < chars.len() {
                let c = chars[j];
                let def = match defs.iter().find(|d| d.short == Some(c)) {
                    Some(d) => d,
                    None => usage_error(&canonical, &format!("unknown shorthand flag: '{}' in {}", c, a)),
                };
                let display = format!("-{}, --{}", c, def.name);
                if def.ft == FT::Bool {
                    let (v, consumed_all) = if j + 1 < chars.len() && chars[j + 1] == '=' {
                        (chars[j + 2..].iter().collect::<String>(), true)
                    } else {
                        ("true".to_string(), false)
                    };
                    if let Err(e) = set_flag(&mut flags, def, &v, &display) {
                        usage_error(&canonical, &e);
                    }
                    if consumed_all {
                        break;
                    }
                    j += 1;
                    continue;
                }
                let v = if j + 1 < chars.len() {
                    let s: String = chars[j + 1..].iter().collect();
                    s.strip_prefix('=').map(|x| x.to_string()).unwrap_or(s)
                } else if i + 1 < rest.len() {
                    i += 1;
                    rest[i].clone()
                } else {
                    usage_error(&canonical, &format!("flag needs an argument: '{}' in -{}", c, c));
                };
                if let Err(e) = set_flag(&mut flags, def, &v, &display) {
                    usage_error(&canonical, &e);
                }
                break;
            }
        } else {
            positional.push(a);
        }
        i += 1;
    }

    if flags.get_bool("help") {
        print!("{}", help_text(&canonical));
        return;
    }
    if canonical.is_empty() {
        if flags.get_bool("version") {
            println!("gitleaks version {}", VERSION);
            return;
        }
        if !positional.is_empty() {
            // handled above as unknown command
        }
        print!("{}", help_text(""));
        return;
    }

    // initLog
    let ll = flags.get_str("log-level");
    match ll.to_lowercase().as_str() {
        "trace" => logging::set_level(Level::Trace),
        "debug" => logging::set_level(Level::Debug),
        "info" => logging::set_level(Level::Info),
        "warn" => logging::set_level(Level::Warn),
        "err" | "error" => logging::set_level(Level::Error),
        "fatal" => logging::set_level(Level::Fatal),
        _ => logging::warn().msg(&format!("unknown log level: {}", ll)),
    }

    match canonical.as_str() {
        "version" => {
            println!("{}", VERSION);
        }
        "completion" => {
            print!("{}", help_text("completion"));
        }
        "git" => {
            if positional.len() > 1 {
                usage_error("git", &format!("accepts at most 1 arg(s), received {}", positional.len()));
            }
            run_git(&flags, &positional);
        }
        "dir" => run_directory(&flags, &positional),
        "stdin" => run_stdin(&flags),
        "detect" => run_detect(&flags),
        "protect" => run_protect(&flags),
        _ => {}
    }
}

// ------------------------------------------------------------------ config / detector setup

fn read_config_file(path: &str) -> Result<toml::Value, String> {
    let content = std::fs::read(path).map_err(|e| format!("open {}: {}", path, io_err_str(&e)))?;
    toml::parse(&String::from_utf8_lossy(&content)).map_err(|e| format!("While parsing config: {}", e))
}

/// initConfig + Config(): returns the loaded config.
fn init_config(flags: &Flags, source: &str) -> config::Config {
    if !flags.get_bool("no-banner") {
        eprint!("{}", BANNER);
    }
    logging::debug().msg(&format!("using {} regex engine", goregex::VERSION));
    let cfg_path = flags.get_str("config");
    let mut used_path = String::new();
    let value: toml::Value;
    if !cfg_path.is_empty() {
        logging::debug().msg(&format!("using gitleaks config {} from `--config`", cfg_path));
        value = read_config_file(&cfg_path)
            .unwrap_or_else(|e| fatal_exit(&format!("unable to load gitleaks config, err: {}", e)));
        used_path = cfg_path.clone();
    } else if !std::env::var("GITLEAKS_CONFIG").unwrap_or_default().is_empty() {
        let p = std::env::var("GITLEAKS_CONFIG").unwrap();
        logging::debug().msg(&format!("using gitleaks config from GITLEAKS_CONFIG env var: {}", p));
        value = read_config_file(&p)
            .unwrap_or_else(|e| fatal_exit(&format!("unable to load gitleaks config, err: {}", e)));
        used_path = p;
    } else if !std::env::var("GITLEAKS_CONFIG_TOML").unwrap_or_default().is_empty() {
        let c = std::env::var("GITLEAKS_CONFIG_TOML").unwrap();
        value = match toml::parse(&c) {
            Ok(v) => v,
            Err(e) => {
                logging::fatal()
                    .err(&format!("While parsing config: {}", e))
                    .str("content", &c)
                    .msg("unable to load gitleaks config from GITLEAKS_CONFIG_TOML env var");
                unreachable!()
            }
        };
        logging::debug().str("content", &c).msg("using gitleaks config from GITLEAKS_CONFIG_TOML env var content");
    } else {
        let md = match std::fs::metadata(source) {
            Ok(m) => m,
            Err(e) => fatal_exit(&format!("stat {}: {}", source, io_err_str(&e))),
        };
        let p = util::join(source, ".gitleaks.toml");
        if !md.is_dir() {
            logging::debug().msg(&format!(
                "unable to load gitleaks config from {} since --source={} is a file, using default config",
                p, source
            ));
            value = toml::parse(config::DEFAULT_CONFIG).unwrap_or_else(|e| fatal_exit(&format!("err reading toml {}", e)));
        } else if !std::path::Path::new(&p).exists() {
            logging::debug().msg(&format!("no gitleaks config found in path {}, using default gitleaks config", p));
            value = toml::parse(config::DEFAULT_CONFIG)
                .unwrap_or_else(|e| fatal_exit(&format!("err reading default config toml {}", e)));
        } else {
            logging::debug().msg(&format!("using existing gitleaks config {} from `(--source)/.gitleaks.toml`", p));
            value = read_config_file(&p)
                .unwrap_or_else(|e| fatal_exit(&format!("unable to load gitleaks config, err: {}", e)));
            used_path = util::abs(&p);
        }
    }
    initial_diagnostics(flags);
    match config::translate(&value, &used_path) {
        Ok(mut c) => {
            c.path = flags.get_str("config");
            c
        }
        Err(e) => {
            logging::fatal().err(&e).msg("Failed to load config");
            unreachable!()
        }
    }
}

fn initial_diagnostics(flags: &Flags) {
    let d = flags.get_str("diagnostics");
    if !d.is_empty() {
        logging::info().msg("Starting diagnostics...");
    }
}

fn fatal_exit(msg: &str) -> ! {
    logging::fatal().msg(msg);
    std::process::exit(1)
}

fn file_exists(p: &str) -> bool {
    std::fs::metadata(p).map(|m| !m.is_dir()).unwrap_or(false)
}

fn build_detector(flags: &Flags, cfg: config::Config, source: &str) -> Detector {
    let ordered = cfg.get_ordered_rules();
    let all_rules = cfg.rules.clone();
    let mut d = Detector::new(cfg);
    d.max_decode_depth = flags.get_int("max-decode-depth");
    d.max_archive_depth = flags.get_int("max-archive-depth");
    d.no_color = flags.get_bool("no-color");
    if d.no_color {
        logging::set_no_color(true);
    }
    d.config.path = flags.get_str("config");
    if d.config.path.is_empty() {
        d.config.path = util::join(source, ".gitleaks.toml");
    }
    d.verbose = flags.get_bool("verbose");
    d.redact = flags.get_int("redact") as u64;
    d.max_target_mega_bytes = flags.get_int("max-target-megabytes");
    d.ignore_gitleaks_allow = flags.get_bool("ignore-gitleaks-allow");

    let gip = flags.get_str("gitleaks-ignore-path");
    if file_exists(&gip) {
        if let Err(e) = d.add_gitleaks_ignore(&gip) {
            logging::fatal().err(&e).msg("could not call AddGitleaksIgnore");
        }
    }
    let p = util::join(&gip, ".gitleaksignore");
    if file_exists(&p) {
        if let Err(e) = d.add_gitleaks_ignore(&p) {
            logging::fatal().err(&e).msg("could not call AddGitleaksIgnore");
        }
    }
    let p = util::join(source, ".gitleaksignore");
    if file_exists(&p) {
        if let Err(e) = d.add_gitleaks_ignore(&p) {
            logging::fatal().err(&e).msg("could not call AddGitleaksIgnore");
        }
    }

    let bp = flags.get_str("baseline-path");
    if !bp.is_empty() {
        if let Err(e) = d.add_baseline(&bp, source) {
            logging::error().msg(&format!(
                "Could not load baseline. The path must point of a gitleaks report generated using the default format: {}",
                e
            ));
        }
    }

    let rules = flags.get_slice("enable-rule");
    if !rules.is_empty() {
        logging::info().msg(&format!("Overriding enabled rules: {}", rules.join(", ")));
        let mut over = std::collections::BTreeMap::new();
        for r in &rules {
            match all_rules.get(r) {
                Some(rule) => {
                    over.insert(r.clone(), rule.clone());
                }
                None => fatal_exit(&format!("Requested rule {} not found in rules", r)),
            }
        }
        d.config.rules = over;
    }

    let report_path = flags.get_str("report-path");
    if !report_path.is_empty() {
        if report_path != report::STDOUT_REPORT_PATH {
            match std::fs::File::create(&report_path) {
                Ok(_) => {
                    let _ = std::fs::remove_file(&report_path);
                }
                Err(e) => {
                    logging::fatal()
                        .err(&format!("open {}: {}", report_path, io_err_str(&e)))
                        .msg(&format!("Report path is not writable: {}", report_path));
                }
            }
        }
        let mut report_format = flags.get_str("report-format");
        let report_template = flags.get_str("report-template");
        if report_format.is_empty() {
            let ext = util::ext(&report_path).to_lowercase();
            report_format = match ext.as_str() {
                ".csv" => "csv".into(),
                ".json" => "json".into(),
                ".sarif" => "sarif".into(),
                _ => fatal_exit(&format!("Unknown report format: {}", report_format)),
            };
            logging::debug().msg(&format!(
                "No report format specified, inferred {} from {}",
                util::go_quote(&report_format),
                util::go_quote(&ext)
            ));
        }
        let reporter = match report_format.trim().to_lowercase().as_str() {
            "csv" => Reporter::Csv,
            "json" => Reporter::Json,
            "junit" => Reporter::Junit,
            "sarif" => Reporter::Sarif(ordered),
            "template" => match template::Template::from_file(&report_template) {
                Ok(t) => Reporter::Template(t),
                Err(e) => {
                    logging::fatal().err(&e).msg("Invalid report template");
                    unreachable!()
                }
            },
            _ => fatal_exit(&format!("unknown report format {}", report_format)),
        };
        if !report_template.is_empty() && report_format != "template" {
            fatal_exit("Report format must be 'template' if --report-template is specified");
        }
        d.report_path = report_path;
        d.reporter = Some(reporter);
    }
    d
}

fn bytes_convert(bytes: u64) -> String {
    const KB: f64 = 1000.0;
    const MB: f64 = KB * 1000.0;
    const GB: f64 = MB * 1000.0;
    let b = bytes as f64;
    let (value, unit) = if b >= GB {
        ((bytes as f32) / GB as f32, "GB")
    } else if b >= MB {
        ((bytes as f32) / MB as f32, "MB")
    } else if b >= KB {
        ((bytes as f32) / KB as f32, "KB")
    } else if bytes >= 1 {
        (bytes as f32, "bytes")
    } else {
        return "0".into();
    };
    let s = format!("{:.2}", value);
    let s = s.strip_suffix(".00").map(|x| x.to_string()).unwrap_or(s);
    format!("{} {}", s, unit)
}

/// Go's time.Duration String after rounding as FormatDuration does.
fn format_duration(d: std::time::Duration) -> String {
    let ns = d.as_nanos() as i64;
    let mut scale: i64 = 100 * 1_000_000_000;
    while scale > ns && scale > 1 {
        scale /= 10;
    }
    let r = (scale / 100).max(1);
    let rounded = ((ns + r / 2) / r) * r;
    go_duration_string(rounded)
}

fn go_duration_string(ns: i64) -> String {
    if ns == 0 {
        return "0s".into();
    }
    if ns < 1_000 {
        return format!("{}ns", ns);
    }
    if ns < 1_000_000 {
        return format!("{}µs", trim_float(ns as f64 / 1e3));
    }
    if ns < 1_000_000_000 {
        return format!("{}ms", trim_float(ns as f64 / 1e6));
    }
    let total_s = ns as f64 / 1e9;
    let h = ns / 3_600_000_000_000;
    let m = (ns / 60_000_000_000) % 60;
    let s = total_s - (h * 3600 + m * 60) as f64;
    let mut out = String::new();
    if h > 0 {
        out.push_str(&format!("{}h", h));
    }
    if h > 0 || m > 0 {
        out.push_str(&format!("{}m", m));
    }
    out.push_str(&format!("{}s", trim_float(s)));
    out
}

fn trim_float(f: f64) -> String {
    let s = format!("{:.9}", f);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    s.to_string()
}

fn finding_summary_and_exit(d: &Detector, findings: &[Finding], exit_code: i64, start: Instant, err: Option<String>) {
    let total = d.total_bytes;
    let bytes_msg = format!("scanned ~{} bytes ({})", total, bytes_convert(total));
    if err.is_none() {
        logging::info().msg(&format!("{} in {}", bytes_msg, format_duration(start.elapsed())));
        if !findings.is_empty() {
            logging::warn().msg(&format!("leaks found: {}", findings.len()));
        } else {
            logging::info().msg("no leaks found");
        }
    } else {
        logging::warn().msg(&bytes_msg);
        logging::warn().msg(&format!("partial scan completed in {}", format_duration(start.elapsed())));
        if !findings.is_empty() {
            logging::warn().msg(&format!("{} leaks found in partial scan", findings.len()));
        } else {
            logging::warn().msg("no leaks found in partial scan");
        }
    }
    if let Some(r) = &d.reporter {
        let res = if d.report_path == report::STDOUT_REPORT_PATH {
            let stdout = std::io::stdout();
            let mut h = stdout.lock();
            let r = r.write(&mut h, findings);
            let _ = h.flush();
            r
        } else {
            match std::fs::File::create(&d.report_path) {
                Ok(mut f) => r.write(&mut f, findings),
                Err(e) => Err(e),
            }
        };
        if let Err(e) = res {
            logging::fatal().err(&e.to_string()).msg("failed to write report");
        }
    }
    if err.is_some() {
        std::process::exit(1);
    }
    if !findings.is_empty() {
        std::process::exit(exit_code as i32);
    }
}

// ------------------------------------------------------------------ commands

fn run_git(flags: &Flags, args: &[String]) {
    let start = Instant::now();
    let mut source = ".".to_string();
    if args.len() == 1 && !args[0].is_empty() {
        source = args[0].clone();
    }
    let cfg = init_config(flags, &source);
    let mut d = build_detector(flags, cfg, &source);
    let exit_code = flags.get_int("exit-code");
    let log_opts = flags.get_str("log-opts");
    let staged = flags.get_bool("staged");
    let pre_commit = flags.get_bool("pre-commit");
    let (cmd, platform) = if pre_commit || staged {
        (sources::new_git_diff_cmd(&source, staged), sources::Platform::None)
    } else {
        let p = match sources::Platform::from_str(&flags.get_str("platform")) {
            Ok(p) => p,
            Err(e) => fatal_exit_err(&e),
        };
        (sources::new_git_log_cmd(&source, &log_opts), p)
    };
    let remote = Arc::new(sources::new_remote_info(platform, &source));
    let err = scan_git(&mut d, cmd, remote);
    if let Some(e) = &err {
        logging::error().err(e).msg("failed to scan Git repository");
    }
    let findings = d.findings().clone();
    finding_summary_and_exit(&d, &findings, exit_code, start, err);
}

fn fatal_exit_err(e: &str) -> ! {
    logging::fatal().err(e).send();
    std::process::exit(1)
}

fn scan_git(d: &mut Detector, cmd: sources::GitCmd, remote: Arc<sources::RemoteInfo>) -> Option<String> {
    let cfg = d.config.clone();
    let git = sources::Git { cmd, config: &cfg, remote, max_archive_depth: d.max_archive_depth };
    let res = git.fragments(&mut |f, e| d.handle_fragment(f, e));
    logging::info().msg(&format!("{} commits scanned.", d.commits_scanned()));
    logging::debug().msg("Note: this number might be smaller than expected due to commits with no additions");
    res.err()
}

fn scan_files(d: &mut Detector, cfg: &config::Config, source: &str) -> Option<String> {
    let files = sources::Files {
        config: cfg,
        follow_symlinks: d.follow_symlinks,
        max_file_size: d.max_target_mega_bytes * 1_000_000,
        path: source.to_string(),
        max_archive_depth: d.max_archive_depth,
    };
    files.fragments(&mut |f, e| d.handle_fragment(f, e));
    None
}

fn scan_stdin(d: &mut Detector) -> Option<String> {
    let mut buf = vec![];
    if let Err(e) = std::io::stdin().read_to_end(&mut buf) {
        return Some(e.to_string());
    }
    let f = sources::File {
        content: buf,
        path: String::new(),
        symlink: String::new(),
        config: None,
        outer_paths: vec![],
        max_archive_depth: d.max_archive_depth,
        archive_depth: 0,
    };
    f.fragments(&mut |fr, e| d.handle_fragment(fr, e));
    None
}

fn run_directory(flags: &Flags, args: &[String]) {
    let mut source = ".".to_string();
    if args.len() == 1 && !args[0].is_empty() {
        source = args[0].clone();
    }
    let cfg = init_config(flags, &source);
    let start = Instant::now();
    let mut d = build_detector(flags, cfg.clone(), &source);
    d.follow_symlinks = flags.get_bool("follow-symlinks");
    let exit_code = flags.get_int("exit-code");
    let err = scan_files(&mut d, &cfg, &source);
    if let Some(e) = &err {
        logging::error().err(e).msg("failed scan directory");
    }
    let findings = d.findings().clone();
    finding_summary_and_exit(&d, &findings, exit_code, start, err);
}

fn run_stdin(flags: &Flags) {
    let start = Instant::now();
    let cfg = init_config(flags, ".");
    let mut d = build_detector(flags, cfg, "");
    let exit_code = flags.get_int("exit-code");
    let err = scan_stdin(&mut d);
    if let Some(e) = &err {
        logging::fatal().err(e).msg("failed scan input from stdin");
    }
    let findings = d.findings().clone();
    finding_summary_and_exit(&d, &findings, exit_code, start, err);
}

fn run_detect(flags: &Flags) {
    let start = Instant::now();
    let source = flags.get_str("source");
    let cfg = init_config(flags, &source);
    let mut d = build_detector(flags, cfg.clone(), &source);
    d.follow_symlinks = flags.get_bool("follow-symlinks");
    let exit_code = flags.get_int("exit-code");
    let err;
    if flags.get_bool("no-git") {
        err = scan_files(&mut d, &cfg, &source);
        if let Some(e) = &err {
            logging::error().err(e).msg("failed to scan directory");
        }
    } else if flags.get_bool("pipe") {
        err = scan_stdin(&mut d);
        if let Some(e) = &err {
            logging::fatal().err(e).msg("failed scan input from stdin");
        }
    } else {
        let cmd = sources::new_git_log_cmd(&source, &flags.get_str("log-opts"));
        let p = match sources::Platform::from_str(&flags.get_str("platform")) {
            Ok(p) => p,
            Err(e) => fatal_exit_err(&e),
        };
        let remote = Arc::new(sources::new_remote_info(p, &source));
        err = scan_git(&mut d, cmd, remote);
        if let Some(e) = &err {
            logging::error().err(e).msg("failed to scan Git repository");
        }
    }
    let findings = d.findings().clone();
    finding_summary_and_exit(&d, &findings, exit_code, start, err);
}

fn run_protect(flags: &Flags) {
    let start = Instant::now();
    let source = flags.get_str("source");
    let cfg = init_config(flags, &source);
    let mut d = build_detector(flags, cfg, &source);
    let exit_code = flags.get_int("exit-code");
    let staged = flags.get_bool("staged");
    let cmd = sources::new_git_diff_cmd(&source, staged);
    let remote = Arc::new(sources::RemoteInfo { platform: sources::Platform::None, url: String::new() });
    let err = scan_git(&mut d, cmd, remote);
    if let Some(e) = &err {
        logging::error().err(e).msg("failed to scan Git repository");
    }
    let findings = d.findings().clone();
    finding_summary_and_exit(&d, &findings, exit_code, start, err);
}
