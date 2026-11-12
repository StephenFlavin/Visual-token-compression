//! A small port of the zerolog console logger used by gitleaks.
//!
//! Output format mirrors zerolog's `ConsoleWriter`:
//! `<time> <LVL> <message> <key>=<value> ...`

use std::collections::BTreeMap;
use std::fmt::Display;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Mutex;

use crate::gostr;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    Trace = -1,
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
    Fatal = 4,
    Panic = 5,
}

impl Level {
    fn formatted(self) -> &'static str {
        match self {
            Level::Trace => "TRC",
            Level::Debug => "DBG",
            Level::Info => "INF",
            Level::Warn => "WRN",
            Level::Error => "ERR",
            Level::Fatal => "FTL",
            Level::Panic => "PNC",
        }
    }
    fn color(self) -> i32 {
        match self {
            Level::Trace => COLOR_BLUE,
            Level::Debug => 0,
            Level::Info => COLOR_GREEN,
            Level::Warn => COLOR_YELLOW,
            Level::Error | Level::Fatal | Level::Panic => COLOR_RED,
        }
    }
}

const COLOR_RED: i32 = 31;
const COLOR_GREEN: i32 = 32;
const COLOR_YELLOW: i32 = 33;
const COLOR_BLUE: i32 = 34;
const COLOR_CYAN: i32 = 36;
const COLOR_BOLD: i32 = 1;
const COLOR_DARK_GRAY: i32 = 90;

static LEVEL: AtomicI32 = AtomicI32::new(Level::Info as i32);
static NO_COLOR: AtomicBool = AtomicBool::new(false);
static WRITE_LOCK: Mutex<()> = Mutex::new(());

pub fn set_level(l: Level) {
    LEVEL.store(l as i32, Ordering::SeqCst);
}

pub fn get_level() -> Level {
    match LEVEL.load(Ordering::SeqCst) {
        -1 => Level::Trace,
        0 => Level::Debug,
        1 => Level::Info,
        2 => Level::Warn,
        3 => Level::Error,
        4 => Level::Fatal,
        _ => Level::Panic,
    }
}

pub fn set_no_color(v: bool) {
    NO_COLOR.store(v, Ordering::SeqCst);
}

fn colorize(s: &str, c: i32, disabled: bool) -> String {
    let env_no_color = std::env::var("NO_COLOR").map(|v| !v.is_empty()).unwrap_or(false);
    if env_no_color || c == 0 || disabled {
        return s.to_string();
    }
    format!("\x1b[{}m{}\x1b[0m", c, s)
}

#[derive(Clone, Debug)]
enum FieldVal {
    Str(String),
    Raw(String),
}

/// Context holds a set of fields that are added to every event created from it
#[derive(Clone, Default)]
pub struct Ctx {
    fields: Vec<(String, FieldVal)>,
}

impl Ctx {
    pub fn str(mut self, k: &str, v: &str) -> Self {
        self.fields.push((k.to_string(), FieldVal::Str(v.to_string())));
        self
    }
    fn event(&self, level: Level) -> Event {
        let mut e = Event::new(level);
        if e.enabled {
            e.fields = self.fields.clone();
        }
        e
    }
    pub fn trace(&self) -> Event {
        self.event(Level::Trace)
    }
    pub fn debug(&self) -> Event {
        self.event(Level::Debug)
    }
    pub fn info(&self) -> Event {
        self.event(Level::Info)
    }
    pub fn warn(&self) -> Event {
        self.event(Level::Warn)
    }
    pub fn error(&self) -> Event {
        self.event(Level::Error)
    }
}

pub fn with() -> Ctx {
    Ctx::default()
}

pub struct Event {
    level: Level,
    enabled: bool,
    fields: Vec<(String, FieldVal)>,
}

impl Event {
    fn new(level: Level) -> Event {
        let enabled = level >= get_level();
        Event { level, enabled, fields: Vec::new() }
    }

    pub fn str(mut self, k: &str, v: &str) -> Self {
        if self.enabled {
            self.fields.push((k.to_string(), FieldVal::Str(v.to_string())));
        }
        self
    }

    pub fn strs(mut self, k: &str, v: &[String]) -> Self {
        if self.enabled {
            let mut s = String::from("[");
            for (i, x) in v.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&crate::gojson::encode_string(x));
            }
            s.push(']');
            self.fields.push((k.to_string(), FieldVal::Raw(s)));
        }
        self
    }

    pub fn int(mut self, k: &str, v: i64) -> Self {
        if self.enabled {
            self.fields.push((k.to_string(), FieldVal::Raw(v.to_string())));
        }
        self
    }

    pub fn bool(mut self, k: &str, v: bool) -> Self {
        if self.enabled {
            self.fields.push((k.to_string(), FieldVal::Raw(v.to_string())));
        }
        self
    }

    pub fn float32(mut self, k: &str, v: f32) -> Self {
        if self.enabled {
            self.fields.push((k.to_string(), FieldVal::Raw(gostr::format_float32_json(v))));
        }
        self
    }

    pub fn err<E: Display>(mut self, e: E) -> Self {
        if self.enabled {
            self.fields.push(("error".to_string(), FieldVal::Str(e.to_string())));
        }
        self
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn msg(self, m: &str) {
        self.write(Some(m));
    }

    pub fn send(self) {
        self.write(None);
    }

    fn write(self, m: Option<&str>) {
        if self.enabled {
            let line = self.format(m);
            let _g = WRITE_LOCK.lock();
            let stderr = std::io::stderr();
            let mut h = stderr.lock();
            let _ = h.write_all(line.as_bytes());
            let _ = h.flush();
        }
        if self.level == Level::Fatal {
            std::process::exit(1);
        }
        if self.level == Level::Panic {
            panic!("{}", m.unwrap_or(""));
        }
    }

    fn format(&self, m: Option<&str>) -> String {
        let no_color = NO_COLOR.load(Ordering::SeqCst);
        let mut buf = String::new();
        // timestamp
        let ts = chrono::Local::now().format("%-I:%M%p").to_string();
        let part = colorize(&ts, COLOR_DARK_GRAY, no_color);
        push_part(&mut buf, &part);
        // level
        let part = colorize(self.level.formatted(), self.level.color(), no_color);
        push_part(&mut buf, &part);
        // message
        if let Some(m) = m {
            if !m.is_empty() {
                let part = match self.level {
                    Level::Info | Level::Warn | Level::Error | Level::Fatal | Level::Panic => {
                        colorize(m, COLOR_BOLD, no_color)
                    }
                    _ => m.to_string(),
                };
                push_part(&mut buf, &part);
            }
        }
        // fields (map semantics: last one wins; sorted; "error" first)
        let mut map: BTreeMap<String, FieldVal> = BTreeMap::new();
        for (k, v) in &self.fields {
            map.insert(k.clone(), v.clone());
        }
        let mut keys: Vec<String> = map.keys().cloned().collect();
        if !buf.is_empty() && !keys.is_empty() {
            buf.push(' ');
        }
        if let Some(pos) = keys.iter().position(|k| k == "error") {
            let k = keys.remove(pos);
            keys.insert(0, k);
        }
        let n = keys.len();
        for (i, k) in keys.iter().enumerate() {
            let is_err = k == "error";
            buf.push_str(&colorize(&format!("{}=", k), COLOR_CYAN, no_color));
            let val = match &map[k] {
                FieldVal::Str(s) => {
                    if needs_quote(s) {
                        gostr::quote(s)
                    } else {
                        s.clone()
                    }
                }
                FieldVal::Raw(s) => s.clone(),
            };
            if is_err {
                buf.push_str(&colorize(&colorize(&val, COLOR_BOLD, no_color), COLOR_RED, no_color));
            } else {
                buf.push_str(&val);
            }
            if i < n - 1 {
                buf.push(' ');
            }
        }
        buf.push('\n');
        buf
    }
}

fn push_part(buf: &mut String, s: &str) {
    if !s.is_empty() {
        if !buf.is_empty() {
            buf.push(' ');
        }
        buf.push_str(s);
    }
}

fn needs_quote(s: &str) -> bool {
    s.bytes().any(|b| b < 0x20 || b > 0x7e || b == b' ' || b == b'\\' || b == b'"')
}

pub fn trace() -> Event {
    Event::new(Level::Trace)
}
pub fn debug() -> Event {
    Event::new(Level::Debug)
}
pub fn info() -> Event {
    Event::new(Level::Info)
}
pub fn warn() -> Event {
    Event::new(Level::Warn)
}
pub fn error() -> Event {
    Event::new(Level::Error)
}
pub fn fatal() -> Event {
    Event::new(Level::Fatal)
}
