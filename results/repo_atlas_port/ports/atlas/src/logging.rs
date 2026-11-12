// Port of the logging package: a zerolog ConsoleWriter-like logger writing to stderr.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub enum Level {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Fatal = 5,
}

static LEVEL: AtomicU8 = AtomicU8::new(2);
static NO_COLOR: AtomicBool = AtomicBool::new(false);

pub fn set_level(l: Level) {
    LEVEL.store(l as u8, Ordering::SeqCst);
}

pub fn level() -> Level {
    match LEVEL.load(Ordering::SeqCst) {
        0 => Level::Trace,
        1 => Level::Debug,
        2 => Level::Info,
        3 => Level::Warn,
        4 => Level::Error,
        _ => Level::Fatal,
    }
}

pub fn set_no_color(b: bool) {
    NO_COLOR.store(b, Ordering::SeqCst);
}

pub fn enabled(l: Level) -> bool {
    l >= level()
}

fn colorize(s: &str, c: u32) -> String {
    let disabled = NO_COLOR.load(Ordering::SeqCst)
        || std::env::var("NO_COLOR").map(|v| !v.is_empty()).unwrap_or(false)
        || c == 0;
    if disabled {
        s.to_string()
    } else {
        format!("\x1b[{}m{}\x1b[0m", c, s)
    }
}

fn kitchen_time() -> String {
    unsafe {
        let mut t: libc::time_t = 0;
        libc::time(&mut t);
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        let mut h = tm.tm_hour % 12;
        if h == 0 {
            h = 12;
        }
        let ampm = if tm.tm_hour >= 12 { "PM" } else { "AM" };
        format!("{}:{:02}{}", h, tm.tm_min, ampm)
    }
}

/// Field value for structured log output.
pub enum FV {
    S(String),
    I(i64),
    B(bool),
    F(f64),
}

pub struct Event {
    level: Level,
    fields: Vec<(String, FV)>,
    err: Option<String>,
}

pub fn needs_quote(s: &str) -> bool {
    s.bytes()
        .any(|b| b < 0x20 || b > 0x7e || b == b' ' || b == b'\\' || b == b'"')
}

impl Event {
    pub fn new(level: Level) -> Event {
        Event { level, fields: vec![], err: None }
    }
    pub fn str(mut self, k: &str, v: &str) -> Event {
        self.fields.push((k.to_string(), FV::S(v.to_string())));
        self
    }
    pub fn int(mut self, k: &str, v: i64) -> Event {
        self.fields.push((k.to_string(), FV::I(v)));
        self
    }
    pub fn boolean(mut self, k: &str, v: bool) -> Event {
        self.fields.push((k.to_string(), FV::B(v)));
        self
    }
    pub fn float(mut self, k: &str, v: f64) -> Event {
        self.fields.push((k.to_string(), FV::F(v)));
        self
    }
    pub fn err(mut self, e: &str) -> Event {
        self.err = Some(e.to_string());
        self
    }
    pub fn msg(self, m: &str) {
        if !enabled(self.level) && self.level != Level::Fatal {
            return;
        }
        if enabled(self.level) {
            let mut parts: Vec<String> = vec![];
            parts.push(colorize(&kitchen_time(), 90));
            let (lvl, c) = match self.level {
                Level::Trace => ("TRC", 34),
                Level::Debug => ("DBG", 0),
                Level::Info => ("INF", 32),
                Level::Warn => ("WRN", 33),
                Level::Error => ("ERR", 31),
                Level::Fatal => ("FTL", 31),
            };
            parts.push(colorize(lvl, c));
            if !m.is_empty() {
                match self.level {
                    Level::Trace | Level::Debug => parts.push(m.to_string()),
                    _ => parts.push(colorize(m, 1)),
                }
            }
            let mut line = parts.join(" ");
            // fields: error first, then sorted
            let mut fields: Vec<(String, String, bool)> = vec![];
            for (k, v) in self.fields.iter() {
                let vs = match v {
                    FV::S(s) => {
                        if needs_quote(s) {
                            crate::util::go_quote(s)
                        } else {
                            s.clone()
                        }
                    }
                    FV::I(i) => i.to_string(),
                    FV::B(b) => b.to_string(),
                    FV::F(f) => crate::util::format_float_g(*f),
                };
                fields.push((k.clone(), vs, false));
            }
            // dedupe keys keeping last (zerolog map semantics)
            let mut uniq: Vec<(String, String, bool)> = vec![];
            for f in fields.into_iter() {
                if let Some(pos) = uniq.iter().position(|u| u.0 == f.0) {
                    uniq[pos] = f;
                } else {
                    uniq.push(f);
                }
            }
            uniq.sort_by(|a, b| a.0.cmp(&b.0));
            if let Some(e) = &self.err {
                let v = if needs_quote(e) { crate::util::go_quote(e) } else { e.clone() };
                uniq.retain(|f| f.0 != "error");
                uniq.insert(0, ("error".to_string(), v, true));
            }
            if !uniq.is_empty() {
                line.push(' ');
                let mut first = true;
                for (k, v, is_err) in uniq {
                    if !first {
                        line.push(' ');
                    }
                    first = false;
                    line.push_str(&colorize(&format!("{}=", k), 36));
                    if is_err {
                        line.push_str(&colorize(&colorize(&v, 1), 31));
                    } else {
                        line.push_str(&v);
                    }
                }
            }
            line.push('\n');
            let stderr = std::io::stderr();
            let mut h = stderr.lock();
            let _ = h.write_all(line.as_bytes());
            let _ = h.flush();
        }
        if self.level == Level::Fatal {
            std::process::exit(1);
        }
    }
    pub fn send(self) {
        self.msg("")
    }
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
