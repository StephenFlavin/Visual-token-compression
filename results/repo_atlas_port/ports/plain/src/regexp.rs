//! Regular expression wrapper.
//!
//! Gitleaks rules are written in Go's RE2 syntax. Rust's `regex` crate is very
//! close, but there are a few semantic differences (Perl classes are ASCII-only
//! in Go, `\<` is a literal in Go, `{` may be a literal in Go, nested classes
//! and set operators don't exist in Go...). This module translates Go syntax into
//! an equivalent Rust pattern before compiling it.

pub const VERSION: &str = "stdlib";

#[derive(Clone, Debug)]
pub struct Regexp {
    src: String,
    re: regex::Regex,
}

impl Regexp {
    pub fn compile(pattern: &str) -> Result<Regexp, String> {
        let translated = translate(pattern)?;
        let re = regex::RegexBuilder::new(&translated)
            .size_limit(1 << 30)
            .dfa_size_limit(1 << 28)
            .nest_limit(1000)
            .build()
            .map_err(|e| simplify_err(&e.to_string()))?;
        Ok(Regexp { src: pattern.to_string(), re })
    }

    /// Mirrors regexp.MustCompile: panics (exit code 2) on invalid patterns.
    pub fn must_compile(pattern: &str) -> Regexp {
        match Regexp::compile(pattern) {
            Ok(r) => r,
            Err(e) => {
                let q = if can_backquote(pattern) {
                    format!("`{}`", pattern)
                } else {
                    crate::gostr::quote(pattern)
                };
                eprintln!("panic: regexp: Compile({}): error parsing regexp: {}\n", q, e);
                eprintln!("goroutine 1 [running]:");
                std::process::exit(2);
            }
        }
    }

    pub fn as_str(&self) -> &str {
        &self.src
    }

    pub fn is_match(&self, s: &str) -> bool {
        self.re.is_match(s)
    }

    /// FindString(s) != ""
    pub fn find_string_nonempty(&self, s: &str) -> bool {
        match self.re.find(s) {
            Some(m) => !m.as_str().is_empty(),
            None => false,
        }
    }

    pub fn find_all_index(&self, s: &str) -> Vec<(usize, usize)> {
        self.re.find_iter(s).map(|m| (m.start(), m.end())).collect()
    }

    pub fn find_all_strings(&self, s: &str) -> Vec<String> {
        self.re.find_iter(s).map(|m| m.as_str().to_string()).collect()
    }

    /// FindStringSubmatch: returns None if no match. Non-participating groups are "".
    pub fn find_submatch(&self, s: &str) -> Option<Vec<String>> {
        self.re.captures(s).map(|caps| {
            (0..caps.len())
                .map(|i| caps.get(i).map(|m| m.as_str().to_string()).unwrap_or_default())
                .collect()
        })
    }

    /// FindAllStringSubmatch
    pub fn find_all_submatch(&self, s: &str) -> Vec<Vec<String>> {
        self.re
            .captures_iter(s)
            .map(|caps| {
                (0..caps.len())
                    .map(|i| caps.get(i).map(|m| m.as_str().to_string()).unwrap_or_default())
                    .collect()
            })
            .collect()
    }

    /// FindAllStringSubmatchIndex: -1 for non-participating groups.
    pub fn find_all_submatch_index(&self, s: &str) -> Vec<Vec<i64>> {
        self.re
            .captures_iter(s)
            .map(|caps| {
                let mut v = Vec::with_capacity(caps.len() * 2);
                for i in 0..caps.len() {
                    match caps.get(i) {
                        Some(m) => {
                            v.push(m.start() as i64);
                            v.push(m.end() as i64);
                        }
                        None => {
                            v.push(-1);
                            v.push(-1);
                        }
                    }
                }
                v
            })
            .collect()
    }

    pub fn num_subexp(&self) -> usize {
        self.re.captures_len() - 1
    }
}

fn can_backquote(s: &str) -> bool {
    for c in s.chars() {
        if c == '`' || c == '\u{FEFF}' {
            return false;
        }
        if (c as u32) < 0x20 && c != '\t' {
            return false;
        }
        if c == '\x7f' {
            return false;
        }
    }
    true
}

fn simplify_err(e: &str) -> String {
    // Rust's errors are multi-line with a caret diagram; keep the last line
    let lines: Vec<&str> = e.lines().filter(|l| !l.trim().is_empty()).collect();
    if let Some(l) = lines.iter().rev().find(|l| l.starts_with("error:")) {
        return l.trim_start_matches("error:").trim().to_string();
    }
    lines.last().map(|s| s.to_string()).unwrap_or_default()
}

const D_CLASS: &str = "0-9";
const ND_CLASS: &str = "\\x00-/:-\\x{10FFFF}";
const W_CLASS: &str = "0-9A-Z_a-z";
const NW_CLASS: &str = "\\x00-/:-@\\[-\\^`\\{-\\x{10FFFF}";
const S_CLASS: &str = "\\t\\n\\x0C\\r ";
const NS_CLASS: &str = "\\x00-\\x08\\x0B\\x0E-\\x1F!-\\x{10FFFF}";

fn perl_class(c: char) -> Option<&'static str> {
    match c {
        'd' => Some(D_CLASS),
        'D' => Some(ND_CLASS),
        'w' => Some(W_CLASS),
        'W' => Some(NW_CLASS),
        's' => Some(S_CLASS),
        'S' => Some(NS_CLASS),
        _ => None,
    }
}

fn emit_literal(out: &mut String, c: char) {
    if c.is_ascii_alphanumeric() || (!c.is_ascii() && !c.is_control()) || c == ' ' {
        out.push(c);
    } else {
        out.push_str(&format!("\\x{{{:X}}}", c as u32));
    }
}

/// Parse an escape sequence that denotes a single character (Go parseEscape).
/// `chars[i]` is the character right after the backslash.
/// Returns the character and the new index.
fn parse_escape_char(chars: &[char], mut i: usize) -> Result<(char, usize), String> {
    if i >= chars.len() {
        return Err("trailing backslash at end of expression: ``".to_string());
    }
    let c = chars[i];
    i += 1;
    match c {
        '1'..='7' | '0' => {
            if c != '0' && (i >= chars.len() || chars[i] < '0' || chars[i] > '7') {
                return Err(format!("invalid escape sequence: `\\{}`", c));
            }
            let mut r = c as u32 - '0' as u32;
            for _ in 1..3 {
                if i >= chars.len() || chars[i] < '0' || chars[i] > '7' {
                    break;
                }
                r = r * 8 + (chars[i] as u32 - '0' as u32);
                i += 1;
            }
            Ok((char::from_u32(r).unwrap_or('\u{FFFD}'), i))
        }
        'x' => {
            if i >= chars.len() {
                return Err("invalid escape sequence: `\\x`".to_string());
            }
            if chars[i] == '{' {
                i += 1;
                let mut r: u32 = 0;
                let mut nhex = 0;
                while i < chars.len() && chars[i] != '}' {
                    let d = chars[i].to_digit(16).ok_or_else(|| "invalid escape sequence".to_string())?;
                    r = r.saturating_mul(16).saturating_add(d);
                    if r > 0x10FFFF {
                        return Err("invalid escape sequence".to_string());
                    }
                    nhex += 1;
                    i += 1;
                }
                if i >= chars.len() || nhex == 0 {
                    return Err("invalid escape sequence".to_string());
                }
                i += 1;
                Ok((char::from_u32(r).unwrap_or('\u{FFFD}'), i))
            } else {
                if i + 1 >= chars.len() + 0 && i + 1 > chars.len() {
                    return Err("invalid escape sequence".to_string());
                }
                let a = chars.get(i).and_then(|c| c.to_digit(16));
                let b = chars.get(i + 1).and_then(|c| c.to_digit(16));
                match (a, b) {
                    (Some(a), Some(b)) => Ok((char::from_u32(a * 16 + b).unwrap(), i + 2)),
                    _ => Err("invalid escape sequence".to_string()),
                }
            }
        }
        'a' => Ok(('\x07', i)),
        'f' => Ok(('\x0C', i)),
        'n' => Ok(('\n', i)),
        'r' => Ok(('\r', i)),
        't' => Ok(('\t', i)),
        'v' => Ok(('\x0B', i)),
        c if (c as u32) < 0x80 && !c.is_ascii_alphanumeric() => Ok((c, i)),
        c => Err(format!("invalid escape sequence: `\\{}`", c)),
    }
}

/// Copy a \p / \P unicode class starting at chars[i] == 'p' or 'P'
fn copy_unicode_class(chars: &[char], mut i: usize, out: &mut String) -> Result<usize, String> {
    out.push('\\');
    out.push(chars[i]);
    i += 1;
    if i >= chars.len() {
        return Err("invalid character class range".to_string());
    }
    if chars[i] == '{' {
        while i < chars.len() && chars[i] != '}' {
            out.push(chars[i]);
            i += 1;
        }
        if i >= chars.len() {
            return Err("invalid character class range".to_string());
        }
        out.push('}');
        i += 1;
    } else {
        out.push(chars[i]);
        i += 1;
    }
    Ok(i)
}

/// Checks whether chars[i..] starts a valid Go repeat `{n}`, `{n,}`, `{n,m}`
fn valid_repeat(chars: &[char], i: usize) -> Option<usize> {
    let mut j = i + 1;
    let start = j;
    while j < chars.len() && chars[j].is_ascii_digit() {
        j += 1;
    }
    if j == start {
        return None;
    }
    if j < chars.len() && chars[j] == ',' {
        j += 1;
        while j < chars.len() && chars[j].is_ascii_digit() {
            j += 1;
        }
    }
    if j < chars.len() && chars[j] == '}' {
        Some(j + 1)
    } else {
        None
    }
}

fn translate(pattern: &str) -> Result<String, String> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::with_capacity(pattern.len() * 2);
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\\' => {
                if i + 1 >= chars.len() {
                    return Err("trailing backslash at end of expression: ``".to_string());
                }
                let n = chars[i + 1];
                if let Some(cls) = perl_class(n) {
                    out.push('[');
                    out.push_str(cls);
                    out.push(']');
                    i += 2;
                    continue;
                }
                match n {
                    'b' => {
                        out.push_str("(?-u:\\b)");
                        i += 2;
                    }
                    'B' => {
                        out.push_str("\\B");
                        i += 2;
                    }
                    'A' | 'z' => {
                        out.push('\\');
                        out.push(n);
                        i += 2;
                    }
                    'Q' => {
                        i += 2;
                        while i < chars.len() {
                            if chars[i] == '\\' && i + 1 < chars.len() && chars[i + 1] == 'E' {
                                i += 2;
                                break;
                            }
                            emit_literal(&mut out, chars[i]);
                            i += 1;
                        }
                    }
                    'p' | 'P' => {
                        i = copy_unicode_class(&chars, i + 1, &mut out)?;
                    }
                    _ => {
                        let (ch, ni) = parse_escape_char(&chars, i + 1)?;
                        emit_literal(&mut out, ch);
                        i = ni;
                    }
                }
            }
            '[' => {
                i = translate_class(&chars, i, &mut out)?;
            }
            '(' if i + 1 < chars.len() && chars[i + 1] == '?' => {
                // group syntax: (?P<name>, (?<name>, (?flags) or (?flags:
                let is_named = (i + 2 < chars.len() && chars[i + 2] == '<')
                    || (i + 3 < chars.len() && chars[i + 2] == 'P' && chars[i + 3] == '<');
                let stop = if is_named { '>' } else { ':' };
                while i < chars.len() {
                    let ch = chars[i];
                    out.push(ch);
                    i += 1;
                    if ch == stop || (!is_named && ch == ')') {
                        break;
                    }
                }
            }
            '{' => {
                if let Some(end) = valid_repeat(&chars, i) {
                    for ch in &chars[i..end] {
                        out.push(*ch);
                    }
                    i = end;
                } else {
                    out.push_str("\\{");
                    i += 1;
                }
            }
            '}' => {
                out.push_str("\\}");
                i += 1;
            }
            '#' | '&' | '~' | '"' | '\'' | '<' | '>' | '/' | '%' | '@' | '!' | ',' | ';' | '=' | '`' | ' ' => {
                emit_literal(&mut out, c);
                i += 1;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    Ok(out)
}

/// Translate a Go character class starting at chars[i] == '['.
fn translate_class(chars: &[char], start: usize, out: &mut String) -> Result<usize, String> {
    let mut i = start + 1;
    let mut cls = String::new();
    cls.push('[');
    if i < chars.len() && chars[i] == '^' {
        cls.push('^');
        i += 1;
    }
    let mut first = true;
    let mut empty = true;
    loop {
        if i >= chars.len() {
            let rest: String = chars[start..].iter().collect();
            return Err(format!("missing closing ]: `{}`", rest));
        }
        if chars[i] == ']' && !first {
            i += 1;
            break;
        }
        first = false;
        // POSIX named class
        if i + 2 < chars.len() && chars[i] == '[' && chars[i + 1] == ':' {
            // find ":]"
            let mut j = i + 2;
            let mut found = None;
            while j + 1 < chars.len() {
                if chars[j] == ':' && chars[j + 1] == ']' {
                    found = Some(j);
                    break;
                }
                j += 1;
            }
            if let Some(j) = found {
                let name: String = chars[i..j + 2].iter().collect();
                cls.push_str(&name);
                empty = false;
                i = j + 2;
                continue;
            }
        }
        if chars[i] == '\\' && i + 1 < chars.len() {
            let n = chars[i + 1];
            if n == 'p' || n == 'P' {
                i = copy_unicode_class(chars, i + 1, &mut cls)?;
                empty = false;
                continue;
            }
            if let Some(pc) = perl_class(n) {
                cls.push_str(pc);
                empty = false;
                i += 2;
                continue;
            }
        }
        // single char or range
        let (lo, ni) = class_char(chars, i)?;
        i = ni;
        let mut hi = lo;
        if i + 1 < chars.len() && chars[i] == '-' && chars[i + 1] != ']' {
            let (h, ni) = class_char(chars, i + 1)?;
            hi = h;
            i = ni;
            if hi < lo {
                return Err("invalid character class range".to_string());
            }
        }
        emit_literal(&mut cls, lo);
        if hi != lo {
            cls.push('-');
            emit_literal(&mut cls, hi);
        }
        empty = false;
    }
    let _ = empty;
    cls.push(']');
    out.push_str(&cls);
    Ok(i)
}

fn class_char(chars: &[char], i: usize) -> Result<(char, usize), String> {
    if chars[i] == '\\' {
        parse_escape_char(chars, i + 1)
    } else {
        Ok((chars[i], i + 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn t() {
        for p in [
            r#"(?i)aws_(.{0,20})?=?.[\'\"0-9a-zA-Z\/+]{40}"#,
            r"[\w\/+-]{16,}={0,2}",
            r"(?:^|[^!$-&\(-9<>-~])((?:A3T[A-Z0-9]|ACCA)[A-Z0-9]{16})\b",
            r"a{,5}x{",
        ] {
            let r = Regexp::compile(p);
            assert!(r.is_ok(), "{} -> {:?} ({:?})", p, r.err(), translate(p));
        }
    }
}
