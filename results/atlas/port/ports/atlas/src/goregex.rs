// Port of the regexp package wrapper. Translates Go RE2 syntax into the Rust regex dialect so
// that semantics (ASCII-only perl classes, literal braces, nested '[' in classes, ...) match Go.

use regex::{Regex, RegexBuilder};

#[derive(Clone, Debug)]
pub struct GoRegex {
    pub re: Regex,
    pub src: String,
}

impl GoRegex {
    pub fn is_match(&self, s: &str) -> bool {
        self.re.is_match(s)
    }
    pub fn as_str(&self) -> &str {
        &self.src
    }
    pub fn num_subexp(&self) -> usize {
        self.re.captures_len() - 1
    }
    pub fn find_all_index(&self, s: &str) -> Vec<(usize, usize)> {
        self.re.find_iter(s).map(|m| (m.start(), m.end())).collect()
    }
    /// FindStringSubmatch: returns groups (empty string for non-participating groups).
    pub fn find_submatch(&self, s: &str) -> Option<Vec<String>> {
        self.re.captures(s).map(|c| {
            (0..c.len())
                .map(|i| c.get(i).map(|m| m.as_str().to_string()).unwrap_or_default())
                .collect()
        })
    }
}

pub const VERSION: &str = "stdlib";

pub fn compile(pat: &str) -> Result<GoRegex, String> {
    let translated = translate(pat)?;
    let re = RegexBuilder::new(&translated)
        .size_limit(1 << 30)
        .dfa_size_limit(1 << 30)
        .nest_limit(1000)
        .build()
        .map_err(|e| format!("error parsing regexp: {}", e.to_string().lines().last().unwrap_or("").trim()))?;
    Ok(GoRegex { re, src: pat.to_string() })
}

pub fn must_compile(pat: &str) -> GoRegex {
    match compile(pat) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("panic: regexp: Compile({}): {}", crate::util::go_quote(pat), e);
            eprintln!();
            std::process::exit(2);
        }
    }
}

fn is_rust_meta(c: char) -> bool {
    matches!(
        c,
        '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$' | '#' | '&' | '-' | '~'
    )
}

fn push_escaped(out: &mut String, c: char) {
    if is_rust_meta(c) {
        out.push('\\');
        out.push(c);
    } else if c == ' ' {
        out.push(' ');
    } else if c.is_ascii_punctuation() {
        // other punctuation can be emitted literally
        out.push(c);
    } else {
        out.push_str(&format!("\\x{{{:x}}}", c as u32));
    }
}

/// Parses a `{n}`, `{n,}` or `{n,m}` repetition at position i (pointing at '{').
fn repetition_len(s: &[char], i: usize) -> Option<usize> {
    let mut j = i + 1;
    let st = j;
    while j < s.len() && s[j].is_ascii_digit() {
        j += 1;
    }
    if j == st {
        return None;
    }
    if j < s.len() && s[j] == '}' {
        return Some(j - i + 1);
    }
    if j < s.len() && s[j] == ',' {
        j += 1;
        while j < s.len() && s[j].is_ascii_digit() {
            j += 1;
        }
        if j < s.len() && s[j] == '}' {
            return Some(j - i + 1);
        }
    }
    None
}

pub fn translate(pat: &str) -> Result<String, String> {
    let s: Vec<char> = pat.chars().collect();
    let mut out = String::with_capacity(pat.len() + 16);
    let mut i = 0;
    let mut in_class = false;
    let mut class_start = 0usize; // index in s where current class content starts
    while i < s.len() {
        let c = s[i];
        if in_class {
            match c {
                ']' if i > class_start => {
                    out.push(']');
                    in_class = false;
                    i += 1;
                }
                '[' => {
                    // POSIX class?
                    if i + 1 < s.len() && s[i + 1] == ':' {
                        // find ":]"
                        let mut j = i + 2;
                        let mut found = None;
                        while j + 1 < s.len() {
                            if s[j] == ':' && s[j + 1] == ']' {
                                found = Some(j);
                                break;
                            }
                            if !(s[j].is_ascii_alphabetic() || s[j] == '^') {
                                break;
                            }
                            j += 1;
                        }
                        if let Some(j) = found {
                            let name: String = s[i..j + 2].iter().collect();
                            out.push_str(&name);
                            i = j + 2;
                            continue;
                        }
                    }
                    out.push_str("\\[");
                    i += 1;
                }
                '&' | '~' => {
                    out.push('\\');
                    out.push(c);
                    i += 1;
                }
                '-' => {
                    if i + 1 < s.len() && s[i + 1] == '-' {
                        out.push_str("\\-");
                    } else {
                        out.push('-');
                    }
                    i += 1;
                }
                '\\' => {
                    if i + 1 >= s.len() {
                        return Err("error parsing regexp: trailing backslash at end of expression".into());
                    }
                    let n = s[i + 1];
                    match n {
                        'd' => out.push_str("0-9"),
                        'w' => out.push_str("0-9A-Za-z_"),
                        's' => out.push_str("\\t\\n\\x0C\\r "),
                        'D' => out.push_str("[:^digit:]"),
                        'W' => out.push_str("[:^word:]"),
                        'S' => out.push_str("[:^space:]"),
                        'p' | 'P' | 'x' => {
                            // copy escape verbatim
                            out.push('\\');
                            out.push(n);
                            i += 2;
                            if i < s.len() && s[i] == '{' {
                                while i < s.len() && s[i] != '}' {
                                    out.push(s[i]);
                                    i += 1;
                                }
                                if i < s.len() {
                                    out.push('}');
                                    i += 1;
                                }
                            } else if n == 'x' {
                                for _ in 0..2 {
                                    if i < s.len() {
                                        out.push(s[i]);
                                        i += 1;
                                    }
                                }
                            } else if i < s.len() {
                                out.push(s[i]);
                                i += 1;
                            }
                            continue;
                        }
                        'a' | 'f' | 't' | 'n' | 'r' | 'v' => {
                            out.push('\\');
                            out.push(n);
                        }
                        c if c.is_ascii_alphanumeric() => {
                            out.push('\\');
                            out.push(c);
                        }
                        c => push_escaped(&mut out, c),
                    }
                    i += 2;
                }
                _ => {
                    out.push(c);
                    i += 1;
                }
            }
            continue;
        }
        match c {
            '[' => {
                out.push('[');
                in_class = true;
                i += 1;
                if i < s.len() && s[i] == '^' {
                    out.push('^');
                    i += 1;
                }
                class_start = i;
                // a leading ']' is literal in Go
                if i < s.len() && s[i] == ']' {
                    out.push_str("\\]");
                    i += 1;
                }
            }
            '{' => {
                if let Some(len) = repetition_len(&s, i) {
                    let rep: String = s[i..i + len].iter().collect();
                    out.push_str(&rep);
                    i += len;
                } else {
                    out.push_str("\\{");
                    i += 1;
                }
            }
            '}' => {
                out.push_str("\\}");
                i += 1;
            }
            '\\' => {
                if i + 1 >= s.len() {
                    return Err("error parsing regexp: trailing backslash at end of expression".into());
                }
                let n = s[i + 1];
                match n {
                    'd' => out.push_str("[0-9]"),
                    'D' => out.push_str("[^0-9]"),
                    'w' => out.push_str("[0-9A-Za-z_]"),
                    'W' => out.push_str("[^0-9A-Za-z_]"),
                    's' => out.push_str("[\\t\\n\\x0C\\r ]"),
                    'S' => out.push_str("[^\\t\\n\\x0C\\r ]"),
                    'b' => out.push_str("(?-u:\\b)"),
                    'B' => out.push_str("(?-u:\\B)"),
                    'Q' => {
                        // literal text until \E
                        let mut j = i + 2;
                        while j < s.len() {
                            if s[j] == '\\' && j + 1 < s.len() && s[j + 1] == 'E' {
                                break;
                            }
                            push_escaped(&mut out, s[j]);
                            j += 1;
                        }
                        i = if j < s.len() { j + 2 } else { j };
                        continue;
                    }
                    'p' | 'P' | 'x' => {
                        out.push('\\');
                        out.push(n);
                        i += 2;
                        if i < s.len() && s[i] == '{' {
                            while i < s.len() && s[i] != '}' {
                                out.push(s[i]);
                                i += 1;
                            }
                            if i < s.len() {
                                out.push('}');
                                i += 1;
                            }
                        } else if n == 'x' {
                            for _ in 0..2 {
                                if i < s.len() {
                                    out.push(s[i]);
                                    i += 1;
                                }
                            }
                        } else if i < s.len() {
                            out.push(s[i]);
                            i += 1;
                        }
                        continue;
                    }
                    c if c.is_ascii_alphanumeric() => {
                        out.push('\\');
                        out.push(c);
                    }
                    c => push_escaped(&mut out, c),
                }
                i += 2;
            }
            '#' | '&' | '~' => {
                out.push('\\');
                out.push(c);
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    if in_class {
        return Err(format!("error parsing regexp: missing closing ]: `{}`", pat));
    }
    Ok(out)
}
