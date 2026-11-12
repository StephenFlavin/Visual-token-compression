// Small helpers mirroring Go standard library behaviour (strconv, path/filepath, strings).

/// strconv.Quote
pub fn go_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\x07' => out.push_str("\\a"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x0b' => out.push_str("\\v"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c if is_go_printable(c) => out.push(c),
            c => {
                let v = c as u32;
                if v < 0x10000 {
                    out.push_str(&format!("\\u{:04x}", v));
                } else {
                    out.push_str(&format!("\\U{:08x}", v));
                }
            }
        }
    }
    out.push('"');
    out
}

fn is_go_printable(c: char) -> bool {
    let v = c as u32;
    if v < 0x80 {
        return (0x20..0x7f).contains(&v);
    }
    // Approximation of unicode.IsPrint: exclude controls, format chars, separators (except space),
    // private use, surrogates and noncharacters.
    if (0x80..0xa0).contains(&v) || v == 0xad {
        return false;
    }
    if v == 0xfeff || (0x200b..=0x200f).contains(&v) || (0x2028..=0x202e).contains(&v) {
        return false;
    }
    if (0x2060..=0x206f).contains(&v) || v == 0x00a0 || v == 0x1680 || (0x2000..=0x200a).contains(&v)
        || v == 0x202f || v == 0x205f || v == 0x3000
    {
        return false;
    }
    if (0xe000..=0xf8ff).contains(&v) || (0xfff0..=0xfffb).contains(&v) || v == 0xfffe || v == 0xffff {
        return false;
    }
    if v >= 0xf0000 {
        return false;
    }
    true
}

/// Formats a float like Go's %v for float64 (strconv 'g' shortest).
pub fn format_float_g(f: f64) -> String {
    if f == 0.0 {
        return "0".to_string();
    }
    let abs = f.abs();
    if abs < 1e-4 || abs >= 1e21 {
        let s = format!("{:e}", f);
        // Rust: 1.5e-5 ; Go: 1.5e-05
        if let Some(pos) = s.find('e') {
            let (m, e) = s.split_at(pos);
            let e = &e[1..];
            let (sign, digits) = if let Some(d) = e.strip_prefix('-') { ("-", d) } else { ("+", e) };
            let digits = if digits.len() < 2 { format!("0{}", digits) } else { digits.to_string() };
            return format!("{}e{}{}", m, sign, digits);
        }
        s
    } else {
        format!("{}", f)
    }
}

/// Formats a float32 like Go's encoding/json does.
pub fn format_json_f32(f: f32) -> String {
    if f == 0.0 {
        return "0".to_string();
    }
    let abs = f.abs();
    if abs < 1e-6 || abs >= 1e21 {
        let s = format!("{:e}", f);
        if let Some(pos) = s.find('e') {
            let (m, e) = s.split_at(pos);
            let e = &e[1..];
            let (sign, digits) = if let Some(d) = e.strip_prefix('-') { ("-", d) } else { ("+", e) };
            let digits = if digits.len() < 2 { format!("0{}", digits) } else { digits.to_string() };
            return format!("{}e{}{}", m, sign, digits);
        }
        s
    } else {
        format!("{}", f)
    }
}

/// Byte-oriented slice of a string (Go semantics). Invalid boundaries are converted lossily.
pub fn bslice(s: &str, start: usize, end: usize) -> String {
    let b = s.as_bytes();
    let end = end.min(b.len());
    let start = start.min(end);
    if s.is_char_boundary(start) && s.is_char_boundary(end) {
        return s[start..end].to_string();
    }
    String::from_utf8_lossy(&b[start..end]).into_owned()
}

/// path/filepath.Clean (unix)
pub fn clean(p: &str) -> String {
    if p.is_empty() {
        return ".".to_string();
    }
    let rooted = p.starts_with('/');
    let mut parts: Vec<&str> = vec![];
    for seg in p.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                if let Some(last) = parts.last() {
                    if *last != ".." {
                        parts.pop();
                        continue;
                    }
                }
                if !rooted {
                    parts.push("..");
                }
            }
            s => parts.push(s),
        }
    }
    let joined = parts.join("/");
    if rooted {
        format!("/{}", joined)
    } else if joined.is_empty() {
        ".".to_string()
    } else {
        joined
    }
}

/// path/filepath.Join
pub fn join(a: &str, b: &str) -> String {
    let parts: Vec<&str> = [a, b].iter().filter(|s| !s.is_empty()).cloned().collect();
    if parts.is_empty() {
        return String::new();
    }
    clean(&parts.join("/"))
}

/// filepath.Ext
pub fn ext(p: &str) -> String {
    let b = p.as_bytes();
    let mut i = b.len();
    while i > 0 {
        i -= 1;
        if b[i] == b'/' {
            break;
        }
        if b[i] == b'.' {
            return p[i..].to_string();
        }
    }
    String::new()
}

/// path.Base
pub fn base(p: &str) -> String {
    if p.is_empty() {
        return ".".to_string();
    }
    let t = p.trim_end_matches('/');
    if t.is_empty() {
        return "/".to_string();
    }
    match t.rfind('/') {
        Some(i) => t[i + 1..].to_string(),
        None => t.to_string(),
    }
}

/// filepath.Abs
pub fn abs(p: &str) -> String {
    if p.starts_with('/') {
        return clean(p);
    }
    let cwd = std::env::current_dir().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
    join(&cwd, p)
}

/// filepath.Rel (both absolute, cleaned)
pub fn rel(basepath: &str, targpath: &str) -> String {
    let b = clean(basepath);
    let t = clean(targpath);
    if b == t {
        return ".".to_string();
    }
    let bs: Vec<&str> = if b == "/" { vec![] } else { b.trim_start_matches('/').split('/').collect() };
    let ts: Vec<&str> = if t == "/" { vec![] } else { t.trim_start_matches('/').split('/').collect() };
    let mut i = 0;
    while i < bs.len() && i < ts.len() && bs[i] == ts[i] {
        i += 1;
    }
    let mut out: Vec<&str> = vec![];
    for _ in i..bs.len() {
        out.push("..");
    }
    for s in &ts[i..] {
        out.push(s);
    }
    out.join("/")
}

/// strings.TrimSpace (ASCII + unicode whitespace)
pub fn trim_space(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace())
}

/// Go's strings.ToLower is unicode aware; Rust's to_lowercase is close enough.
pub fn to_lower(s: &str) -> String {
    s.to_lowercase()
}
