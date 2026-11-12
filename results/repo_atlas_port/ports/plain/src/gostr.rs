//! Helpers replicating Go's string / number formatting behaviour.

/// Approximation of Go's unicode.IsPrint
pub fn is_print(c: char) -> bool {
    if c == ' ' {
        return true;
    }
    if c.is_control() || c.is_whitespace() {
        return false;
    }
    let u = c as u32;
    // format characters / special non-printables
    if matches!(u, 0xAD | 0x600..=0x605 | 0x61C | 0x6DD | 0x70F | 0x180E | 0x200B..=0x200F | 0x202A..=0x202E | 0x2060..=0x2064 | 0x2066..=0x206F | 0xFEFF | 0xFFF9..=0xFFFB) {
        return false;
    }
    // private use / surrogates / noncharacters
    if (0xE000..=0xF8FF).contains(&u) || (0xF0000..=0x10FFFF).contains(&u) {
        return false;
    }
    if (u & 0xFFFE) == 0xFFFE {
        return false;
    }
    true
}

/// Go's strconv.Quote
pub fn quote(s: &str) -> String {
    quote_with(s, '"')
}

fn quote_with(s: &str, q: char) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push(q);
    for c in s.chars() {
        if c == q || c == '\\' {
            out.push('\\');
            out.push(c);
            continue;
        }
        if is_print(c) {
            out.push(c);
            continue;
        }
        match c {
            '\x07' => out.push_str("\\a"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x0b' => out.push_str("\\v"),
            _ => {
                let u = c as u32;
                if u < 0x20 || u == 0x7f {
                    out.push_str(&format!("\\x{:02x}", u));
                } else if u == 0xFFFD {
                    // Go would print invalid bytes as \x..; the replacement char itself is printable
                    out.push(c);
                } else if u < 0x10000 {
                    out.push_str(&format!("\\u{:04x}", u));
                } else {
                    out.push_str(&format!("\\U{:08x}", u));
                }
            }
        }
    }
    out.push(q);
    out
}

/// Format a float32 like Go's encoding/json does.
pub fn format_float32_json(f: f32) -> String {
    if f.is_nan() || f.is_infinite() {
        return "null".to_string();
    }
    let abs = f.abs();
    if abs != 0.0 && (abs < 1e-6 || abs >= 1e21) {
        let s = format!("{:e}", f);
        // Rust: 1e-7 / 1.5e21 ; Go: 1e-07 -> cleaned to 1e-7, positive exponents get '+'
        if let Some(pos) = s.find('e') {
            let (m, e) = s.split_at(pos);
            let e = &e[1..];
            if e.starts_with('-') {
                return format!("{}e{}", m, e);
            }
            return format!("{}e+{}", m, e);
        }
        return s;
    }
    let s = format!("{}", f);
    s
}

/// Format a float like Go's %v (shortest repr, %e when exponent < -4 || >= 21)
pub fn format_float_v(f: f64, bits32: bool) -> String {
    if f.is_nan() {
        return "NaN".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "+Inf".to_string() } else { "-Inf".to_string() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0".to_string() } else { "0".to_string() };
    }
    let e_repr = if bits32 { format!("{:e}", f as f32) } else { format!("{:e}", f) };
    // exponent
    let epos = e_repr.find('e').unwrap();
    let exp: i32 = e_repr[epos + 1..].parse().unwrap_or(0);
    if exp < -4 || exp >= 21 {
        let m = &e_repr[..epos];
        let sign = if exp < 0 { '-' } else { '+' };
        let a = exp.abs();
        return format!("{}e{}{:02}", m, sign, a);
    }
    if bits32 {
        format!("{}", f as f32)
    } else {
        format!("{}", f)
    }
}

/// Go's time.Duration.String()
pub fn duration_string(d: i64) -> String {
    if d == 0 {
        return "0s".to_string();
    }
    let neg = d < 0;
    let mut u: u64 = if neg { (d as i128).unsigned_abs() as u64 } else { d as u64 };
    let mut buf: Vec<u8> = Vec::new(); // built reversed
    const SECOND: u64 = 1_000_000_000;
    if u < SECOND {
        let prec;
        let unit: &str;
        if u < 1000 {
            prec = 0;
            unit = "ns";
        } else if u < 1_000_000 {
            prec = 3;
            unit = "µs";
        } else {
            prec = 6;
            unit = "ms";
        }
        let (frac, rest) = fmt_frac(u, prec);
        u = rest;
        let s = format!("{}{}{}", u, frac, unit);
        return if neg { format!("-{}", s) } else { s };
    }
    let (frac, rest) = fmt_frac(u, 9);
    u = rest;
    let secs = u % 60;
    u /= 60;
    let mut s = format!("{}{}s", secs, frac);
    if u > 0 {
        let mins = u % 60;
        u /= 60;
        s = format!("{}m{}", mins, s);
        if u > 0 {
            s = format!("{}h{}", u, s);
        }
    }
    buf.clear();
    if neg {
        format!("-{}", s)
    } else {
        s
    }
}

fn fmt_frac(mut v: u64, prec: u32) -> (String, u64) {
    let mut digits: Vec<u8> = Vec::new();
    let mut print = false;
    for _ in 0..prec {
        let digit = v % 10;
        print = print || digit != 0;
        if print {
            digits.push(b'0' + digit as u8);
        }
        v /= 10;
    }
    if print {
        digits.push(b'.');
    }
    digits.reverse();
    (String::from_utf8(digits).unwrap(), v)
}

/// Go's time.Duration.Round
pub fn duration_round(d: i64, m: i64) -> i64 {
    if m <= 0 {
        return d;
    }
    let mut r = d % m;
    if d < 0 {
        r = -r;
        if r + r < m {
            return d + r;
        }
        let d1 = d - m + r;
        if d1 < d {
            return d1;
        }
        return i64::MIN;
    }
    if r + r < m {
        return d - r;
    }
    let d1 = d.wrapping_add(m - r);
    if d1 > d {
        return d1;
    }
    i64::MAX
}

/// Byte-slice a string, falling back to a lossy conversion when the
/// indices don't fall on character boundaries.
pub fn bslice(s: &str, a: usize, b: usize) -> String {
    let len = s.len();
    let a = a.min(len);
    let b = b.min(len).max(a);
    match s.get(a..b) {
        Some(x) => x.to_string(),
        None => String::from_utf8_lossy(&s.as_bytes()[a..b]).into_owned(),
    }
}

/// strings.TrimSpace (unicode white space)
pub fn trim_space(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace())
}

/// strings.Fields
pub fn fields(s: &str) -> Vec<&str> {
    s.split(|c: char| c.is_whitespace()).filter(|x| !x.is_empty()).collect()
}
