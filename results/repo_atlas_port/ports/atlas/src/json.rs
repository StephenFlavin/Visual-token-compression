// Minimal JSON encoding (Go encoding/json compatible output) and decoding.

pub fn json_escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub struct JsonWriter {
    out: String,
    indent: String,
    stack: Vec<bool>,
}

impl JsonWriter {
    pub fn new(indent: &str) -> JsonWriter {
        JsonWriter { out: String::new(), indent: indent.to_string(), stack: vec![] }
    }
    fn newline(&mut self) {
        self.out.push('\n');
        for _ in 0..self.stack.len() {
            self.out.push_str(&self.indent);
        }
    }
    pub fn begin_obj(&mut self) {
        self.out.push('{');
        self.stack.push(false);
    }
    pub fn begin_arr(&mut self) {
        self.out.push('[');
        self.stack.push(false);
    }
    fn sep(&mut self) {
        if let Some(top) = self.stack.last_mut() {
            if *top {
                self.out.push(',');
            }
            *top = true;
        }
        self.newline();
    }
    pub fn key(&mut self, k: &str) {
        self.sep();
        self.out.push_str(&json_escape_str(k));
        self.out.push_str(": ");
    }
    pub fn elem(&mut self) {
        self.sep();
    }
    fn end(&mut self, c: char) {
        let had = self.stack.pop().unwrap_or(false);
        if had {
            self.newline();
        }
        self.out.push(c);
    }
    pub fn end_obj(&mut self) {
        self.end('}');
    }
    pub fn end_arr(&mut self) {
        self.end(']');
    }
    pub fn string(&mut self, s: &str) {
        self.out.push_str(&json_escape_str(s));
    }
    pub fn raw(&mut self, s: &str) {
        self.out.push_str(s);
    }
    pub fn string_list(&mut self, l: Option<&Vec<String>>) {
        match l {
            None => self.out.push_str("null"),
            Some(l) => {
                self.begin_arr();
                for s in l {
                    self.elem();
                    self.string(s);
                }
                self.end_arr();
            }
        }
    }
    pub fn finish(self) -> String {
        self.out
    }
}

#[derive(Clone, Debug)]
pub enum JValue {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<JValue>),
    Obj(Vec<(String, JValue)>),
}

impl JValue {
    /// Case-insensitive field lookup like encoding/json.
    pub fn get(&self, k: &str) -> Option<&JValue> {
        if let JValue::Obj(o) = self {
            if let Some((_, v)) = o.iter().rev().find(|(n, _)| n == k) {
                return Some(v);
            }
            let kl = k.to_lowercase();
            o.iter().rev().find(|(n, _)| n.to_lowercase() == kl).map(|(_, v)| v)
        } else {
            None
        }
    }
}

pub fn parse(s: &str) -> Result<JValue, String> {
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    let v = parse_value(&b, &mut i)?;
    skip_ws(&b, &mut i);
    if i != b.len() {
        return Err("invalid character after top-level value".into());
    }
    Ok(v)
}

fn skip_ws(b: &[char], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], ' ' | '\t' | '\n' | '\r') {
        *i += 1;
    }
}

fn parse_value(b: &[char], i: &mut usize) -> Result<JValue, String> {
    skip_ws(b, i);
    if *i >= b.len() {
        return Err("unexpected end of JSON input".into());
    }
    match b[*i] {
        '{' => {
            *i += 1;
            let mut o = vec![];
            skip_ws(b, i);
            if *i < b.len() && b[*i] == '}' {
                *i += 1;
                return Ok(JValue::Obj(o));
            }
            loop {
                skip_ws(b, i);
                let k = match parse_value(b, i)? {
                    JValue::Str(s) => s,
                    _ => return Err("expected string key".into()),
                };
                skip_ws(b, i);
                if *i >= b.len() || b[*i] != ':' {
                    return Err("expected ':'".into());
                }
                *i += 1;
                let v = parse_value(b, i)?;
                o.push((k, v));
                skip_ws(b, i);
                if *i < b.len() && b[*i] == ',' {
                    *i += 1;
                    continue;
                }
                if *i < b.len() && b[*i] == '}' {
                    *i += 1;
                    return Ok(JValue::Obj(o));
                }
                return Err("expected ',' or '}'".into());
            }
        }
        '[' => {
            *i += 1;
            let mut a = vec![];
            skip_ws(b, i);
            if *i < b.len() && b[*i] == ']' {
                *i += 1;
                return Ok(JValue::Arr(a));
            }
            loop {
                a.push(parse_value(b, i)?);
                skip_ws(b, i);
                if *i < b.len() && b[*i] == ',' {
                    *i += 1;
                    continue;
                }
                if *i < b.len() && b[*i] == ']' {
                    *i += 1;
                    return Ok(JValue::Arr(a));
                }
                return Err("expected ',' or ']'".into());
            }
        }
        '"' => {
            *i += 1;
            let mut s = String::new();
            while *i < b.len() {
                let c = b[*i];
                *i += 1;
                match c {
                    '"' => return Ok(JValue::Str(s)),
                    '\\' => {
                        if *i >= b.len() {
                            break;
                        }
                        let e = b[*i];
                        *i += 1;
                        match e {
                            'n' => s.push('\n'),
                            'r' => s.push('\r'),
                            't' => s.push('\t'),
                            'b' => s.push('\x08'),
                            'f' => s.push('\x0c'),
                            '/' => s.push('/'),
                            '\\' => s.push('\\'),
                            '"' => s.push('"'),
                            'u' => {
                                let read_hex = |i: &mut usize| -> Result<u32, String> {
                                    if *i + 4 > b.len() {
                                        return Err("bad unicode escape".into());
                                    }
                                    let h: String = b[*i..*i + 4].iter().collect();
                                    *i += 4;
                                    u32::from_str_radix(&h, 16).map_err(|_| "bad unicode escape".to_string())
                                };
                                let v = read_hex(i)?;
                                if (0xD800..0xDC00).contains(&v)
                                    && *i + 1 < b.len()
                                    && b[*i] == '\\'
                                    && b[*i + 1] == 'u'
                                {
                                    let save = *i;
                                    *i += 2;
                                    let lo = read_hex(i)?;
                                    if (0xDC00..0xE000).contains(&lo) {
                                        let cp = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                                        s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                                    } else {
                                        *i = save;
                                        s.push('\u{FFFD}');
                                    }
                                } else {
                                    s.push(char::from_u32(v).unwrap_or('\u{FFFD}'));
                                }
                            }
                            _ => return Err("invalid escape".into()),
                        }
                    }
                    c => s.push(c),
                }
            }
            Err("unterminated string".into())
        }
        't' | 'f' | 'n' => {
            let rest: String = b[*i..(*i + 5).min(b.len())].iter().collect();
            if rest.starts_with("true") {
                *i += 4;
                Ok(JValue::Bool(true))
            } else if rest.starts_with("false") {
                *i += 5;
                Ok(JValue::Bool(false))
            } else if rest.starts_with("null") {
                *i += 4;
                Ok(JValue::Null)
            } else {
                Err("invalid literal".into())
            }
        }
        _ => {
            let st = *i;
            while *i < b.len() && matches!(b[*i], '0'..='9' | '-' | '+' | '.' | 'e' | 'E') {
                *i += 1;
            }
            if st == *i {
                return Err(format!("invalid character '{}' looking for beginning of value", b[st]));
            }
            Ok(JValue::Num(b[st..*i].iter().collect()))
        }
    }
}
