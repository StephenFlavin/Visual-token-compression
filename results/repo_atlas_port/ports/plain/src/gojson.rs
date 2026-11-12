//! Minimal JSON writer that mirrors Go's encoding/json output byte-for-byte
//! (HTML escaping, indentation rules, etc).

/// A JSON value tree with ordered object keys.
#[derive(Clone, Debug)]
pub enum JVal {
    Null,
    Bool(bool),
    /// a pre-formatted number
    Num(String),
    Str(String),
    Arr(Vec<JVal>),
    Obj(Vec<(String, JVal)>),
}

/// Encode a string the way Go's encoding/json does with HTML escaping enabled.
pub fn encode_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u00{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl JVal {
    pub fn s(v: &str) -> JVal {
        JVal::Str(v.to_string())
    }
    pub fn i(v: i64) -> JVal {
        JVal::Num(v.to_string())
    }
    pub fn strs(v: &[String]) -> JVal {
        JVal::Arr(v.iter().map(|x| JVal::Str(x.clone())).collect())
    }

    /// Compact encoding (json.Marshal)
    pub fn compact(&self) -> String {
        let mut out = String::new();
        self.write_compact(&mut out);
        out
    }

    fn write_compact(&self, out: &mut String) {
        match self {
            JVal::Null => out.push_str("null"),
            JVal::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            JVal::Num(n) => out.push_str(n),
            JVal::Str(s) => out.push_str(&encode_string(s)),
            JVal::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write_compact(out);
                }
                out.push(']');
            }
            JVal::Obj(o) => {
                out.push('{');
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&encode_string(k));
                    out.push(':');
                    v.write_compact(out);
                }
                out.push('}');
            }
        }
    }

    /// Indented encoding (json.MarshalIndent / Encoder.SetIndent)
    pub fn indent(&self, prefix: &str, indent: &str) -> String {
        let mut out = String::new();
        self.write_indent(&mut out, prefix, indent, 0);
        out
    }

    fn newline(out: &mut String, prefix: &str, indent: &str, depth: usize) {
        out.push('\n');
        out.push_str(prefix);
        for _ in 0..depth {
            out.push_str(indent);
        }
    }

    fn write_indent(&self, out: &mut String, prefix: &str, indent: &str, depth: usize) {
        match self {
            JVal::Arr(a) => {
                if a.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    Self::newline(out, prefix, indent, depth + 1);
                    v.write_indent(out, prefix, indent, depth + 1);
                }
                Self::newline(out, prefix, indent, depth);
                out.push(']');
            }
            JVal::Obj(o) => {
                if o.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    Self::newline(out, prefix, indent, depth + 1);
                    out.push_str(&encode_string(k));
                    out.push_str(": ");
                    v.write_indent(out, prefix, indent, depth + 1);
                }
                Self::newline(out, prefix, indent, depth);
                out.push('}');
            }
            other => other.write_compact(out),
        }
    }
}
