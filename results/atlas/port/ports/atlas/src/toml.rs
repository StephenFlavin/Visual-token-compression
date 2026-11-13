// A small TOML parser sufficient for gitleaks configuration files.

#[derive(Clone, Debug)]
pub enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Array(Vec<Value>),
    Table(Vec<(String, Value)>),
}

impl Value {
    /// Case-insensitive key lookup (viper lowercases all keys).
    pub fn get(&self, key: &str) -> Option<&Value> {
        if let Value::Table(t) = self {
            let k = key.to_lowercase();
            // last definition wins
            t.iter().rev().find(|(name, _)| name.to_lowercase() == k).map(|(_, v)| v)
        } else {
            None
        }
    }
}

struct Parser {
    s: Vec<char>,
    i: usize,
    line: usize,
}

type R<T> = Result<T, String>;

pub fn parse(input: &str) -> R<Value> {
    let mut p = Parser { s: input.chars().collect(), i: 0, line: 1 };
    let mut root = Value::Table(vec![]);
    // path to the current table: list of (key, is_array)
    let mut current: Vec<String> = vec![];
    loop {
        p.skip_ws_comments_newlines();
        if p.eof() {
            break;
        }
        let c = p.peek();
        if c == '[' {
            let is_array = p.peek_at(1) == Some('[');
            p.i += if is_array { 2 } else { 1 };
            p.skip_ws();
            let keys = p.parse_key()?;
            p.skip_ws();
            if is_array {
                p.expect(']')?;
                p.expect(']')?;
            } else {
                p.expect(']')?;
            }
            p.expect_eol()?;
            // navigate
            {
                let mut tbl = &mut root;
                for (idx, k) in keys.iter().enumerate() {
                    let last = idx == keys.len() - 1;
                    tbl = if last && is_array {
                        let arr = get_or_insert(tbl, k, Value::Array(vec![]))?;
                        match arr {
                            Value::Array(a) => {
                                a.push(Value::Table(vec![]));
                                a.last_mut().unwrap()
                            }
                            _ => return Err(p.err(&format!("key {} is not an array", k))),
                        }
                    } else {
                        let v = get_or_insert(tbl, k, Value::Table(vec![]))?;
                        match v {
                            Value::Array(a) => match a.last_mut() {
                                Some(x @ Value::Table(_)) => x,
                                _ => return Err(p.err(&format!("key {} is not a table", k))),
                            },
                            Value::Table(_) => v,
                            _ => return Err(p.err(&format!("key {} is not a table", k))),
                        }
                    };
                }
            }
            current = keys;
            continue;
        }
        // key/value
        let keys = p.parse_key()?;
        p.skip_ws();
        p.expect('=')?;
        p.skip_ws();
        let val = p.parse_value()?;
        p.expect_eol()?;
        let tbl = navigate(&mut root, &current).ok_or_else(|| p.err("invalid table"))?;
        insert_dotted(tbl, &keys, val).map_err(|e| p.err(&e))?;
    }
    Ok(root)
}

fn navigate<'a>(root: &'a mut Value, path: &[String]) -> Option<&'a mut Value> {
    let mut cur = root;
    for k in path {
        let next = match cur {
            Value::Table(t) => t.iter_mut().rev().find(|(n, _)| n == k).map(|(_, v)| v)?,
            _ => return None,
        };
        cur = match next {
            Value::Array(a) => a.last_mut()?,
            v => v,
        };
    }
    Some(cur)
}

fn get_or_insert<'a>(tbl: &'a mut Value, k: &str, default: Value) -> R<&'a mut Value> {
    match tbl {
        Value::Table(t) => {
            if let Some(pos) = t.iter().position(|(n, _)| n == k) {
                Ok(&mut t[pos].1)
            } else {
                t.push((k.to_string(), default));
                Ok(&mut t.last_mut().unwrap().1)
            }
        }
        _ => Err(format!("key {} is not a table", k)),
    }
}

fn insert_dotted(tbl: &mut Value, keys: &[String], val: Value) -> Result<(), String> {
    if keys.len() == 1 {
        if let Value::Table(t) = tbl {
            if t.iter().any(|(n, _)| n == &keys[0]) {
                return Err(format!("key {} is already defined", keys[0]));
            }
            t.push((keys[0].clone(), val));
            return Ok(());
        }
        return Err("not a table".into());
    }
    let sub = get_or_insert(tbl, &keys[0], Value::Table(vec![]))?;
    insert_dotted(sub, &keys[1..], val)
}

impl Parser {
    fn eof(&self) -> bool {
        self.i >= self.s.len()
    }
    fn peek(&self) -> char {
        self.s[self.i]
    }
    fn peek_at(&self, n: usize) -> Option<char> {
        self.s.get(self.i + n).cloned()
    }
    fn err(&self, m: &str) -> String {
        format!("toml: line {}: {}", self.line, m)
    }
    fn expect(&mut self, c: char) -> R<()> {
        if !self.eof() && self.peek() == c {
            self.i += 1;
            Ok(())
        } else {
            let got = if self.eof() { "EOF".to_string() } else { format!("{:?}", self.peek()) };
            Err(self.err(&format!("expected {:?}, got {}", c, got)))
        }
    }
    fn skip_ws(&mut self) {
        while !self.eof() && (self.peek() == ' ' || self.peek() == '\t') {
            self.i += 1;
        }
    }
    fn skip_comment(&mut self) {
        if !self.eof() && self.peek() == '#' {
            while !self.eof() && self.peek() != '\n' {
                self.i += 1;
            }
        }
    }
    fn skip_ws_comments_newlines(&mut self) {
        loop {
            self.skip_ws();
            self.skip_comment();
            if !self.eof() && (self.peek() == '\n' || self.peek() == '\r') {
                if self.peek() == '\n' {
                    self.line += 1;
                }
                self.i += 1;
                continue;
            }
            break;
        }
    }
    fn expect_eol(&mut self) -> R<()> {
        self.skip_ws();
        self.skip_comment();
        if self.eof() {
            return Ok(());
        }
        if self.peek() == '\r' {
            self.i += 1;
        }
        if !self.eof() && self.peek() == '\n' {
            self.i += 1;
            self.line += 1;
            return Ok(());
        }
        if self.eof() {
            return Ok(());
        }
        Err(self.err(&format!("expected newline, got {:?}", self.peek())))
    }
    fn parse_key(&mut self) -> R<Vec<String>> {
        let mut keys = vec![];
        loop {
            self.skip_ws();
            if self.eof() {
                return Err(self.err("unexpected EOF in key"));
            }
            let c = self.peek();
            let k = if c == '"' {
                self.i += 1;
                self.parse_basic_string()?
            } else if c == '\'' {
                self.i += 1;
                self.parse_literal_string()?
            } else {
                let start = self.i;
                while !self.eof() {
                    let c = self.peek();
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        self.i += 1;
                    } else {
                        break;
                    }
                }
                if start == self.i {
                    return Err(self.err(&format!("invalid key character {:?}", c)));
                }
                self.s[start..self.i].iter().collect()
            };
            keys.push(k);
            self.skip_ws();
            if !self.eof() && self.peek() == '.' {
                self.i += 1;
                continue;
            }
            break;
        }
        Ok(keys)
    }
    fn parse_basic_string(&mut self) -> R<String> {
        let mut out = String::new();
        loop {
            if self.eof() {
                return Err(self.err("unterminated string"));
            }
            let c = self.peek();
            self.i += 1;
            match c {
                '"' => return Ok(out),
                '\n' => return Err(self.err("newline in string")),
                '\\' => out.push(self.parse_escape()?),
                c => out.push(c),
            }
        }
    }
    fn parse_escape(&mut self) -> R<char> {
        if self.eof() {
            return Err(self.err("unterminated escape"));
        }
        let c = self.peek();
        self.i += 1;
        Ok(match c {
            'b' => '\x08',
            't' => '\t',
            'n' => '\n',
            'f' => '\x0c',
            'r' => '\r',
            'e' => '\x1b',
            '"' => '"',
            '\\' => '\\',
            'u' | 'U' => {
                let n = if c == 'u' { 4 } else { 8 };
                if self.i + n > self.s.len() {
                    return Err(self.err("invalid unicode escape"));
                }
                let hex: String = self.s[self.i..self.i + n].iter().collect();
                self.i += n;
                let v = u32::from_str_radix(&hex, 16).map_err(|_| self.err("invalid unicode escape"))?;
                char::from_u32(v).ok_or_else(|| self.err("invalid unicode escape"))?
            }
            _ => return Err(self.err(&format!("invalid escape character {:?}", c))),
        })
    }
    fn parse_literal_string(&mut self) -> R<String> {
        let mut out = String::new();
        loop {
            if self.eof() {
                return Err(self.err("unterminated string"));
            }
            let c = self.peek();
            self.i += 1;
            match c {
                '\'' => return Ok(out),
                '\n' => return Err(self.err("newline in string")),
                c => out.push(c),
            }
        }
    }
    fn starts_with(&self, pat: &str) -> bool {
        let p: Vec<char> = pat.chars().collect();
        self.i + p.len() <= self.s.len() && self.s[self.i..self.i + p.len()] == p[..]
    }
    fn parse_ml_literal(&mut self) -> R<String> {
        // after opening '''
        if self.starts_with("\r\n") {
            self.i += 2;
            self.line += 1;
        } else if self.starts_with("\n") {
            self.i += 1;
            self.line += 1;
        }
        let mut out = String::new();
        loop {
            if self.eof() {
                return Err(self.err("unterminated multiline string"));
            }
            if self.starts_with("'''") {
                // allow up to two extra quotes
                let mut n = 3;
                while self.peek_at(n) == Some('\'') && n < 5 {
                    n += 1;
                }
                for _ in 3..n {
                    out.push('\'');
                }
                self.i += n;
                return Ok(out);
            }
            let c = self.peek();
            if c == '\n' {
                self.line += 1;
            }
            out.push(c);
            self.i += 1;
        }
    }
    fn parse_ml_basic(&mut self) -> R<String> {
        if self.starts_with("\r\n") {
            self.i += 2;
            self.line += 1;
        } else if self.starts_with("\n") {
            self.i += 1;
            self.line += 1;
        }
        let mut out = String::new();
        loop {
            if self.eof() {
                return Err(self.err("unterminated multiline string"));
            }
            if self.starts_with("\"\"\"") {
                let mut n = 3;
                while self.peek_at(n) == Some('"') && n < 5 {
                    n += 1;
                }
                for _ in 3..n {
                    out.push('"');
                }
                self.i += n;
                return Ok(out);
            }
            let c = self.peek();
            self.i += 1;
            if c == '\\' {
                // line ending backslash
                let save = self.i;
                let mut j = self.i;
                while j < self.s.len() && (self.s[j] == ' ' || self.s[j] == '\t') {
                    j += 1;
                }
                if j < self.s.len() && (self.s[j] == '\n' || self.s[j] == '\r') {
                    self.i = j;
                    while !self.eof() && self.peek().is_whitespace() {
                        if self.peek() == '\n' {
                            self.line += 1;
                        }
                        self.i += 1;
                    }
                    continue;
                }
                self.i = save;
                out.push(self.parse_escape()?);
                continue;
            }
            if c == '\n' {
                self.line += 1;
            }
            out.push(c);
        }
    }
    fn parse_value(&mut self) -> R<Value> {
        if self.eof() {
            return Err(self.err("expected value"));
        }
        let c = self.peek();
        if self.starts_with("'''") {
            self.i += 3;
            return Ok(Value::Str(self.parse_ml_literal()?));
        }
        if self.starts_with("\"\"\"") {
            self.i += 3;
            return Ok(Value::Str(self.parse_ml_basic()?));
        }
        match c {
            '"' => {
                self.i += 1;
                Ok(Value::Str(self.parse_basic_string()?))
            }
            '\'' => {
                self.i += 1;
                Ok(Value::Str(self.parse_literal_string()?))
            }
            '[' => {
                self.i += 1;
                let mut arr = vec![];
                loop {
                    self.skip_ws_comments_newlines();
                    if self.eof() {
                        return Err(self.err("unterminated array"));
                    }
                    if self.peek() == ']' {
                        self.i += 1;
                        break;
                    }
                    arr.push(self.parse_value()?);
                    self.skip_ws_comments_newlines();
                    if self.eof() {
                        return Err(self.err("unterminated array"));
                    }
                    if self.peek() == ',' {
                        self.i += 1;
                        continue;
                    }
                    if self.peek() == ']' {
                        self.i += 1;
                        break;
                    }
                    return Err(self.err(&format!("unexpected character in array {:?}", self.peek())));
                }
                Ok(Value::Array(arr))
            }
            '{' => {
                self.i += 1;
                let mut tbl = Value::Table(vec![]);
                self.skip_ws();
                if !self.eof() && self.peek() == '}' {
                    self.i += 1;
                    return Ok(tbl);
                }
                loop {
                    self.skip_ws();
                    let keys = self.parse_key()?;
                    self.skip_ws();
                    self.expect('=')?;
                    self.skip_ws();
                    let v = self.parse_value()?;
                    insert_dotted(&mut tbl, &keys, v).map_err(|e| self.err(&e))?;
                    self.skip_ws();
                    if self.eof() {
                        return Err(self.err("unterminated inline table"));
                    }
                    if self.peek() == ',' {
                        self.i += 1;
                        continue;
                    }
                    if self.peek() == '}' {
                        self.i += 1;
                        break;
                    }
                    return Err(self.err("unexpected character in inline table"));
                }
                Ok(tbl)
            }
            _ => {
                let start = self.i;
                while !self.eof() {
                    let c = self.peek();
                    if c == ',' || c == ']' || c == '}' || c == '\n' || c == '\r' || c == '#' {
                        break;
                    }
                    self.i += 1;
                }
                let raw: String = self.s[start..self.i].iter().collect::<String>();
                let raw = raw.trim_end().to_string();
                // re-adjust index for trailing spaces trimmed (keep consumed; fine)
                parse_scalar(&raw).ok_or_else(|| self.err(&format!("invalid value {:?}", raw)))
            }
        }
    }
}

fn parse_scalar(raw: &str) -> Option<Value> {
    match raw {
        "true" => return Some(Value::Bool(true)),
        "false" => return Some(Value::Bool(false)),
        "inf" | "+inf" => return Some(Value::Float(f64::INFINITY)),
        "-inf" => return Some(Value::Float(f64::NEG_INFINITY)),
        "nan" | "+nan" | "-nan" => return Some(Value::Float(f64::NAN)),
        _ => {}
    }
    let clean: String = raw.replace('_', "");
    if let Some(h) = clean.strip_prefix("0x") {
        return i64::from_str_radix(h, 16).ok().map(Value::Int);
    }
    if let Some(h) = clean.strip_prefix("0o") {
        return i64::from_str_radix(h, 8).ok().map(Value::Int);
    }
    if let Some(h) = clean.strip_prefix("0b") {
        return i64::from_str_radix(h, 2).ok().map(Value::Int);
    }
    if let Ok(i) = clean.parse::<i64>() {
        return Some(Value::Int(i));
    }
    if clean.chars().all(|c| c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
        && clean.chars().any(|c| c.is_ascii_digit())
    {
        if let Ok(f) = clean.parse::<f64>() {
            return Some(Value::Float(f));
        }
    }
    // dates/times: keep as string
    if raw.len() >= 8 && raw.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        return Some(Value::Str(raw.to_string()));
    }
    None
}
