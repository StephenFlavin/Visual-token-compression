//! A compact implementation of Go's text/template language (plus a subset of
//! sprig helper functions) used by the template reporter.

use std::collections::HashMap;
use std::rc::Rc;

use super::Finding;
use crate::gostr;

// ------------------------------------------------------------------ values

#[derive(Clone, Debug)]
pub enum Val {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64, bool), // value, is float32
    Str(String),
    List(Rc<Vec<Val>>),
    Map(Rc<Vec<(String, Val)>>),
    Time(chrono::DateTime<chrono::Local>),
}

impl Val {
    fn truth(&self) -> bool {
        match self {
            Val::Nil => false,
            Val::Bool(b) => *b,
            Val::Int(i) => *i != 0,
            Val::Float(f, _) => *f != 0.0,
            Val::Str(s) => !s.is_empty(),
            Val::List(l) => !l.is_empty(),
            Val::Map(m) => !m.is_empty(),
            Val::Time(_) => true,
        }
    }

    fn to_string_v(&self) -> String {
        match self {
            Val::Nil => "<nil>".to_string(),
            Val::Bool(b) => b.to_string(),
            Val::Int(i) => i.to_string(),
            Val::Float(f, b32) => gostr::format_float_v(*f, *b32),
            Val::Str(s) => s.clone(),
            Val::List(l) => {
                let parts: Vec<String> = l.iter().map(|v| v.to_string_v()).collect();
                format!("[{}]", parts.join(" "))
            }
            Val::Map(m) => {
                let mut items: Vec<&(String, Val)> = m.iter().collect();
                items.sort_by(|a, b| a.0.cmp(&b.0));
                let parts: Vec<String> = items.iter().map(|(k, v)| format!("{}:{}", k, v.to_string_v())).collect();
                format!("map[{}]", parts.join(" "))
            }
            Val::Time(t) => t.format("%Y-%m-%d %H:%M:%S%.f %z %Z").to_string(),
        }
    }

    fn field(&self, name: &str) -> Result<Val, String> {
        match self {
            Val::Map(m) => Ok(m.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone()).unwrap_or(Val::Nil)),
            Val::Nil => Err(format!("nil pointer evaluating interface {{}}.{}", name)),
            _ => Err(format!("can't evaluate field {} in type {}", name, self.type_name())),
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            Val::Nil => "nil",
            Val::Bool(_) => "bool",
            Val::Int(_) => "int",
            Val::Float(..) => "float64",
            Val::Str(_) => "string",
            Val::List(_) => "[]interface {}",
            Val::Map(_) => "map[string]interface {}",
            Val::Time(_) => "time.Time",
        }
    }

    fn as_i64(&self) -> i64 {
        match self {
            Val::Int(i) => *i,
            Val::Float(f, _) => *f as i64,
            Val::Bool(b) => *b as i64,
            Val::Str(s) => s.trim().parse::<i64>().or_else(|_| s.trim().parse::<f64>().map(|f| f as i64)).unwrap_or(0),
            _ => 0,
        }
    }

    fn as_f64(&self) -> f64 {
        match self {
            Val::Int(i) => *i as f64,
            Val::Float(f, _) => *f,
            Val::Bool(b) => *b as i64 as f64,
            Val::Str(s) => s.trim().parse::<f64>().unwrap_or(0.0),
            _ => 0.0,
        }
    }
}

fn finding_val(f: &Finding) -> Val {
    let s = |x: &str| Val::Str(x.to_string());
    Val::Map(Rc::new(vec![
        ("RuleID".into(), s(&f.rule_id)),
        ("Description".into(), s(&f.description)),
        ("StartLine".into(), Val::Int(f.start_line)),
        ("EndLine".into(), Val::Int(f.end_line)),
        ("StartColumn".into(), Val::Int(f.start_column)),
        ("EndColumn".into(), Val::Int(f.end_column)),
        ("Line".into(), s(&f.line)),
        ("Match".into(), s(&f.match_)),
        ("Secret".into(), s(&f.secret)),
        ("File".into(), s(&f.file)),
        ("SymlinkFile".into(), s(&f.symlink_file)),
        ("Commit".into(), s(&f.commit)),
        ("Link".into(), s(&f.link)),
        ("Entropy".into(), Val::Float(f.entropy as f64, true)),
        ("Author".into(), s(&f.author)),
        ("Email".into(), s(&f.email)),
        ("Date".into(), s(&f.date)),
        ("Message".into(), s(&f.message)),
        ("Tags".into(), Val::List(Rc::new(f.tags.iter().map(|t| s(t)).collect()))),
        ("Fingerprint".into(), s(&f.fingerprint)),
    ]))
}

// ------------------------------------------------------------------ AST

#[derive(Clone, Debug)]
enum Arg {
    Dot,
    Field(Vec<String>),               // .A.B
    Var(String, Vec<String>),         // $x.A
    Ident(String),                    // function name
    Lit(Val),
    Pipe(Box<Pipeline>),              // (pipeline)
    Chain(Box<Arg>, Vec<String>),     // (x).Field
}

#[derive(Clone, Debug)]
struct Cmd {
    args: Vec<Arg>,
}

#[derive(Clone, Debug)]
struct Pipeline {
    decl: Vec<String>,
    is_assign: bool,
    cmds: Vec<Cmd>,
}

#[derive(Clone, Debug)]
enum Node {
    Text(String),
    Action(Pipeline),
    If(Pipeline, Vec<Node>, Vec<Node>),
    Range(Pipeline, Vec<Node>, Vec<Node>),
    With(Pipeline, Vec<Node>, Vec<Node>),
    Template(String, Option<Pipeline>),
    Break,
    Continue,
}

pub struct Template {
    root: Vec<Node>,
    defines: HashMap<String, Vec<Node>>,
}

// ------------------------------------------------------------------ lexer

#[derive(Clone, Debug)]
enum Piece {
    Text(String),
    Action(String),
}

fn split_pieces(src: &str) -> Result<Vec<Piece>, String> {
    let mut pieces = Vec::new();
    let mut rest = src;
    let mut trim_next_left = false;
    loop {
        match rest.find("{{") {
            None => {
                let mut t = rest.to_string();
                if trim_next_left {
                    t = t.trim_start().to_string();
                }
                if !t.is_empty() {
                    pieces.push(Piece::Text(t));
                }
                break;
            }
            Some(p) => {
                let mut text = rest[..p].to_string();
                if trim_next_left {
                    text = text.trim_start().to_string();
                }
                let after = &rest[p + 2..];
                let trim_left = after.starts_with("- ") || after.starts_with("-\t") || after.starts_with("-\n") || after.starts_with("-\r");
                if trim_left {
                    text = text.trim_end().to_string();
                }
                if !text.is_empty() {
                    pieces.push(Piece::Text(text));
                }
                let body_start = if trim_left { 1 } else { 0 };
                let after = &after[body_start..];
                // find closing "}}" outside of string literals
                let close = find_close(after).ok_or_else(|| "unclosed action".to_string())?;
                let mut body = &after[..close];
                let mut trim_right = false;
                if body.ends_with(" -") || body.ends_with("\t-") || body.ends_with("\n-") || body.ends_with("\r-") {
                    trim_right = true;
                    body = &body[..body.len() - 1];
                }
                pieces.push(Piece::Action(body.trim().to_string()));
                trim_next_left = trim_right;
                rest = &after[close + 2..];
            }
        }
    }
    Ok(pieces)
}

fn find_close(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            b'`' => {
                i += 1;
                while i < b.len() && b[i] != b'`' {
                    i += 1;
                }
            }
            b'\'' => {
                i += 1;
                while i < b.len() && b[i] != b'\'' {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            b'}' if i + 1 < b.len() && b[i + 1] == b'}' => return Some(i),
            _ => {}
        }
        i += 1;
    }
    None
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Dot,
    Field(String), // ".Name"
    ChainField(String), // ").Name"
    Var(String),   // "$name" or "$"
    Ident(String),
    Str(String),
    Num(String),
    Char(char),
    LParen,
    RParen,
    Pipe,
    Assign,  // :=
    Eq,      // =
    Comma,
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        match ch {
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
                // field chain directly after paren
            }
            '|' => {
                out.push(Tok::Pipe);
                i += 1;
            }
            ',' => {
                out.push(Tok::Comma);
                i += 1;
            }
            ':' if i + 1 < c.len() && c[i + 1] == '=' => {
                out.push(Tok::Assign);
                i += 2;
            }
            '=' => {
                out.push(Tok::Eq);
                i += 1;
            }
            '"' => {
                let mut j = i + 1;
                let mut raw = String::from("\"");
                while j < c.len() && c[j] != '"' {
                    if c[j] == '\\' && j + 1 < c.len() {
                        raw.push(c[j]);
                        j += 1;
                    }
                    raw.push(c[j]);
                    j += 1;
                }
                raw.push('"');
                out.push(Tok::Str(unquote(&raw)?));
                i = j + 1;
            }
            '`' => {
                let mut j = i + 1;
                let mut v = String::new();
                while j < c.len() && c[j] != '`' {
                    v.push(c[j]);
                    j += 1;
                }
                out.push(Tok::Str(v));
                i = j + 1;
            }
            '\'' => {
                let mut j = i + 1;
                let mut raw = String::new();
                while j < c.len() && c[j] != '\'' {
                    if c[j] == '\\' && j + 1 < c.len() {
                        raw.push(c[j]);
                        j += 1;
                    }
                    raw.push(c[j]);
                    j += 1;
                }
                let u = unquote(&format!("\"{}\"", raw))?;
                out.push(Tok::Char(u.chars().next().unwrap_or('\0')));
                i = j + 1;
            }
            '.' => {
                if i > 0 && c[i - 1] == ')' && i + 1 < c.len() && (c[i + 1].is_alphanumeric() || c[i + 1] == '_') {
                    let mut j = i + 1;
                    while j < c.len() && (c[j].is_alphanumeric() || c[j] == '_') {
                        j += 1;
                    }
                    out.push(Tok::ChainField(c[i + 1..j].iter().collect()));
                    i = j;
                } else if i + 1 < c.len() && (c[i + 1].is_alphanumeric() || c[i + 1] == '_') {
                    let mut j = i + 1;
                    while j < c.len() && (c[j].is_alphanumeric() || c[j] == '_') {
                        j += 1;
                    }
                    out.push(Tok::Field(c[i + 1..j].iter().collect()));
                    i = j;
                } else {
                    out.push(Tok::Dot);
                    i += 1;
                }
            }
            '$' => {
                let mut j = i + 1;
                while j < c.len() && (c[j].is_alphanumeric() || c[j] == '_') {
                    j += 1;
                }
                out.push(Tok::Var(c[i..j].iter().collect()));
                i = j;
            }
            ch if ch.is_ascii_digit() || ((ch == '-' || ch == '+') && i + 1 < c.len() && c[i + 1].is_ascii_digit()) => {
                let mut j = i + 1;
                while j < c.len() && (c[j].is_ascii_alphanumeric() || c[j] == '.' || c[j] == '_' || ((c[j] == '-' || c[j] == '+') && (c[j - 1] == 'e' || c[j - 1] == 'E'))) {
                    j += 1;
                }
                out.push(Tok::Num(c[i..j].iter().collect()));
                i = j;
            }
            ch if ch.is_alphabetic() || ch == '_' => {
                let mut j = i;
                while j < c.len() && (c[j].is_alphanumeric() || c[j] == '_') {
                    j += 1;
                }
                out.push(Tok::Ident(c[i..j].iter().collect()));
                i = j;
            }
            other => return Err(format!("unexpected {:?} in command", other)),
        }
    }
    Ok(out)
}

fn unquote(raw: &str) -> Result<String, String> {
    let inner = &raw[1..raw.len() - 1];
    let mut out = String::new();
    let c: Vec<char> = inner.chars().collect();
    let mut i = 0;
    while i < c.len() {
        if c[i] == '\\' && i + 1 < c.len() {
            i += 1;
            match c[i] {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                'a' => out.push('\x07'),
                'b' => out.push('\x08'),
                'f' => out.push('\x0c'),
                'v' => out.push('\x0b'),
                '\\' => out.push('\\'),
                '"' => out.push('"'),
                '\'' => out.push('\''),
                'x' if i + 2 < c.len() => {
                    let h: String = c[i + 1..i + 3].iter().collect();
                    out.push(u8::from_str_radix(&h, 16).map_err(|e| e.to_string())? as char);
                    i += 2;
                }
                'u' if i + 4 < c.len() => {
                    let h: String = c[i + 1..i + 5].iter().collect();
                    out.push(char::from_u32(u32::from_str_radix(&h, 16).map_err(|e| e.to_string())?).unwrap_or('\u{FFFD}'));
                    i += 4;
                }
                o => {
                    out.push('\\');
                    out.push(o)
                }
            }
            i += 1;
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------ parser

struct TokParser {
    toks: Vec<Tok>,
    pos: usize,
}

impl TokParser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn pipeline(&mut self, allow_decl: bool) -> Result<Pipeline, String> {
        let mut decl = Vec::new();
        let mut is_assign = false;
        if allow_decl {
            // look for "$x :=" or "$x, $y :="
            let save = self.pos;
            let mut vars = Vec::new();
            loop {
                match self.peek() {
                    Some(Tok::Var(v)) => {
                        vars.push(v.clone());
                        self.pos += 1;
                    }
                    _ => break,
                }
                match self.peek() {
                    Some(Tok::Comma) => {
                        self.pos += 1;
                        continue;
                    }
                    Some(Tok::Assign) => {
                        self.pos += 1;
                        decl = vars.clone();
                        break;
                    }
                    Some(Tok::Eq) => {
                        self.pos += 1;
                        decl = vars.clone();
                        is_assign = true;
                        break;
                    }
                    _ => break,
                }
            }
            if decl.is_empty() {
                self.pos = save;
            }
        }
        let mut cmds = Vec::new();
        loop {
            let cmd = self.command()?;
            cmds.push(cmd);
            match self.peek() {
                Some(Tok::Pipe) => {
                    self.pos += 1;
                }
                _ => break,
            }
        }
        Ok(Pipeline { decl, is_assign, cmds })
    }

    fn command(&mut self) -> Result<Cmd, String> {
        let mut args = Vec::new();
        loop {
            match self.peek() {
                None | Some(Tok::Pipe) | Some(Tok::RParen) => break,
                _ => args.push(self.operand()?),
            }
        }
        if args.is_empty() {
            return Err("missing value for command".to_string());
        }
        Ok(Cmd { args })
    }

    fn operand(&mut self) -> Result<Arg, String> {
        let t = self.toks.get(self.pos).cloned().ok_or("unexpected end")?;
        self.pos += 1;
        let base = match t {
            Tok::Dot => Arg::Dot,
            Tok::Field(f) => {
                let mut fields = vec![f];
                while let Some(Tok::Field(f)) = self.peek() {
                    fields.push(f.clone());
                    self.pos += 1;
                }
                return Ok(Arg::Field(fields));
            }
            Tok::Var(v) => {
                let mut fields = Vec::new();
                while let Some(Tok::Field(f)) = self.peek() {
                    fields.push(f.clone());
                    self.pos += 1;
                }
                return Ok(Arg::Var(v, fields));
            }
            Tok::Ident(id) => match id.as_str() {
                "true" => Arg::Lit(Val::Bool(true)),
                "false" => Arg::Lit(Val::Bool(false)),
                "nil" => Arg::Lit(Val::Nil),
                _ => Arg::Ident(id),
            },
            Tok::Str(s) => Arg::Lit(Val::Str(s)),
            Tok::Char(c) => Arg::Lit(Val::Int(c as i64)),
            Tok::Num(n) => Arg::Lit(parse_num(&n)?),
            Tok::LParen => {
                let p = self.pipeline(true)?;
                match self.peek() {
                    Some(Tok::RParen) => self.pos += 1,
                    _ => return Err("unclosed left paren".to_string()),
                }
                Arg::Pipe(Box::new(p))
            }
            other => return Err(format!("unexpected {:?} in operand", other)),
        };
        let mut fields = Vec::new();
        if let Arg::Pipe(_) = base {
            while let Some(Tok::ChainField(f)) = self.peek() {
                fields.push(f.clone());
                self.pos += 1;
            }
        }
        if fields.is_empty() {
            Ok(base)
        } else {
            Ok(Arg::Chain(Box::new(base), fields))
        }
    }
}

fn parse_num(n: &str) -> Result<Val, String> {
    let clean = n.replace('_', "");
    if let Ok(i) = clean.parse::<i64>() {
        return Ok(Val::Int(i));
    }
    if let Some(h) = clean.strip_prefix("0x").or_else(|| clean.strip_prefix("0X")) {
        if let Ok(i) = i64::from_str_radix(h, 16) {
            return Ok(Val::Int(i));
        }
    }
    clean.parse::<f64>().map(|f| Val::Float(f, false)).map_err(|_| format!("illegal number syntax: {:?}", n))
}

fn parse_pipeline(s: &str, allow_decl: bool) -> Result<Pipeline, String> {
    let toks = tokenize(s)?;
    let mut p = TokParser { toks, pos: 0 };
    let pl = p.pipeline(allow_decl)?;
    if p.pos != p.toks.len() {
        return Err(format!("unexpected {:?} in operand", p.toks[p.pos]));
    }
    Ok(pl)
}

enum Term {
    End,
    Else(String),
    Eof,
}

struct Parser {
    pieces: Vec<Piece>,
    pos: usize,
    defines: HashMap<String, Vec<Node>>,
}

impl Parser {
    fn list(&mut self) -> Result<(Vec<Node>, Term), String> {
        let mut nodes = Vec::new();
        while self.pos < self.pieces.len() {
            let p = self.pieces[self.pos].clone();
            self.pos += 1;
            match p {
                Piece::Text(t) => nodes.push(Node::Text(t)),
                Piece::Action(a) => {
                    if a.starts_with("/*") {
                        continue;
                    }
                    let (kw, rest) = match a.find(|c: char| c.is_whitespace()) {
                        Some(i) => (&a[..i], a[i..].trim()),
                        None => (a.as_str(), ""),
                    };
                    match kw {
                        "end" => return Ok((nodes, Term::End)),
                        "else" => return Ok((nodes, Term::Else(rest.to_string()))),
                        "if" | "with" | "range" => {
                            let pl = parse_pipeline(rest, true)?;
                            let node = self.control(kw, pl)?;
                            nodes.push(node);
                        }
                        "define" | "block" => {
                            let toks = tokenize(rest)?;
                            let name = match toks.first() {
                                Some(Tok::Str(s)) => s.clone(),
                                _ => return Err("missing name for define".to_string()),
                            };
                            let (body, term) = self.list()?;
                            if !matches!(term, Term::End) {
                                return Err("unexpected EOF".to_string());
                            }
                            self.defines.insert(name.clone(), body);
                            if kw == "block" {
                                let p = if toks.len() > 1 {
                                    let src = rest.splitn(2, |c: char| c.is_whitespace()).nth(1).unwrap_or("");
                                    Some(parse_pipeline(src, false)?)
                                } else {
                                    None
                                };
                                nodes.push(Node::Template(name, p));
                            }
                        }
                        "template" => {
                            let toks = tokenize(rest)?;
                            let name = match toks.first() {
                                Some(Tok::Str(s)) => s.clone(),
                                _ => return Err("missing name for template".to_string()),
                            };
                            let p = if toks.len() > 1 {
                                let idx = find_after_string(rest);
                                Some(parse_pipeline(&rest[idx..], false)?)
                            } else {
                                None
                            };
                            nodes.push(Node::Template(name, p));
                        }
                        "break" => nodes.push(Node::Break),
                        "continue" => nodes.push(Node::Continue),
                        _ => nodes.push(Node::Action(parse_pipeline(&a, true)?)),
                    }
                }
            }
        }
        Ok((nodes, Term::Eof))
    }

    fn control(&mut self, kw: &str, pl: Pipeline) -> Result<Node, String> {
        let (body, term) = self.list()?;
        let else_nodes = match term {
            Term::End => Vec::new(),
            Term::Eof => return Err(format!("unexpected EOF")),
            Term::Else(rest) => {
                if rest.starts_with("if ") || rest.starts_with("with ") {
                    // else if / else with chains
                    let (kw2, r2) = rest.split_at(rest.find(' ').unwrap());
                    let pl2 = parse_pipeline(r2.trim(), true)?;
                    vec![self.control(kw2, pl2)?]
                } else {
                    let (nodes, t2) = self.list()?;
                    if !matches!(t2, Term::End) {
                        return Err("expected end".to_string());
                    }
                    nodes
                }
            }
        };
        Ok(match kw {
            "if" => Node::If(pl, body, else_nodes),
            "with" => Node::With(pl, body, else_nodes),
            _ => Node::Range(pl, body, else_nodes),
        })
    }
}

fn find_after_string(s: &str) -> usize {
    let b = s.as_bytes();
    if b.is_empty() {
        return 0;
    }
    let q = b[0];
    let mut i = 1;
    while i < b.len() && b[i] != q {
        if b[i] == b'\\' && q == b'"' {
            i += 1;
        }
        i += 1;
    }
    (i + 1).min(b.len())
}

// ------------------------------------------------------------------ exec

enum Flow {
    Normal,
    Break,
    Continue,
}

struct Exec<'a> {
    tmpl: &'a Template,
    vars: Vec<(String, Val)>,
    out: String,
}

impl Template {
    pub fn parse(src: &str) -> Result<Template, String> {
        let pieces = split_pieces(src).map_err(|e| format!("template: custom:1: {}", e))?;
        let mut p = Parser { pieces, pos: 0, defines: HashMap::new() };
        let (root, term) = p.list().map_err(|e| format!("template: custom:1: {}", e))?;
        match term {
            Term::Eof => {}
            Term::End => return Err("template: custom:1: unexpected {{end}}".to_string()),
            Term::Else(_) => return Err("template: custom:1: unexpected {{else}}".to_string()),
        }
        Ok(Template { root, defines: p.defines })
    }

    pub fn execute(&self, findings: &[Finding]) -> Result<String, String> {
        let data = Val::List(Rc::new(findings.iter().map(finding_val).collect()));
        let mut ex = Exec { tmpl: self, vars: vec![("$".to_string(), data.clone())], out: String::new() };
        ex.walk(&data, &self.root).map_err(|e| format!("template: custom: {}", e))?;
        Ok(ex.out)
    }
}

impl<'a> Exec<'a> {
    fn walk(&mut self, dot: &Val, nodes: &[Node]) -> Result<Flow, String> {
        for n in nodes {
            match self.node(dot, n)? {
                Flow::Normal => {}
                f => return Ok(f),
            }
        }
        Ok(Flow::Normal)
    }

    fn node(&mut self, dot: &Val, n: &Node) -> Result<Flow, String> {
        match n {
            Node::Text(t) => self.out.push_str(t),
            Node::Action(p) => {
                let v = self.pipeline(dot, p)?;
                if p.decl.is_empty() {
                    self.out.push_str(&print_val(&v));
                }
            }
            Node::If(p, a, b) => {
                let mark = self.vars.len();
                let v = self.pipeline(dot, p)?;
                let f = if v.truth() { self.walk(dot, a)? } else { self.walk(dot, b)? };
                self.vars.truncate(mark);
                return Ok(f);
            }
            Node::With(p, a, b) => {
                let mark = self.vars.len();
                let v = self.pipeline(dot, p)?;
                let f = if v.truth() { self.walk(&v, a)? } else { self.walk(dot, b)? };
                self.vars.truncate(mark);
                return Ok(f);
            }
            Node::Range(p, a, b) => {
                let mark = self.vars.len();
                let mut p2 = p.clone();
                let decl = std::mem::take(&mut p2.decl);
                let v = self.pipeline(dot, &p2)?;
                let items: Vec<(Val, Val)> = match &v {
                    Val::List(l) => l.iter().enumerate().map(|(i, x)| (Val::Int(i as i64), x.clone())).collect(),
                    Val::Map(m) => {
                        let mut items: Vec<(String, Val)> = m.iter().cloned().collect();
                        items.sort_by(|a, b| a.0.cmp(&b.0));
                        items.into_iter().map(|(k, v)| (Val::Str(k), v)).collect()
                    }
                    Val::Int(n) => (0..*n).map(|i| (Val::Int(i), Val::Int(i))).collect(),
                    Val::Nil => Vec::new(),
                    other => return Err(format!("range can't iterate over {}", print_val(other))),
                };
                if items.is_empty() {
                    self.walk(dot, b)?;
                } else {
                    for (k, item) in items {
                        let m2 = self.vars.len();
                        if decl.len() == 1 {
                            self.vars.push((decl[0].clone(), item.clone()));
                        } else if decl.len() >= 2 {
                            self.vars.push((decl[0].clone(), k));
                            self.vars.push((decl[1].clone(), item.clone()));
                        }
                        let f = self.walk(&item, a)?;
                        self.vars.truncate(m2);
                        if let Flow::Break = f {
                            break;
                        }
                    }
                }
                self.vars.truncate(mark);
            }
            Node::Template(name, p) => {
                let body = self.tmpl.defines.get(name).ok_or_else(|| format!("no such template {:?}", name))?.clone();
                let d = match p {
                    Some(p) => self.pipeline(dot, p)?,
                    None => Val::Nil,
                };
                let saved = std::mem::replace(&mut self.vars, vec![("$".to_string(), d.clone())]);
                self.walk(&d, &body)?;
                self.vars = saved;
            }
            Node::Break => return Ok(Flow::Break),
            Node::Continue => return Ok(Flow::Continue),
        }
        Ok(Flow::Normal)
    }

    fn lookup_var(&self, name: &str) -> Result<Val, String> {
        for (k, v) in self.vars.iter().rev() {
            if k == name {
                return Ok(v.clone());
            }
        }
        Err(format!("undefined variable: {}", name))
    }

    fn pipeline(&mut self, dot: &Val, p: &Pipeline) -> Result<Val, String> {
        let mut prev: Option<Val> = None;
        for c in &p.cmds {
            prev = Some(self.command(dot, c, prev)?);
        }
        let v = prev.unwrap_or(Val::Nil);
        if !p.decl.is_empty() {
            if p.is_assign {
                let name = &p.decl[0];
                if let Some(slot) = self.vars.iter_mut().rev().find(|(k, _)| k == name) {
                    slot.1 = v.clone();
                } else {
                    return Err(format!("undefined variable: {}", name));
                }
            } else {
                self.vars.push((p.decl[0].clone(), v.clone()));
            }
        }
        Ok(v)
    }

    fn arg(&mut self, dot: &Val, a: &Arg) -> Result<Val, String> {
        match a {
            Arg::Dot => Ok(dot.clone()),
            Arg::Field(fs) => {
                let mut v = dot.clone();
                for f in fs {
                    v = v.field(f)?;
                }
                Ok(v)
            }
            Arg::Var(name, fs) => {
                let mut v = self.lookup_var(name)?;
                for f in fs {
                    v = v.field(f)?;
                }
                Ok(v)
            }
            Arg::Lit(v) => Ok(v.clone()),
            Arg::Pipe(p) => self.pipeline(dot, p),
            Arg::Chain(b, fs) => {
                let mut v = self.arg(dot, b)?;
                for f in fs {
                    v = v.field(f)?;
                }
                Ok(v)
            }
            Arg::Ident(name) => call_func(name, vec![]),
        }
    }

    fn command(&mut self, dot: &Val, c: &Cmd, prev: Option<Val>) -> Result<Val, String> {
        let first = &c.args[0];
        if let Arg::Ident(name) = first {
            // special-cased short-circuit and/or
            let mut args = Vec::new();
            for a in &c.args[1..] {
                args.push(self.arg(dot, a)?);
            }
            if let Some(p) = prev {
                args.push(p);
            }
            return call_func(name, args);
        }
        if c.args.len() > 1 {
            return Err(format!("can't give argument to non-function {:?}", first));
        }
        if prev.is_some() {
            return Err("can't give argument to non-function".to_string());
        }
        self.arg(dot, first)
    }
}

fn print_val(v: &Val) -> String {
    match v {
        Val::Nil => "<no value>".to_string(),
        other => other.to_string_v(),
    }
}

// ------------------------------------------------------------------ functions

fn cmp_vals(a: &Val, b: &Val) -> Result<std::cmp::Ordering, String> {
    use std::cmp::Ordering;
    Ok(match (a, b) {
        (Val::Int(x), Val::Int(y)) => x.cmp(y),
        (Val::Str(x), Val::Str(y)) => x.cmp(y),
        (Val::Int(_), Val::Float(..)) | (Val::Float(..), Val::Int(_)) | (Val::Float(..), Val::Float(..)) => {
            a.as_f64().partial_cmp(&b.as_f64()).unwrap_or(Ordering::Equal)
        }
        (Val::Bool(x), Val::Bool(y)) => x.cmp(y),
        _ => return Err("incompatible types for comparison".to_string()),
    })
}

fn eq_vals(a: &Val, b: &Val) -> Result<bool, String> {
    Ok(match (a, b) {
        (Val::Nil, Val::Nil) => true,
        (Val::Nil, _) | (_, Val::Nil) => false,
        _ => cmp_vals(a, b)? == std::cmp::Ordering::Equal,
    })
}

fn val_len(v: &Val) -> Result<i64, String> {
    match v {
        Val::Str(s) => Ok(s.len() as i64),
        Val::List(l) => Ok(l.len() as i64),
        Val::Map(m) => Ok(m.len() as i64),
        Val::Nil => Ok(0),
        other => Err(format!("len of type {}", other.type_name())),
    }
}

fn arg_n(args: &[Val], n: usize, name: &str) -> Result<(), String> {
    if args.len() != n {
        return Err(format!("wrong number of args for {}: want {} got {}", name, n, args.len()));
    }
    Ok(())
}

fn to_str(v: &Val) -> String {
    match v {
        Val::Nil => String::new(),
        o => o.to_string_v(),
    }
}

fn call_func(name: &str, args: Vec<Val>) -> Result<Val, String> {
    let s = |v: String| Ok(Val::Str(v));
    match name {
        "len" => {
            arg_n(&args, 1, name)?;
            Ok(Val::Int(val_len(&args[0])?))
        }
        "not" => {
            arg_n(&args, 1, name)?;
            Ok(Val::Bool(!args[0].truth()))
        }
        "and" => {
            for a in &args {
                if !a.truth() {
                    return Ok(a.clone());
                }
            }
            Ok(args.last().cloned().unwrap_or(Val::Nil))
        }
        "or" => {
            for a in &args {
                if a.truth() {
                    return Ok(a.clone());
                }
            }
            Ok(args.last().cloned().unwrap_or(Val::Nil))
        }
        "eq" => {
            if args.len() < 2 {
                return Err("missing argument for comparison".to_string());
            }
            for b in &args[1..] {
                if eq_vals(&args[0], b)? {
                    return Ok(Val::Bool(true));
                }
            }
            Ok(Val::Bool(false))
        }
        "ne" => {
            arg_n(&args, 2, name)?;
            Ok(Val::Bool(!eq_vals(&args[0], &args[1])?))
        }
        "lt" | "le" | "gt" | "ge" => {
            arg_n(&args, 2, name)?;
            let o = cmp_vals(&args[0], &args[1])?;
            use std::cmp::Ordering::*;
            Ok(Val::Bool(match name {
                "lt" => o == Less,
                "le" => o != Greater,
                "gt" => o == Greater,
                _ => o != Less,
            }))
        }
        "index" => {
            let mut v = args.first().cloned().unwrap_or(Val::Nil);
            for idx in &args[1..] {
                v = match (&v, idx) {
                    (Val::List(l), i) => {
                        let i = i.as_i64();
                        l.get(i as usize).cloned().ok_or_else(|| format!("index out of range: {}", i))?
                    }
                    (Val::Map(_), Val::Str(k)) => v.field(k)?,
                    (Val::Str(st), i) => Val::Int(*st.as_bytes().get(i.as_i64() as usize).ok_or("index out of range")? as i64),
                    _ => return Err("can't index item".to_string()),
                };
            }
            Ok(v)
        }
        "print" => {
            let mut out = String::new();
            for (i, a) in args.iter().enumerate() {
                let is_str = matches!(a, Val::Str(_));
                let prev_str = i > 0 && matches!(args[i - 1], Val::Str(_));
                if i > 0 && !is_str && !prev_str {
                    out.push(' ');
                }
                out.push_str(&a.to_string_v());
            }
            s(out)
        }
        "println" => {
            let parts: Vec<String> = args.iter().map(|a| a.to_string_v()).collect();
            s(parts.join(" ") + "\n")
        }
        "printf" => {
            if args.is_empty() {
                return Err("wrong number of args for printf".to_string());
            }
            s(go_sprintf(&to_str(&args[0]), &args[1..]))
        }
        "html" => s(to_str(args.first().unwrap_or(&Val::Nil))
            .replace('&', "&amp;")
            .replace('\'', "&#39;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&#34;")),
        "js" => {
            let mut out = String::new();
            for c in to_str(args.first().unwrap_or(&Val::Nil)).chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '\'' => out.push_str("\\'"),
                    '"' => out.push_str("\\\""),
                    '<' => out.push_str("\\u003C"),
                    '>' => out.push_str("\\u003E"),
                    '&' => out.push_str("\\u0026"),
                    '=' => out.push_str("\\u003D"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c => out.push(c),
                }
            }
            s(out)
        }
        "urlquery" => {
            let mut out = String::new();
            for b in to_str(args.first().unwrap_or(&Val::Nil)).bytes() {
                if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
                    out.push(b as char);
                } else if b == b' ' {
                    out.push('+');
                } else {
                    out.push_str(&format!("%{:02X}", b));
                }
            }
            s(out)
        }
        // ---------------- sprig subset
        "quote" => {
            let parts: Vec<String> = args.iter().filter(|a| !matches!(a, Val::Nil)).map(|a| gostr::quote(&a.to_string_v())).collect();
            s(parts.join(" "))
        }
        "squote" => {
            let parts: Vec<String> = args.iter().filter(|a| !matches!(a, Val::Nil)).map(|a| format!("'{}'", a.to_string_v())).collect();
            s(parts.join(" "))
        }
        "add" => Ok(Val::Int(args.iter().map(|a| a.as_i64()).sum())),
        "add1" => Ok(Val::Int(args.first().map(|a| a.as_i64()).unwrap_or(0) + 1)),
        "sub" => {
            arg_n(&args, 2, name)?;
            Ok(Val::Int(args[0].as_i64() - args[1].as_i64()))
        }
        "mul" => Ok(Val::Int(args.iter().map(|a| a.as_i64()).product())),
        "div" => {
            arg_n(&args, 2, name)?;
            let d = args[1].as_i64();
            if d == 0 {
                return Err("integer divide by zero".to_string());
            }
            Ok(Val::Int(args[0].as_i64() / d))
        }
        "mod" => {
            arg_n(&args, 2, name)?;
            let d = args[1].as_i64();
            if d == 0 {
                return Err("integer divide by zero".to_string());
            }
            Ok(Val::Int(args[0].as_i64() % d))
        }
        "max" => Ok(Val::Int(args.iter().map(|a| a.as_i64()).max().unwrap_or(0))),
        "min" => Ok(Val::Int(args.iter().map(|a| a.as_i64()).min().unwrap_or(0))),
        "upper" => s(to_str(&args[0]).to_uppercase()),
        "lower" => s(to_str(&args[0]).to_lowercase()),
        "title" => s(to_str(&args[0])
            .split(' ')
            .map(|w| {
                let mut c = w.chars();
                match c.next() {
                    Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")),
        "trim" => s(gostr::trim_space(&to_str(&args[0])).to_string()),
        "trimPrefix" => {
            arg_n(&args, 2, name)?;
            let (p, v) = (to_str(&args[0]), to_str(&args[1]));
            s(v.strip_prefix(p.as_str()).unwrap_or(&v).to_string())
        }
        "trimSuffix" => {
            arg_n(&args, 2, name)?;
            let (p, v) = (to_str(&args[0]), to_str(&args[1]));
            s(v.strip_suffix(p.as_str()).unwrap_or(&v).to_string())
        }
        "replace" => {
            arg_n(&args, 3, name)?;
            s(to_str(&args[2]).replace(&to_str(&args[0]), &to_str(&args[1])))
        }
        "contains" => {
            arg_n(&args, 2, name)?;
            Ok(Val::Bool(to_str(&args[1]).contains(&to_str(&args[0]))))
        }
        "hasPrefix" => {
            arg_n(&args, 2, name)?;
            Ok(Val::Bool(to_str(&args[1]).starts_with(&to_str(&args[0]))))
        }
        "hasSuffix" => {
            arg_n(&args, 2, name)?;
            Ok(Val::Bool(to_str(&args[1]).ends_with(&to_str(&args[0]))))
        }
        "repeat" => {
            arg_n(&args, 2, name)?;
            s(to_str(&args[1]).repeat(args[0].as_i64().max(0) as usize))
        }
        "trunc" => {
            arg_n(&args, 2, name)?;
            let n = args[0].as_i64();
            let v = to_str(&args[1]);
            if n < 0 {
                let l = v.len() as i64;
                if -n < l {
                    return s(v[(l + n) as usize..].to_string());
                }
                return s(v);
            }
            if (n as usize) < v.len() {
                return s(v[..n as usize].to_string());
            }
            s(v)
        }
        "join" => {
            arg_n(&args, 2, name)?;
            let sep = to_str(&args[0]);
            match &args[1] {
                Val::List(l) => s(l.iter().map(to_str).collect::<Vec<_>>().join(&sep)),
                other => s(to_str(other)),
            }
        }
        "list" => Ok(Val::List(Rc::new(args))),
        "default" => {
            if args.len() < 2 {
                return Ok(args.first().cloned().unwrap_or(Val::Nil));
            }
            if args[1].truth() {
                Ok(args[1].clone())
            } else {
                Ok(args[0].clone())
            }
        }
        "empty" => Ok(Val::Bool(!args.first().map(|a| a.truth()).unwrap_or(false))),
        "toString" => s(to_str(&args[0])),
        "toJson" | "toPrettyJson" => {
            let j = val_to_json(&args[0]);
            if name == "toJson" {
                s(j.compact())
            } else {
                s(j.indent("", "  "))
            }
        }
        "b64enc" => s(b64enc(to_str(&args[0]).as_bytes())),
        "now" => Ok(Val::Time(chrono::Local::now())),
        "date" => {
            arg_n(&args, 2, name)?;
            let fmt = to_str(&args[0]);
            match &args[1] {
                Val::Time(t) => s(go_time_format(t, &fmt)),
                _ => s(go_time_format(&chrono::Local::now(), &fmt)),
            }
        }
        "abbrev" => {
            arg_n(&args, 2, name)?;
            let w = args[0].as_i64() as usize;
            let v = to_str(&args[1]);
            if w < 4 || v.len() <= w {
                return s(v);
            }
            s(format!("{}...", &v[..w - 3]))
        }
        _ => Err(format!("function {:?} not defined", name)),
    }
}

fn val_to_json(v: &Val) -> crate::gojson::JVal {
    use crate::gojson::JVal;
    match v {
        Val::Nil => JVal::Null,
        Val::Bool(b) => JVal::Bool(*b),
        Val::Int(i) => JVal::i(*i),
        Val::Float(f, b32) => JVal::Num(if *b32 { gostr::format_float32_json(*f as f32) } else { gostr::format_float_v(*f, false) }),
        Val::Str(s) => JVal::s(s),
        Val::List(l) => JVal::Arr(l.iter().map(val_to_json).collect()),
        Val::Map(m) => JVal::Obj(m.iter().map(|(k, v)| (k.clone(), val_to_json(v))).collect()),
        Val::Time(t) => JVal::s(&t.to_rfc3339()),
    }
}

fn b64enc(b: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for ch in b.chunks(3) {
        let v = (ch[0] as u32) << 16 | (*ch.get(1).unwrap_or(&0) as u32) << 8 | *ch.get(2).unwrap_or(&0) as u32;
        out.push(A[(v >> 18) as usize & 63] as char);
        out.push(A[(v >> 12) as usize & 63] as char);
        out.push(if ch.len() > 1 { A[(v >> 6) as usize & 63] as char } else { '=' });
        out.push(if ch.len() > 2 { A[v as usize & 63] as char } else { '=' });
    }
    out
}

/// Go time layout formatting (subset)
fn go_time_format(t: &chrono::DateTime<chrono::Local>, layout: &str) -> String {
    let reps: [(&str, &str); 24] = [
        ("January", "%B"),
        ("January", "%B"),
        ("Monday", "%A"),
        ("2006", "%Y"),
        ("-07:00", "%:z"),
        ("-0700", "%z"),
        ("Z07:00", "%:z"),
        ("Jan", "%b"),
        ("Mon", "%a"),
        ("MST", "%Z"),
        ("01", "%m"),
        ("02", "%d"),
        ("_2", "%e"),
        ("15", "%H"),
        ("03", "%I"),
        ("04", "%M"),
        ("05", "%S"),
        ("06", "%y"),
        ("PM", "%p"),
        ("pm", "%P"),
        (".000", "%.3f"),
        ("1", "%-m"),
        ("2", "%-d"),
        ("3", "%-I"),
    ];
    let mut out = String::new();
    let mut rest = layout;
    'outer: while !rest.is_empty() {
        for (k, v) in reps.iter() {
            if rest.starts_with(k) {
                if *k == "MST" {
                    let off = t.offset().local_minus_utc();
                    let z = t.format("%Z").to_string();
                    if off == 0 && (z.starts_with('+') || z.starts_with('-')) {
                        out.push_str("UTC");
                    } else if z.starts_with('+') || z.starts_with('-') {
                        out.push_str(&t.format("%z").to_string());
                    } else {
                        out.push_str(&z);
                    }
                    rest = &rest[k.len()..];
                    continue 'outer;
                }
                out.push_str(&t.format(v).to_string());
                rest = &rest[k.len()..];
                continue 'outer;
            }
        }
        let c = rest.chars().next().unwrap();
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

/// A subset of Go's fmt.Sprintf
pub fn go_sprintf(format: &str, args: &[Val]) -> String {
    let mut out = String::new();
    let c: Vec<char> = format.chars().collect();
    let mut i = 0;
    let mut ai = 0;
    while i < c.len() {
        if c[i] != '%' {
            out.push(c[i]);
            i += 1;
            continue;
        }
        i += 1;
        if i >= c.len() {
            out.push_str("%!(NOVERB)");
            break;
        }
        let mut minus = false;
        let mut zero = false;
        let mut plus = false;
        let mut space = false;
        let mut sharp = false;
        while i < c.len() && "-0+ #".contains(c[i]) {
            match c[i] {
                '-' => minus = true,
                '0' => zero = true,
                '+' => plus = true,
                ' ' => space = true,
                _ => sharp = true,
            }
            i += 1;
        }
        let mut width: Option<usize> = None;
        while i < c.len() && c[i].is_ascii_digit() {
            width = Some(width.unwrap_or(0) * 10 + c[i].to_digit(10).unwrap() as usize);
            i += 1;
        }
        let mut prec: Option<usize> = None;
        if i < c.len() && c[i] == '.' {
            i += 1;
            let mut p = 0;
            while i < c.len() && c[i].is_ascii_digit() {
                p = p * 10 + c[i].to_digit(10).unwrap() as usize;
                i += 1;
            }
            prec = Some(p);
        }
        if i >= c.len() {
            out.push_str("%!(NOVERB)");
            break;
        }
        let verb = c[i];
        i += 1;
        if verb == '%' {
            out.push('%');
            continue;
        }
        if ai >= args.len() {
            out.push_str(&format!("%!{}(MISSING)", verb));
            continue;
        }
        let a = &args[ai];
        ai += 1;
        let _ = sharp;
        let mut body = match verb {
            'v' | 's' => {
                let mut s = match a {
                    Val::Nil => "<nil>".to_string(),
                    o => o.to_string_v(),
                };
                if let Some(p) = prec {
                    if verb == 's' || matches!(a, Val::Str(_)) {
                        s = s.chars().take(p).collect();
                    }
                }
                if let (Val::Float(f, b32), Some(_)) = (a, prec) {
                    if verb == 'v' {
                        s = gostr::format_float_v(*f, *b32);
                    }
                }
                s
            }
            'q' => gostr::quote(&a.to_string_v()),
            'd' => {
                let v = a.as_i64();
                let mut s = v.abs().to_string();
                if v < 0 {
                    s = format!("-{}", s);
                } else if plus {
                    s = format!("+{}", s);
                } else if space {
                    s = format!(" {}", s);
                }
                s
            }
            'f' | 'F' => {
                let f = match a {
                    Val::Float(f, b32) => {
                        if *b32 {
                            (*f as f32) as f64
                        } else {
                            *f
                        }
                    }
                    o => o.as_f64(),
                };
                let p = prec.unwrap_or(6);
                let mut s = format!("{:.*}", p, f);
                if plus && f >= 0.0 {
                    s = format!("+{}", s);
                }
                s
            }
            'e' => format!("{:.*e}", prec.unwrap_or(6), a.as_f64()),
            'g' => gostr::format_float_v(a.as_f64(), false),
            't' => a.to_string_v(),
            'x' | 'X' => {
                let s = match a {
                    Val::Int(v) => format!("{:x}", v),
                    o => o.to_string_v().bytes().map(|b| format!("{:02x}", b)).collect(),
                };
                if verb == 'X' {
                    s.to_uppercase()
                } else {
                    s
                }
            }
            'c' => char::from_u32(a.as_i64() as u32).map(|c| c.to_string()).unwrap_or_default(),
            other => format!("%!{}({})", other, a.to_string_v()),
        };
        if let Some(w) = width {
            let len = body.chars().count();
            if len < w {
                let pad = w - len;
                if minus {
                    body.push_str(&" ".repeat(pad));
                } else if zero && matches!(verb, 'd' | 'f' | 'F' | 'x' | 'X') {
                    let (sign, digits) = if body.starts_with('-') || body.starts_with('+') {
                        body.split_at(1)
                    } else {
                        ("", body.as_str())
                    };
                    body = format!("{}{}{}", sign, "0".repeat(pad), digits);
                } else {
                    body = format!("{}{}", " ".repeat(pad), body);
                }
            }
        }
        out.push_str(&body);
    }
    if ai < args.len() {
        out.push_str("%!(EXTRA ");
        let extras: Vec<String> = args[ai..].iter().map(|a| format!("{}={}", a.type_name(), a.to_string_v())).collect();
        out.push_str(&extras.join(", "));
        out.push(')');
    }
    out
}
