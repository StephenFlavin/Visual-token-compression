// A compact implementation of the Go text/template subset (with common sprig functions)
// used by the template report format.

use crate::report::Finding;
use crate::util::go_quote;

#[derive(Clone, Debug)]
enum Val {
    Nil,
    Str(String),
    Int(i64),
    F32(f32),
    Float(f64),
    Bool(bool),
    List(Vec<Val>),
    Finding(Box<Finding>),
}

impl Val {
    fn truthy(&self) -> bool {
        match self {
            Val::Nil => false,
            Val::Str(s) => !s.is_empty(),
            Val::Int(i) => *i != 0,
            Val::F32(f) => *f != 0.0,
            Val::Float(f) => *f != 0.0,
            Val::Bool(b) => *b,
            Val::List(l) => !l.is_empty(),
            Val::Finding(_) => true,
        }
    }
    fn to_string_v(&self) -> String {
        match self {
            Val::Nil => "<nil>".into(),
            Val::Str(s) => s.clone(),
            Val::Int(i) => i.to_string(),
            Val::F32(f) => fmt_v_float(*f as f64, true),
            Val::Float(f) => fmt_v_float(*f, false),
            Val::Bool(b) => b.to_string(),
            Val::List(l) => format!("[{}]", l.iter().map(|v| v.to_string_v()).collect::<Vec<_>>().join(" ")),
            Val::Finding(f) => format!("{{{} {}}}", f.rule_id, f.description),
        }
    }
    fn to_i64(&self) -> i64 {
        match self {
            Val::Int(i) => *i,
            Val::F32(f) => *f as i64,
            Val::Float(f) => *f as i64,
            Val::Bool(b) => *b as i64,
            Val::Str(s) => s.parse().unwrap_or(0),
            _ => 0,
        }
    }
    fn to_f64(&self) -> f64 {
        match self {
            Val::Int(i) => *i as f64,
            Val::F32(f) => *f as f64,
            Val::Float(f) => *f,
            Val::Str(s) => s.parse().unwrap_or(0.0),
            _ => 0.0,
        }
    }
}

fn fmt_v_float(f: f64, is32: bool) -> String {
    if f == 0.0 {
        return "0".into();
    }
    let e = if is32 { format!("{:e}", f as f32) } else { format!("{:e}", f) };
    let (mant, exp) = e.split_at(e.find('e').unwrap());
    let exp: i32 = exp[1..].parse().unwrap_or(0);
    if !(-4..6).contains(&exp) {
        let sign = if exp < 0 { "-" } else { "+" };
        return format!("{}e{}{:02}", mant, sign, exp.abs());
    }
    if is32 {
        format!("{}", f as f32)
    } else {
        format!("{}", f)
    }
}

fn field(v: &Val, name: &str) -> Result<Val, String> {
    match v {
        Val::Finding(f) => Ok(match name {
            "RuleID" => Val::Str(f.rule_id.clone()),
            "Description" => Val::Str(f.description.clone()),
            "StartLine" => Val::Int(f.start_line),
            "EndLine" => Val::Int(f.end_line),
            "StartColumn" => Val::Int(f.start_column),
            "EndColumn" => Val::Int(f.end_column),
            "Line" => Val::Str(f.line.clone()),
            "Match" => Val::Str(f.match_.clone()),
            "Secret" => Val::Str(f.secret.clone()),
            "File" => Val::Str(f.file.clone()),
            "SymlinkFile" => Val::Str(f.symlink_file.clone()),
            "Commit" => Val::Str(f.commit.clone()),
            "Link" => Val::Str(f.link.clone()),
            "Entropy" => Val::F32(f.entropy),
            "Author" => Val::Str(f.author.clone()),
            "Email" => Val::Str(f.email.clone()),
            "Date" => Val::Str(f.date.clone()),
            "Message" => Val::Str(f.message.clone()),
            "Tags" => Val::List(f.tags.iter().map(|t| Val::Str(t.clone())).collect()),
            "Fingerprint" => Val::Str(f.fingerprint.clone()),
            "Fragment" => Val::Nil,
            _ => return Err(format!("can't evaluate field {} in type report.Finding", name)),
        }),
        Val::Nil => Err(format!("nil pointer evaluating interface {{}}.{}", name)),
        _ => Err(format!("can't evaluate field {}", name)),
    }
}

// ---------------------------------------------------------------- AST

#[derive(Clone, Debug)]
enum Arg {
    Dot(Vec<String>),
    Var(String, Vec<String>),
    Str(String),
    Num(String),
    Bool(bool),
    Nil,
    Ident(String),
    Pipe(Box<Pipeline>),
}

#[derive(Clone, Debug)]
struct Pipeline {
    decl: Vec<String>,
    assign: bool,
    cmds: Vec<Vec<Arg>>,
}

#[derive(Clone, Debug)]
enum Node {
    Text(String),
    Action(Pipeline),
    If(Pipeline, Vec<Node>, Vec<Node>),
    Range(Pipeline, Vec<Node>, Vec<Node>),
    With(Pipeline, Vec<Node>, Vec<Node>),
}

pub struct Template {
    nodes: Vec<Node>,
}

#[derive(Debug)]
enum Item {
    Text(String),
    Action(String),
}

fn lex(src: &str) -> Result<Vec<Item>, String> {
    let mut items = vec![];
    let mut rest = src;
    let mut trim_next = false;
    loop {
        match rest.find("{{") {
            None => {
                let mut t = rest.to_string();
                if trim_next {
                    t = t.trim_start_matches(|c: char| c == ' ' || c == '\t' || c == '\r' || c == '\n').to_string();
                }
                if !t.is_empty() {
                    items.push(Item::Text(t));
                }
                break;
            }
            Some(i) => {
                let mut text = rest[..i].to_string();
                let after = &rest[i + 2..];
                let left_trim = after.starts_with("- ") || after.starts_with("-\t") || after.starts_with("-\n") || after.starts_with("-\r");
                if trim_next {
                    text = text.trim_start_matches(|c: char| c == ' ' || c == '\t' || c == '\r' || c == '\n').to_string();
                }
                if left_trim {
                    text = text.trim_end_matches(|c: char| c == ' ' || c == '\t' || c == '\r' || c == '\n').to_string();
                }
                if !text.is_empty() {
                    items.push(Item::Text(text));
                }
                let body_start = if left_trim { 1 } else { 0 };
                let after = &after[body_start..];
                // find closing, respecting strings
                let b = after.as_bytes();
                let mut j = 0;
                let mut end = None;
                while j < b.len() {
                    match b[j] {
                        b'"' => {
                            j += 1;
                            while j < b.len() && b[j] != b'"' {
                                if b[j] == b'\\' {
                                    j += 1;
                                }
                                j += 1;
                            }
                        }
                        b'`' => {
                            j += 1;
                            while j < b.len() && b[j] != b'`' {
                                j += 1;
                            }
                        }
                        b'}' if j + 1 < b.len() && b[j + 1] == b'}' => {
                            end = Some(j);
                            break;
                        }
                        _ => {}
                    }
                    j += 1;
                }
                let end = end.ok_or("unclosed action")?;
                let mut body = &after[..end];
                let right_trim = body.ends_with(" -") || body.ends_with("\t-") || body.ends_with("\n-") || body.ends_with("\r-");
                if right_trim {
                    body = &body[..body.len() - 1];
                }
                trim_next = right_trim;
                let body = body.trim();
                if !(body.starts_with("/*") && body.ends_with("*/")) {
                    items.push(Item::Action(body.to_string()));
                }
                rest = &after[end + 2..];
            }
        }
    }
    Ok(items)
}

fn tokenize(s: &str) -> Result<Vec<String>, String> {
    let c: Vec<char> = s.chars().collect();
    let mut toks = vec![];
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch == '(' || ch == ')' || ch == '|' || ch == ',' {
            toks.push(ch.to_string());
            i += 1;
            continue;
        }
        if ch == ':' && i + 1 < c.len() && c[i + 1] == '=' {
            toks.push(":=".into());
            i += 2;
            continue;
        }
        if ch == '=' {
            toks.push("=".into());
            i += 1;
            continue;
        }
        if ch == '"' {
            let st = i;
            i += 1;
            while i < c.len() && c[i] != '"' {
                if c[i] == '\\' {
                    i += 1;
                }
                i += 1;
            }
            i += 1;
            toks.push(c[st..i.min(c.len())].iter().collect());
            continue;
        }
        if ch == '`' {
            let st = i;
            i += 1;
            while i < c.len() && c[i] != '`' {
                i += 1;
            }
            i += 1;
            toks.push(c[st..i.min(c.len())].iter().collect());
            continue;
        }
        let st = i;
        while i < c.len() && !c[i].is_whitespace() && !"()|,=\"`".contains(c[i]) && !(c[i] == ':' && i + 1 < c.len() && c[i + 1] == '=') {
            i += 1;
        }
        toks.push(c[st..i].iter().collect());
    }
    Ok(toks)
}

fn unquote_go(s: &str) -> String {
    if s.starts_with('`') {
        return s[1..s.len() - 1].to_string();
    }
    let inner = &s[1..s.len().max(2) - 1];
    let mut out = String::new();
    let mut it = inner.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some(o) => {
                    out.push('\\');
                    out.push(o);
                }
                None => {}
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn parse_arg(toks: &[String], i: &mut usize) -> Result<Arg, String> {
    let t = &toks[*i];
    *i += 1;
    if t == "(" {
        let mut depth = 1;
        let st = *i;
        while *i < toks.len() {
            if toks[*i] == "(" {
                depth += 1;
            } else if toks[*i] == ")" {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            *i += 1;
        }
        let inner = toks[st..*i].to_vec();
        *i += 1;
        return Ok(Arg::Pipe(Box::new(parse_pipeline(&inner)?)));
    }
    if t.starts_with('"') || t.starts_with('`') {
        return Ok(Arg::Str(unquote_go(t)));
    }
    if t == "true" || t == "false" {
        return Ok(Arg::Bool(t == "true"));
    }
    if t == "nil" {
        return Ok(Arg::Nil);
    }
    if t.starts_with('.') {
        let fields: Vec<String> = t[1..].split('.').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
        return Ok(Arg::Dot(fields));
    }
    if t.starts_with('$') {
        let mut parts = t.split('.');
        let name = parts.next().unwrap().to_string();
        return Ok(Arg::Var(name, parts.map(|s| s.to_string()).collect()));
    }
    if t.chars().next().map(|c| c.is_ascii_digit() || c == '-' || c == '+').unwrap_or(false) {
        return Ok(Arg::Num(t.clone()));
    }
    Ok(Arg::Ident(t.clone()))
}

fn parse_pipeline(toks: &[String]) -> Result<Pipeline, String> {
    let mut decl = vec![];
    let mut assign = false;
    let mut start = 0;
    // variable declarations: $a := or $a, $b :=
    if let Some(pos) = toks.iter().position(|t| t == ":=" || t == "=") {
        let lhs = &toks[..pos];
        if !lhs.is_empty() && lhs.iter().all(|t| t.starts_with('$') || t == ",") {
            decl = lhs.iter().filter(|t| *t != ",").cloned().collect();
            assign = toks[pos] == "=";
            start = pos + 1;
        }
    }
    let mut cmds = vec![];
    let mut cur = vec![];
    let mut i = start;
    while i < toks.len() {
        if toks[i] == "|" {
            cmds.push(std::mem::take(&mut cur));
            i += 1;
            continue;
        }
        cur.push(parse_arg(toks, &mut i)?);
    }
    if !cur.is_empty() {
        cmds.push(cur);
    }
    if cmds.is_empty() {
        return Err("missing value for command".into());
    }
    Ok(Pipeline { decl, assign, cmds })
}

fn parse_nodes(items: &[Item], i: &mut usize, stop: &[&str]) -> Result<(Vec<Node>, String), String> {
    let mut nodes = vec![];
    while *i < items.len() {
        match &items[*i] {
            Item::Text(t) => {
                nodes.push(Node::Text(t.clone()));
                *i += 1;
            }
            Item::Action(a) => {
                let kw = a.split_whitespace().next().unwrap_or("");
                if stop.contains(&kw) {
                    *i += 1;
                    return Ok((nodes, a.clone()));
                }
                *i += 1;
                match kw {
                    "if" | "range" | "with" => {
                        let toks = tokenize(&a[kw.len()..])?;
                        let pipe = parse_pipeline(&toks)?;
                        let (body, term) = parse_nodes(items, i, &["end", "else"])?;
                        let mut else_body = vec![];
                        if term.starts_with("else") {
                            let rest = term["else".len()..].trim();
                            if rest.starts_with("if ") || rest.starts_with("with ") {
                                // else if chain: parse as nested node ending at the same end
                                let kw2 = rest.split_whitespace().next().unwrap();
                                let toks2 = tokenize(&rest[kw2.len()..])?;
                                let pipe2 = parse_pipeline(&toks2)?;
                                let (b2, t2) = parse_nodes(items, i, &["end", "else"])?;
                                let mut e2 = vec![];
                                if t2.starts_with("else") {
                                    let (b3, _) = parse_nodes(items, i, &["end"])?;
                                    e2 = b3;
                                }
                                else_body.push(if kw2 == "if" { Node::If(pipe2, b2, e2) } else { Node::With(pipe2, b2, e2) });
                            } else {
                                let (b, _) = parse_nodes(items, i, &["end"])?;
                                else_body = b;
                            }
                        }
                        nodes.push(match kw {
                            "if" => Node::If(pipe, body, else_body),
                            "range" => Node::Range(pipe, body, else_body),
                            _ => Node::With(pipe, body, else_body),
                        });
                    }
                    "end" | "else" => return Err(format!("unexpected {{{{{}}}}}", kw)),
                    _ => {
                        let toks = tokenize(a)?;
                        nodes.push(Node::Action(parse_pipeline(&toks)?));
                    }
                }
            }
        }
    }
    if !stop.is_empty() {
        return Err("unexpected EOF".into());
    }
    Ok((nodes, String::new()))
}

struct Ctx {
    vars: Vec<(String, Val)>,
}

impl Ctx {
    fn lookup(&self, n: &str) -> Result<Val, String> {
        self.vars
            .iter()
            .rev()
            .find(|(k, _)| k == n)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| format!("undefined variable: {}", n))
    }
    fn set(&mut self, n: &str, v: Val) {
        if let Some(e) = self.vars.iter_mut().rev().find(|(k, _)| k == n) {
            e.1 = v;
        }
    }
}

fn eval_arg(ctx: &mut Ctx, dot: &Val, a: &Arg) -> Result<Val, String> {
    match a {
        Arg::Dot(fields) => {
            let mut v = dot.clone();
            for f in fields {
                v = field(&v, f)?;
            }
            Ok(v)
        }
        Arg::Var(n, fields) => {
            let mut v = ctx.lookup(n)?;
            for f in fields {
                v = field(&v, f)?;
            }
            Ok(v)
        }
        Arg::Str(s) => Ok(Val::Str(s.clone())),
        Arg::Num(n) => {
            if let Ok(i) = n.parse::<i64>() {
                Ok(Val::Int(i))
            } else {
                n.parse::<f64>().map(Val::Float).map_err(|_| format!("bad number syntax: {}", n))
            }
        }
        Arg::Bool(b) => Ok(Val::Bool(*b)),
        Arg::Nil => Ok(Val::Nil),
        Arg::Ident(name) => call(ctx, dot, name, vec![]),
        Arg::Pipe(p) => eval_pipeline(ctx, dot, p),
    }
}

fn eq_vals(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Str(x), Val::Str(y)) => x == y,
        (Val::Bool(x), Val::Bool(y)) => x == y,
        (Val::Int(_), _) | (Val::F32(_), _) | (Val::Float(_), _) => match b {
            Val::Int(_) | Val::F32(_) | Val::Float(_) => {
                if let (Val::Int(x), Val::Int(y)) = (a, b) {
                    x == y
                } else {
                    a.to_f64() == b.to_f64()
                }
            }
            _ => false,
        },
        (Val::Nil, Val::Nil) => true,
        _ => a.to_string_v() == b.to_string_v(),
    }
}

fn call(ctx: &mut Ctx, dot: &Val, name: &str, args: Vec<Val>) -> Result<Val, String> {
    let _ = ctx;
    let s = |i: usize| args.get(i).map(|v| v.to_string_v()).unwrap_or_default();
    Ok(match name {
        "len" => Val::Int(match args.first() {
            Some(Val::Str(x)) => x.len() as i64,
            Some(Val::List(l)) => l.len() as i64,
            _ => return Err("len of nil pointer".into()),
        }),
        "quote" => Val::Str(
            args.iter().filter(|v| !matches!(v, Val::Nil)).map(|v| go_quote(&v.to_string_v())).collect::<Vec<_>>().join(" "),
        ),
        "squote" => Val::Str(
            args.iter().filter(|v| !matches!(v, Val::Nil)).map(|v| format!("'{}'", v.to_string_v())).collect::<Vec<_>>().join(" "),
        ),
        "sub" => Val::Int(args[0].to_i64() - args[1].to_i64()),
        "add" => Val::Int(args.iter().map(|v| v.to_i64()).sum()),
        "add1" => Val::Int(args[0].to_i64() + 1),
        "mul" => Val::Int(args.iter().map(|v| v.to_i64()).product()),
        "div" => Val::Int(args[0].to_i64() / args[1].to_i64().max(1)),
        "mod" => Val::Int(args[0].to_i64() % args[1].to_i64().max(1)),
        "eq" => Val::Bool(args[1..].iter().any(|b| eq_vals(&args[0], b))),
        "ne" => Val::Bool(!eq_vals(&args[0], &args[1])),
        "lt" => Val::Bool(args[0].to_f64() < args[1].to_f64()),
        "le" => Val::Bool(args[0].to_f64() <= args[1].to_f64()),
        "gt" => Val::Bool(args[0].to_f64() > args[1].to_f64()),
        "ge" => Val::Bool(args[0].to_f64() >= args[1].to_f64()),
        "not" => Val::Bool(!args[0].truthy()),
        "and" => args.iter().find(|v| !v.truthy()).cloned().unwrap_or_else(|| args.last().cloned().unwrap_or(Val::Nil)),
        "or" => args.iter().find(|v| v.truthy()).cloned().unwrap_or_else(|| args.last().cloned().unwrap_or(Val::Nil)),
        "print" => Val::Str(args.iter().map(|v| v.to_string_v()).collect::<Vec<_>>().join("")),
        "println" => Val::Str(args.iter().map(|v| v.to_string_v()).collect::<Vec<_>>().join(" ") + "\n"),
        "printf" => Val::Str(sprintf(&s(0), &args[1..])),
        "upper" => Val::Str(s(0).to_uppercase()),
        "lower" => Val::Str(s(0).to_lowercase()),
        "trim" => Val::Str(s(0).trim().to_string()),
        "toString" => Val::Str(s(0)),
        "replace" => Val::Str(s(2).replace(&s(0), &s(1))),
        "contains" => Val::Bool(s(1).contains(&s(0))),
        "hasPrefix" => Val::Bool(s(1).starts_with(&s(0))),
        "hasSuffix" => Val::Bool(s(1).ends_with(&s(0))),
        "join" => Val::Str(match args.get(1) {
            Some(Val::List(l)) => l.iter().map(|v| v.to_string_v()).collect::<Vec<_>>().join(&s(0)),
            Some(v) => v.to_string_v(),
            None => String::new(),
        }),
        "default" => {
            if args.len() > 1 && args[1].truthy() {
                args[1].clone()
            } else {
                args[0].clone()
            }
        }
        "list" => Val::List(args.clone()),
        "html" => Val::Str(
            s(0).replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&#34;").replace('\'', "&#39;"),
        ),
        "index" => match (&args[0], args.get(1)) {
            (Val::List(l), Some(i)) => l.get(i.to_i64() as usize).cloned().ok_or("index out of range")?,
            _ => Val::Nil,
        },
        _ => {
            let _ = dot;
            return Err(format!("function {} not defined", go_quote(name)));
        }
    })
}

fn sprintf(fmtstr: &str, args: &[Val]) -> String {
    let mut out = String::new();
    let mut it = fmtstr.chars().peekable();
    let mut ai = 0;
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('%') => out.push('%'),
            Some(v) => {
                let a = args.get(ai).cloned().unwrap_or(Val::Nil);
                ai += 1;
                match v {
                    'q' => out.push_str(&go_quote(&a.to_string_v())),
                    'd' => out.push_str(&a.to_i64().to_string()),
                    _ => out.push_str(&a.to_string_v()),
                }
            }
            None => out.push('%'),
        }
    }
    out
}

fn eval_pipeline(ctx: &mut Ctx, dot: &Val, p: &Pipeline) -> Result<Val, String> {
    let mut last: Option<Val> = None;
    for cmd in &p.cmds {
        let v = match &cmd[0] {
            Arg::Ident(name) => {
                let mut args = vec![];
                for a in &cmd[1..] {
                    args.push(eval_arg(ctx, dot, a)?);
                }
                if let Some(l) = last.take() {
                    args.push(l);
                }
                call(ctx, dot, name, args)?
            }
            a => {
                if cmd.len() > 1 {
                    return Err("can't give argument to non-function".into());
                }
                eval_arg(ctx, dot, a)?
            }
        };
        last = Some(v);
    }
    let v = last.unwrap_or(Val::Nil);
    if !p.decl.is_empty() && p.decl.len() == 1 {
        if p.assign {
            ctx.set(&p.decl[0], v.clone());
        } else {
            ctx.vars.push((p.decl[0].clone(), v.clone()));
        }
    }
    Ok(v)
}

fn exec(ctx: &mut Ctx, dot: &Val, nodes: &[Node], out: &mut String) -> Result<(), String> {
    for n in nodes {
        match n {
            Node::Text(t) => out.push_str(t),
            Node::Action(p) => {
                let v = eval_pipeline(ctx, dot, p)?;
                if p.decl.is_empty() {
                    out.push_str(&v.to_string_v());
                }
            }
            Node::If(p, body, els) => {
                let mark = ctx.vars.len();
                let v = eval_pipeline(ctx, dot, &Pipeline { decl: vec![], assign: false, cmds: p.cmds.clone() })?;
                if v.truthy() {
                    exec(ctx, dot, body, out)?;
                } else {
                    exec(ctx, dot, els, out)?;
                }
                ctx.vars.truncate(mark);
            }
            Node::With(p, body, els) => {
                let mark = ctx.vars.len();
                let v = eval_pipeline(ctx, dot, p)?;
                if v.truthy() {
                    exec(ctx, &v, body, out)?;
                } else {
                    exec(ctx, dot, els, out)?;
                }
                ctx.vars.truncate(mark);
            }
            Node::Range(p, body, els) => {
                let mark = ctx.vars.len();
                let pp = Pipeline { decl: vec![], assign: false, cmds: p.cmds.clone() };
                let v = eval_pipeline(ctx, dot, &pp)?;
                let items: Vec<Val> = match v {
                    Val::List(l) => l,
                    Val::Nil => vec![],
                    Val::Int(n) => (0..n).map(Val::Int).collect(),
                    _ => return Err("range can't iterate over value".into()),
                };
                if items.is_empty() {
                    exec(ctx, dot, els, out)?;
                }
                for (idx, item) in items.into_iter().enumerate() {
                    let inner = ctx.vars.len();
                    match p.decl.len() {
                        1 => ctx.vars.push((p.decl[0].clone(), item.clone())),
                        2 => {
                            ctx.vars.push((p.decl[0].clone(), Val::Int(idx as i64)));
                            ctx.vars.push((p.decl[1].clone(), item.clone()));
                        }
                        _ => {}
                    }
                    exec(ctx, &item, body, out)?;
                    ctx.vars.truncate(inner);
                }
                ctx.vars.truncate(mark);
            }
        }
    }
    Ok(())
}

impl Template {
    pub fn from_file(path: &str) -> Result<Template, String> {
        if path.is_empty() {
            return Err("template path cannot be empty".into());
        }
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("error reading file: open {}: {}", path, crate::io_err_str(&e)))?;
        Template::parse(&content).map_err(|e| format!("error parsing file: template: custom:{}", e))
    }

    pub fn parse(src: &str) -> Result<Template, String> {
        let items = lex(src)?;
        let mut i = 0;
        let (nodes, _) = parse_nodes(&items, &mut i, &[])?;
        Ok(Template { nodes })
    }

    pub fn execute(&self, findings: &[Finding]) -> Result<String, String> {
        let data = Val::List(findings.iter().map(|f| Val::Finding(Box::new(f.clone()))).collect());
        let mut ctx = Ctx { vars: vec![("$".to_string(), data.clone())] };
        let mut out = String::new();
        exec(&mut ctx, &data, &self.nodes, &mut out)?;
        Ok(out)
    }
}
