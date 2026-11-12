//! A port of the parts of github.com/gitleaks/go-gitdiff used by gitleaks:
//! parsing `git log -p` / `git diff` output into files and text fragments.

use std::io::BufRead;

#[derive(Clone, Debug, Default)]
pub struct PatchIdentity {
    pub name: String,
    pub email: String,
}

#[derive(Clone, Debug, Default)]
pub struct PatchHeader {
    pub sha: String,
    pub author: Option<PatchIdentity>,
    /// author date as (unix seconds); None = zero time
    pub author_date: Option<chrono::DateTime<chrono::FixedOffset>>,
    pub title: String,
    pub body: String,
}

impl PatchHeader {
    pub fn message(&self) -> String {
        let mut m = self.title.clone();
        if !self.body.is_empty() {
            m.push_str("\n\n");
            m.push_str(&self.body);
        }
        m
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineOp {
    Context,
    Delete,
    Add,
}

#[derive(Clone, Debug, Default)]
pub struct TextFragment {
    pub old_position: i64,
    pub old_lines: i64,
    pub new_position: i64,
    pub new_lines: i64,
    pub lines: Vec<(LineOp, String)>,
    pub lines_added: i64,
    pub lines_deleted: i64,
}

impl TextFragment {
    pub fn raw(&self, op: LineOp) -> String {
        let mut s = String::new();
        for (o, l) in &self.lines {
            if *o == op {
                s.push_str(l);
            }
        }
        s
    }
}

#[derive(Clone, Debug, Default)]
pub struct File {
    pub old_name: String,
    pub new_name: String,
    pub is_new: bool,
    pub is_delete: bool,
    pub is_binary: bool,
    pub text_fragments: Vec<TextFragment>,
    pub patch_header: Option<std::rc::Rc<PatchHeader>>,
}

pub struct Parser<R: BufRead> {
    r: R,
    eof: bool,
    lineno: i64,
    lines: [String; 3],
    ph: Option<std::rc::Rc<PatchHeader>>,
    started: bool,
    done: bool,
}

fn read_line<R: BufRead>(r: &mut R) -> String {
    let mut buf = Vec::new();
    match r.read_until(b'\n', &mut buf) {
        Ok(_) => String::from_utf8_lossy(&buf).into_owned(),
        Err(_) => String::new(),
    }
}

impl<R: BufRead> Parser<R> {
    pub fn new(r: R) -> Self {
        Parser {
            r,
            eof: false,
            lineno: 0,
            lines: [String::new(), String::new(), String::new()],
            ph: Some(std::rc::Rc::new(PatchHeader::default())),
            started: false,
            done: false,
        }
    }

    fn next(&mut self) -> bool {
        if self.eof {
            return false;
        }
        if self.lineno == 0 {
            for _ in 0..2 {
                self.shift();
            }
        }
        self.shift();
        self.lineno += 1;
        if self.lines[0].is_empty() {
            self.eof = true;
            return false;
        }
        true
    }

    fn shift(&mut self) {
        self.lines[0] = std::mem::take(&mut self.lines[1]);
        self.lines[1] = std::mem::take(&mut self.lines[2]);
        self.lines[2] = read_line(&mut self.r);
    }

    fn line(&self, d: usize) -> &str {
        &self.lines[d]
    }

    /// Returns the next parsed file or None at the end of the stream
    pub fn next_file(&mut self) -> Option<File> {
        if self.done {
            return None;
        }
        if !self.started {
            self.started = true;
            if !self.next() {
                self.done = true;
                return None;
            }
        }
        loop {
            let (file, pre) = match self.parse_next_file_header() {
                Ok(Some(x)) => x,
                Ok(None) => {
                    self.done = true;
                    return None;
                }
                Err(_) => {
                    if self.eof {
                        self.done = true;
                        return None;
                    }
                    self.next();
                    continue;
                }
            };
            if pre.contains("commit") {
                self.ph = parse_patch_header(&pre).map(std::rc::Rc::new);
            }
            let mut file = file;
            match self.parse_text_fragments(&mut file) {
                Err(_) => {
                    self.done = true;
                    return None;
                }
                Ok(n) => {
                    if n == 0 {
                        if self.parse_binary_marker(&mut file).is_err() {
                            self.done = true;
                            return None;
                        }
                    }
                }
            }
            file.patch_header = self.ph.clone();
            return Some(file);
        }
    }

    fn parse_next_file_header(&mut self) -> Result<Option<(File, String)>, String> {
        let mut preamble = String::new();
        loop {
            let mut skip = false;
            match self.parse_text_fragment_header() {
                Err(_) => skip = true,
                Ok(Some(_)) => return Err("patch fragment without file header".to_string()),
                Ok(None) => {}
            }
            if !skip {
                if self.line(0).starts_with("commit") {
                    preamble.clear();
                } else {
                    if let Some(f) = self.parse_git_file_header()? {
                        return Ok(Some((f, preamble)));
                    }
                    if let Some(f) = self.parse_traditional_file_header()? {
                        return Ok(Some((f, preamble)));
                    }
                }
            }
            preamble.push_str(&self.lines[0]);
            if !self.next() {
                break;
            }
        }
        Ok(None)
    }

    fn parse_git_file_header(&mut self) -> Result<Option<File>, String> {
        const PREFIX: &str = "diff --git ";
        if !self.line(0).starts_with(PREFIX) {
            return Ok(None);
        }
        let header = self.line(0)[PREFIX.len()..].to_string();
        let default_name = parse_git_header_name(&header)?;
        let mut f = File::default();
        loop {
            let l1 = self.line(1).to_string();
            let end = parse_git_header_data(&mut f, &l1, &default_name)?;
            if !self.next() {
                break;
            }
            if end {
                break;
            }
        }
        if f.old_name.is_empty() && f.new_name.is_empty() {
            if default_name.is_empty() {
                return Err("git file header: missing filename information".to_string());
            }
            f.old_name = default_name.clone();
            f.new_name = default_name;
        }
        if (f.new_name.is_empty() && !f.is_delete) || (f.old_name.is_empty() && !f.is_new) {
            return Err("git file header: missing filename information".to_string());
        }
        Ok(Some(f))
    }

    fn parse_traditional_file_header(&mut self) -> Result<Option<File>, String> {
        let old = self.line(0).to_string();
        let new = self.line(1).to_string();
        if !old.starts_with("--- ") || !new.starts_with("+++ ") {
            return Ok(None);
        }
        if self.line(2).len() < "@@ -1 +1 @@\n".len() || !self.line(2).starts_with("@@ -") {
            return Ok(None);
        }
        self.next();
        self.next();
        let (old_name, _) = parse_name(&old[4..], Some('\t'), 0)?;
        let (new_name, _) = parse_name(&new[4..], Some('\t'), 0)?;
        let mut f = File::default();
        if old_name == "/dev/null" {
            f.is_new = true;
            f.new_name = new_name;
        } else if new_name == "/dev/null" {
            f.is_delete = true;
            f.old_name = old_name;
        } else if new_name.starts_with(&old_name) {
            f.old_name = old_name.clone();
            f.new_name = old_name;
        } else {
            f.old_name = new_name.clone();
            f.new_name = new_name;
        }
        Ok(Some(f))
    }

    fn parse_text_fragment_header(&mut self) -> Result<Option<TextFragment>, String> {
        let l = self.line(0).to_string();
        if !l.starts_with("@@ -") {
            return Ok(None);
        }
        let idx = match l.find(" @@") {
            Some(i) => i,
            None => return Err("invalid fragment header".to_string()),
        };
        let header = &l[4..idx];
        let ranges: Vec<&str> = header.split(" +").collect();
        if ranges.len() != 2 {
            return Err("invalid fragment header".to_string());
        }
        let (op, ol) = parse_range(ranges[0])?;
        let (np, nl) = parse_range(ranges[1])?;
        let f = TextFragment { old_position: op, old_lines: ol, new_position: np, new_lines: nl, ..Default::default() };
        self.next();
        Ok(Some(f))
    }

    fn parse_text_fragments(&mut self, f: &mut File) -> Result<usize, String> {
        let mut n = 0;
        loop {
            let frag = match self.parse_text_fragment_header()? {
                Some(fr) => fr,
                None => return Ok(n),
            };
            let mut frag = frag;
            if f.is_new && frag.old_lines > 0 {
                return Err("new file depends on old contents".to_string());
            }
            if f.is_delete && frag.new_lines > 0 {
                return Err("deleted file still has contents".to_string());
            }
            self.parse_text_chunk(&mut frag)?;
            f.text_fragments.push(frag);
            n += 1;
        }
    }

    fn parse_text_chunk(&mut self, frag: &mut TextFragment) -> Result<(), String> {
        if self.line(0).is_empty() {
            return Err("no content following fragment header".to_string());
        }
        let (mut old, mut new) = (frag.old_lines, frag.new_lines);
        while old > 0 || new > 0 {
            let line = self.line(0).to_string();
            if line.is_empty() {
                break;
            }
            let op = line.as_bytes()[0];
            let data = line[1..].to_string();
            match op {
                b'\n' | b' ' => {
                    let data = if op == b'\n' { "\n".to_string() } else { data };
                    old -= 1;
                    new -= 1;
                    frag.lines.push((LineOp::Context, data));
                }
                b'-' => {
                    old -= 1;
                    frag.lines_deleted += 1;
                    frag.lines.push((LineOp::Delete, data));
                }
                b'+' => {
                    new -= 1;
                    frag.lines_added += 1;
                    frag.lines.push((LineOp::Add, data));
                }
                b'\\' if is_no_newline_marker(&line) => {
                    remove_last_newline(frag);
                }
                _ => return Err("invalid line operation".to_string()),
            }
            if !self.next() {
                break;
            }
        }
        if old != 0 || new != 0 {
            return Err("fragment header miscounts lines".to_string());
        }
        if frag.lines_added == 0 && frag.lines_deleted == 0 {
            return Err("fragment contains no changes".to_string());
        }
        if is_no_newline_marker(self.line(0)) {
            remove_last_newline(frag);
            self.next();
        }
        Ok(())
    }

    fn parse_binary_marker(&mut self, f: &mut File) -> Result<(), String> {
        let line = self.line(0).to_string();
        let is_bin = line == "GIT binary patch\n"
            || line == "Binary files differ\n"
            || line == "Files differ\n"
            || (line.starts_with("Binary files ") && line.ends_with("differ\n"));
        if !is_bin {
            return Ok(());
        }
        f.is_binary = true;
        let has_data = line == "GIT binary patch\n";
        self.next();
        if has_data {
            // skip binary data until an empty line or new header
            while !self.eof {
                let l = self.line(0).to_string();
                if l.starts_with("diff --git ") || l.starts_with("commit ") {
                    break;
                }
                self.next();
            }
        }
        Ok(())
    }
}

fn is_no_newline_marker(s: &str) -> bool {
    s.len() >= 12 && s.starts_with("\\ ")
}

fn remove_last_newline(frag: &mut TextFragment) {
    if let Some(last) = frag.lines.last_mut() {
        if last.1.ends_with('\n') {
            last.1.pop();
        }
    }
}

fn parse_range(s: &str) -> Result<(i64, i64), String> {
    let mut parts = s.splitn(2, ',');
    let start = parts.next().unwrap_or("").parse::<i64>().map_err(|_| "bad start of range".to_string())?;
    let end = match parts.next() {
        Some(e) => e.parse::<i64>().map_err(|_| "bad end of range".to_string())?,
        None => 1,
    };
    Ok((start, end))
}

fn is_space(c: u8) -> bool {
    c == b' ' || c == b'\t' || c == b'\n'
}

fn parse_git_header_name(header: &str) -> Result<String, String> {
    let header = header.strip_suffix('\n').unwrap_or(header);
    if header.is_empty() {
        return Ok(String::new());
    }
    let hb = header.as_bytes();
    let mut first: String;
    let mut second = String::new();
    match header.find('"') {
        None => first = header.to_string(),
        Some(q) if q > 0 => {
            if !is_space(hb[q - 1]) {
                return Err("missing separator".to_string());
            }
            first = header[..q - 1].to_string();
            second = parse_quoted_name(&header[q..])?.0;
        }
        Some(_) => {
            let (f, mut n) = parse_quoted_name(header)?;
            first = f;
            while n < hb.len() && is_space(hb[n]) {
                n += 1;
            }
            if n == hb.len() {
                return Ok(String::new());
            }
            if hb[n] == b'"' {
                second = parse_quoted_name(&header[n..])?.0;
            } else {
                second = header[n..].to_string();
            }
        }
    }
    first = trim_tree_prefix(&first, 1);
    if !second.is_empty() {
        if first == trim_tree_prefix(&second, 1) {
            return Ok(first);
        }
        return Ok(String::new());
    }
    let fb = first.as_bytes();
    for i in 0..fb.len().saturating_sub(1) {
        if !is_space(fb[i]) {
            continue;
        }
        let sec = trim_tree_prefix(&first[i + 1..], 1);
        if first[..i] == sec {
            return Ok(sec);
        }
    }
    Ok(String::new())
}

fn parse_git_header_data(f: &mut File, line: &str, default_name: &str) -> Result<bool, String> {
    let line = line.strip_suffix('\n').unwrap_or(line);
    let prefixes: [(&str, bool); 16] = [
        ("@@ -", true),
        ("--- ", false),
        ("+++ ", false),
        ("old mode ", false),
        ("new mode ", false),
        ("deleted file mode ", false),
        ("new file mode ", false),
        ("copy from ", false),
        ("copy to ", false),
        ("rename old ", false),
        ("rename new ", false),
        ("rename from ", false),
        ("rename to ", false),
        ("similarity index ", false),
        ("dissimilarity index ", false),
        ("index ", false),
    ];
    for (p, end) in prefixes.iter() {
        if let Some(rest) = line.strip_prefix(p) {
            match *p {
                "--- " => {
                    let (name, _) = parse_name(rest, Some('\t'), 1)?;
                    if f.old_name.is_empty() && !f.is_new {
                        f.old_name = name;
                    } else {
                        verify_name(&name, &f.old_name, f.is_new, "old")?;
                    }
                }
                "+++ " => {
                    let (name, _) = parse_name(rest, Some('\t'), 1)?;
                    if f.new_name.is_empty() && !f.is_delete {
                        f.new_name = name;
                    } else {
                        verify_name(&name, &f.new_name, f.is_delete, "new")?;
                    }
                }
                "deleted file mode " => {
                    f.is_delete = true;
                    f.old_name = default_name.to_string();
                }
                "new file mode " => {
                    f.is_new = true;
                    f.new_name = default_name.to_string();
                }
                "copy from " | "rename old " | "rename from " => {
                    f.old_name = parse_name(rest, None, 0)?.0;
                }
                "copy to " | "rename new " | "rename to " => {
                    f.new_name = parse_name(rest, None, 0)?.0;
                }
                _ => {}
            }
            return Ok(*end);
        }
    }
    Ok(true)
}

fn verify_name(parsed: &str, existing: &str, is_null: bool, side: &str) -> Result<(), String> {
    if !existing.is_empty() {
        if is_null {
            return Err(format!("expected /dev/null, but filename is set to {}", existing));
        }
        if existing != parsed {
            return Err(format!("inconsistent {} filename", side));
        }
    }
    if is_null && parsed != "/dev/null" {
        return Err("expected /dev/null".to_string());
    }
    Ok(())
}

fn parse_name(s: &str, term: Option<char>, drop: usize) -> Result<(String, usize), String> {
    let (name, n) = if s.starts_with('"') {
        parse_quoted_name(s)?
    } else {
        let mut n = 0;
        for (i, c) in s.char_indices() {
            if c == '\n' || Some(c) == term {
                break;
            }
            n = i + c.len_utf8();
        }
        if n == 0 {
            return Err("missing name".to_string());
        }
        (s[..n].to_string(), n)
    };
    if name == "/dev/null" {
        return Ok((name, n));
    }
    Ok((clean_name(&name, drop), n))
}

fn parse_quoted_name(s: &str) -> Result<(String, usize), String> {
    let b = s.as_bytes();
    let mut n = 1;
    while n < b.len() {
        if b[n] == b'"' && b[n - 1] != b'\\' {
            n += 1;
            break;
        }
        n += 1;
    }
    if n == 2 {
        return Err("missing name".to_string());
    }
    Ok((go_unquote(&s[..n])?, n))
}

/// strconv.Unquote for double-quoted strings (supports octal escapes used by git)
fn go_unquote(s: &str) -> Result<String, String> {
    if s.len() < 2 || !s.starts_with('"') || !s.ends_with('"') {
        return Err("invalid syntax".to_string());
    }
    let b = &s.as_bytes()[1..s.len() - 1];
    let mut out: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'\\' {
            out.push(b[i]);
            i += 1;
            continue;
        }
        i += 1;
        if i >= b.len() {
            return Err("invalid syntax".to_string());
        }
        match b[i] {
            b'n' => out.push(b'\n'),
            b't' => out.push(b'\t'),
            b'r' => out.push(b'\r'),
            b'a' => out.push(7),
            b'b' => out.push(8),
            b'f' => out.push(12),
            b'v' => out.push(11),
            b'\\' => out.push(b'\\'),
            b'"' => out.push(b'"'),
            b'\'' => out.push(b'\''),
            b'0'..=b'7' => {
                if i + 2 >= b.len() + 0 && i + 2 > b.len() - 1 + 1 {
                    return Err("invalid syntax".to_string());
                }
                let oct = std::str::from_utf8(&b[i..(i + 3).min(b.len())]).map_err(|_| "invalid syntax")?;
                let v = u8::from_str_radix(oct, 8).map_err(|_| "invalid syntax".to_string())?;
                out.push(v);
                i += 2;
            }
            b'x' => {
                let hex = std::str::from_utf8(&b[i + 1..(i + 3).min(b.len())]).map_err(|_| "invalid syntax")?;
                out.push(u8::from_str_radix(hex, 16).map_err(|_| "invalid syntax".to_string())?);
                i += 2;
            }
            _ => return Err("invalid syntax".to_string()),
        }
        i += 1;
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

fn clean_name(name: &str, mut drop: usize) -> String {
    let b = name.as_bytes();
    let mut out: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' {
            if i < b.len() - 1 && b[i + 1] == b'/' {
                i += 1;
                continue;
            }
            if drop > 0 {
                drop -= 1;
                out.clear();
                i += 1;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn trim_tree_prefix(name: &str, mut n: usize) -> String {
    let b = name.as_bytes();
    let mut i = 0;
    while i < b.len() && n > 0 {
        if b[i] == b'/' {
            n -= 1;
        }
        i += 1;
    }
    name[i..].to_string()
}

// ------------------------------------------------------------------ patch header

pub fn parse_patch_header(s: &str) -> Option<PatchHeader> {
    let mut lines = s.split_inclusive('\n');
    let mut first = String::new();
    for l in lines.by_ref() {
        let t = l.trim();
        if !t.is_empty() {
            first = t.to_string();
            break;
        }
    }
    if let Some(rest) = first.strip_prefix("commit ") {
        let mut h = PatchHeader::default();
        h.sha = match rest.find(' ') {
            Some(i) if i > 0 => rest[..i].to_string(),
            _ => rest.to_string(),
        };
        let remaining: Vec<String> = lines.map(|l| l.trim_end_matches('\n').trim_end_matches('\r').to_string()).collect();
        let mut idx = 0;
        while idx < remaining.len() {
            let line = &remaining[idx];
            idx += 1;
            if line.trim().is_empty() {
                break;
            }
            if let Some(v) = line.strip_prefix("Author:") {
                h.author = Some(parse_patch_identity(v)?);
            } else if let Some(v) = line.strip_prefix("Date:") {
                h.author_date = Some(parse_patch_date(v.trim())?);
            } else if let Some(v) = line.strip_prefix("AuthorDate:") {
                h.author_date = Some(parse_patch_date(v.trim())?);
            }
        }
        // title
        let mut title = String::new();
        let mut indent = String::new();
        let mut i = 0;
        while idx < remaining.len() {
            let line = &remaining[idx];
            idx += 1;
            let t = line.trim();
            if t.is_empty() {
                break;
            }
            if i == 0 {
                if let Some(start) = line.find(|c: char| !c.is_whitespace()) {
                    if start > 0 {
                        indent = line[..start].to_string();
                    }
                }
            }
            if !title.is_empty() {
                title.push(' ');
            }
            title.push_str(t);
            i += 1;
        }
        h.title = title;
        if !h.title.is_empty() {
            let mut body = String::new();
            let mut empty = 0;
            while idx < remaining.len() {
                let line = remaining[idx].trim_end();
                idx += 1;
                let line = line.strip_prefix(indent.as_str()).unwrap_or(line);
                if line.is_empty() {
                    empty += 1;
                    continue;
                }
                if !body.is_empty() {
                    body.push('\n');
                    if empty > 0 {
                        body.push('\n');
                    }
                }
                empty = 0;
                body.push_str(line);
            }
            h.body = body;
        }
        return Some(h);
    }
    None
}

fn parse_patch_identity(s: &str) -> Option<PatchIdentity> {
    let mut email_start = 0;
    let mut email_end = 0;
    for (i, c) in s.char_indices() {
        if c == '<' && email_start == 0 {
            email_start = i + 1;
        }
        if c == '>' && email_start > 0 {
            email_end = i;
            break;
        }
    }
    if email_start > 0 && email_end == 0 {
        return None;
    }
    let mut id = PatchIdentity::default();
    if email_start > 0 {
        id.name = s[..email_start - 1].trim().to_string();
        id.email = s[email_start..email_end].trim().to_string();
    }
    Some(id)
}

fn parse_patch_date(s: &str) -> Option<chrono::DateTime<chrono::FixedOffset>> {
    use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone};
    if s.is_empty() {
        return None;
    }
    for f in ["%Y-%m-%d %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%:z", "%a, %e %b %Y %H:%M:%S %z", "%a %b %e %H:%M:%S %Y %z"] {
        if let Ok(t) = DateTime::parse_from_str(s, f) {
            return Some(t);
        }
    }
    let local_offset = |nd: NaiveDateTime| -> Option<DateTime<chrono::FixedOffset>> {
        chrono::Local.from_local_datetime(&nd).single().map(|t| t.fixed_offset())
    };
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return local_offset(d.and_hms_opt(0, 0, 0)?);
    }
    if let Ok(nd) = NaiveDateTime::parse_from_str(s, "%a %b %e %H:%M:%S %Y") {
        return local_offset(nd);
    }
    if let Ok(u) = s.parse::<i64>() {
        return chrono::DateTime::from_timestamp(u, 0).map(|t| t.fixed_offset());
    }
    if let Some(sp) = s.find(' ') {
        if let (Ok(u), Ok(z)) = (s[..sp].parse::<i64>(), chrono::FixedOffset::from_str_tz(&s[sp + 1..])) {
            return chrono::DateTime::from_timestamp(u, 0).map(|t| t.with_timezone(&z));
        }
    }
    None
}

trait FromStrTz {
    fn from_str_tz(s: &str) -> Result<chrono::FixedOffset, ()>;
}

impl FromStrTz for chrono::FixedOffset {
    fn from_str_tz(s: &str) -> Result<chrono::FixedOffset, ()> {
        let b = s.as_bytes();
        if b.len() != 5 || (b[0] != b'+' && b[0] != b'-') {
            return Err(());
        }
        let h: i32 = s[1..3].parse().map_err(|_| ())?;
        let m: i32 = s[3..5].parse().map_err(|_| ())?;
        let secs = h * 3600 + m * 60;
        let secs = if b[0] == b'-' { -secs } else { secs };
        chrono::FixedOffset::east_opt(secs).ok_or(())
    }
}
