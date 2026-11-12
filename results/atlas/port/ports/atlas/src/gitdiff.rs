// A port of the subset of github.com/gitleaks/go-gitdiff used by gitleaks: parses `git log -p`
// and `git diff` output into files with added-line fragments and commit headers.

use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct PatchHeader {
    pub sha: String,
    pub author: Option<(String, String)>,
    pub date_utc: String,
    pub title: String,
    pub body: String,
}

impl PatchHeader {
    fn empty() -> PatchHeader {
        PatchHeader {
            sha: String::new(),
            author: None,
            date_utc: "0001-01-01T00:00:00Z".to_string(),
            title: String::new(),
            body: String::new(),
        }
    }
    pub fn message(&self) -> String {
        let mut m = self.title.clone();
        if !self.body.is_empty() {
            m.push_str("\n\n");
            m.push_str(&self.body);
        }
        m
    }
}

#[derive(Debug, Default)]
pub struct TextFragment {
    pub new_position: i64,
    pub raw_added: String,
}

#[derive(Debug, Default)]
pub struct DiffFile {
    pub old_name: String,
    pub new_name: String,
    pub is_new: bool,
    pub is_delete: bool,
    pub is_binary: bool,
    pub fragments: Vec<TextFragment>,
    pub header: Option<Rc<PatchHeader>>,
}

struct P<'a> {
    lines: Vec<&'a str>,
    i: usize,
}

impl<'a> P<'a> {
    fn line(&self, d: usize) -> &'a str {
        self.lines.get(self.i + d).cloned().unwrap_or("")
    }
    fn eof(&self) -> bool {
        self.i >= self.lines.len()
    }
}

pub fn parse(input: &str) -> Vec<DiffFile> {
    let mut p = P { lines: input.split_inclusive('\n').collect(), i: 0 };
    let mut out = vec![];
    let mut ph: Option<Rc<PatchHeader>> = Some(Rc::new(PatchHeader::empty()));
    loop {
        // ParseNextFileHeader
        let mut preamble = String::new();
        let mut file: Option<DiffFile> = None;
        let mut frag_err = false;
        while !p.eof() {
            let line = p.line(0);
            if line.starts_with("@@ -") {
                if parse_fragment_header(line).is_some() {
                    frag_err = true;
                    break;
                }
            } else if line.starts_with("commit") {
                preamble.clear();
            } else if line.starts_with("diff --git ") {
                match parse_git_file_header(&mut p) {
                    Some(f) => {
                        file = Some(f);
                        break;
                    }
                    None => return out,
                }
            } else if let Some(f) = parse_traditional_header(&mut p) {
                file = Some(f);
                break;
            }
            preamble.push_str(line);
            p.i += 1;
        }
        if frag_err {
            // error: "patch fragment without file header" -> skip the line and continue
            p.i += 1;
            continue;
        }
        if preamble.contains("commit") {
            ph = parse_patch_header(&preamble).map(Rc::new);
        }
        let mut f = match file {
            Some(f) => f,
            None => break,
        };
        match parse_text_fragments(&mut p, &mut f) {
            Err(_) => return out,
            Ok(0) => {
                if parse_binary(&mut p) {
                    f.is_binary = true;
                }
            }
            Ok(_) => {}
        }
        f.header = ph.clone();
        out.push(f);
    }
    out
}

fn parse_range(s: &str) -> Option<(i64, i64)> {
    let mut parts = s.splitn(2, ',');
    let start = parts.next()?.parse::<i64>().ok()?;
    let len = match parts.next() {
        Some(x) => x.parse::<i64>().ok()?,
        None => 1,
    };
    Some((start, len))
}

/// Returns (old_pos, old_lines, new_pos, new_lines)
fn parse_fragment_header(line: &str) -> Option<(i64, i64, i64, i64)> {
    let rest = line.strip_prefix("@@ -")?;
    let end = rest.find(" @@")?;
    let header = &rest[..end];
    let ranges: Vec<&str> = header.split(" +").collect();
    if ranges.len() != 2 {
        return None;
    }
    let (op, ol) = parse_range(ranges[0])?;
    let (np, nl) = parse_range(ranges[1])?;
    Some((op, ol, np, nl))
}

fn parse_text_fragments(p: &mut P, f: &mut DiffFile) -> Result<usize, ()> {
    let mut n = 0;
    loop {
        let line = p.line(0);
        if !line.starts_with("@@ -") {
            return Ok(n);
        }
        let (_op, ol, np, nl) = match parse_fragment_header(line) {
            Some(h) => h,
            None => return Err(()),
        };
        p.i += 1;
        if f.is_new && ol > 0 {
            return Err(());
        }
        if f.is_delete && nl > 0 {
            return Err(());
        }
        if p.eof() {
            return Err(());
        }
        let mut old_lines = ol;
        let mut new_lines = nl;
        let mut lines: Vec<(u8, String)> = vec![];
        let (mut added, mut deleted) = (0, 0);
        while old_lines > 0 || new_lines > 0 {
            if p.eof() {
                break;
            }
            let line = p.line(0);
            let op = line.as_bytes()[0];
            let data = &line[1..];
            match op {
                b'\n' | b' ' => {
                    old_lines -= 1;
                    new_lines -= 1;
                    lines.push((b' ', if op == b'\n' { "\n".to_string() } else { data.to_string() }));
                }
                b'-' => {
                    old_lines -= 1;
                    deleted += 1;
                    lines.push((b'-', data.to_string()));
                }
                b'+' => {
                    new_lines -= 1;
                    added += 1;
                    lines.push((b'+', data.to_string()));
                }
                b'\\' if line.len() >= 12 && line.starts_with("\\ ") => {
                    if let Some(last) = lines.last_mut() {
                        if last.1.ends_with('\n') {
                            last.1.pop();
                        }
                    }
                }
                _ => return Err(()),
            }
            p.i += 1;
        }
        if old_lines != 0 || new_lines != 0 {
            return Err(());
        }
        if added == 0 && deleted == 0 {
            return Err(());
        }
        let l = p.line(0);
        if l.len() >= 12 && l.starts_with("\\ ") {
            if let Some(last) = lines.last_mut() {
                if last.1.ends_with('\n') {
                    last.1.pop();
                }
            }
            p.i += 1;
        }
        let raw: String = lines.iter().filter(|(o, _)| *o == b'+').map(|(_, s)| s.as_str()).collect();
        f.fragments.push(TextFragment { new_position: np, raw_added: raw });
        n += 1;
    }
}

fn parse_binary(p: &mut P) -> bool {
    let line = p.line(0);
    if line == "GIT binary patch\n" {
        p.i += 1;
        // skip binary data blocks (forward and optional reverse)
        for _ in 0..2 {
            let l = p.line(0);
            if !(l.starts_with("literal ") || l.starts_with("delta ")) {
                break;
            }
            p.i += 1;
            while !p.eof() && p.line(0) != "\n" {
                p.i += 1;
            }
            if !p.eof() {
                p.i += 1;
            }
        }
        return true;
    }
    if line.starts_with("Binary files ") && line.ends_with(" differ\n") {
        p.i += 1;
        return true;
    }
    false
}

/// Unquotes a C-style quoted git path.
fn unquote(s: &str) -> Option<(String, usize)> {
    let b = s.as_bytes();
    if b.first() != Some(&b'"') {
        return None;
    }
    let mut out: Vec<u8> = vec![];
    let mut i = 1;
    while i < b.len() {
        match b[i] {
            b'"' => return Some((String::from_utf8_lossy(&out).into_owned(), i + 1)),
            b'\\' => {
                i += 1;
                if i >= b.len() {
                    return None;
                }
                match b[i] {
                    b'a' => out.push(7),
                    b'b' => out.push(8),
                    b'f' => out.push(12),
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'v' => out.push(11),
                    b'0'..=b'7' => {
                        let mut v: u32 = 0;
                        let mut k = 0;
                        while k < 3 && i < b.len() && (b'0'..=b'7').contains(&b[i]) {
                            v = v * 8 + (b[i] - b'0') as u32;
                            i += 1;
                            k += 1;
                        }
                        out.push(v as u8);
                        continue;
                    }
                    c => out.push(c),
                }
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    None
}

/// Parses a file name, dropping the first path component (a/ or b/).
fn parse_name(s: &str, term: Option<char>, drop: usize) -> String {
    let (mut name, _) = if let Some((n, used)) = unquote(s) {
        (n, used)
    } else {
        let end = match term {
            Some(t) => s.find(t).unwrap_or(s.len()),
            None => s.len(),
        };
        (s[..end].to_string(), end)
    };
    if name == "/dev/null" {
        return name;
    }
    for _ in 0..drop {
        match name.find('/') {
            Some(i) => name = name[i + 1..].to_string(),
            None => break,
        }
    }
    name
}

fn parse_git_header_name(header: &str) -> String {
    let h = header.trim_end_matches('\n');
    if h.starts_with('"') {
        if let Some((first, used)) = unquote(h) {
            let rest = h[used..].trim_start();
            let second = if rest.starts_with('"') { unquote(rest).map(|x| x.0).unwrap_or_default() } else { rest.to_string() };
            let a = first.splitn(2, '/').nth(1).unwrap_or("").to_string();
            let b = second.splitn(2, '/').nth(1).unwrap_or("").to_string();
            return if a == b { a } else { String::new() };
        }
        return String::new();
    }
    // unquoted: find a split where both halves (minus prefixes) match
    let bytes = h.as_bytes();
    for (i, &c) in bytes.iter().enumerate() {
        if c == b' ' {
            let first = &h[..i];
            let second = &h[i + 1..];
            if second.starts_with('"') {
                let s2 = unquote(second).map(|x| x.0).unwrap_or_default();
                let a = first.splitn(2, '/').nth(1).unwrap_or("");
                let b = s2.splitn(2, '/').nth(1).unwrap_or("");
                if a == b {
                    return a.to_string();
                }
                continue;
            }
            let a = first.splitn(2, '/').nth(1);
            let b = second.splitn(2, '/').nth(1);
            if let (Some(a), Some(b)) = (a, b) {
                if a == b {
                    return a.to_string();
                }
            }
        }
    }
    String::new()
}

fn parse_git_file_header(p: &mut P) -> Option<DiffFile> {
    let header = &p.line(0)["diff --git ".len()..];
    let default_name = parse_git_header_name(header);
    let mut f = DiffFile::default();
    loop {
        let raw = p.line(1);
        let line = raw.strip_suffix('\n').unwrap_or(raw);
        p.i += 1;
        if p.i >= p.lines.len() {
            break;
        }
        let mut end = true;
        let prefixes = [
            "--- ", "+++ ", "old mode ", "new mode ", "deleted file mode ", "new file mode ", "copy from ", "copy to ",
            "rename old ", "rename new ", "rename from ", "rename to ", "similarity index ", "dissimilarity index ",
            "index ",
        ];
        for pre in prefixes.iter() {
            if let Some(v) = line.strip_prefix(pre) {
                end = false;
                match *pre {
                    "--- " => {
                        let n = parse_name(v, Some('\t'), 1);
                        if n != "/dev/null" {
                            f.old_name = n;
                        }
                    }
                    "+++ " => {
                        let n = parse_name(v, Some('\t'), 1);
                        if n != "/dev/null" {
                            f.new_name = n;
                        }
                    }
                    "deleted file mode " => {
                        f.is_delete = true;
                        f.old_name = default_name.clone();
                    }
                    "new file mode " => {
                        f.is_new = true;
                        f.new_name = default_name.clone();
                    }
                    "copy from " | "rename old " | "rename from " => {
                        f.old_name = parse_name(v, None, 0);
                    }
                    "copy to " | "rename new " | "rename to " => {
                        f.new_name = parse_name(v, None, 0);
                    }
                    _ => {}
                }
                break;
            }
        }
        if end {
            break;
        }
    }
    if f.old_name.is_empty() && f.new_name.is_empty() {
        if default_name.is_empty() {
            return None;
        }
        f.old_name = default_name.clone();
        f.new_name = default_name;
    }
    if (f.new_name.is_empty() && !f.is_delete) || (f.old_name.is_empty() && !f.is_new) {
        return None;
    }
    Some(f)
}

fn parse_traditional_header(p: &mut P) -> Option<DiffFile> {
    let old = p.line(0);
    let new = p.line(1);
    if !old.starts_with("--- ") || !new.starts_with("+++ ") {
        return None;
    }
    let l2 = p.line(2);
    if l2.len() < "@@ -1 +1 @@\n".len() || !l2.starts_with("@@ -") {
        return None;
    }
    p.i += 2;
    let on = parse_name(&old[4..], Some('\t'), 0);
    let nn = parse_name(&new[4..], Some('\t'), 0);
    let mut f = DiffFile::default();
    if on == "/dev/null" {
        f.is_new = true;
        f.new_name = nn;
    } else if nn == "/dev/null" {
        f.is_delete = true;
        f.old_name = on;
    } else {
        f.old_name = on.clone();
        f.new_name = on;
    }
    Some(f)
}

fn parse_patch_header(s: &str) -> Option<PatchHeader> {
    let mut lines = s.split_inclusive('\n');
    let mut first = "";
    for l in lines.by_ref() {
        let t = l.trim();
        if !t.is_empty() {
            first = t;
            break;
        }
    }
    let rest = first.strip_prefix("commit ")?;
    let mut h = PatchHeader::empty();
    h.sha = match rest.find(' ') {
        Some(i) if i > 0 => rest[..i].to_string(),
        _ => rest.to_string(),
    };
    let remaining: Vec<&str> = lines.map(|l| l.strip_suffix('\n').unwrap_or(l)).map(|l| l.strip_suffix('\r').unwrap_or(l)).collect();
    let mut idx = 0;
    while idx < remaining.len() {
        let line = remaining[idx];
        idx += 1;
        if line.trim().is_empty() {
            break;
        }
        if let Some(v) = line.strip_prefix("Author:") {
            h.author = Some(parse_identity(v)?);
        } else if let Some(v) = line.strip_prefix("Date:") {
            h.date_utc = parse_date(v.trim())?;
        } else if let Some(v) = line.strip_prefix("AuthorDate:") {
            h.date_utc = parse_date(v.trim())?;
        } else if line.starts_with("Commit:") {
            parse_identity(&line["Commit:".len()..])?;
        }
    }
    // title
    let mut title = String::new();
    let mut indent = String::new();
    let mut first_line = true;
    while idx < remaining.len() {
        let line = remaining[idx];
        idx += 1;
        let t = line.trim();
        if t.is_empty() {
            break;
        }
        if first_line {
            let start = line.find(|c: char| !c.is_whitespace()).unwrap_or(0);
            if start > 0 {
                indent = line[..start].to_string();
            }
            first_line = false;
        }
        if !title.is_empty() {
            title.push(' ');
        }
        title.push_str(t);
    }
    h.title = title;
    if !h.title.is_empty() {
        let mut body = String::new();
        let mut empty = 0;
        while idx < remaining.len() {
            let line = remaining[idx];
            idx += 1;
            let line = line.trim_end();
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
    Some(h)
}

fn parse_identity(s: &str) -> Option<(String, String)> {
    let mut email_start = 0usize;
    let mut email_end = 0usize;
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
    let mut name = String::new();
    let mut email = String::new();
    if email_start > 0 {
        name = s[..email_start - 1].trim().to_string();
        email = s[email_start..email_end].trim().to_string();
    }
    Some((name, email))
}

const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn format_utc(epoch: i64) -> String {
    let days = epoch.div_euclid(86400);
    let secs = epoch.rem_euclid(86400);
    let (y, m, d) = civil_from_days(days);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, secs / 3600, (secs / 60) % 60, secs % 60)
}

fn parse_zone(z: &str) -> Option<i64> {
    let b = z.as_bytes();
    if b.len() == 5 && (b[0] == b'+' || b[0] == b'-') {
        let h: i64 = z[1..3].parse().ok()?;
        let m: i64 = z[3..5].parse().ok()?;
        let off = h * 3600 + m * 60;
        return Some(if b[0] == b'-' { -off } else { off });
    }
    if b.len() == 6 && (b[0] == b'+' || b[0] == b'-') && b[3] == b':' {
        let h: i64 = z[1..3].parse().ok()?;
        let m: i64 = z[4..6].parse().ok()?;
        let off = h * 3600 + m * 60;
        return Some(if b[0] == b'-' { -off } else { off });
    }
    if z == "Z" {
        return Some(0);
    }
    None
}

fn hms(t: &str) -> Option<i64> {
    let p: Vec<&str> = t.split(':').collect();
    if p.len() != 3 {
        return None;
    }
    Some(p[0].parse::<i64>().ok()? * 3600 + p[1].parse::<i64>().ok()? * 60 + p[2].parse::<i64>().ok()?)
}

fn local_offset(epoch_local_guess: i64) -> i64 {
    unsafe {
        let t: libc::time_t = epoch_local_guess as libc::time_t;
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        tm.tm_gmtoff as i64
    }
}

/// Parses git dates and returns the RFC3339 UTC representation.
fn parse_date(s: &str) -> Option<String> {
    if s.is_empty() {
        return Some("0001-01-01T00:00:00Z".into());
    }
    let parts: Vec<&str> = s.split_whitespace().collect();
    // default: Mon Jan 2 15:04:05 2006 -0700
    if parts.len() >= 5 {
        if let Some(mi) = MONTHS.iter().position(|m| *m == parts[1]) {
            let d: i64 = parts[2].parse().ok()?;
            let secs = hms(parts[3])?;
            let y: i64 = parts[4].parse().ok()?;
            let base = days_from_civil(y, mi as i64 + 1, d) * 86400 + secs;
            let off = if parts.len() >= 6 { parse_zone(parts[5])? } else { local_offset(base) };
            return Some(format_utc(base - off));
        }
    }
    // rfc2822: Mon, 2 Jan 2006 15:04:05 -0700
    if parts.len() == 6 && parts[0].ends_with(',') {
        let d: i64 = parts[1].parse().ok()?;
        let mi = MONTHS.iter().position(|m| *m == parts[2])?;
        let y: i64 = parts[3].parse().ok()?;
        let secs = hms(parts[4])?;
        let off = parse_zone(parts[5])?;
        return Some(format_utc(days_from_civil(y, mi as i64 + 1, d) * 86400 + secs - off));
    }
    // iso: 2006-01-02 15:04:05 -0700 / strict 2006-01-02T15:04:05-07:00 / short 2006-01-02
    let ymd = |x: &str| -> Option<(i64, i64, i64)> {
        let p: Vec<&str> = x.split('-').collect();
        if p.len() != 3 {
            return None;
        }
        Some((p[0].parse().ok()?, p[1].parse().ok()?, p[2].parse().ok()?))
    };
    if parts.len() == 3 {
        if let Some((y, m, d)) = ymd(parts[0]) {
            let secs = hms(parts[1])?;
            let off = parse_zone(parts[2])?;
            return Some(format_utc(days_from_civil(y, m, d) * 86400 + secs - off));
        }
    }
    if parts.len() == 1 && s.contains('T') {
        let (dpart, rest) = s.split_at(s.find('T')?);
        let rest = &rest[1..];
        let (y, m, d) = ymd(dpart)?;
        let tz_pos = rest.find(|c| c == '+' || c == '-' || c == 'Z')?;
        let secs = hms(&rest[..tz_pos])?;
        let off = parse_zone(&rest[tz_pos..])?;
        return Some(format_utc(days_from_civil(y, m, d) * 86400 + secs - off));
    }
    if parts.len() == 1 {
        if let Some((y, m, d)) = ymd(parts[0]) {
            let base = days_from_civil(y, m, d) * 86400;
            return Some(format_utc(base - local_offset(base)));
        }
        if let Ok(u) = s.parse::<i64>() {
            return Some(format_utc(u));
        }
    }
    if parts.len() == 2 {
        if let (Ok(u), Some(_)) = (parts[0].parse::<i64>(), parse_zone(parts[1])) {
            return Some(format_utc(u));
        }
    }
    None
}
