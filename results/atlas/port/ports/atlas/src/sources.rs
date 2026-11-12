// Port of the sources package: File, Files (directory walk), Git (git log -p parsing) and archives.

use crate::config::Config;
use crate::gitdiff;
use crate::logging;
use crate::util;
use std::sync::Arc;

pub const INNER_PATH_SEPARATOR: &str = "!";
const DEFAULT_BUFFER_SIZE: usize = 100 * 1_000;
const MAX_PEEK_SIZE: usize = 25 * 1_000;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Platform {
    Unknown,
    None,
    GitHub,
    GitLab,
    AzureDevOps,
    Gitea,
    Bitbucket,
}

impl Platform {
    pub fn as_str(&self) -> &'static str {
        match self {
            Platform::Unknown => "unknown",
            Platform::None => "none",
            Platform::GitHub => "github",
            Platform::GitLab => "gitlab",
            Platform::AzureDevOps => "azuredevops",
            Platform::Gitea => "gitea",
            Platform::Bitbucket => "bitbucket",
        }
    }
    pub fn from_str(s: &str) -> Result<Platform, String> {
        match s.to_lowercase().as_str() {
            "" | "unknown" => Ok(Platform::Unknown),
            "none" => Ok(Platform::None),
            "github" => Ok(Platform::GitHub),
            "gitlab" => Ok(Platform::GitLab),
            "azuredevops" => Ok(Platform::AzureDevOps),
            "gitea" => Ok(Platform::Gitea),
            "bitbucket" => Ok(Platform::Bitbucket),
            _ => Err(format!("invalid scm platform value: {}", s)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RemoteInfo {
    pub platform: Platform,
    pub url: String,
}

#[derive(Clone, Debug)]
pub struct CommitInfo {
    pub author_email: String,
    pub author_name: String,
    pub date: String,
    pub message: String,
    pub remote: Arc<RemoteInfo>,
    pub sha: String,
}

#[derive(Clone, Debug, Default)]
pub struct Fragment {
    pub raw: String,
    pub bytes_len: Option<usize>,
    pub file_path: String,
    pub symlink_file: String,
    pub windows_file_path: String,
    pub commit_sha: String,
    pub start_line: i64,
    pub commit_info: Option<Arc<CommitInfo>>,
    pub inherited_from_finding: bool,
}

pub type Yield<'a> = &'a mut dyn FnMut(Fragment, Option<String>);

// ---------------------------------------------------------------- archives

#[derive(Clone, Copy, PartialEq, Debug)]
enum Comp {
    Brotli,
    Bz2,
    Gz,
    Lz4,
    Lzip,
    MinLZ,
    Sz,
    Xz,
    Zlib,
    Zstd,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Arch {
    SevenZip,
    Rar,
    Tar,
    Zip,
}

#[derive(Clone, Copy, Debug)]
struct Format {
    comp: Option<Comp>,
    arch: Option<Arch>,
}

fn identify(path: &str) -> Option<Format> {
    let name = util::base(path).to_lowercase();
    let name = if name == "." { String::new() } else { name };
    let ext = util::ext(&name);
    let comp = if name.contains(".br") {
        Some(Comp::Brotli)
    } else if name.contains(".bz2") {
        Some(Comp::Bz2)
    } else if name.contains(".gz") {
        Some(Comp::Gz)
    } else if name.contains(".lz4") {
        Some(Comp::Lz4)
    } else if ext == ".lz" {
        Some(Comp::Lzip)
    } else if ext == ".mz" {
        Some(Comp::MinLZ)
    } else if name.contains(".sz") || name.contains(".s2") {
        Some(Comp::Sz)
    } else if name.contains(".xz") {
        Some(Comp::Xz)
    } else if name.contains(".zz") {
        Some(Comp::Zlib)
    } else if name.contains(".zst") {
        Some(Comp::Zstd)
    } else {
        None
    };
    let arch = if name.contains(".7z") {
        Some(Arch::SevenZip)
    } else if name.contains(".rar") {
        Some(Arch::Rar)
    } else if name.contains(".tar") {
        Some(Arch::Tar)
    } else if name.contains(".zip") {
        Some(Arch::Zip)
    } else {
        None
    };
    if comp.is_none() && arch.is_none() {
        return None;
    }
    Some(Format { comp, arch })
}

pub fn is_archive(path: &str) -> bool {
    identify(path).is_some()
}

fn decompress(c: Comp, data: &[u8]) -> Result<Vec<u8>, String> {
    match c {
        Comp::Gz => crate::inflate::gunzip(data),
        Comp::Zlib => crate::inflate::zlib_decompress(data),
        Comp::Xz => crate::lzma::xz_decode(data),
        Comp::Zstd => crate::zstd::decode(data),
        other => Err(format!("unsupported compression format: {:?}", other)),
    }
}

struct ArchiveEntry {
    name: String,
    is_dir: bool,
    data: Result<Vec<u8>, String>,
}

fn parse_octal(b: &[u8]) -> u64 {
    // GNU base-256 encoding
    if !b.is_empty() && b[0] & 0x80 != 0 {
        let mut v: u64 = (b[0] & 0x7f) as u64;
        for &x in &b[1..] {
            v = (v << 8) | x as u64;
        }
        return v;
    }
    let s: String = b.iter().take_while(|&&c| c != 0).map(|&c| c as char).collect();
    u64::from_str_radix(s.trim(), 8).unwrap_or(0)
}

fn cstr(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

fn tar_entries(data: &[u8]) -> Result<Vec<ArchiveEntry>, String> {
    let mut out = vec![];
    let mut pos = 0;
    let mut long_name: Option<String> = None;
    let mut pax_path: Option<String> = None;
    while pos + 512 <= data.len() {
        let h = &data[pos..pos + 512];
        if h.iter().all(|&b| b == 0) {
            break;
        }
        let mut name = cstr(&h[0..100]);
        let size = parse_octal(&h[124..136]) as usize;
        let typ = h[156];
        let magic = &h[257..263];
        if magic == b"ustar\0" || magic.starts_with(b"ustar") {
            let prefix = cstr(&h[345..500]);
            if !prefix.is_empty() && magic == b"ustar\0" {
                name = format!("{}/{}", prefix, name);
            }
        }
        let body_start = pos + 512;
        let body_end = (body_start + size).min(data.len());
        let body = &data[body_start..body_end];
        pos = body_start + size.div_ceil(512) * 512;
        match typ {
            b'L' => {
                long_name = Some(cstr(body));
                continue;
            }
            b'x' => {
                // pax extended header
                let text = String::from_utf8_lossy(body).into_owned();
                for rec in text.split('\n') {
                    if let Some(sp) = rec.find(' ') {
                        let kv = &rec[sp + 1..];
                        if let Some(p) = kv.strip_prefix("path=") {
                            pax_path = Some(p.to_string());
                        }
                    }
                }
                continue;
            }
            b'g' | b'K' => continue,
            _ => {}
        }
        if let Some(l) = long_name.take() {
            name = l;
        }
        if let Some(p) = pax_path.take() {
            name = p;
        }
        let is_dir = typ == b'5' || name.ends_with('/');
        if typ == b'0' || typ == 0 || typ == b'7' || is_dir {
            out.push(ArchiveEntry { name, is_dir, data: Ok(body.to_vec()) });
        }
    }
    Ok(out)
}

fn zip_entries(data: &[u8]) -> Result<Vec<ArchiveEntry>, String> {
    let rd16 = |p: usize| -> u16 { u16::from_le_bytes([data[p], data[p + 1]]) };
    let rd32 = |p: usize| -> u32 { u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]) };
    if data.len() < 22 {
        return Err("zip: not a valid zip file".into());
    }
    let mut eocd = None;
    let mut i = data.len() - 22;
    loop {
        if rd32(i) == 0x06054b50 {
            eocd = Some(i);
            break;
        }
        if i == 0 || data.len() - i > 65535 + 22 {
            break;
        }
        i -= 1;
    }
    let eocd = eocd.ok_or("zip: not a valid zip file")?;
    let count = rd16(eocd + 10) as usize;
    let mut cd = rd32(eocd + 16) as usize;
    let mut out = vec![];
    for _ in 0..count {
        if cd + 46 > data.len() || rd32(cd) != 0x02014b50 {
            return Err("zip: not a valid zip file".into());
        }
        let method = rd16(cd + 10);
        let csize = rd32(cd + 20) as usize;
        let nlen = rd16(cd + 28) as usize;
        let elen = rd16(cd + 30) as usize;
        let clen = rd16(cd + 32) as usize;
        let local = rd32(cd + 42) as usize;
        let name = String::from_utf8_lossy(&data[cd + 46..(cd + 46 + nlen).min(data.len())]).into_owned();
        cd += 46 + nlen + elen + clen;
        let is_dir = name.ends_with('/');
        let entry_data = (|| -> Result<Vec<u8>, String> {
            if local + 30 > data.len() || rd32(local) != 0x04034b50 {
                return Err("zip: not a valid zip file".into());
            }
            let ln = rd16(local + 26) as usize;
            let le = rd16(local + 28) as usize;
            let start = local + 30 + ln + le;
            let end = (start + csize).min(data.len());
            let raw = &data[start.min(end)..end];
            match method {
                0 => Ok(raw.to_vec()),
                8 => crate::inflate::inflate(raw).map(|(o, _)| o),
                _ => Err("zip: unsupported compression algorithm".into()),
            }
        })();
        out.push(ArchiveEntry { name, is_dir, data: entry_data });
    }
    Ok(out)
}

// ---------------------------------------------------------------- File

pub struct File<'c> {
    pub content: Vec<u8>,
    pub path: String,
    pub symlink: String,
    pub config: Option<&'c Config>,
    pub outer_paths: Vec<String>,
    pub max_archive_depth: i64,
    pub archive_depth: i64,
}

/// Emulates a Go bufio.Reader (4096 byte buffer) over an in-memory io.Reader.
struct BufReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: Vec<u8>,
    r: usize,
}

impl<'a> BufReader<'a> {
    fn read(&mut self, n: usize) -> Vec<u8> {
        if self.r < self.buf.len() {
            let k = n.min(self.buf.len() - self.r);
            let out = self.buf[self.r..self.r + k].to_vec();
            self.r += k;
            return out;
        }
        if n >= 4096 {
            let k = n.min(self.data.len() - self.pos);
            let out = self.data[self.pos..self.pos + k].to_vec();
            self.pos += k;
            return out;
        }
        self.fill();
        let k = n.min(self.buf.len() - self.r);
        let out = self.buf[self.r..self.r + k].to_vec();
        self.r += k;
        out
    }
    fn fill(&mut self) {
        let k = 4096.min(self.data.len() - self.pos);
        self.buf = self.data[self.pos..self.pos + k].to_vec();
        self.pos += k;
        self.r = 0;
    }
    fn read_byte(&mut self) -> Option<u8> {
        if self.r >= self.buf.len() {
            self.fill();
            if self.buf.is_empty() {
                return None;
            }
        }
        let b = self.buf[self.r];
        self.r += 1;
        Some(b)
    }
}

fn is_ws(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == b'\n' || b == b'\r'
}

fn read_until_safe_boundary(r: &mut BufReader, n: usize, peek: &mut Vec<u8>) {
    if peek.is_empty() {
        return;
    }
    let mut newline_count = 0;
    let last = peek[peek.len() - 1];
    if is_ws(last) {
        for i in (0..peek.len()).rev() {
            let c = peek[i];
            if c == b'\n' {
                newline_count += 1;
                if newline_count >= 2 {
                    return;
                }
            } else if is_ws(c) {
            } else {
                break;
            }
        }
    }
    newline_count = 0;
    loop {
        let last = peek[peek.len() - 1];
        if last == b'\n' {
            newline_count += 1;
            if newline_count >= 2 {
                break;
            }
        } else if is_ws(last) {
        } else {
            newline_count = 0;
        }
        if peek.len() - n >= MAX_PEEK_SIZE {
            break;
        }
        match r.read_byte() {
            Some(b) => peek.push(b),
            None => break,
        }
    }
}

/// h2non/filetype: is the buffer an "application" MIME type?
fn is_application_type(b: &[u8]) -> bool {
    let pre = |m: &[u8]| b.len() >= m.len() && &b[..m.len()] == m;
    // fonts
    if pre(b"wOFF") || pre(b"wOF2") || pre(&[0x00, 0x01, 0x00, 0x00, 0x00]) || pre(b"OTTO") {
        return true;
    }
    // documents (OLE)
    if pre(&[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) {
        return true;
    }
    // archives
    if b.len() > 3 && b[0] == 0x50 && b[1] == 0x4B && matches!(b[2], 3 | 5 | 7) && matches!(b[3], 4 | 6 | 8) {
        return true;
    }
    if b.len() > 261 && &b[257..262] == b"ustar" {
        return true;
    }
    if b.len() > 6 && &b[0..6] == b"Rar!\x1a\x07" && (b[6] == 0 || b[6] == 1) {
        return true;
    }
    let magics: [&[u8]; 14] = [
        &[0x1F, 0x8B, 0x08],
        b"BZh",
        &[0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C],
        &[0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00],
        &[0x28, 0xB5, 0x2F, 0xFD],
        b"%PDF",
        b"MZ",
        b"{\\rtf",
        b"%!",
        b"SQLi",
        b"NES\x1a",
        b"Cr24",
        b"!<arch>",
        b"LZIP",
    ];
    if magics.iter().any(|m| pre(m)) {
        return true;
    }
    if b.len() > 2 && (b[0] == 0x43 || b[0] == 0x46) && b[1] == 0x57 && b[2] == 0x53 {
        return true;
    }
    if pre(b"MSCF") || pre(b"ISc(") {
        return true;
    }
    if b.len() > 35
        && b[34] == 0x4C
        && b[35] == 0x50
        && ((b[8] == 2 && b[9] == 0 && b[10] == 1) || (b[8] == 1 && b[9] == 0 && b[10] == 0) || (b[8] == 2 && b[9] == 0 && b[10] == 2))
    {
        return true;
    }
    if b.len() > 1 && b[0] == 0x1F && (b[1] == 0xA0 || b[1] == 0x9D) {
        return true;
    }
    if b.len() > 96 && pre(&[0xED, 0xAB, 0xEE, 0xDB]) {
        return true;
    }
    if b.len() > 52 && pre(&[0x7F, 0x45, 0x4C, 0x46]) {
        return true;
    }
    if b.len() > 131 && &b[128..132] == b"DICM" {
        return true;
    }
    if b.len() > 32773 && &b[32769..32774] == b"CD001" {
        return true;
    }
    if b.len() > 3 {
        let m = [b[0], b[1], b[2], b[3]];
        if matches!(
            m,
            [0xFE, 0xED, 0xFA, 0xCF] | [0xFE, 0xED, 0xFA, 0xCE] | [0xBE, 0xBA, 0xFE, 0xCA] | [0xCF, 0xFA, 0xED, 0xFE] | [0xCE, 0xFA, 0xED, 0xFE] | [0xCA, 0xFE, 0xBA, 0xBE]
        ) {
            return true;
        }
    }
    // application
    if pre(&[0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00]) || pre(&[0x64, 0x65, 0x78, 0x0A]) || pre(&[0x64, 0x65, 0x79, 0x0A]) {
        return true;
    }
    false
}

impl<'c> File<'c> {
    pub fn full_path(&self) -> String {
        if !self.outer_paths.is_empty() {
            let mut v = self.outer_paths.clone();
            v.push(self.path.clone());
            return v.join(INNER_PATH_SEPARATOR);
        }
        self.path.clone()
    }

    pub fn fragments(&self, y: Yield) {
        if let Some(format) = identify(&self.path) {
            if self.archive_depth + 1 > self.max_archive_depth {
                let ev = if self.max_archive_depth != 0 { logging::warn() } else { logging::trace() };
                ev.str("path", &self.full_path())
                    .int("max_archive_depth", self.max_archive_depth)
                    .msg("skipping archive: exceeds max archive depth");
                return;
            }
            if let Some(arch) = format.arch {
                // extractor (possibly compressed archive)
                let data = match format.comp {
                    Some(c) => match decompress(c, &self.content) {
                        Ok(d) => d,
                        Err(e) => {
                            logging::error().str("path", &self.full_path()).err(&e).msg("could not extract archive");
                            return;
                        }
                    },
                    None => self.content.clone(),
                };
                let entries = match arch {
                    Arch::Tar => tar_entries(&data),
                    Arch::Zip => zip_entries(&data),
                    Arch::SevenZip => crate::lzma::sevenzip_entries(&data).map(|v| {
                        v.into_iter().map(|(name, is_dir, d)| ArchiveEntry { name, is_dir, data: Ok(d) }).collect()
                    }),
                    other => Err(format!("unsupported archive format: {:?}", other)),
                };
                let entries = match entries {
                    Ok(e) => e,
                    Err(e) => {
                        y(Fragment { file_path: self.full_path(), ..Default::default() }, Some(e));
                        return;
                    }
                };
                for e in entries {
                    if e.is_dir {
                        continue;
                    }
                    let content = match e.data {
                        Ok(d) => d,
                        Err(err) => {
                            logging::error().err(&err).str("path", &self.full_path()).msg("could not open archive inner file");
                            continue;
                        }
                    };
                    let path = util::clean(&e.name);
                    if let Some(cfg) = self.config {
                        if should_skip_path(cfg, &path) {
                            logging::debug().str("path", &self.full_path()).msg("skipping file: global allowlist");
                            continue;
                        }
                    }
                    let mut outer = self.outer_paths.clone();
                    outer.push(self.path.clone());
                    let f = File {
                        content,
                        path,
                        symlink: self.symlink.clone(),
                        config: None,
                        outer_paths: outer,
                        max_archive_depth: self.max_archive_depth,
                        archive_depth: self.archive_depth + 1,
                    };
                    f.fragments(y);
                }
                return;
            }
            if let Some(c) = format.comp {
                match decompress(c, &self.content) {
                    Ok(d) => self.file_fragments(&d, y),
                    Err(_) => {
                        logging::error().str("path", &self.full_path()).msg("could read compressed file");
                    }
                }
                return;
            }
        }
        self.file_fragments(&self.content, y)
    }

    fn file_fragments(&self, data: &[u8], y: Yield) {
        let mut reader = BufReader { data, pos: 0, buf: vec![], r: 0 };
        let mut total_lines: i64 = 0;
        loop {
            let chunk = reader.read(DEFAULT_BUFFER_SIZE);
            let n = chunk.len();
            if n == 0 {
                return;
            }
            if total_lines == 0 && is_application_type(&chunk) {
                logging::debug().str("path", &self.full_path()).msg("skipping binary file");
                return;
            }
            let mut peek = chunk;
            read_until_safe_boundary(&mut reader, n, &mut peek);
            let raw = String::from_utf8_lossy(&peek).into_owned();
            let frag = Fragment {
                file_path: self.full_path(),
                raw,
                bytes_len: Some(peek.len()),
                start_line: total_lines + 1,
                symlink_file: self.symlink.clone(),
                ..Default::default()
            };
            total_lines += peek.iter().filter(|&&b| b == b'\n').count() as i64;
            y(frag, None);
        }
    }
}

pub fn should_skip_path(cfg: &Config, path: &str) -> bool {
    cfg.allowlists.iter().any(|a| a.path_allowed(path))
}

// ---------------------------------------------------------------- Files

pub struct Files<'c> {
    pub config: &'c Config,
    pub follow_symlinks: bool,
    pub max_file_size: i64,
    pub path: String,
    pub max_archive_depth: i64,
}

struct ScanTarget {
    path: String,
    symlink: String,
}

impl<'c> Files<'c> {
    fn visit(&self, path: &str, is_root: bool, targets: &mut dyn FnMut(ScanTarget)) {
        let md = match std::fs::symlink_metadata(path) {
            Ok(m) => m,
            Err(e) => {
                if is_root {
                    logging::warn().str("path", path).err(&format!("lstat {}: {}", path, crate::io_err_str(&e))).msg("skipping");
                } else {
                    logging::error().str("path", path).err(&crate::io_err_str(&e)).msg("skipping file: could not get info");
                }
                return;
            }
        };
        let ft = md.file_type();
        let mut target = ScanTarget { path: path.to_string(), symlink: String::new() };
        if !ft.is_dir() {
            if md.len() == 0 {
                logging::debug().str("path", path).msg("skipping empty file");
                return;
            }
            if self.max_file_size > 0 && md.len() as i64 > self.max_file_size {
                logging::warn().str("path", path).msg(&format!(
                    "skipping file: too large max_size={}MB, size={}MB",
                    self.max_file_size / 1_000_000,
                    md.len() / 1_000_000
                ));
                return;
            }
        }
        if ft.is_symlink() {
            if !self.follow_symlinks {
                logging::debug().str("path", path).msg("skipping symlink: follow symlinks disabled");
                return;
            }
            let real = match std::fs::canonicalize(path) {
                Ok(p) => p.to_string_lossy().into_owned(),
                Err(e) => {
                    logging::error().str("path", path).err(&crate::io_err_str(&e)).msg("skipping symlink: could not evaluate");
                    return;
                }
            };
            // filepath.EvalSymlinks returns a relative path when given a relative one
            let real = eval_symlinks(path).unwrap_or(real);
            if std::fs::metadata(&real).map(|m| m.is_dir()).unwrap_or(false) {
                logging::debug().str("path", path).str("target", &real).msg("skipping symlink: target is directory");
                return;
            }
            target = ScanTarget { path: real, symlink: path.to_string() };
        }
        if ft.is_dir() {
            if should_skip_path(self.config, path) {
                logging::debug().str("path", path).msg("skipping directory: global allowlist");
                return;
            }
            let mut entries: Vec<String> = match std::fs::read_dir(path) {
                Ok(rd) => rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect(),
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::PermissionDenied {
                        logging::warn().str("path", path).err("permission denied").msg("skipping directory");
                    } else {
                        logging::warn().str("path", path).err(&crate::io_err_str(&e)).msg("skipping");
                    }
                    return;
                }
            };
            entries.sort();
            for name in entries {
                let child = util::join(path, &name);
                self.visit(&child, false, targets);
            }
            return;
        }
        if should_skip_path(self.config, path) {
            logging::debug().str("path", path).msg("skipping file: global allowlist");
            return;
        }
        targets(target);
    }

    pub fn fragments(&self, y: Yield) {
        let mut targets: Vec<ScanTarget> = vec![];
        self.visit(&self.path, true, &mut |t| targets.push(t));
        for t in targets {
            logging::trace().str("path", &t.path).msg("scanning path");
            let content = match std::fs::read(&t.path) {
                Ok(c) => c,
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::PermissionDenied {
                        logging::warn().str("path", &t.path).msg("skipping file: permission denied");
                    }
                    continue;
                }
            };
            let f = File {
                content,
                path: t.path.clone(),
                symlink: t.symlink.clone(),
                config: Some(self.config),
                outer_paths: vec![],
                max_archive_depth: self.max_archive_depth,
                archive_depth: 0,
            };
            f.fragments(y);
        }
    }
}

/// filepath.EvalSymlinks (lexical-preserving for relative inputs).
fn eval_symlinks(path: &str) -> Option<String> {
    let mut current = path.to_string();
    for _ in 0..255 {
        let md = std::fs::symlink_metadata(&current).ok()?;
        if !md.file_type().is_symlink() {
            return Some(util::clean(&current));
        }
        let target = std::fs::read_link(&current).ok()?.to_string_lossy().into_owned();
        if target.starts_with('/') {
            current = target;
        } else {
            let dir = match current.rfind('/') {
                Some(i) => current[..i].to_string(),
                None => String::new(),
            };
            current = if dir.is_empty() { target } else { util::join(&dir, &target) };
        }
    }
    None
}

// ---------------------------------------------------------------- Git

pub struct GitCmd {
    pub args: Vec<String>,
    pub repo_path: String,
}

pub fn new_git_log_cmd(source: &str, log_opts: &str) -> GitCmd {
    let source_clean = util::clean(source);
    let mut args: Vec<String> = vec!["-C".into(), source_clean.clone(), "log".into(), "-p".into(), "-U0".into()];
    if !log_opts.is_empty() {
        let user_args: Vec<&str> = log_opts.split(' ').collect();
        let quoted: Vec<&str> = user_args
            .iter()
            .filter(|e| {
                let b = e.as_bytes();
                b.len() >= 3
                    && ((b[0] == b'"' && b[b.len() - 1] == b'"' && !e[1..e.len() - 1].contains('"'))
                        || (b[0] == b'\'' && b[b.len() - 1] == b'\'' && !e[1..e.len() - 1].contains('\'')))
            })
            .cloned()
            .collect();
        if !quoted.is_empty() {
            logging::warn().msg(&format!(
                "the following `--log-opts` values may not work as expected: [{}]\n\tsee https://github.com/gitleaks/gitleaks/issues/1153 for more information",
                quoted.join(" ")
            ));
        }
        args.extend(user_args.iter().map(|s| s.to_string()));
    } else {
        args.extend(["--full-history", "--all", "--diff-filter=tuxdb"].iter().map(|s| s.to_string()));
    }
    log_executing(&args);
    GitCmd { args, repo_path: source_clean }
}

fn log_executing(args: &[String]) {
    if !logging::enabled(logging::Level::Debug) {
        return;
    }
    let mut git = "git".to_string();
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            let cand = util::join(if dir.is_empty() { "." } else { dir }, "git");
            if std::fs::metadata(&cand).map(|m| m.is_file()).unwrap_or(false) {
                git = cand;
                break;
            }
        }
    }
    logging::debug().msg(&format!("executing: {} {}", git, args.join(" ")));
}

pub fn new_git_diff_cmd(source: &str, staged: bool) -> GitCmd {
    let source_clean = util::clean(source);
    let mut args: Vec<String> = vec!["-C".into(), source_clean.clone(), "diff".into(), "-U0".into(), "--no-ext-diff".into()];
    if staged {
        args.push("--staged".into());
    }
    args.push(".".into());
    log_executing(&args);
    GitCmd { args, repo_path: source_clean }
}

fn is_known_git_warning(line: &str) -> bool {
    line.contains("exhaustive rename detection was skipped")
        || line.contains("inexact rename detection was skipped")
        || line.contains("you may want to set your diff.renameLimit")
        || line.contains("See \"git help gc\" for manual housekeeping")
        || line.contains("Auto packing the repository in background for optimum performance")
}

pub struct Git<'c> {
    pub cmd: GitCmd,
    pub config: &'c Config,
    pub remote: Arc<RemoteInfo>,
    pub max_archive_depth: i64,
}

impl<'c> Git<'c> {
    pub fn fragments(&self, y: Yield) -> Result<(), String> {
        let output = std::process::Command::new("git")
            .args(&self.cmd.args)
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|e| format!("exec: \"git\": {}", e))?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let files = gitdiff::parse(&stdout);
        for file in files {
            if file.is_delete {
                continue;
            }
            let mut yield_as_archive = false;
            if file.is_binary {
                if !is_archive(&file.new_name) {
                    continue;
                }
                yield_as_archive = true;
            }
            let mut commit_sha = String::new();
            let mut commit_info: Option<Arc<CommitInfo>> = None;
            if let Some(ph) = &file.header {
                commit_sha = ph.sha.clone();
                commit_info = Some(Arc::new(CommitInfo {
                    date: ph.date_utc.clone(),
                    message: ph.message(),
                    remote: self.remote.clone(),
                    sha: commit_sha.clone(),
                    author_name: ph.author.as_ref().map(|a| a.0.clone()).unwrap_or_default(),
                    author_email: ph.author.as_ref().map(|a| a.1.clone()).unwrap_or_default(),
                }));
            }
            if yield_as_archive {
                let blob = std::process::Command::new("git")
                    .args(["-C", &self.cmd.repo_path, "cat-file", "blob", &format!("{}:{}", commit_sha, file.new_name)])
                    .stdin(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .output();
                let blob = match blob {
                    Ok(o) => o.stdout,
                    Err(e) => {
                        logging::error().err(&e.to_string()).msg("could not read archive blob");
                        continue;
                    }
                };
                let f = File {
                    content: blob,
                    path: file.new_name.clone(),
                    symlink: String::new(),
                    config: Some(self.config),
                    outer_paths: vec![],
                    max_archive_depth: self.max_archive_depth,
                    archive_depth: 0,
                };
                let sha = commit_sha.clone();
                let ci = commit_info.clone();
                f.fragments(&mut |mut frag: Fragment, err: Option<String>| {
                    frag.commit_sha = sha.clone();
                    frag.commit_info = ci.clone();
                    y(frag, err);
                });
                continue;
            }
            for tf in &file.fragments {
                let frag = Fragment {
                    commit_sha: commit_sha.clone(),
                    file_path: file.new_name.clone(),
                    raw: tf.raw_added.clone(),
                    start_line: tf.new_position,
                    commit_info: commit_info.clone(),
                    ..Default::default()
                };
                y(frag, None);
            }
        }
        // stderr handling
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let mut err_encountered = false;
        for line in stderr.lines() {
            if is_known_git_warning(line) {
                logging::warn().msg(line);
            } else {
                logging::error().msg(&format!("[git] {}", line));
                err_encountered = true;
            }
        }
        if err_encountered {
            y(Fragment::default(), Some("stderr is not empty".into()));
        }
        Ok(())
    }
}

pub fn new_remote_info(platform: Platform, source: &str) -> RemoteInfo {
    if platform == Platform::None {
        return RemoteInfo { platform, url: String::new() };
    }
    let mut platform = platform;
    let remote = get_remote_url(source);
    let mut url = String::new();
    match remote {
        Err(e) => {
            if e.contains("No remote configured") {
                logging::debug().msg("skipping finding links: repository has no configured remote.");
                platform = Platform::None;
            } else {
                logging::error().err(&e).msg("skipping finding links: unable to parse remote URL");
            }
        }
        Ok((u, host)) => {
            if platform == Platform::Unknown {
                platform = platform_from_host(&host);
                if platform == Platform::Unknown {
                    logging::info()
                        .str("host", &host)
                        .msg("Unknown SCM platform. Use --platform to include links in findings.");
                } else {
                    logging::debug().str("host", &host).str("platform", platform.as_str()).msg("SCM platform parsed from host");
                }
            }
            url = u;
        }
    }
    RemoteInfo { platform, url }
}

fn get_remote_url(source: &str) -> Result<(String, String), String> {
    let mut cmd = std::process::Command::new("git");
    cmd.args(["ls-remote", "--quiet", "--get-url"]);
    if source != "." {
        cmd.current_dir(source);
    }
    let out = cmd.stdin(std::process::Stdio::null()).output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        let code = out.status.code().unwrap_or(-1);
        return Err(format!(
            "command failed ({}): exit status {}, stderr: {}",
            code,
            code,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let mut remote = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let ssh = crate::goregex::must_compile(r"^git@([a-zA-Z0-9.-]+):(?:\d{1,5}/)?([\w/.-]+?)(?:\.git)?$");
    if let Some(m) = ssh.find_submatch(&remote) {
        remote = format!("https://{}/{}", m[1], m[2]);
    }
    if let Some(r) = remote.strip_suffix(".git") {
        remote = r.to_string();
    }
    // minimal url.Parse: strip user info, extract host
    let (scheme, rest) = match remote.find("://") {
        Some(i) => (remote[..i + 3].to_string(), remote[i + 3..].to_string()),
        None => (String::new(), remote.clone()),
    };
    if scheme.is_empty() {
        return Ok((remote.clone(), String::new()));
    }
    let (authority, path) = match rest.find('/') {
        Some(i) => (rest[..i].to_string(), rest[i..].to_string()),
        None => (rest.clone(), String::new()),
    };
    let host_port = match authority.rfind('@') {
        Some(i) => authority[i + 1..].to_string(),
        None => authority.clone(),
    };
    let host = if host_port.starts_with('[') {
        host_port.trim_start_matches('[').split(']').next().unwrap_or("").to_string()
    } else {
        host_port.split(':').next().unwrap_or("").to_string()
    };
    Ok((format!("{}{}{}", scheme, host_port, path), host))
}

fn platform_from_host(host: &str) -> Platform {
    match host.to_lowercase().as_str() {
        "github.com" => Platform::GitHub,
        "gitlab.com" => Platform::GitLab,
        "dev.azure.com" | "visualstudio.com" => Platform::AzureDevOps,
        "gitea.com" | "code.forgejo.org" | "codeberg.org" => Platform::Gitea,
        "bitbucket.org" => Platform::Bitbucket,
        _ => Platform::Unknown,
    }
}
