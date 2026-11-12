//! Port of gitleaks' sources package.

pub mod filetype;
pub mod git;
pub mod gitdiff;

use std::io::{self, Read};
use std::rc::Rc;

use crate::cmd::scm::Platform;
use crate::config::{go_errno_text, Config};
use crate::logging;

pub const INNER_PATH_SEPARATOR: &str = "!";
const DEFAULT_BUFFER_SIZE: usize = 100 * 1_000;
const MAX_PEEK_SIZE: usize = 25 * 1_000;

/// RemoteInfo provides the info needed for reconstructing links from findings
#[derive(Clone, Debug)]
pub struct RemoteInfo {
    pub platform: Platform,
    pub url: String,
}

/// CommitInfo captures metadata about the commit
#[derive(Clone, Debug)]
pub struct CommitInfo {
    pub author_email: String,
    pub author_name: String,
    pub date: String,
    pub message: String,
    pub remote: Rc<RemoteInfo>,
    pub sha: String,
}

/// Fragment represents a fragment of a source with its meta data
#[derive(Clone, Debug, Default)]
pub struct Fragment {
    pub raw: String,
    /// length of the original bytes, if the fragment came from a byte buffer
    pub bytes_len: Option<usize>,
    pub file_path: String,
    pub symlink_file: String,
    pub windows_file_path: String,
    pub commit_sha: String,
    pub start_line: i64,
    pub commit_info: Option<Rc<CommitInfo>>,
    pub inherited_from_finding: bool,
}

pub type YieldFn<'a> = dyn FnMut(Fragment, Option<String>) -> Result<(), String> + 'a;

// ------------------------------------------------------------------ timeouts

static DEADLINE: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

pub fn set_deadline(d: std::time::Instant) {
    let _ = DEADLINE.set(d);
}

/// ctx.Done() equivalent
pub fn ctx_done() -> bool {
    match DEADLINE.get() {
        Some(d) => std::time::Instant::now() >= *d,
        None => false,
    }
}

pub const CTX_ERR: &str = "context deadline exceeded";

// ------------------------------------------------------------------ helpers

/// shouldSkipPath checks a path against all the allowlists to see if it can be skipped
pub fn should_skip_path(cfg: Option<&Config>, path: &str) -> bool {
    let cfg = match cfg {
        Some(c) => c,
        None => {
            logging::trace().str("path", path).msg("not skipping path because config is nil");
            return false;
        }
    };
    cfg.allowlists.iter().any(|a| a.path_allowed(path))
}

/// Go's filepath.Clean
pub fn clean_path(p: &str) -> String {
    if p.is_empty() {
        return ".".to_string();
    }
    let rooted = p.starts_with('/');
    let mut out: Vec<&str> = Vec::new();
    for comp in p.split('/') {
        match comp {
            "" | "." => {}
            ".." => {
                if let Some(last) = out.last() {
                    if *last != ".." {
                        out.pop();
                        continue;
                    }
                }
                if !rooted {
                    out.push("..");
                }
            }
            c => out.push(c),
        }
    }
    let joined = out.join("/");
    if rooted {
        format!("/{}", joined)
    } else if joined.is_empty() {
        ".".to_string()
    } else {
        joined
    }
}

/// Go's filepath.Join
pub fn join_path(a: &str, b: &str) -> String {
    if a.is_empty() {
        return clean_path(b);
    }
    if b.is_empty() {
        return clean_path(a);
    }
    clean_path(&format!("{}/{}", a, b))
}

pub fn base_name(p: &str) -> String {
    let p = p.trim_end_matches('/');
    if p.is_empty() {
        return "/".to_string();
    }
    match p.rfind('/') {
        Some(i) => p[i + 1..].to_string(),
        None => p.to_string(),
    }
}

/// filepath.Ext
pub fn ext(p: &str) -> String {
    let mut i = p.len();
    for (idx, c) in p.char_indices().rev() {
        if c == '/' {
            break;
        }
        if c == '.' {
            return p[idx..].to_string();
        }
        i = idx;
    }
    let _ = i;
    String::new()
}

/// A reader mirroring Go's bufio.Reader semantics (4096 byte buffer)
pub struct GoBufReader<'a> {
    rd: Box<dyn Read + 'a>,
    buf: Vec<u8>,
    r: usize,
    w: usize,
    err: Option<io::Error>,
    eof: bool,
}

impl<'a> GoBufReader<'a> {
    pub fn new(rd: Box<dyn Read + 'a>) -> Self {
        GoBufReader { rd, buf: vec![0; 4096], r: 0, w: 0, err: None, eof: false }
    }

    fn read_err(&mut self) -> Option<io::Error> {
        self.err.take()
    }

    fn fill(&mut self) {
        if self.r > 0 {
            self.buf.copy_within(self.r..self.w, 0);
            self.w -= self.r;
            self.r = 0;
        }
        for _ in 0..100 {
            match self.rd.read(&mut self.buf[self.w..]) {
                Ok(0) => {
                    self.eof = true;
                    return;
                }
                Ok(n) => {
                    self.w += n;
                    return;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    self.err = Some(e);
                    return;
                }
            }
        }
    }

    /// Returns (n, err) where err is None, Some(EOF) as Ok(false) etc.
    pub fn read(&mut self, p: &mut [u8]) -> (usize, Result<(), io::Error>, bool) {
        if p.is_empty() {
            return (0, Ok(()), false);
        }
        if self.r == self.w {
            if let Some(e) = self.read_err() {
                return (0, Err(e), false);
            }
            if self.eof {
                return (0, Ok(()), true);
            }
            if p.len() >= self.buf.len() {
                loop {
                    match self.rd.read(p) {
                        Ok(0) => {
                            self.eof = true;
                            return (0, Ok(()), true);
                        }
                        Ok(n) => return (n, Ok(()), false),
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                        Err(e) => return (0, Err(e), false),
                    }
                }
            }
            self.r = 0;
            self.w = 0;
            self.fill();
            if self.r == self.w {
                if let Some(e) = self.read_err() {
                    return (0, Err(e), false);
                }
                return (0, Ok(()), true);
            }
        }
        let n = (self.w - self.r).min(p.len());
        p[..n].copy_from_slice(&self.buf[self.r..self.r + n]);
        self.r += n;
        (n, Ok(()), false)
    }

    /// Ok(Some(b)) / Ok(None) on EOF / Err
    pub fn read_byte(&mut self) -> Result<Option<u8>, io::Error> {
        while self.r == self.w {
            if let Some(e) = self.read_err() {
                return Err(e);
            }
            if self.eof {
                return Ok(None);
            }
            self.fill();
        }
        let c = self.buf[self.r];
        self.r += 1;
        Ok(Some(c))
    }
}

fn is_whitespace(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == b'\n' || b == b'\r'
}

/// readUntilSafeBoundary consumes |r| until it finds two consecutive `\n` characters, up to |max_peek|.
fn read_until_safe_boundary(r: &mut GoBufReader, n: usize, max_peek: usize, peek: &mut Vec<u8>) -> Result<(), io::Error> {
    if peek.is_empty() {
        return Ok(());
    }
    let mut newline_count = 0;
    let last = peek[peek.len() - 1];
    if is_whitespace(last) {
        let mut i = peek.len() as i64 - 1;
        while i >= 0 {
            let c = peek[i as usize];
            if c == b'\n' {
                newline_count += 1;
                if newline_count >= 2 {
                    return Ok(());
                }
            } else if is_whitespace(c) {
            } else {
                break;
            }
            i -= 1;
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
        } else if is_whitespace(last) {
        } else {
            newline_count = 0;
        }
        if peek.len() - n >= max_peek {
            break;
        }
        match r.read_byte()? {
            Some(b) => peek.push(b),
            None => break,
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ archives

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    Gz,
    Bz2,
    Xz,
    Zstd,
    Brotli,
    Lz4,
    Sz,
    Zlib,
    Lzip,
    MinLz,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Archival {
    Tar,
    Zip,
    SevenZip,
    Rar,
}

#[derive(Clone, Copy, Debug)]
pub enum Format {
    Decompressor(Compression),
    Extractor(Option<Compression>, Archival),
}

/// archives.Identify with a nil stream: matches on file name only.
pub fn identify(filename: &str) -> Option<Format> {
    let name = base_name(filename).to_lowercase();
    let name = if name == "." { String::new() } else { name };
    let e = ext(&name);
    let comp = if name.contains(".gz") {
        Some(Compression::Gz)
    } else if name.contains(".bz2") {
        Some(Compression::Bz2)
    } else if name.contains(".xz") {
        Some(Compression::Xz)
    } else if name.contains(".zst") {
        Some(Compression::Zstd)
    } else if name.contains(".br") {
        Some(Compression::Brotli)
    } else if name.contains(".lz4") {
        Some(Compression::Lz4)
    } else if name.contains(".sz") || name.contains(".s2") {
        Some(Compression::Sz)
    } else if name.contains(".zz") {
        Some(Compression::Zlib)
    } else if e == ".lz" {
        Some(Compression::Lzip)
    } else if e == ".mz" {
        Some(Compression::MinLz)
    } else {
        None
    };
    let arch = if name.contains(".tar") {
        Some(Archival::Tar)
    } else if name.contains(".zip") {
        Some(Archival::Zip)
    } else if name.contains(".7z") {
        Some(Archival::SevenZip)
    } else if name.contains(".rar") {
        Some(Archival::Rar)
    } else {
        None
    };
    match (comp, arch) {
        (Some(c), None) => Some(Format::Decompressor(c)),
        (c, Some(a)) => Some(Format::Extractor(c, a)),
        (None, None) => None,
    }
}

pub fn is_archive(path: &str) -> bool {
    identify(path).is_some()
}

fn open_decompressor<'a>(c: Compression, r: Box<dyn Read + 'a>) -> Result<Box<dyn Read + 'a>, String> {
    Ok(match c {
        Compression::Gz => Box::new(flate2::read::MultiGzDecoder::new(r)),
        Compression::Zlib => Box::new(flate2::read::ZlibDecoder::new(r)),
        Compression::Bz2 => Box::new(bzip2::read::MultiBzDecoder::new(r)),
        Compression::Xz => {
            let mut input = io::BufReader::new(r);
            let mut out = Vec::new();
            lzma_rs::xz_decompress(&mut input, &mut out).map_err(|e| e.to_string())?;
            Box::new(io::Cursor::new(out))
        }
        Compression::Zstd => Box::new(ruzstd::StreamingDecoder::new(r).map_err(|e| e.to_string())?),
        Compression::Brotli => Box::new(brotli_decompressor::Decompressor::new(r, 4096)),
        Compression::Lz4 => Box::new(lz4_flex::frame::FrameDecoder::new(r)),
        Compression::Sz => Box::new(snap::read::FrameDecoder::new(r)),
        Compression::Lzip | Compression::MinLz => return Err("unsupported compression format".to_string()),
    })
}

/// An entry extracted from an archive
struct ArchiveEntry {
    name: String,
    data: Vec<u8>,
}

fn extract_entries<'a>(arch: Archival, mut r: Box<dyn Read + 'a>) -> Result<Vec<ArchiveEntry>, String> {
    let mut entries = Vec::new();
    match arch {
        Archival::Tar => {
            let mut a = tar::Archive::new(r);
            for e in a.entries().map_err(|e| e.to_string())? {
                let mut e = e.map_err(|e| e.to_string())?;
                let et = e.header().entry_type();
                if et.is_dir() || et == tar::EntryType::XGlobalHeader {
                    continue;
                }
                let name = String::from_utf8_lossy(&e.path_bytes()).into_owned();
                let mut data = Vec::new();
                if et.is_file() || et == tar::EntryType::Continuous {
                    e.read_to_end(&mut data).map_err(|e| e.to_string())?;
                }
                entries.push(ArchiveEntry { name, data });
            }
        }
        Archival::Zip => {
            let mut buf = Vec::new();
            r.read_to_end(&mut buf).map_err(|e| e.to_string())?;
            let mut z = zip::ZipArchive::new(io::Cursor::new(buf)).map_err(|e| e.to_string())?;
            for i in 0..z.len() {
                let mut f = z.by_index(i).map_err(|e| e.to_string())?;
                if f.is_dir() {
                    continue;
                }
                let name = f.name().to_string();
                let mut data = Vec::new();
                f.read_to_end(&mut data).map_err(|e| e.to_string())?;
                entries.push(ArchiveEntry { name, data });
            }
        }
        Archival::SevenZip => {
            let mut buf = Vec::new();
            r.read_to_end(&mut buf).map_err(|e| e.to_string())?;
            let len = buf.len() as u64;
            let mut sz = sevenz_rust::SevenZReader::new(io::Cursor::new(buf), len, sevenz_rust::Password::empty())
                .map_err(|e| e.to_string())?;
            sz.for_each_entries(|entry, reader| {
                if entry.is_directory() {
                    return Ok(true);
                }
                let mut data = Vec::new();
                reader.read_to_end(&mut data)?;
                entries.push(ArchiveEntry { name: entry.name().to_string(), data });
                Ok(true)
            })
            .map_err(|e| e.to_string())?;
        }
        Archival::Rar => return Err("rar archives are not supported".to_string()),
    }
    Ok(entries)
}

// ------------------------------------------------------------------ File source

/// File is a source for yielding fragments from a file or other reader
pub struct File<'a> {
    pub content: Box<dyn Read + 'a>,
    pub path: String,
    pub symlink: String,
    pub buffer_size: usize,
    pub config: Option<&'a Config>,
    pub outer_paths: Vec<String>,
    pub max_archive_depth: i64,
    pub archive_depth: i64,
}

impl<'a> File<'a> {
    pub fn new(content: Box<dyn Read + 'a>, path: &str) -> File<'a> {
        File {
            content,
            path: path.to_string(),
            symlink: String::new(),
            buffer_size: DEFAULT_BUFFER_SIZE,
            config: None,
            outer_paths: Vec::new(),
            max_archive_depth: 0,
            archive_depth: 0,
        }
    }

    /// FullPath returns the File.Path with any preceding outer paths
    pub fn full_path(&self) -> String {
        if !self.outer_paths.is_empty() {
            let mut v = self.outer_paths.clone();
            v.push(self.path.clone());
            return v.join(INNER_PATH_SEPARATOR);
        }
        self.path.clone()
    }

    pub fn fragments(self, yield_: &mut YieldFn) -> Result<(), String> {
        if let Some(format) = identify(&self.path) {
            if self.archive_depth + 1 > self.max_archive_depth {
                let ev = if self.max_archive_depth != 0 { logging::warn() } else { logging::trace() };
                ev.str("path", &self.full_path())
                    .int("max_archive_depth", self.max_archive_depth)
                    .msg("skipping archive: exceeds max archive depth");
                return Ok(());
            }
            return match format {
                Format::Extractor(comp, arch) => self.extractor_fragments(comp, arch, yield_),
                Format::Decompressor(comp) => self.decompressor_fragments(comp, yield_),
            };
        }
        let full = self.full_path();
        let symlink = self.symlink.clone();
        let size = self.buffer_size;
        let mut reader = GoBufReader::new(self.content);
        file_fragments(&mut reader, &full, &symlink, size, yield_)
    }

    fn extractor_fragments(self, comp: Option<Compression>, arch: Archival, yield_: &mut YieldFn) -> Result<(), String> {
        let full = self.full_path();
        let reader: Box<dyn Read + 'a> = match comp {
            Some(c) => match open_decompressor(c, self.content) {
                Ok(r) => r,
                Err(e) => {
                    logging::error().err(&e).str("path", &full).msg("could not open archive inner file");
                    return Ok(());
                }
            },
            None => self.content,
        };
        let entries = match extract_entries(arch, reader) {
            Ok(e) => e,
            Err(e) => return Err(e),
        };
        let mut outer = self.outer_paths.clone();
        outer.push(self.path.clone());
        for entry in entries {
            let path = clean_path(&entry.name);
            if self.config.is_some() && should_skip_path(self.config, &path) {
                logging::debug().str("path", &full).msg("skipping file: global allowlist");
                continue;
            }
            let file = File {
                content: Box::new(io::Cursor::new(entry.data)),
                path,
                symlink: self.symlink.clone(),
                buffer_size: DEFAULT_BUFFER_SIZE,
                config: None,
                outer_paths: outer.clone(),
                max_archive_depth: self.max_archive_depth,
                archive_depth: self.archive_depth + 1,
            };
            file.fragments(yield_)?;
        }
        Ok(())
    }

    fn decompressor_fragments(self, comp: Compression, yield_: &mut YieldFn) -> Result<(), String> {
        let full = self.full_path();
        let symlink = self.symlink.clone();
        let size = self.buffer_size;
        let inner = match open_decompressor(comp, self.content) {
            Ok(r) => r,
            Err(_) => {
                logging::error().str("path", &full).msg("could read compressed file");
                return Ok(());
            }
        };
        let mut reader = GoBufReader::new(inner);
        file_fragments(&mut reader, &full, &symlink, size, yield_)
    }
}

/// fileFragments reads the file into fragments to yield
fn file_fragments(reader: &mut GoBufReader, full_path: &str, symlink: &str, buf_size: usize, yield_: &mut YieldFn) -> Result<(), String> {
    let mut buffer = vec![0u8; buf_size];
    let mut total_lines: i64 = 0;
    loop {
        if ctx_done() {
            return Err(CTX_ERR.to_string());
        }
        let mut fragment = Fragment { file_path: full_path.to_string(), ..Default::default() };
        let (n, err, eof) = reader.read(&mut buffer);
        if n == 0 {
            if let Err(e) = err {
                return yield_(fragment, Some(format!("could not read file: {}", e)));
            }
            let _ = eof;
            return Ok(());
        }
        if total_lines == 0 && filetype::is_application(&buffer[..n]) {
            logging::debug()
                .str("mime_type", "application")
                .str("path", full_path)
                .msg("skipping binary file");
            return Ok(());
        }
        let mut peek = buffer[..n].to_vec();
        if let Err(e) = read_until_safe_boundary(reader, n, MAX_PEEK_SIZE, &mut peek) {
            return yield_(
                fragment,
                Some(format!("could not read file: could not read until safe boundary: {}", go_errno_text(&e))),
            );
        }
        fragment.raw = String::from_utf8_lossy(&peek).into_owned();
        fragment.bytes_len = Some(peek.len());
        fragment.start_line = total_lines + 1;
        total_lines += peek.iter().filter(|b| **b == b'\n').count() as i64;
        if !symlink.is_empty() {
            fragment.symlink_file = symlink.to_string();
        }
        yield_(fragment, None)?;
    }
}

// ------------------------------------------------------------------ Files source

/// ScanTarget is a path to scan (and the symlink that pointed to it)
pub struct ScanTarget {
    pub path: String,
    pub symlink: String,
}

/// Files is a source for yielding fragments from a collection of files
pub struct Files<'a> {
    pub config: Option<&'a Config>,
    pub follow_symlinks: bool,
    pub max_file_size: i64,
    pub path: String,
    pub max_archive_depth: i64,
}

enum WalkCtl {
    Continue,
    SkipDir,
}

impl<'a> Files<'a> {
    fn scan_targets(&self, yield_: &mut dyn FnMut(ScanTarget) -> Result<(), String>) -> Result<(), String> {
        let root = self.path.clone();
        match std::fs::symlink_metadata(&root) {
            Err(e) => {
                self.visit(&root, None, Some(e), yield_)?;
                Ok(())
            }
            Ok(md) => self.walk(&root, md, yield_),
        }
    }

    fn walk(&self, path: &str, md: std::fs::Metadata, yield_: &mut dyn FnMut(ScanTarget) -> Result<(), String>) -> Result<(), String> {
        let ctl = self.visit(path, Some(md.clone()), None, yield_)?;
        if !md.is_dir() {
            return Ok(());
        }
        if let WalkCtl::SkipDir = ctl {
            return Ok(());
        }
        let rd = match std::fs::read_dir(path) {
            Ok(rd) => rd,
            Err(e) => {
                self.visit(path, Some(md), Some(e), yield_)?;
                return Ok(());
            }
        };
        let mut names: Vec<String> = rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        for name in names {
            if ctx_done() {
                return Err(CTX_ERR.to_string());
            }
            let child = join_path(path, &name);
            match std::fs::symlink_metadata(&child) {
                Ok(cmd) => self.walk(&child, cmd, yield_)?,
                Err(e) => {
                    logging::with().str("path", &child).error().err(go_errno_text(&e)).msg("skipping file: could not get info");
                }
            }
        }
        Ok(())
    }

    fn visit(
        &self,
        path: &str,
        md: Option<std::fs::Metadata>,
        err: Option<io::Error>,
        yield_: &mut dyn FnMut(ScanTarget) -> Result<(), String>,
    ) -> Result<WalkCtl, String> {
        let logger = logging::with().str("path", path);
        if let Some(e) = err {
            if e.kind() == io::ErrorKind::PermissionDenied {
                logger.warn().err("permission denied").msg("skipping directory");
                return Ok(WalkCtl::SkipDir);
            }
            let op = if md.is_some() { "open" } else { "lstat" };
            logger.warn().err(crate::config::go_os_error(op, path, &e)).msg("skipping");
            return Ok(WalkCtl::Continue);
        }
        let md = md.unwrap();
        let is_dir = md.is_dir();
        let is_symlink = md.file_type().is_symlink();
        let mut target = ScanTarget { path: path.to_string(), symlink: String::new() };
        if !is_dir {
            if md.len() == 0 {
                logger.debug().msg("skipping empty file");
                return Ok(WalkCtl::Continue);
            }
            if self.max_file_size > 0 && md.len() as i64 > self.max_file_size {
                logger.warn().msg(&format!(
                    "skipping file: too large max_size={}MB, size={}MB",
                    self.max_file_size / 1_000_000,
                    md.len() / 1_000_000
                ));
                return Ok(WalkCtl::Continue);
            }
        }
        if is_symlink {
            if !self.follow_symlinks {
                logger.debug().msg("skipping symlink: follow symlinks disabled");
                return Ok(WalkCtl::Continue);
            }
            let real = match eval_symlinks(path) {
                Ok(r) => r,
                Err(e) => {
                    logger.error().err(e).msg("skipping symlink: could not evaluate");
                    return Ok(WalkCtl::Continue);
                }
            };
            if let Ok(rmd) = std::fs::metadata(&real) {
                if rmd.is_dir() {
                    logger.debug().str("target", &real).msg("skipping symlink: target is directory");
                    return Ok(WalkCtl::Continue);
                }
            }
            target = ScanTarget { path: real, symlink: path.to_string() };
        }
        if is_dir {
            if should_skip_path(self.config, path) {
                logger.debug().msg("skipping directory: global allowlist");
                return Ok(WalkCtl::SkipDir);
            }
            return Ok(WalkCtl::Continue);
        }
        if should_skip_path(self.config, path) {
            logger.debug().msg("skipping file: global allowlist");
            return Ok(WalkCtl::Continue);
        }
        yield_(target)?;
        Ok(WalkCtl::Continue)
    }

    pub fn fragments(&self, yield_: &mut YieldFn) -> Result<(), String> {
        let mut inner = |st: ScanTarget| -> Result<(), String> {
            if ctx_done() {
                return Err(CTX_ERR.to_string());
            }
            let logger = logging::with().str("path", &st.path);
            logger.trace().msg("scanning path");
            let f = match std::fs::File::open(&st.path) {
                Ok(f) => f,
                Err(e) => {
                    if e.kind() == io::ErrorKind::PermissionDenied {
                        logger.warn().msg("skipping file: permission denied");
                    }
                    return Ok(());
                }
            };
            let file = File {
                content: Box::new(f),
                path: st.path.clone(),
                symlink: st.symlink.clone(),
                buffer_size: DEFAULT_BUFFER_SIZE,
                config: self.config,
                outer_paths: Vec::new(),
                max_archive_depth: self.max_archive_depth,
                archive_depth: 0,
            };
            file.fragments(yield_)
        };
        self.scan_targets(&mut inner)
    }
}

/// Go's filepath.EvalSymlinks (simplified)
pub fn eval_symlinks(path: &str) -> Result<String, String> {
    let mut remaining: Vec<String> = path.split('/').map(|s| s.to_string()).collect();
    remaining.reverse();
    let mut resolved = if path.starts_with('/') { "/".to_string() } else { String::new() };
    let mut links = 0;
    while let Some(comp) = remaining.pop() {
        if comp.is_empty() || comp == "." {
            continue;
        }
        if comp == ".." {
            resolved = if resolved.is_empty() || resolved == "." {
                "..".to_string()
            } else if resolved.ends_with("..") {
                format!("{}/..", resolved)
            } else {
                let d = match resolved.rfind('/') {
                    Some(0) => "/".to_string(),
                    Some(i) => resolved[..i].to_string(),
                    None => String::new(),
                };
                d
            };
            continue;
        }
        let candidate = if resolved.is_empty() {
            comp.clone()
        } else if resolved.ends_with('/') {
            format!("{}{}", resolved, comp)
        } else {
            format!("{}/{}", resolved, comp)
        };
        let md = std::fs::symlink_metadata(&candidate).map_err(|e| crate::config::go_os_error("lstat", &candidate, &e))?;
        if md.file_type().is_symlink() {
            links += 1;
            if links > 255 {
                return Err("EvalSymlinks: too many links".to_string());
            }
            let target = std::fs::read_link(&candidate).map_err(|e| e.to_string())?;
            let target = target.to_string_lossy().into_owned();
            if target.starts_with('/') {
                resolved = "/".to_string();
            }
            let mut tc: Vec<String> = target.split('/').map(|s| s.to_string()).collect();
            tc.reverse();
            remaining.extend(tc);
        } else {
            resolved = candidate;
        }
    }
    Ok(clean_path(&resolved))
}
