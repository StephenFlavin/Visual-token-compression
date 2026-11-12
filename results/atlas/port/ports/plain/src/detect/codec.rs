//! Port of gitleaks' detect/codec package: recursive decoding of encoded
//! segments (percent, unicode, hex, base64).

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::OnceLock;

use crate::logging;
use crate::regexp::Regexp;

// ----------------------------------------------------------------- start/end

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StartEnd {
    pub start: i64,
    pub end: i64,
}

impl StartEnd {
    fn new(start: i64, end: i64) -> Self {
        StartEnd { start, end }
    }
    fn sub(self, o: StartEnd) -> StartEnd {
        StartEnd::new(self.start - o.start, self.end - o.end)
    }
    fn add(self, o: StartEnd) -> StartEnd {
        StartEnd::new(self.start + o.start, self.end + o.end)
    }
    fn overlaps(self, o: StartEnd) -> bool {
        o.start <= self.end && o.end >= self.start
    }
    fn contains(self, o: StartEnd) -> bool {
        self.start <= o.start && o.end <= self.end
    }
    fn overflow(self, o: StartEnd) -> StartEnd {
        self.merge(o).sub(self)
    }
    fn merge(self, o: StartEnd) -> StartEnd {
        StartEnd::new(self.start.min(o.start), self.end.max(o.end))
    }
}

impl std::fmt::Display for StartEnd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{},{}]", self.start, self.end)
    }
}

// ----------------------------------------------------------------- ascii helpers

fn printable(b: u8) -> bool {
    b > 0x08 && b < 0x7f
}

fn is_printable_ascii(b: &[u8]) -> bool {
    b.iter().all(|c| printable(*c))
}

fn has_byte(data: &str, set: &[u8]) -> bool {
    data.bytes().any(|b| set.contains(&b))
}

// ----------------------------------------------------------------- encodings

pub type EncodingKind = u32;
pub const PERCENT_KIND: EncodingKind = 1;
pub const UNICODE_KIND: EncodingKind = 2;
pub const HEX_KIND: EncodingKind = 4;
pub const BASE64_KIND: EncodingKind = 8;

const ENCODING_NAMES: [&str; 4] = ["percent", "unicode", "hex", "base64"];

fn kind_string(e: EncodingKind) -> String {
    if e == 0 {
        return String::new();
    }
    let i = (e as f64).log2() as usize;
    if i >= ENCODING_NAMES.len() {
        return String::new();
    }
    ENCODING_NAMES[i].to_string()
}

fn kinds(e: EncodingKind) -> Vec<EncodingKind> {
    let mut out = Vec::new();
    for i in 0..ENCODING_NAMES.len() {
        let k = e & (1 << i);
        if k != 0 {
            out.push(k);
        }
    }
    out
}

struct Encoding {
    kind: EncodingKind,
    pattern: &'static str,
    decode: fn(&str) -> String,
    precedence: i32,
}

fn encodings() -> &'static (Vec<Encoding>, Regexp) {
    static E: OnceLock<(Vec<Encoding>, Regexp)> = OnceLock::new();
    E.get_or_init(|| {
        let mut encs = vec![
            Encoding {
                kind: PERCENT_KIND,
                pattern: r"%[0-9A-Fa-f]{2}(?:.*%[0-9A-Fa-f]{2})?",
                decode: decode_percent,
                precedence: 0,
            },
            Encoding {
                kind: UNICODE_KIND,
                pattern: r"(?:(?:U\+[a-fA-F0-9]{4}(?:\s|$))+|(?i)(?:\\{1,2}u[a-fA-F0-9]{4})+)",
                decode: decode_unicode,
                precedence: 0,
            },
            Encoding { kind: HEX_KIND, pattern: r"[0-9A-Fa-f]{32,}", decode: decode_hex, precedence: 0 },
            Encoding { kind: BASE64_KIND, pattern: r"[\w\/+-]{16,}={0,2}", decode: decode_base64, precedence: 0 },
        ];
        let count = encs.len() as i32;
        let mut named = Vec::new();
        for (i, e) in encs.iter_mut().enumerate() {
            e.precedence = count - i as i32;
            named.push(format!("(?P<{}>{})", kind_string(e.kind), e.pattern));
        }
        let re = Regexp::must_compile(&named.join("|"));
        (encs, re)
    })
}

#[derive(Clone, Copy)]
struct EncodingMatch {
    enc: usize,
    se: StartEnd,
}

fn find_encoding_matches(data: &str) -> Vec<EncodingMatch> {
    let (encs, re) = encodings();
    let mut all = Vec::new();
    for mi in re.find_all_submatch_index(data) {
        let mut i = 2;
        let mut j = 0;
        while i < mi.len() {
            if mi[i] > -1 {
                all.push(EncodingMatch { enc: j, se: StartEnd::new(mi[i], mi[i + 1]) });
            }
            i += 2;
            j += 1;
        }
    }
    let total = all.len();
    if total == 1 {
        return all;
    }
    let mut filtered = Vec::with_capacity(total);
    for (i, m) in all.iter().enumerate() {
        if i > 0 {
            let prev = all[i - 1];
            if m.se.overlaps(prev.se) && encs[prev.enc].precedence > encs[m.enc].precedence {
                continue;
            }
        }
        if i + 1 < total {
            let next = all[i + 1];
            if m.se.overlaps(next.se) && encs[next.enc].precedence > encs[m.enc].precedence {
                continue;
            }
        }
        filtered.push(*m);
    }
    filtered
}

// ----------------------------------------------------------------- segments

#[derive(Debug)]
pub struct EncodedSegment {
    predecessors: Vec<Rc<EncodedSegment>>,
    original: StartEnd,
    encoded: StartEnd,
    decoded: StartEnd,
    decoded_value: String,
    encodings: EncodingKind,
    depth: i64,
}

/// Tags returns additional meta data tags related to the types of segments
pub fn tags(segments: &[Rc<EncodedSegment>]) -> Vec<String> {
    if segments.is_empty() {
        return Vec::new();
    }
    let depth = segments[0].depth;
    let mut enc = segments[0].encodings;
    for s in &segments[1..] {
        enc |= s.encodings;
    }
    let ks = kinds(enc);
    let mut out: Vec<String> = ks.iter().map(|k| format!("decoded:{}", kind_string(*k))).collect();
    out.push(format!("decode-depth:{}", depth));
    out
}

/// CurrentLine returns from the start of the line containing the segments
/// to the end of the line where the segment ends.
pub fn current_line(segments: &[Rc<EncodedSegment>], current_raw: &str) -> String {
    if segments.is_empty() {
        return current_raw.to_string();
    }
    let bytes = current_raw.as_bytes();
    let mut start: i64 = 0;
    let mut end: i64 = bytes.len() as i64;
    let mut decoded = segments[0].decoded;
    for s in &segments[1..] {
        decoded = decoded.merge(s.decoded);
    }
    let mut i = decoded.start;
    while i > -1 {
        if (i as usize) < bytes.len() && bytes[i as usize] == b'\n' {
            start = i;
            break;
        }
        i -= 1;
    }
    let mut i = decoded.end;
    while i < end {
        if bytes[i as usize] == b'\n' {
            end = i;
            break;
        }
        i += 1;
    }
    crate::gostr::bslice(current_raw, start as usize, end as usize)
}

/// AdjustMatchIndex maps a match index from the current decode pass back to
/// its location in the original text
pub fn adjust_match_index(segments: &[Rc<EncodedSegment>], m: (i64, i64)) -> (i64, i64) {
    if segments.is_empty() {
        return m;
    }
    let adjusted = to_original(segments, StartEnd::new(m.0, m.1));
    (adjusted.start, adjusted.end)
}

/// SegmentsWithDecodedOverlap the segments where the start and end overlap its decoded range
pub fn segments_with_decoded_overlap(segments: &[Rc<EncodedSegment>], start: i64, end: i64) -> Vec<Rc<EncodedSegment>> {
    let se = StartEnd::new(start, end);
    segments.iter().filter(|s| s.decoded.overlaps(se)).cloned().collect()
}

fn to_original(predecessors: &[Rc<EncodedSegment>], decoded: StartEnd) -> StartEnd {
    if predecessors.is_empty() {
        return decoded;
    }
    let mut encoded = StartEnd::default();
    for p in predecessors {
        if !p.decoded.overlaps(decoded) {
            continue;
        }
        if p.decoded.contains(decoded) {
            return p.original;
        }
        if encoded.end == 0 {
            encoded = p.encoded.add(p.decoded.overflow(decoded));
        } else {
            encoded = encoded.merge(p.encoded.add(p.decoded.overflow(decoded)));
        }
    }
    if encoded.end == 0 {
        return decoded;
    }
    to_original(&predecessors[0].predecessors, encoded)
}

// ----------------------------------------------------------------- decoder

pub struct Decoder {
    decoded_map: HashMap<String, String>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    pub fn new() -> Decoder {
        Decoder { decoded_map: HashMap::new() }
    }

    /// Decode returns the data with the values decoded in place along with the
    /// encoded segment meta data for the next pass of decoding
    pub fn decode(&mut self, data: &str, predecessors: &[Rc<EncodedSegment>]) -> (String, Vec<Rc<EncodedSegment>>) {
        let segments = self.find_encoded_segments(data, predecessors);
        if !segments.is_empty() {
            let bytes = data.as_bytes();
            let mut result: Vec<u8> = Vec::with_capacity(data.len());
            let mut encoded_start = 0usize;
            for s in &segments {
                result.extend_from_slice(&bytes[encoded_start..s.encoded.start as usize]);
                result.extend_from_slice(s.decoded_value.as_bytes());
                encoded_start = s.encoded.end as usize;
            }
            result.extend_from_slice(&bytes[encoded_start..]);
            let s = match String::from_utf8(result) {
                Ok(s) => s,
                Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
            };
            return (s, segments);
        }
        (data.to_string(), segments)
    }

    fn find_encoded_segments(&mut self, data: &str, predecessors: &[Rc<EncodedSegment>]) -> Vec<Rc<EncodedSegment>> {
        if data.is_empty() {
            return Vec::new();
        }
        let (encs, _) = encodings();
        let mut decoded_shift: i64 = 0;
        let matches = find_encoding_matches(data);
        let mut segments = Vec::with_capacity(matches.len());
        for m in matches {
            let enc = &encs[m.enc];
            let encoded_value = crate::gostr::bslice(data, m.se.start as usize, m.se.end as usize);
            let decoded_value = match self.decoded_map.get(&encoded_value) {
                Some(v) => v.clone(),
                None => {
                    let v = (enc.decode)(&encoded_value);
                    self.decoded_map.insert(encoded_value.clone(), v.clone());
                    v
                }
            };
            if decoded_value.is_empty() {
                continue;
            }
            let mut seg = EncodedSegment {
                predecessors: predecessors.to_vec(),
                original: to_original(predecessors, m.se),
                encoded: m.se,
                decoded: StartEnd::new(
                    m.se.start + decoded_shift,
                    m.se.start + decoded_shift + decoded_value.len() as i64,
                ),
                decoded_value: decoded_value.clone(),
                encodings: enc.kind,
                depth: 1,
            };
            decoded_shift += decoded_value.len() as i64 - encoded_value.len() as i64;
            if !seg.predecessors.is_empty() {
                seg.depth = 1 + seg.predecessors[0].depth;
                let mut add = 0;
                for p in &seg.predecessors {
                    if seg.encoded.overlaps(p.decoded) {
                        add |= p.encodings;
                    }
                }
                seg.encodings |= add;
            }
            let ev = logging::debug();
            if ev.is_enabled() {
                ev.str("decoder", &kind_string(enc.kind)).msg(&format!(
                    "segment found: original={} pos={}: {} -> {}",
                    seg.original,
                    seg.encoded,
                    crate::gostr::quote(&encoded_value),
                    crate::gostr::quote(&seg.decoded_value)
                ));
            }
            segments.push(Rc::new(seg));
        }
        segments
    }
}

// ----------------------------------------------------------------- decoders

fn hex_val(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => 0xff,
    }
}

/// decodePercent decodes percent encoded strings
fn decode_percent(encoded: &str) -> String {
    let e = encoded.as_bytes();
    let n = e.len();
    let mut out: Vec<u8> = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if e[i] == b'%' && i + 2 < n {
            let n1 = hex_val(e[i + 1]);
            let n2 = hex_val(e[i + 2]);
            if n1 | n2 != 0xff {
                let b = (n1 << 4) | n2;
                if !printable(b) {
                    return String::new();
                }
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(e[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// decodeHex decodes hex data
fn decode_hex(encoded: &str) -> String {
    let e = encoded.as_bytes();
    let size = e.len();
    if size % 2 != 0 {
        return String::new();
    }
    if !has_byte(encoded, b"0123456789") {
        return String::new();
    }
    let mut out = Vec::with_capacity(size / 2);
    let mut i = 0;
    while i < size {
        let n1 = hex_val(e[i]);
        let n2 = hex_val(e[i + 1]);
        if n1 | n2 == 0xff {
            return String::new();
        }
        let b = (n1 << 4) | n2;
        if !printable(b) {
            return String::new();
        }
        out.push(b);
        i += 2;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn b64_val(c: u8, url: bool) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some((c - b'A') as u32),
        b'a'..=b'z' => Some((c - b'a') as u32 + 26),
        b'0'..=b'9' => Some((c - b'0') as u32 + 52),
        b'+' if !url => Some(62),
        b'/' if !url => Some(63),
        b'-' if url => Some(62),
        b'_' if url => Some(63),
        _ => None,
    }
}

/// Go's base64.StdEncoding.DecodeString (padded, non-strict)
fn b64_std_decode(s: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut i = 0;
    let n = s.len();
    while i < n {
        let mut quantum = [0u32; 4];
        let mut j = 0;
        while j < 4 {
            if i >= n {
                if j == 0 {
                    return Some(out);
                }
                return None; // incomplete quantum without padding
            }
            let c = s[i];
            if c == b'\n' || c == b'\r' {
                i += 1;
                continue;
            }
            if c == b'=' {
                // padding
                match j {
                    0 | 1 => return None,
                    2 => {
                        // expect another '='
                        i += 1;
                        while i < n && (s[i] == b'\n' || s[i] == b'\r') {
                            i += 1;
                        }
                        if i >= n || s[i] != b'=' {
                            return None;
                        }
                        i += 1;
                    }
                    _ => {
                        i += 1;
                    }
                }
                while i < n && (s[i] == b'\n' || s[i] == b'\r') {
                    i += 1;
                }
                if i < n {
                    return None; // trailing garbage
                }
                let v = (quantum[0] << 18) | (quantum[1] << 12) | (quantum[2] << 6) | quantum[3];
                out.push((v >> 16) as u8);
                if j == 3 {
                    out.push((v >> 8) as u8);
                }
                return Some(out);
            }
            quantum[j] = b64_val(c, false)?;
            j += 1;
            i += 1;
        }
        let v = (quantum[0] << 18) | (quantum[1] << 12) | (quantum[2] << 6) | quantum[3];
        out.push((v >> 16) as u8);
        out.push((v >> 8) as u8);
        out.push(v as u8);
    }
    Some(out)
}

/// Go's base64.RawURLEncoding.DecodeString (no padding)
fn b64_rawurl_decode(s: &[u8]) -> Option<Vec<u8>> {
    let filtered: Vec<u8> = s.iter().copied().filter(|c| *c != b'\n' && *c != b'\r').collect();
    let mut out = Vec::with_capacity(filtered.len() * 3 / 4);
    for chunk in filtered.chunks(4) {
        let mut q = [0u32; 4];
        for (k, c) in chunk.iter().enumerate() {
            q[k] = b64_val(*c, true)?;
        }
        let v = (q[0] << 18) | (q[1] << 12) | (q[2] << 6) | q[3];
        match chunk.len() {
            4 => {
                out.push((v >> 16) as u8);
                out.push((v >> 8) as u8);
                out.push(v as u8);
            }
            3 => {
                out.push((v >> 16) as u8);
                out.push((v >> 8) as u8);
            }
            2 => {
                out.push((v >> 16) as u8);
            }
            _ => return None,
        }
    }
    Some(out)
}

/// decodeBase64 decodes base64 encoded printable ASCII characters
fn decode_base64(encoded: &str) -> String {
    if !has_byte(encoded, b"0123456789+/-_") {
        return String::new();
    }
    if let Some(d) = b64_std_decode(encoded.as_bytes()) {
        if is_printable_ascii(&d) {
            return String::from_utf8_lossy(&d).into_owned();
        }
    }
    if let Some(d) = b64_rawurl_decode(encoded.as_bytes()) {
        if is_printable_ascii(&d) {
            return String::from_utf8_lossy(&d).into_owned();
        }
    }
    String::new()
}

// ----------------------------------------------------------------- unicode

struct UnicodePats {
    code_point: Regexp,
    multi_code_point: Regexp,
    escape: Regexp,
    multi_escape: Regexp,
}

fn unicode_pats() -> &'static UnicodePats {
    static P: OnceLock<UnicodePats> = OnceLock::new();
    P.get_or_init(|| UnicodePats {
        code_point: Regexp::must_compile(r"U\+([a-fA-F0-9]{4}).?"),
        multi_code_point: Regexp::must_compile(r"(?:U\+[a-fA-F0-9]{4}(?:\s|$))+"),
        escape: Regexp::must_compile(r"(?i)\\{1,2}u([a-fA-F0-9]{4})"),
        multi_escape: Regexp::must_compile(r"(?i)(?:\\{1,2}u[a-fA-F0-9]{4})+"),
    })
}

fn rune_from(v: i64) -> char {
    if v < 0 {
        return '\u{FFFD}';
    }
    char::from_u32(v as u32).unwrap_or('\u{FFFD}')
}

fn decode_unicode(encoded_value: &str) -> String {
    let p = unicode_pats();
    let mut encoded_value = encoded_value.to_string();

    let matches = p.multi_code_point.find_all_strings(&encoded_value);
    if !matches.is_empty() {
        for m in matches {
            let d = decode_multi_code_point(&m);
            if !d.is_empty() && d != m {
                encoded_value = encoded_value.replacen(&m, &d, 1);
            }
        }
        return encoded_value;
    }

    let matches = p.multi_escape.find_all_strings(&encoded_value);
    if !matches.is_empty() {
        for m in matches {
            let d = decode_multi_escape(&m);
            if !d.is_empty() && d != m {
                encoded_value = encoded_value.replacen(&m, &d, 1);
            }
        }
        return encoded_value;
    }

    let data = encoded_value.clone();
    let mut result = String::new();
    if p.code_point.is_match(&data) {
        result = decode_individual_code_points(&data);
    }
    if result.is_empty() || p.escape.is_match(&data) {
        if !result.is_empty() {
            result = decode_individual_escapes(&result);
        } else {
            result = decode_individual_escapes(&data);
        }
    }
    if result.is_empty() || result == data {
        return encoded_value;
    }
    result
}

fn decode_multi_code_point(seq: &str) -> String {
    if seq.is_empty() {
        return String::new();
    }
    let cps = crate::gostr::fields(seq);
    if cps.is_empty() {
        return seq.to_string();
    }
    let mut out = String::new();
    let mut any = false;
    for cp in cps {
        if !cp.starts_with("U+") || cp.len() < 6 {
            continue;
        }
        match i64::from_str_radix(&cp[2..], 16) {
            Ok(v) if v <= i32::MAX as i64 => {
                out.push(rune_from(v));
                any = true;
            }
            _ => continue,
        }
    }
    if !any {
        return seq.to_string();
    }
    out
}

fn decode_multi_escape(seq: &str) -> String {
    if seq.is_empty() {
        return String::new();
    }
    let escapes = unicode_pats().escape.find_all_submatch(seq);
    if escapes.is_empty() {
        return seq.to_string();
    }
    let mut out = String::new();
    let mut any = false;
    for esc in escapes {
        if let Ok(v) = i64::from_str_radix(&esc[1], 16) {
            out.push(rune_from(v));
            any = true;
        }
    }
    if !any {
        return seq.to_string();
    }
    out
}

fn decode_individual_code_points(input: &str) -> String {
    let indices = unicode_pats().code_point.find_all_submatch_index(input);
    if indices.is_empty() {
        return input.to_string();
    }
    let mut buf: Vec<u8> = input.as_bytes().to_vec();
    for m in indices.iter().rev() {
        let start = m[0] as usize;
        let mut end = m[1] as usize;
        let hs = m[2] as usize;
        let he = m[3] as usize;
        if end != he && end < buf.len() && buf[end - 1] == b' ' {
            end -= 1;
        }
        let hex = String::from_utf8_lossy(&buf[hs..he]).into_owned();
        let v = match i64::from_str_radix(&hex, 16) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let mut tmp = [0u8; 4];
        let enc = rune_from(v).encode_utf8(&mut tmp).as_bytes().to_vec();
        let mut nb = Vec::with_capacity(buf.len());
        nb.extend_from_slice(&buf[..start]);
        nb.extend_from_slice(&enc);
        nb.extend_from_slice(&buf[end..]);
        buf = nb;
    }
    String::from_utf8_lossy(&buf).into_owned()
}

fn decode_individual_escapes(input: &str) -> String {
    let indices = unicode_pats().escape.find_all_submatch_index(input);
    if indices.is_empty() {
        return input.to_string();
    }
    let mut buf: Vec<u8> = input.as_bytes().to_vec();
    for m in indices.iter().rev() {
        let start = m[0] as usize;
        let hs = m[2] as usize;
        let end = m[3] as usize;
        let hex = String::from_utf8_lossy(&buf[hs..end]).into_owned();
        let v = match i64::from_str_radix(&hex, 16) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let mut tmp = [0u8; 4];
        let enc = rune_from(v).encode_utf8(&mut tmp).as_bytes().to_vec();
        let mut nb = Vec::with_capacity(buf.len());
        nb.extend_from_slice(&buf[..start]);
        nb.extend_from_slice(&enc);
        nb.extend_from_slice(&buf[end..]);
        buf = nb;
    }
    String::from_utf8_lossy(&buf).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_decode(d: &mut Decoder, data: &str) -> String {
        let mut data = data.to_string();
        let mut segs: Vec<Rc<EncodedSegment>> = Vec::new();
        loop {
            let (nd, ns) = d.decode(&data, &segs);
            data = nd;
            segs = ns;
            if segs.is_empty() {
                return data;
            }
        }
    }

    #[test]
    fn test_decode() {
        let cases = [
            ("bG9uZ2VyLWVuY29kZWQtc2VjcmV0LXRlc3Q=", "longer-encoded-secret-test"),
            ("token: bG9uZ2VyLWVuY29kZWQtc2VjcmV0LXRlc3Q=", "token: longer-encoded-secret-test"),
            ("", ""),
            ("some-encoded-secret=dGVzdC1zZWNyZXQtdmFsdWU=", "some-encoded-secret=test-secret-value"),
            ("Z2l0bGVha3M-PmZpbmRzLXNlY3JldHM", "gitleaks>>finds-secrets"),
            ("YjY0dXJsc2FmZS10ZXN0LXNlY3JldC11bmRlcnNjb3Jlcz8_", "b64urlsafe-test-secret-underscores??"),
            ("a3d3fa7c2bb99e469ba55e5834ce79ee4853a8a3", "a3d3fa7c2bb99e469ba55e5834ce79ee4853a8a3"),
            ("secret%3D%22q%24%21%40%23%24%25%5E%26%2A%28%20asdf%22", "secret=\"q$!@#$%^&*( asdf\""),
            ("secret=\"466973684D617048756E6B79212121363334\"", "secret=\"FishMapHunky!!!634\""),
            ("secret=U+0061 U+0062 U+0063 U+0064 U+0065 U+0066", "secret=abcdef"),
            (
                r"secret=\\u0068\\u0065\\u006c\\u006c\\u006f\\u0020\\u0077\\u006f\\u0072\\u006c\\u0064\\u0020\\u0064\\u0075\\u0064\\u0065",
                "secret=hello world dude",
            ),
            (
                r"secret=hello world 6C6F76656C792070656F706C65206F66206561727468",
                "secret=hello world lovely people of earth",
            ),
        ];
        let mut d = Decoder::new();
        for (i, o) in cases {
            assert_eq!(full_decode(&mut d, i), o, "input {}", i);
        }
    }
}
