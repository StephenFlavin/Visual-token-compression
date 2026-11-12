// Port of detect/codec: recursive decoding of percent, unicode, hex and base64 encoded text.

use crate::goregex::{must_compile, GoRegex};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
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

pub const PERCENT_KIND: u32 = 1;
pub const UNICODE_KIND: u32 = 2;
pub const HEX_KIND: u32 = 4;
pub const BASE64_KIND: u32 = 8;
const ENCODING_NAMES: [&str; 4] = ["percent", "unicode", "hex", "base64"];

fn kind_name(k: u32) -> &'static str {
    let i = (k as f64).log2() as usize;
    if i >= ENCODING_NAMES.len() {
        ""
    } else {
        ENCODING_NAMES[i]
    }
}

fn kinds(e: u32) -> Vec<u32> {
    let mut out = vec![];
    for i in 0..ENCODING_NAMES.len() {
        let k = e & (1 << i);
        if k != 0 {
            out.push(k);
        }
    }
    out
}

#[derive(Debug)]
pub struct EncodedSegment {
    predecessors: Vec<Rc<EncodedSegment>>,
    original: StartEnd,
    encoded: StartEnd,
    decoded: StartEnd,
    decoded_value: String,
    encodings: u32,
    depth: i64,
}

struct Encoding {
    kind: u32,
    precedence: i32,
    decode: fn(&str) -> String,
}

const ENCODINGS: [Encoding; 4] = [
    Encoding { kind: PERCENT_KIND, precedence: 4, decode: decode_percent },
    Encoding { kind: UNICODE_KIND, precedence: 3, decode: decode_unicode },
    Encoding { kind: HEX_KIND, precedence: 2, decode: decode_hex },
    Encoding { kind: BASE64_KIND, precedence: 1, decode: decode_base64 },
];

fn encodings_re() -> &'static GoRegex {
    static RE: OnceLock<GoRegex> = OnceLock::new();
    RE.get_or_init(|| {
        let pats = [
            r"%[0-9A-Fa-f]{2}(?:.*%[0-9A-Fa-f]{2})?",
            r"(?:(?:U\+[a-fA-F0-9]{4}(?:\s|$))+|(?i)(?:\\{1,2}u[a-fA-F0-9]{4})+)",
            r"[0-9A-Fa-f]{32,}",
            r"[\w\/+-]{16,}={0,2}",
        ];
        let named: Vec<String> = pats
            .iter()
            .enumerate()
            .map(|(i, p)| format!("(?P<{}>{})", ENCODING_NAMES[i], p))
            .collect();
        must_compile(&named.join("|"))
    })
}

struct EncodingMatch {
    enc: usize,
    se: StartEnd,
}

fn find_encoding_matches(data: &str) -> Vec<EncodingMatch> {
    let re = &encodings_re().re;
    let mut all = vec![];
    for caps in re.captures_iter(data) {
        for j in 0..4 {
            if let Some(m) = caps.get(j + 1) {
                all.push(EncodingMatch { enc: j, se: StartEnd::new(m.start() as i64, m.end() as i64) });
            }
        }
    }
    let total = all.len();
    if total == 1 {
        return all;
    }
    let mut keep = vec![true; total];
    for i in 0..total {
        let m = &all[i];
        if i > 0 {
            let prev = &all[i - 1];
            if m.se.overlaps(prev.se) && ENCODINGS[prev.enc].precedence > ENCODINGS[m.enc].precedence {
                keep[i] = false;
                continue;
            }
        }
        if i + 1 < total {
            let next = &all[i + 1];
            if m.se.overlaps(next.se) && ENCODINGS[next.enc].precedence > ENCODINGS[m.enc].precedence {
                keep[i] = false;
                continue;
            }
        }
    }
    all.into_iter().zip(keep).filter(|(_, k)| *k).map(|(m, _)| m).collect()
}

pub struct Decoder {
    decoded_map: HashMap<String, String>,
}

impl Decoder {
    pub fn new() -> Decoder {
        Decoder { decoded_map: HashMap::new() }
    }

    pub fn decode(&mut self, data: &str, predecessors: &[Rc<EncodedSegment>]) -> (String, Vec<Rc<EncodedSegment>>) {
        let segments = self.find_encoded_segments(data, predecessors);
        if !segments.is_empty() {
            let mut result = String::with_capacity(data.len());
            let mut encoded_start = 0usize;
            for s in &segments {
                result.push_str(&data[encoded_start..s.encoded.start as usize]);
                result.push_str(&s.decoded_value);
                encoded_start = s.encoded.end as usize;
            }
            result.push_str(&data[encoded_start..]);
            return (result, segments);
        }
        (data.to_string(), segments)
    }

    fn find_encoded_segments(&mut self, data: &str, predecessors: &[Rc<EncodedSegment>]) -> Vec<Rc<EncodedSegment>> {
        if data.is_empty() {
            return vec![];
        }
        let mut decoded_shift: i64 = 0;
        let mut segments = vec![];
        for m in find_encoding_matches(data) {
            let encoded_value = &data[m.se.start as usize..m.se.end as usize];
            let decoded_value = match self.decoded_map.get(encoded_value) {
                Some(v) => v.clone(),
                None => {
                    let v = (ENCODINGS[m.enc].decode)(encoded_value);
                    self.decoded_map.insert(encoded_value.to_string(), v.clone());
                    v
                }
            };
            if decoded_value.is_empty() {
                continue;
            }
            let dlen = decoded_value.len() as i64;
            let mut seg = EncodedSegment {
                predecessors: predecessors.to_vec(),
                original: to_original(predecessors, m.se),
                encoded: m.se,
                decoded: StartEnd::new(m.se.start + decoded_shift, m.se.start + decoded_shift + dlen),
                decoded_value,
                encodings: ENCODINGS[m.enc].kind,
                depth: 1,
            };
            decoded_shift += dlen - encoded_value.len() as i64;
            if !seg.predecessors.is_empty() {
                seg.depth = 1 + seg.predecessors[0].depth;
                for p in &seg.predecessors {
                    if seg.encoded.overlaps(p.decoded) {
                        seg.encodings |= p.encodings;
                    }
                }
            }
            crate::logging::debug().str("decoder", kind_name(ENCODINGS[m.enc].kind)).msg(&format!(
                "segment found: original=[{},{}] pos=[{},{}]: {} -> {}",
                seg.original.start,
                seg.original.end,
                seg.encoded.start,
                seg.encoded.end,
                crate::util::go_quote(encoded_value),
                crate::util::go_quote(&seg.decoded_value)
            ));
            segments.push(Rc::new(seg));
        }
        segments
    }
}

pub fn tags(segments: &[Rc<EncodedSegment>]) -> Vec<String> {
    if segments.is_empty() {
        return vec![];
    }
    let depth = segments[0].depth;
    let mut enc = segments[0].encodings;
    for s in &segments[1..] {
        enc |= s.encodings;
    }
    let mut out: Vec<String> = kinds(enc).into_iter().map(|k| format!("decoded:{}", kind_name(k))).collect();
    out.push(format!("decode-depth:{}", depth));
    out
}

pub fn current_line(segments: &[Rc<EncodedSegment>], current_raw: &str) -> String {
    if segments.is_empty() {
        return current_raw.to_string();
    }
    let b = current_raw.as_bytes();
    let mut start = 0usize;
    let mut end = b.len();
    let mut decoded = segments[0].decoded;
    for s in &segments[1..] {
        decoded = decoded.merge(s.decoded);
    }
    let mut i = decoded.start;
    while i > -1 {
        if (i as usize) < b.len() && b[i as usize] == b'\n' {
            start = i as usize;
            break;
        }
        i -= 1;
    }
    let mut i = decoded.end.max(0) as usize;
    while i < end {
        if b[i] == b'\n' {
            end = i;
            break;
        }
        i += 1;
    }
    crate::util::bslice(current_raw, start, end)
}

pub fn adjust_match_index(segments: &[Rc<EncodedSegment>], m: (usize, usize)) -> (usize, usize) {
    if segments.is_empty() {
        return m;
    }
    let a = to_original(segments, StartEnd::new(m.0 as i64, m.1 as i64));
    (a.start.max(0) as usize, a.end.max(0) as usize)
}

pub fn segments_with_decoded_overlap(segments: &[Rc<EncodedSegment>], start: usize, end: usize) -> Vec<Rc<EncodedSegment>> {
    let se = StartEnd::new(start as i64, end as i64);
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

// ---- individual decoders ----

fn printable(b: u8) -> bool {
    b > 0x08 && b < 0x7f
}

fn hex_val(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => 0xff,
    }
}

fn decode_percent(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let n1 = hex_val(b[i + 1]);
            let n2 = hex_val(b[i + 2]);
            if n1 | n2 != 0xff {
                let v = (n1 << 4) | n2;
                if !printable(v) {
                    return String::new();
                }
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn decode_hex(s: &str) -> String {
    let b = s.as_bytes();
    if b.len() % 2 != 0 {
        return String::new();
    }
    if !b.iter().any(|c| c.is_ascii_digit()) {
        return String::new();
    }
    let mut out = Vec::with_capacity(b.len() / 2);
    let mut i = 0;
    while i < b.len() {
        let n1 = hex_val(b[i]);
        let n2 = hex_val(b[i + 1]);
        if n1 | n2 == 0xff {
            return String::new();
        }
        let v = (n1 << 4) | n2;
        if !printable(v) {
            return String::new();
        }
        out.push(v);
        i += 2;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn b64_val(c: u8, url: bool) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' if !url => Some(62),
        b'/' if !url => Some(63),
        b'-' if url => Some(62),
        b'_' if url => Some(63),
        _ => None,
    }
}

/// Mirrors Go's encoding/base64 (non-strict) decoding.
fn b64_decode(s: &[u8], url: bool, padded: bool) -> Option<Vec<u8>> {
    let src: Vec<u8> = s.iter().cloned().filter(|&c| c != b'\r' && c != b'\n').collect();
    let mut out = vec![];
    let mut i = 0;
    while i < src.len() {
        let mut quantum = [0u8; 4];
        let mut n = 0;
        let mut padded_end = false;
        while n < 4 {
            if i >= src.len() {
                break;
            }
            let c = src[i];
            if c == b'=' && padded {
                // padding
                if n < 2 {
                    return None;
                }
                if n == 2 {
                    // need another '='
                    if i + 1 >= src.len() || src[i + 1] != b'=' {
                        return None;
                    }
                    i += 2;
                } else {
                    i += 1;
                }
                if i < src.len() {
                    return None;
                }
                padded_end = true;
                break;
            }
            let v = b64_val(c, url)?;
            quantum[n] = v;
            n += 1;
            i += 1;
        }
        if n == 0 {
            break;
        }
        if n < 4 && padded && !padded_end {
            return None;
        }
        match n {
            1 => return None,
            2 => {
                out.push((quantum[0] << 2) | (quantum[1] >> 4));
            }
            3 => {
                out.push((quantum[0] << 2) | (quantum[1] >> 4));
                out.push((quantum[1] << 4) | (quantum[2] >> 2));
            }
            _ => {
                out.push((quantum[0] << 2) | (quantum[1] >> 4));
                out.push((quantum[1] << 4) | (quantum[2] >> 2));
                out.push((quantum[2] << 6) | quantum[3]);
            }
        }
        if padded_end {
            break;
        }
    }
    Some(out)
}

fn decode_base64(s: &str) -> String {
    if !s.bytes().any(|c| c.is_ascii_digit() || c == b'+' || c == b'/' || c == b'-' || c == b'_') {
        return String::new();
    }
    if let Some(d) = b64_decode(s.as_bytes(), false, true) {
        if d.iter().all(|&c| printable(c)) {
            return String::from_utf8_lossy(&d).into_owned();
        }
    }
    if let Some(d) = b64_decode(s.as_bytes(), true, false) {
        if d.iter().all(|&c| printable(c)) {
            return String::from_utf8_lossy(&d).into_owned();
        }
    }
    String::new()
}

struct UnicodeRes {
    code_point: GoRegex,
    multi_code_point: GoRegex,
    escape: GoRegex,
    multi_escape: GoRegex,
}

fn unicode_res() -> &'static UnicodeRes {
    static RE: OnceLock<UnicodeRes> = OnceLock::new();
    RE.get_or_init(|| UnicodeRes {
        code_point: must_compile(r"U\+([a-fA-F0-9]{4}).?"),
        multi_code_point: must_compile(r"(?:U\+[a-fA-F0-9]{4}(?:\s|$))+"),
        escape: must_compile(r"(?i)\\{1,2}u([a-fA-F0-9]{4})"),
        multi_escape: must_compile(r"(?i)(?:\\{1,2}u[a-fA-F0-9]{4})+"),
    })
}

fn rune_from(v: i64) -> char {
    if v < 0 {
        return '\u{FFFD}';
    }
    char::from_u32(v as u32).unwrap_or('\u{FFFD}')
}

fn decode_unicode(encoded: &str) -> String {
    let res = unicode_res();
    let mut value = encoded.to_string();
    let matches: Vec<String> = res.multi_code_point.re.find_iter(&value).map(|m| m.as_str().to_string()).collect();
    if !matches.is_empty() {
        for m in matches {
            let d = decode_multi_code_point(&m);
            if !d.is_empty() && d != m {
                value = value.replacen(&m, &d, 1);
            }
        }
        return value;
    }
    let matches: Vec<String> = res.multi_escape.re.find_iter(&value).map(|m| m.as_str().to_string()).collect();
    if !matches.is_empty() {
        for m in matches {
            let d = decode_multi_escape(&m);
            if !d.is_empty() && d != m {
                value = value.replacen(&m, &d, 1);
            }
        }
        return value;
    }
    let data = value.clone();
    let mut result = String::new();
    if res.code_point.is_match(&data) {
        result = decode_individual_code_points(&data);
    }
    if result.is_empty() || res.escape.is_match(&data) {
        if !result.is_empty() {
            result = decode_individual_escapes(&result);
        } else {
            result = decode_individual_escapes(&data);
        }
    }
    if result.is_empty() || result == data {
        return value;
    }
    result
}

fn decode_multi_code_point(seq: &str) -> String {
    if seq.is_empty() {
        return String::new();
    }
    let cps: Vec<&str> = seq.split_whitespace().collect();
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
    let res = unicode_res();
    let mut out = String::new();
    let mut any = false;
    for c in res.escape.re.captures_iter(seq) {
        if let Ok(v) = i64::from_str_radix(c.get(1).unwrap().as_str(), 16) {
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
    let res = unicode_res();
    let caps: Vec<(usize, usize, usize, usize)> = res
        .code_point
        .re
        .captures_iter(input)
        .map(|c| {
            let m = c.get(0).unwrap();
            let h = c.get(1).unwrap();
            (m.start(), m.end(), h.start(), h.end())
        })
        .collect();
    if caps.is_empty() {
        return input.to_string();
    }
    let mut buf = input.as_bytes().to_vec();
    for &(start, end0, hs, he) in caps.iter().rev() {
        let mut end = end0;
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
        let mut nb = buf[..start].to_vec();
        nb.extend_from_slice(&enc);
        nb.extend_from_slice(&buf[end.min(buf.len())..]);
        buf = nb;
    }
    String::from_utf8_lossy(&buf).into_owned()
}

fn decode_individual_escapes(input: &str) -> String {
    let res = unicode_res();
    let caps: Vec<(usize, usize, usize)> = res
        .escape
        .re
        .captures_iter(input)
        .map(|c| {
            let m = c.get(0).unwrap();
            let h = c.get(1).unwrap();
            (m.start(), h.start(), h.end())
        })
        .collect();
    if caps.is_empty() {
        return input.to_string();
    }
    let mut buf = input.as_bytes().to_vec();
    for &(start, hs, end) in caps.iter().rev() {
        let hex = String::from_utf8_lossy(&buf[hs..end]).into_owned();
        let v = match i64::from_str_radix(&hex, 16) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let mut tmp = [0u8; 4];
        let enc = rune_from(v).encode_utf8(&mut tmp).as_bytes().to_vec();
        let mut nb = buf[..start].to_vec();
        nb.extend_from_slice(&enc);
        nb.extend_from_slice(&buf[end..]);
        buf = nb;
    }
    String::from_utf8_lossy(&buf).into_owned()
}
