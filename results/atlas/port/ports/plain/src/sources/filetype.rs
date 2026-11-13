//! Port of the subset of h2non/filetype used by gitleaks: determine whether
//! a buffer's detected MIME type is in the "application" family.
//!
//! Matchers are evaluated in the same category order as the Go library:
//! application, image, video, audio, font, document, archive.

fn cmp(buf: &[u8], sub: &[u8], off: usize) -> bool {
    off + sub.len() <= buf.len() && &buf[off..off + sub.len()] == sub
}

fn application(b: &[u8]) -> bool {
    let wasm = b.len() >= 8 && b[..8] == [0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
    let dex = |b: &[u8]| b.len() > 36 && b[0] == 0x64 && b[1] == 0x65 && b[2] == 0x78 && b[3] == 0x0A && b[36] == 0x70;
    let dey = b.len() > 100 && b[0] == 0x64 && b[1] == 0x65 && b[2] == 0x79 && b[3] == 0x0A && dex(&b[40..100]);
    wasm || dex(b) || dey
}

fn is_isobmff(b: &[u8]) -> bool {
    if b.len() < 16 || &b[4..8] != b"ftyp" {
        return false;
    }
    let l = u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize;
    b.len() >= l
}

fn heif(b: &[u8]) -> bool {
    if !is_isobmff(b) || b.len() < 17 {
        return false;
    }
    let l = u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let major = &b[8..12];
    if major == b"heic" {
        return true;
    }
    if major == b"mif1" || major == b"msf1" {
        let mut i = 16;
        while i < l {
            if b.len() >= i + 4 && &b[i..i + 4] == b"heic" {
                return true;
            }
            i += 4;
        }
    }
    false
}

fn image(b: &[u8]) -> bool {
    let n = b.len();
    let jpeg = n > 2 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF;
    let jp2 = n > 12 && b[..13] == [0, 0, 0, 0xC, 0x6A, 0x50, 0x20, 0x20, 0xD, 0xA, 0x87, 0xA, 0];
    let png = n > 3 && b[..4] == [0x89, 0x50, 0x4E, 0x47];
    let gif = n > 2 && b[..3] == [0x47, 0x49, 0x46];
    let webp = n > 11 && b[8..12] == [0x57, 0x45, 0x42, 0x50];
    let tiff_hdr = n > 10 && (b[..4] == [0x49, 0x49, 0x2A, 0x0] || b[..4] == [0x4D, 0x4D, 0x0, 0x2A]);
    let cr2 = tiff_hdr && b[8] == 0x43 && b[9] == 0x52 && b[10] == 0x02;
    let tiff = tiff_hdr && !cr2;
    let bmp = n > 1 && b[0] == 0x42 && b[1] == 0x4D;
    let jxr = n > 2 && b[0] == 0x49 && b[1] == 0x49 && b[2] == 0xBC;
    let psd = n > 3 && b[..4] == [0x38, 0x42, 0x50, 0x53];
    let ico = n > 3 && b[..4] == [0, 0, 1, 0];
    let dwg = n > 3 && b[..4] == [0x41, 0x43, 0x31, 0x30];
    jpeg || jp2 || png || gif || webp || cr2 || tiff || bmp || jxr || psd || ico || heif(b) || dwg
}

fn matroska_sig(b: &[u8], sub: &[u8]) -> bool {
    let limit = b.len().min(4096);
    let hay = &b[..limit];
    let idx = hay.windows(sub.len()).position(|w| w == sub);
    match idx {
        Some(i) if i >= 3 => b[i - 3] == 0x42 && b[i - 2] == 0x82,
        _ => false,
    }
}

fn video(b: &[u8]) -> bool {
    let n = b.len();
    let ftyp = n > 7 && &b[4..8] == b"ftyp";
    let m4v = n > 10 && ftyp && b[8] == 0x4D && b[9] == 0x34 && b[10] == 0x56;
    let ebml = n > 3 && b[..4] == [0x1A, 0x45, 0xDF, 0xA3];
    let mkv = ebml && matroska_sig(b, b"matroska");
    let webm = ebml && matroska_sig(b, b"webm");
    let mov = n > 15
        && ((b[..8] == [0, 0, 0, 0x14, 0x66, 0x74, 0x79, 0x70])
            || &b[4..8] == b"moov"
            || &b[4..8] == b"mdat"
            || &b[12..16] == b"mdat");
    let avi = n > 10 && b[..4] == [0x52, 0x49, 0x46, 0x46] && b[8..11] == [0x41, 0x56, 0x49];
    let wmv = n > 9 && b[..10] == [0x30, 0x26, 0xB2, 0x75, 0x8E, 0x66, 0xCF, 0x11, 0xA6, 0xD9];
    let mpeg = n > 3 && b[0] == 0 && b[1] == 0 && b[2] == 1 && (0xb0..=0xbf).contains(&b[3]);
    let flv = n > 3 && b[..4] == [0x46, 0x4C, 0x56, 0x01];
    let mp4_brands: [&[u8; 4]; 28] = [
        b"avc1", b"dash", b"iso2", b"iso3", b"iso4", b"iso5", b"iso6", b"isom", b"mmp4", b"mp41", b"mp42", b"mp4v",
        b"mp71", b"MSNV", b"NDAS", b"NDSC", b"NSDC", b"NDSH", b"NDSM", b"NDSP", b"NDSS", b"NDXC", b"NDXH", b"NDXM",
        b"NDXP", b"NDXS", b"F4V ", b"F4P ",
    ];
    let mp4 = n > 11 && ftyp && mp4_brands.iter().any(|br| &b[8..12] == &br[..]);
    let g3p = n > 10 && ftyp && b[8] == 0x33 && b[9] == 0x67 && b[10] == 0x70;
    m4v || mkv || webm || mov || avi || wmv || mpeg || flv || mp4 || g3p
}

fn audio(b: &[u8]) -> bool {
    let n = b.len();
    let midi = n > 3 && b[..4] == [0x4D, 0x54, 0x68, 0x64];
    let mp3 = n > 2 && (b[..3] == [0x49, 0x44, 0x33] || (b[0] == 0xFF && b[1] == 0xfb));
    let m4a = n > 10
        && ((&b[4..8] == b"ftyp" && b[8] == 0x4D && b[9] == 0x34 && b[10] == 0x41) || b[..4] == [0x4D, 0x34, 0x41, 0x20]);
    let ogg = n > 3 && &b[..4] == b"OggS";
    let flac = n > 3 && &b[..4] == b"fLaC";
    let wav = n > 11 && &b[..4] == b"RIFF" && &b[8..12] == b"WAVE";
    let amr = n > 11 && b[..6] == [0x23, 0x21, 0x41, 0x4D, 0x52, 0x0A];
    let aac = n > 1 && b[0] == 0xFF && (b[1] == 0xF1 || b[1] == 0xF9);
    let aiff = n > 11 && &b[..4] == b"FORM" && &b[8..12] == b"AIFF";
    midi || mp3 || m4a || ogg || flac || wav || amr || aac || aiff
}

fn font(b: &[u8]) -> bool {
    let n = b.len();
    let woff = n > 7 && b[..8] == [0x77, 0x4F, 0x46, 0x46, 0, 1, 0, 0];
    let woff2 = n > 7 && b[..8] == [0x77, 0x4F, 0x46, 0x32, 0, 1, 0, 0];
    let ttf = n > 4 && b[..5] == [0, 1, 0, 0, 0];
    let otf = n > 4 && b[..5] == [0x4F, 0x54, 0x54, 0x4F, 0];
    woff || woff2 || ttf || otf
}

fn ole(b: &[u8]) -> bool {
    b.len() > 3 && b[..4] == [0xD0, 0xCF, 0x11, 0xE0]
}

fn check_msooml(b: &[u8], off: usize) -> bool {
    cmp(b, b"word/", off) || cmp(b, b"ppt/", off) || cmp(b, b"xl/", off)
}

fn search(b: &[u8], start: usize, range: usize) -> Option<usize> {
    let end = (start + range).min(b.len());
    if start >= end {
        return None;
    }
    b[start..end].windows(4).position(|w| w == b"PK\x03\x04")
}

/// returns Some(is_specific_office_type)
fn msooxml(b: &[u8]) -> bool {
    if !cmp(b, b"PK\x03\x04", 0) {
        return false;
    }
    if check_msooml(b, 0x1E) {
        return true;
    }
    if !cmp(b, b"[Content_Types].xml", 0x1E) && !cmp(b, b"_rels/.rels", 0x1E) && !cmp(b, b"docProps", 0x1E) {
        return false;
    }
    if b.len() < 22 {
        return false;
    }
    let mut start = u32::from_le_bytes([b[18], b[19], b[20], b[21]]) as usize + 49;
    let idx = match search(b, start, 6000) {
        Some(i) => i,
        None => return false,
    };
    start += idx + 4 + 26;
    let idx = match search(b, start, 6000) {
        Some(i) => i,
        None => return false,
    };
    start += idx + 4 + 26;
    if check_msooml(b, start) {
        return true;
    }
    // TYPE_OOXML which is not one of docx/xlsx/pptx: doesn't match those matchers
    false
}

fn document(b: &[u8]) -> bool {
    ole(b) || msooxml(b)
}

fn zst(b: &[u8]) -> bool {
    if cmp(b, &[0x28, 0xB5, 0x2F, 0xFD], 0) {
        return true;
    }
    if b.len() < 8 {
        return false;
    }
    let magic = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    if magic & 0xFFFFFFF0 == 0x184D2A50 {
        let l = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize;
        if b.len() < 8 + l {
            return false;
        }
        return zst(&b[8 + l..]);
    }
    false
}

fn archive(b: &[u8]) -> bool {
    let n = b.len();
    let prefixes: [&[u8]; 14] = [
        &[0x50, 0x4B, 0x03, 0x04, 0x6D, 0x69, 0x6D, 0x65, 0x74, 0x79, 0x70, 0x65, 0x61, 0x70, 0x70, 0x6C, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6F, 0x6E, 0x2F, 0x65, 0x70, 0x75, 0x62, 0x2B, 0x7A, 0x69, 0x70],
        &[0x1F, 0x8B, 0x08],
        &[0x42, 0x5A, 0x68],
        &[0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C],
        &[0x25, 0x50, 0x44, 0x46],
        &[0x4D, 0x5A],
        &[0x7B, 0x5C, 0x72, 0x74, 0x66],
        &[0x4E, 0x45, 0x53, 0x1A],
        &[0x43, 0x72, 0x32, 0x34],
        &[0x25, 0x21],
        &[0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00],
        &[0x53, 0x51, 0x4C, 0x69],
        &[0x21, 0x3C, 0x61, 0x72, 0x63, 0x68, 0x3E],
        &[0x4C, 0x5A, 0x49, 0x50],
    ];
    if prefixes.iter().any(|p| cmp(b, p, 0)) {
        return true;
    }
    let zip = n > 3 && b[0] == 0x50 && b[1] == 0x4B && matches!(b[2], 3 | 5 | 7) && matches!(b[3], 4 | 6 | 8);
    let tar = n > 261 && &b[257..262] == b"ustar";
    let rar = n > 6 && b[..6] == [0x52, 0x61, 0x72, 0x21, 0x1A, 0x07] && (b[6] == 0 || b[6] == 1);
    let swf = n > 2 && (b[0] == 0x43 || b[0] == 0x46) && b[1] == 0x57 && b[2] == 0x53;
    let cab = n > 3 && (&b[..4] == b"MSCF" || &b[..4] == b"ISc(");
    let eot = n > 35
        && b[34] == 0x4C
        && b[35] == 0x50
        && ((b[8] == 2 && b[9] == 0 && b[10] == 1) || (b[8] == 1 && b[9] == 0 && b[10] == 0) || (b[8] == 2 && b[9] == 0 && b[10] == 2));
    let z = n > 1 && b[0] == 0x1F && (b[1] == 0xA0 || b[1] == 0x9D);
    let rpm = n > 96 && b[..4] == [0xED, 0xAB, 0xEE, 0xDB];
    let elf = n > 52 && b[..4] == [0x7F, 0x45, 0x4C, 0x46];
    let dcm = n > 131 && &b[128..132] == b"DICM";
    let iso = n > 32773 && &b[32769..32774] == b"CD001";
    let macho = n > 3
        && [
            [0xFE, 0xED, 0xFA, 0xCF],
            [0xFE, 0xED, 0xFA, 0xCE],
            [0xBE, 0xBA, 0xFE, 0xCA],
            [0xCF, 0xFA, 0xED, 0xFE],
            [0xCE, 0xFA, 0xED, 0xFE],
            [0xCA, 0xFE, 0xBA, 0xBE],
        ]
        .iter()
        .any(|m| b[..4] == *m);
    zip || tar || rar || swf || cab || eot || z || rpm || elf || dcm || iso || macho || zst(b)
}

/// Returns the MIME "value" if the buffer is detected as an application/* type.
pub fn is_application(buf: &[u8]) -> bool {
    if buf.is_empty() {
        return false;
    }
    if application(buf) {
        return true;
    }
    if image(buf) || video(buf) || audio(buf) {
        return false;
    }
    font(buf) || document(buf) || archive(buf)
}
