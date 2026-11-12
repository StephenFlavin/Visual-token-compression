// LZMA / LZMA2 decoders with xz and 7z container support (used for archive scanning).

type R<T> = Result<T, String>;

struct RangeDec<'a> {
    data: &'a [u8],
    pos: usize,
    range: u32,
    code: u32,
}

impl<'a> RangeDec<'a> {
    fn new(data: &'a [u8], pos: usize) -> R<RangeDec<'a>> {
        if pos + 5 > data.len() {
            return Err("lzma: truncated".into());
        }
        let code = u32::from_be_bytes([data[pos + 1], data[pos + 2], data[pos + 3], data[pos + 4]]);
        Ok(RangeDec { data, pos: pos + 5, range: 0xFFFF_FFFF, code })
    }
    #[inline]
    fn normalize(&mut self) {
        if self.range < (1 << 24) {
            self.range <<= 8;
            let b = if self.pos < self.data.len() { self.data[self.pos] } else { 0 };
            self.pos += 1;
            self.code = (self.code << 8) | b as u32;
        }
    }
    #[inline]
    fn bit(&mut self, p: &mut u16) -> u32 {
        let bound = (self.range >> 11) * (*p as u32);
        let r;
        if self.code < bound {
            self.range = bound;
            *p += (2048 - *p) >> 5;
            r = 0;
        } else {
            self.range -= bound;
            self.code -= bound;
            *p -= *p >> 5;
            r = 1;
        }
        self.normalize();
        r
    }
    fn direct(&mut self, n: u32) -> u32 {
        let mut res = 0u32;
        for _ in 0..n {
            self.range >>= 1;
            let t = if self.code >= self.range {
                self.code -= self.range;
                1
            } else {
                0
            };
            res = (res << 1) | t;
            self.normalize();
        }
        res
    }
    fn tree(&mut self, probs: &mut [u16], bits: u32) -> u32 {
        let mut m = 1usize;
        for _ in 0..bits {
            m = (m << 1) | self.bit(&mut probs[m]) as usize;
        }
        (m - (1 << bits)) as u32
    }
    fn rev_tree(&mut self, probs: &mut [u16], bits: u32) -> u32 {
        let mut m = 1usize;
        let mut sym = 0u32;
        for i in 0..bits {
            let b = self.bit(&mut probs[m]);
            m = (m << 1) | b as usize;
            sym |= b << i;
        }
        sym
    }
}

#[derive(Clone)]
struct LenDec {
    choice: u16,
    choice2: u16,
    low: Vec<[u16; 8]>,
    mid: Vec<[u16; 8]>,
    high: [u16; 256],
}

impl LenDec {
    fn new() -> LenDec {
        LenDec { choice: 1024, choice2: 1024, low: vec![[1024; 8]; 16], mid: vec![[1024; 8]; 16], high: [1024; 256] }
    }
    fn decode(&mut self, rc: &mut RangeDec, ps: usize) -> u32 {
        if rc.bit(&mut self.choice) == 0 {
            return rc.tree(&mut self.low[ps], 3);
        }
        if rc.bit(&mut self.choice2) == 0 {
            return 8 + rc.tree(&mut self.mid[ps], 3);
        }
        16 + rc.tree(&mut self.high, 8)
    }
}

pub struct Lzma {
    lc: u32,
    lp: u32,
    pb: u32,
    lit: Vec<u16>,
    is_match: [u16; 192],
    is_rep: [u16; 12],
    is_rep_g0: [u16; 12],
    is_rep_g1: [u16; 12],
    is_rep_g2: [u16; 12],
    is_rep0_long: [u16; 192],
    pos_slot: [[u16; 64]; 4],
    pos_special: [u16; 115],
    align: [u16; 16],
    len_dec: LenDec,
    rep_len_dec: LenDec,
    state: usize,
    reps: [u32; 4],
    pub out: Vec<u8>,
    dict_start: usize,
}

impl Lzma {
    pub fn new(props: u8) -> R<Lzma> {
        let mut l = Lzma {
            lc: 0,
            lp: 0,
            pb: 0,
            lit: vec![],
            is_match: [1024; 192],
            is_rep: [1024; 12],
            is_rep_g0: [1024; 12],
            is_rep_g1: [1024; 12],
            is_rep_g2: [1024; 12],
            is_rep0_long: [1024; 192],
            pos_slot: [[1024; 64]; 4],
            pos_special: [1024; 115],
            align: [1024; 16],
            len_dec: LenDec::new(),
            rep_len_dec: LenDec::new(),
            state: 0,
            reps: [0; 4],
            out: vec![],
            dict_start: 0,
        };
        l.set_props(props)?;
        Ok(l)
    }
    fn set_props(&mut self, d: u8) -> R<()> {
        if d >= 225 {
            return Err("lzma: invalid properties".into());
        }
        let d = d as u32;
        self.lc = d % 9;
        self.lp = (d / 9) % 5;
        self.pb = d / 45;
        self.reset_state();
        Ok(())
    }
    fn reset_state(&mut self) {
        self.lit = vec![1024; 0x300 << (self.lc + self.lp)];
        self.is_match = [1024; 192];
        self.is_rep = [1024; 12];
        self.is_rep_g0 = [1024; 12];
        self.is_rep_g1 = [1024; 12];
        self.is_rep_g2 = [1024; 12];
        self.is_rep0_long = [1024; 192];
        self.pos_slot = [[1024; 64]; 4];
        self.pos_special = [1024; 115];
        self.align = [1024; 16];
        self.len_dec = LenDec::new();
        self.rep_len_dec = LenDec::new();
        self.state = 0;
        self.reps = [0; 4];
    }

    /// Decodes until `limit` total output bytes (or end marker when allow_end).
    fn decode(&mut self, rc: &mut RangeDec, limit: usize, allow_end: bool) -> R<()> {
        let pb_mask = (1usize << self.pb) - 1;
        let lp_mask = (1usize << self.lp) - 1;
        while self.out.len() < limit {
            let pos = self.out.len();
            let ps = pos & pb_mask;
            let st = self.state;
            if rc.bit(&mut self.is_match[(st << 4) | ps]) == 0 {
                let prev = if pos > self.dict_start { self.out[pos - 1] as usize } else { 0 };
                let ls = ((pos & lp_mask) << self.lc) + (prev >> (8 - self.lc));
                let base = ls * 0x300;
                let probs = &mut self.lit[base..base + 0x300];
                let mut sym = 1usize;
                if st >= 7 {
                    let r0 = self.reps[0] as usize;
                    let mut mb = if pos > r0 { self.out[pos - r0 - 1] as usize } else { 0 };
                    while sym < 0x100 {
                        let mbit = (mb >> 7) & 1;
                        mb <<= 1;
                        let b = rc.bit(&mut probs[((1 + mbit) << 8) + sym]) as usize;
                        sym = (sym << 1) | b;
                        if mbit != b {
                            break;
                        }
                    }
                }
                while sym < 0x100 {
                    sym = (sym << 1) | rc.bit(&mut probs[sym]) as usize;
                }
                self.out.push((sym - 0x100) as u8);
                self.state = if st < 4 { 0 } else if st < 10 { st - 3 } else { st - 6 };
                continue;
            }
            let len;
            if rc.bit(&mut self.is_rep[st]) != 0 {
                if pos <= self.dict_start {
                    return Err("lzma: data error".into());
                }
                if rc.bit(&mut self.is_rep_g0[st]) == 0 {
                    if rc.bit(&mut self.is_rep0_long[(st << 4) | ps]) == 0 {
                        self.state = if st < 7 { 9 } else { 11 };
                        let b = self.out[pos - self.reps[0] as usize - 1];
                        self.out.push(b);
                        continue;
                    }
                } else {
                    let dist;
                    if rc.bit(&mut self.is_rep_g1[st]) == 0 {
                        dist = self.reps[1];
                    } else {
                        if rc.bit(&mut self.is_rep_g2[st]) == 0 {
                            dist = self.reps[2];
                        } else {
                            dist = self.reps[3];
                            self.reps[3] = self.reps[2];
                        }
                        self.reps[2] = self.reps[1];
                    }
                    self.reps[1] = self.reps[0];
                    self.reps[0] = dist;
                }
                len = self.rep_len_dec.decode(rc, ps);
                self.state = if st < 7 { 8 } else { 11 };
            } else {
                self.reps[3] = self.reps[2];
                self.reps[2] = self.reps[1];
                self.reps[1] = self.reps[0];
                len = self.len_dec.decode(rc, ps);
                self.state = if st < 7 { 7 } else { 10 };
                let ls = (len.min(3)) as usize;
                let slot = rc.tree(&mut self.pos_slot[ls], 6);
                let dist = if slot < 4 {
                    slot
                } else {
                    let nd = (slot >> 1) - 1;
                    let mut d = (2 | (slot & 1)) << nd;
                    if slot < 14 {
                        let off = (d - slot) as usize;
                        d += rc.rev_tree(&mut self.pos_special[off..], nd);
                    } else {
                        d += rc.direct(nd - 4) << 4;
                        d += rc.rev_tree(&mut self.align, 4);
                    }
                    d
                };
                if dist == 0xFFFF_FFFF {
                    if allow_end {
                        return Ok(());
                    }
                    return Err("lzma: unexpected end marker".into());
                }
                self.reps[0] = dist;
            }
            let l = len as usize + 2;
            let d = self.reps[0] as usize + 1;
            if d > pos - self.dict_start {
                return Err("lzma: distance out of range".into());
            }
            for _ in 0..l {
                if self.out.len() >= limit {
                    break;
                }
                let b = self.out[self.out.len() - d];
                self.out.push(b);
            }
        }
        Ok(())
    }
}

/// Decodes a raw LZMA stream (7z coder): props (5 bytes) + data with known size.
pub fn lzma_decode(props: &[u8], data: &[u8], unpack: usize) -> R<Vec<u8>> {
    if props.is_empty() {
        return Err("lzma: missing properties".into());
    }
    let mut l = Lzma::new(props[0])?;
    let mut rc = RangeDec::new(data, 0)?;
    l.decode(&mut rc, unpack, true)?;
    Ok(l.out)
}

/// Decodes an LZMA2 stream. Returns (output, bytes consumed).
pub fn lzma2_decode(data: &[u8]) -> R<(Vec<u8>, usize)> {
    let mut pos = 0;
    let mut l: Option<Lzma> = None;
    let mut out_keep: Vec<u8> = vec![];
    loop {
        if pos >= data.len() {
            return Err("lzma2: truncated".into());
        }
        let c = data[pos];
        pos += 1;
        if c == 0 {
            break;
        }
        if c == 1 || c == 2 {
            if pos + 2 > data.len() {
                return Err("lzma2: truncated".into());
            }
            let size = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize + 1;
            pos += 2;
            if pos + size > data.len() {
                return Err("lzma2: truncated".into());
            }
            let chunk = &data[pos..pos + size];
            pos += size;
            match l.as_mut() {
                Some(lz) => {
                    if c == 1 {
                        lz.dict_start = lz.out.len();
                    }
                    lz.out.extend_from_slice(chunk)
                }
                None => out_keep.extend_from_slice(chunk),
            }
            continue;
        }
        if c < 0x80 {
            return Err("lzma2: invalid control".into());
        }
        if pos + 4 > data.len() {
            return Err("lzma2: truncated".into());
        }
        let unpacked = (((c & 0x1F) as usize) << 16) + u16::from_be_bytes([data[pos], data[pos + 1]]) as usize + 1;
        let packed = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize + 1;
        pos += 4;
        let reset = (c >> 5) & 3;
        if reset >= 2 {
            let p = data[pos];
            pos += 1;
            match l.as_mut() {
                Some(lz) => lz.set_props(p)?,
                None => {
                    let mut lz = Lzma::new(p)?;
                    lz.out = std::mem::take(&mut out_keep);
                    lz.dict_start = lz.out.len();
                    l = Some(lz);
                }
            }
        } else if reset == 1 {
            if let Some(lz) = l.as_mut() {
                lz.reset_state();
            }
        }
        let lz = l.as_mut().ok_or("lzma2: missing properties")?;
        if reset == 3 {
            lz.dict_start = lz.out.len();
        }
        if pos + packed > data.len() {
            return Err("lzma2: truncated".into());
        }
        let chunk = &data[pos..pos + packed];
        let mut rc = RangeDec::new(chunk, 0)?;
        let target = lz.out.len() + unpacked;
        lz.decode(&mut rc, target, false)?;
        pos += packed;
    }
    Ok((l.map(|x| x.out).unwrap_or(out_keep), pos))
}

fn varint(d: &[u8], pos: &mut usize) -> R<u64> {
    let mut v: u64 = 0;
    for i in 0..9 {
        if *pos >= d.len() {
            return Err("xz: truncated".into());
        }
        let b = d[*pos];
        *pos += 1;
        v |= ((b & 0x7F) as u64) << (7 * i);
        if b & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err("xz: invalid varint".into())
}

pub fn xz_decode(d: &[u8]) -> R<Vec<u8>> {
    if d.len() < 12 || &d[0..6] != b"\xFD7zXZ\x00" {
        return Err("xz: invalid header".into());
    }
    let check = d[7] & 0x0F;
    let check_size = match check {
        0 => 0,
        1 => 4,
        4 => 8,
        10 => 32,
        c => 4 << ((c - 1) / 3),
    };
    let mut pos = 12;
    let mut out = vec![];
    loop {
        if pos >= d.len() {
            return Err("xz: truncated".into());
        }
        let hs = d[pos] as usize;
        if hs == 0 {
            break; // index
        }
        let header_size = (hs + 1) * 4;
        let hstart = pos;
        pos += 1;
        let flags = d[pos];
        pos += 1;
        let nfilters = (flags & 3) as usize + 1;
        if flags & 0x40 != 0 {
            varint(d, &mut pos)?;
        }
        if flags & 0x80 != 0 {
            varint(d, &mut pos)?;
        }
        let mut lzma2 = false;
        for _ in 0..nfilters {
            let id = varint(d, &mut pos)?;
            let psize = varint(d, &mut pos)? as usize;
            pos += psize;
            if id == 0x21 {
                lzma2 = true;
            } else {
                return Err(format!("xz: unsupported filter {:#x}", id));
            }
        }
        if !lzma2 {
            return Err("xz: missing lzma2 filter".into());
        }
        pos = hstart + header_size;
        if pos > d.len() {
            return Err("xz: truncated".into());
        }
        let (o, used) = lzma2_decode(&d[pos..])?;
        out.extend(o);
        pos += used;
        while pos % 4 != 0 {
            pos += 1;
        }
        pos += check_size;
    }
    Ok(out)
}

// ---------------------------------------------------------------- 7z

struct Rd<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn byte(&mut self) -> R<u8> {
        if self.p >= self.d.len() {
            return Err("7z: truncated header".into());
        }
        self.p += 1;
        Ok(self.d[self.p - 1])
    }
    fn num(&mut self) -> R<u64> {
        let first = self.byte()?;
        let mut mask = 0x80u8;
        let mut v: u64 = 0;
        for i in 0..8 {
            if first & mask == 0 {
                let high = (first & (mask.wrapping_sub(1))) as u64;
                v |= high << (8 * i);
                return Ok(v);
            }
            v |= (self.byte()? as u64) << (8 * i);
            mask >>= 1;
        }
        Ok(v)
    }
    fn bytes(&mut self, n: usize) -> R<&'a [u8]> {
        if self.p + n > self.d.len() {
            return Err("7z: truncated header".into());
        }
        self.p += n;
        Ok(&self.d[self.p - n..self.p])
    }
    fn bits(&mut self, n: usize) -> R<Vec<bool>> {
        let mut v = vec![];
        let mut b = 0u8;
        let mut mask = 0u8;
        for _ in 0..n {
            if mask == 0 {
                b = self.byte()?;
                mask = 0x80;
            }
            v.push(b & mask != 0);
            mask >>= 1;
        }
        Ok(v)
    }
    fn digests(&mut self, n: usize) -> R<()> {
        let all = self.byte()?;
        let defined = if all == 0 { self.bits(n)? } else { vec![true; n] };
        for d in defined {
            if d {
                self.bytes(4)?;
            }
        }
        Ok(())
    }
}

#[derive(Default, Clone)]
struct Folder {
    coders: Vec<(Vec<u8>, Vec<u8>)>,
    unpack_sizes: Vec<u64>,
}

#[derive(Default)]
struct Streams {
    pack_pos: u64,
    pack_sizes: Vec<u64>,
    folders: Vec<Folder>,
    num_unpack: Vec<usize>,
    sub_sizes: Vec<u64>,
}

fn read_streams(r: &mut Rd) -> R<Streams> {
    let mut s = Streams::default();
    loop {
        let id = r.byte()?;
        match id {
            0x00 => break,
            0x06 => {
                s.pack_pos = r.num()?;
                let n = r.num()? as usize;
                loop {
                    let t = r.byte()?;
                    match t {
                        0x00 => break,
                        0x09 => {
                            for _ in 0..n {
                                s.pack_sizes.push(r.num()?);
                            }
                        }
                        0x0A => r.digests(n)?,
                        _ => return Err("7z: bad pack info".into()),
                    }
                }
            }
            0x07 => {
                if r.byte()? != 0x0B {
                    return Err("7z: bad unpack info".into());
                }
                let nf = r.num()? as usize;
                if r.byte()? != 0 {
                    return Err("7z: external folders unsupported".into());
                }
                let mut outs = vec![];
                for _ in 0..nf {
                    let nc = r.num()? as usize;
                    let mut f = Folder::default();
                    let mut total_out = 0;
                    let mut total_in = 0;
                    for _ in 0..nc {
                        let flag = r.byte()?;
                        let id = r.bytes((flag & 0x0F) as usize)?.to_vec();
                        let (ni, no) = if flag & 0x10 != 0 { (r.num()? as usize, r.num()? as usize) } else { (1, 1) };
                        total_in += ni;
                        total_out += no;
                        let props = if flag & 0x20 != 0 {
                            let n = r.num()? as usize;
                            r.bytes(n)?.to_vec()
                        } else {
                            vec![]
                        };
                        f.coders.push((id, props));
                    }
                    for _ in 0..total_out.saturating_sub(1) {
                        r.num()?;
                        r.num()?;
                    }
                    let npacked = total_in - (total_out.saturating_sub(1));
                    if npacked > 1 {
                        for _ in 0..npacked {
                            r.num()?;
                        }
                    }
                    outs.push(total_out);
                    s.folders.push(f);
                }
                if r.byte()? != 0x0C {
                    return Err("7z: bad unpack info".into());
                }
                for (i, n) in outs.iter().enumerate() {
                    for _ in 0..*n {
                        let v = r.num()?;
                        s.folders[i].unpack_sizes.push(v);
                    }
                }
                loop {
                    let t = r.byte()?;
                    match t {
                        0x00 => break,
                        0x0A => r.digests(nf)?,
                        _ => return Err("7z: bad unpack info".into()),
                    }
                }
            }
            0x08 => {
                let nf = s.folders.len();
                s.num_unpack = vec![1; nf];
                let mut t = r.byte()?;
                if t == 0x0D {
                    for i in 0..nf {
                        s.num_unpack[i] = r.num()? as usize;
                    }
                    t = r.byte()?;
                }
                let has_sizes = t == 0x09;
                for i in 0..nf {
                    let n = s.num_unpack[i];
                    if n == 0 {
                        continue;
                    }
                    let total = *s.folders[i].unpack_sizes.last().unwrap_or(&0);
                    let mut sum = 0;
                    if has_sizes {
                        for _ in 0..n - 1 {
                            let v = r.num()?;
                            sum += v;
                            s.sub_sizes.push(v);
                        }
                    }
                    s.sub_sizes.push(total - sum);
                }
                if has_sizes {
                    t = r.byte()?;
                }
                while t != 0x00 {
                    if t == 0x0A {
                        let n: usize = s.num_unpack.iter().sum();
                        r.digests(n)?;
                    } else {
                        return Err("7z: bad substreams info".into());
                    }
                    t = r.byte()?;
                }
            }
            _ => return Err(format!("7z: unexpected property {:#x}", id)),
        }
    }
    if s.num_unpack.is_empty() {
        s.num_unpack = vec![1; s.folders.len()];
        s.sub_sizes = s.folders.iter().map(|f| *f.unpack_sizes.last().unwrap_or(&0)).collect();
    }
    Ok(s)
}

fn decode_folder(archive: &[u8], f: &Folder, packed: &[u8]) -> R<Vec<u8>> {
    let _ = archive;
    if f.coders.len() != 1 {
        return Err("7z: unsupported coder chain".into());
    }
    let (id, props) = &f.coders[0];
    let size = *f.unpack_sizes.last().unwrap_or(&0) as usize;
    match id.as_slice() {
        [0x00] => Ok(packed.to_vec()),
        [0x03, 0x01, 0x01] => lzma_decode(props, packed, size),
        [0x21] => lzma2_decode(packed).map(|x| x.0),
        _ => Err("7z: unsupported compression method".into()),
    }
}

fn folder_data(archive: &[u8], s: &Streams) -> R<Vec<Vec<u8>>> {
    let mut pos = 32 + s.pack_pos as usize;
    let mut out = vec![];
    for (i, f) in s.folders.iter().enumerate() {
        let sz = *s.pack_sizes.get(i).unwrap_or(&0) as usize;
        if pos + sz > archive.len() {
            return Err("7z: truncated".into());
        }
        out.push(decode_folder(archive, f, &archive[pos..pos + sz])?);
        pos += sz;
    }
    Ok(out)
}

/// Returns entries (name, is_dir, data).
pub fn sevenzip_entries(archive: &[u8]) -> R<Vec<(String, bool, Vec<u8>)>> {
    if archive.len() < 32 || &archive[0..6] != b"7z\xBC\xAF\x27\x1C" {
        return Err("7z: not a valid 7z archive".into());
    }
    let off = u64::from_le_bytes(archive[12..20].try_into().unwrap()) as usize;
    let size = u64::from_le_bytes(archive[20..28].try_into().unwrap()) as usize;
    if 32 + off + size > archive.len() {
        return Err("7z: truncated".into());
    }
    let mut header = archive[32 + off..32 + off + size].to_vec();
    loop {
        if header.first() == Some(&0x17) {
            let mut r = Rd { d: &header, p: 1 };
            let s = read_streams(&mut r)?;
            let data = folder_data(archive, &s)?;
            header = data.into_iter().next().unwrap_or_default();
            continue;
        }
        break;
    }
    let mut r = Rd { d: &header, p: 0 };
    if r.byte()? != 0x01 {
        return Err("7z: invalid header".into());
    }
    let mut streams = Streams::default();
    let mut names: Vec<String> = vec![];
    let mut empty_stream: Vec<bool> = vec![];
    let mut empty_file: Vec<bool> = vec![];
    loop {
        let id = r.byte()?;
        match id {
            0x00 => break,
            0x02 => loop {
                let t = r.byte()?;
                if t == 0 {
                    break;
                }
                let n = r.num()? as usize;
                r.bytes(n)?;
            },
            0x03 => {
                read_streams(&mut r)?;
            }
            0x04 => streams = read_streams(&mut r)?,
            0x05 => {
                let nfiles = r.num()? as usize;
                loop {
                    let t = r.byte()?;
                    if t == 0 {
                        break;
                    }
                    let sz = r.num()? as usize;
                    let body = r.bytes(sz)?;
                    let mut br = Rd { d: body, p: 0 };
                    match t {
                        0x0E => empty_stream = br.bits(nfiles)?,
                        0x0F => {
                            let n = empty_stream.iter().filter(|x| **x).count();
                            empty_file = br.bits(n)?;
                        }
                        0x11 => {
                            let ext = br.byte()?;
                            if ext != 0 {
                                return Err("7z: external names unsupported".into());
                            }
                            let rest = &body[1..];
                            let u16s: Vec<u16> = rest.chunks(2).filter(|c| c.len() == 2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                            for part in u16s.split(|c| *c == 0) {
                                if names.len() < nfiles {
                                    names.push(String::from_utf16_lossy(part));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                while names.len() < nfiles {
                    names.push(String::new());
                }
                if empty_stream.is_empty() {
                    empty_stream = vec![false; nfiles];
                }
            }
            _ => return Err(format!("7z: unexpected header property {:#x}", id)),
        }
    }
    let folders = folder_data(archive, &streams)?;
    // split folders into substreams
    let mut subs: Vec<Vec<u8>> = vec![];
    let mut si = 0;
    for (fi, data) in folders.iter().enumerate() {
        let n = *streams.num_unpack.get(fi).unwrap_or(&1);
        let mut p = 0usize;
        for _ in 0..n {
            let sz = *streams.sub_sizes.get(si).unwrap_or(&0) as usize;
            si += 1;
            let end = (p + sz).min(data.len());
            subs.push(data[p.min(end)..end].to_vec());
            p += sz;
        }
    }
    let mut out = vec![];
    let mut sub_iter = subs.into_iter();
    let mut ei = 0;
    for (i, name) in names.into_iter().enumerate() {
        if empty_stream.get(i).cloned().unwrap_or(false) {
            let is_file = empty_file.get(ei).cloned().unwrap_or(false);
            ei += 1;
            out.push((name, !is_file, vec![]));
        } else {
            out.push((name, false, sub_iter.next().unwrap_or_default()));
        }
    }
    Ok(out)
}
