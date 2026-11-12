// A compact Zstandard (RFC 8878) frame decoder used for archive scanning.

type R<T> = Result<T, String>;

struct BackBits<'a> {
    d: &'a [u8],
    pos: i64, // number of unread bits remaining
}

impl<'a> BackBits<'a> {
    fn new(d: &'a [u8]) -> R<BackBits<'a>> {
        let last = *d.last().ok_or("zstd: empty bitstream")?;
        if last == 0 {
            return Err("zstd: invalid bitstream padding".into());
        }
        let hb = 7 - last.leading_zeros() as i64;
        Ok(BackBits { d, pos: (d.len() as i64 - 1) * 8 + hb })
    }
    fn bit(&self, i: i64) -> u64 {
        if i < 0 {
            return 0;
        }
        ((self.d[(i / 8) as usize] >> (i % 8)) & 1) as u64
    }
    fn read(&mut self, n: u32) -> u64 {
        if n == 0 {
            return 0;
        }
        self.pos -= n as i64;
        let mut v = 0u64;
        for j in 0..n as i64 {
            v |= self.bit(self.pos + j) << j;
        }
        v
    }
}

#[derive(Clone, Copy, Default)]
struct FseEntry {
    sym: u16,
    nb: u8,
    base: u32,
}

#[derive(Clone, Default)]
struct Fse {
    al: u32,
    t: Vec<FseEntry>,
}

fn build_fse(norm: &[i32], al: u32) -> Fse {
    let size = 1usize << al;
    let mut t = vec![FseEntry::default(); size];
    let mut high = size as i64 - 1;
    let mut next = vec![0u32; norm.len()];
    for (s, &n) in norm.iter().enumerate() {
        if n == -1 {
            t[high as usize].sym = s as u16;
            high -= 1;
            next[s] = 1;
        } else {
            next[s] = n.max(0) as u32;
        }
    }
    let step = (size >> 1) + (size >> 3) + 3;
    let mask = size - 1;
    let mut pos = 0usize;
    for (s, &n) in norm.iter().enumerate() {
        for _ in 0..n.max(0) {
            t[pos].sym = s as u16;
            pos = (pos + step) & mask;
            while pos as i64 > high {
                pos = (pos + step) & mask;
            }
        }
    }
    for e in t.iter_mut() {
        let s = e.sym as usize;
        let ns = next[s];
        next[s] += 1;
        let nb = al - (31 - ns.leading_zeros());
        e.nb = nb as u8;
        e.base = (ns << nb) - size as u32;
    }
    Fse { al, t }
}

fn rle_fse(sym: u16) -> Fse {
    Fse { al: 0, t: vec![FseEntry { sym, nb: 0, base: 0 }] }
}

/// Reads an FSE table description; returns (table, bytes consumed).
fn read_fse(d: &[u8], max_al: u32, max_sym: usize) -> R<(Fse, usize)> {
    let mut bitpos: usize = 0;
    let get = |bp: usize, n: u32| -> u32 {
        let mut v = 0u32;
        for j in 0..n as usize {
            let i = bp + j;
            let b = if i / 8 < d.len() { (d[i / 8] >> (i % 8)) & 1 } else { 0 };
            v |= (b as u32) << j;
        }
        v
    };
    let al = get(0, 4) + 5;
    bitpos += 4;
    if al > max_al {
        return Err("zstd: accuracy log too large".into());
    }
    let mut remaining: i32 = (1 << al) + 1;
    let mut threshold: i32 = 1 << al;
    let mut nb = al + 1;
    let mut norm: Vec<i32> = vec![];
    let mut prev0 = false;
    while remaining > 1 && norm.len() <= max_sym {
        if prev0 {
            loop {
                let rep = get(bitpos, 2);
                bitpos += 2;
                for _ in 0..rep {
                    norm.push(0);
                }
                if rep != 3 {
                    break;
                }
            }
            prev0 = false;
            if norm.len() > max_sym {
                break;
            }
        }
        let max = (2 * threshold - 1) - remaining;
        let low = get(bitpos, nb - 1) as i32;
        let count;
        if low < max {
            count = low;
            bitpos += (nb - 1) as usize;
        } else {
            let mut c = get(bitpos, nb) as i32;
            if c >= threshold {
                c -= max;
            }
            count = c;
            bitpos += nb as usize;
        }
        let prob = count - 1;
        remaining -= prob.abs();
        norm.push(prob);
        prev0 = prob == 0;
        while remaining < threshold {
            nb -= 1;
            threshold >>= 1;
        }
    }
    if remaining != 1 {
        return Err("zstd: invalid FSE table".into());
    }
    Ok((build_fse(&norm, al), bitpos.div_ceil(8)))
}

#[derive(Clone, Default)]
struct Huff {
    max_bits: u32,
    t: Vec<(u8, u8)>, // (symbol, nbBits)
}

fn build_huff(weights: &[u8]) -> R<Huff> {
    let mut total: u32 = 0;
    for &w in weights {
        if w > 0 {
            total += 1 << (w - 1);
        }
    }
    if total == 0 {
        return Err("zstd: invalid huffman weights".into());
    }
    let max_bits = 32 - total.leading_zeros();
    let left = (1u32 << max_bits) - total;
    if !left.is_power_of_two() {
        return Err("zstd: invalid huffman weights".into());
    }
    let last_w = left.trailing_zeros() as u8 + 1;
    let mut w: Vec<u8> = weights.to_vec();
    w.push(last_w);
    let mut rank_count = vec![0u32; max_bits as usize + 2];
    for &x in &w {
        rank_count[x as usize] += 1;
    }
    let mut rank_start = vec![0u32; max_bits as usize + 2];
    let mut next = 0u32;
    for n in 1..=max_bits as usize {
        rank_start[n] = next;
        next += rank_count[n] << (n - 1);
    }
    let mut t = vec![(0u8, 0u8); 1 << max_bits];
    for (s, &x) in w.iter().enumerate() {
        if x == 0 {
            continue;
        }
        let len = 1u32 << (x - 1);
        let nbits = (max_bits + 1 - x as u32) as u8;
        for u in rank_start[x as usize]..rank_start[x as usize] + len {
            t[u as usize] = (s as u8, nbits);
        }
        rank_start[x as usize] += len;
    }
    Ok(Huff { max_bits, t })
}

fn read_huff(d: &[u8]) -> R<(Huff, usize)> {
    let hb = *d.first().ok_or("zstd: truncated")? as usize;
    let mut weights = vec![];
    let used;
    if hb < 128 {
        if 1 + hb > d.len() {
            return Err("zstd: truncated".into());
        }
        let src = &d[1..1 + hb];
        let (fse, n) = read_fse(src, 6, 255)?;
        let mut bits = BackBits::new(&src[n..])?;
        let mut s1 = bits.read(fse.al) as usize;
        let mut s2 = bits.read(fse.al) as usize;
        loop {
            weights.push(fse.t[s1].sym as u8);
            s1 = fse.t[s1].base as usize + bits.read(fse.t[s1].nb as u32) as usize;
            if bits.pos < 0 {
                weights.push(fse.t[s2].sym as u8);
                break;
            }
            weights.push(fse.t[s2].sym as u8);
            s2 = fse.t[s2].base as usize + bits.read(fse.t[s2].nb as u32) as usize;
            if bits.pos < 0 {
                weights.push(fse.t[s1].sym as u8);
                break;
            }
            if weights.len() > 255 {
                return Err("zstd: too many weights".into());
            }
        }
        used = 1 + hb;
    } else {
        let n = hb - 127;
        let nbytes = n.div_ceil(2);
        if 1 + nbytes > d.len() {
            return Err("zstd: truncated".into());
        }
        for i in 0..n {
            let b = d[1 + i / 2];
            weights.push(if i % 2 == 0 { b >> 4 } else { b & 0xF });
        }
        used = 1 + nbytes;
    }
    Ok((build_huff(&weights)?, used))
}

fn huff_stream(h: &Huff, src: &[u8], n: usize, out: &mut Vec<u8>) -> R<()> {
    let mut bits = BackBits::new(src)?;
    for _ in 0..n {
        // peek max_bits
        let save = bits.pos;
        let v = bits.read(h.max_bits) as usize;
        let (s, nb) = h.t[v];
        bits.pos = save - nb as i64;
        out.push(s);
    }
    Ok(())
}

const LL_BASE: [u32; 36] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048,
    4096, 8192, 16384, 32768, 65536,
];
const LL_BITS: [u32; 36] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const ML_BASE: [u32; 53] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539,
];
const ML_BITS: [u32; 53] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4,
    4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];
const LL_DEF: [i32; 36] = [4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1];
const ML_DEF: [i32; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];
const OF_DEF: [i32; 29] = [1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1];

struct FrameState {
    huff: Option<Huff>,
    ll: Option<Fse>,
    of: Option<Fse>,
    ml: Option<Fse>,
    reps: [u64; 3],
}

fn read_literals(d: &[u8], st: &mut FrameState) -> R<(Vec<u8>, usize)> {
    let b0 = *d.first().ok_or("zstd: truncated")?;
    let typ = b0 & 3;
    let sf = (b0 >> 2) & 3;
    let g = |i: usize| -> R<usize> { d.get(i).map(|x| *x as usize).ok_or_else(|| "zstd: truncated".to_string()) };
    match typ {
        0 | 1 => {
            let (size, hl) = match sf {
                0 | 2 => ((b0 >> 3) as usize, 1),
                1 => (((b0 >> 4) as usize) + (g(1)? << 4), 2),
                _ => (((b0 >> 4) as usize) + (g(1)? << 4) + (g(2)? << 12), 3),
            };
            if typ == 0 {
                if hl + size > d.len() {
                    return Err("zstd: truncated".into());
                }
                Ok((d[hl..hl + size].to_vec(), hl + size))
            } else {
                Ok((vec![g(hl)? as u8; size], hl + 1))
            }
        }
        _ => {
            let (regen, comp, hl, streams) = match sf {
                0 | 1 => {
                    let h = (b0 as usize) | (g(1)? << 8) | (g(2)? << 16);
                    ((h >> 4) & 0x3FF, (h >> 14) & 0x3FF, 3, if sf == 0 { 1 } else { 4 })
                }
                2 => {
                    let h = (b0 as usize) | (g(1)? << 8) | (g(2)? << 16) | (g(3)? << 24);
                    ((h >> 4) & 0x3FFF, (h >> 18) & 0x3FFF, 4, 4)
                }
                _ => {
                    let h = (b0 as u64) | ((g(1)? as u64) << 8) | ((g(2)? as u64) << 16) | ((g(3)? as u64) << 24) | ((g(4)? as u64) << 32);
                    (((h >> 4) & 0x3FFFF) as usize, ((h >> 22) & 0x3FFFF) as usize, 5, 4)
                }
            };
            if hl + comp > d.len() {
                return Err("zstd: truncated".into());
            }
            let mut src = &d[hl..hl + comp];
            if typ == 2 {
                let (h, used) = read_huff(src)?;
                st.huff = Some(h);
                src = &src[used..];
            }
            let h = st.huff.as_ref().ok_or("zstd: missing huffman table")?;
            let mut out = Vec::with_capacity(regen);
            if streams == 1 {
                huff_stream(h, src, regen, &mut out)?;
            } else {
                if src.len() < 6 {
                    return Err("zstd: truncated".into());
                }
                let s1 = u16::from_le_bytes([src[0], src[1]]) as usize;
                let s2 = u16::from_le_bytes([src[2], src[3]]) as usize;
                let s3 = u16::from_le_bytes([src[4], src[5]]) as usize;
                let rest = &src[6..];
                if s1 + s2 + s3 > rest.len() {
                    return Err("zstd: truncated".into());
                }
                let seg = regen.div_ceil(4);
                let parts = [&rest[..s1], &rest[s1..s1 + s2], &rest[s1 + s2..s1 + s2 + s3], &rest[s1 + s2 + s3..]];
                for (i, p) in parts.iter().enumerate() {
                    let n = if i < 3 { seg } else { regen - 3 * seg };
                    huff_stream(h, p, n, &mut out)?;
                }
            }
            Ok((out, hl + comp))
        }
    }
}

fn seq_table(mode: u8, d: &[u8], pos: &mut usize, def: &[i32], def_al: u32, max_al: u32, prev: &Option<Fse>) -> R<Fse> {
    match mode {
        0 => Ok(build_fse(def, def_al)),
        1 => {
            let s = *d.get(*pos).ok_or("zstd: truncated")?;
            *pos += 1;
            Ok(rle_fse(s as u16))
        }
        2 => {
            let (f, n) = read_fse(&d[*pos..], max_al, def.len() - 1)?;
            *pos += n;
            Ok(f)
        }
        _ => prev.clone().ok_or_else(|| "zstd: missing repeat table".to_string()),
    }
}

fn compressed_block(d: &[u8], st: &mut FrameState, out: &mut Vec<u8>) -> R<()> {
    let (lits, used) = read_literals(d, st)?;
    let mut pos = used;
    let b0 = *d.get(pos).ok_or("zstd: truncated")? as usize;
    pos += 1;
    let nseq = if b0 == 0 {
        0
    } else if b0 < 128 {
        b0
    } else if b0 < 255 {
        let b1 = *d.get(pos).ok_or("zstd: truncated")? as usize;
        pos += 1;
        ((b0 - 128) << 8) + b1
    } else {
        let b1 = *d.get(pos).ok_or("zstd: truncated")? as usize;
        let b2 = *d.get(pos + 1).ok_or("zstd: truncated")? as usize;
        pos += 2;
        b1 + (b2 << 8) + 0x7F00
    };
    if nseq == 0 {
        out.extend_from_slice(&lits);
        return Ok(());
    }
    let modes = *d.get(pos).ok_or("zstd: truncated")?;
    pos += 1;
    let ll = seq_table(modes >> 6, d, &mut pos, &LL_DEF, 6, 9, &st.ll)?;
    let of = seq_table((modes >> 4) & 3, d, &mut pos, &OF_DEF, 5, 8, &st.of)?;
    let ml = seq_table((modes >> 2) & 3, d, &mut pos, &ML_DEF, 6, 9, &st.ml)?;
    let mut bits = BackBits::new(&d[pos..])?;
    let mut sll = bits.read(ll.al) as usize;
    let mut sof = bits.read(of.al) as usize;
    let mut sml = bits.read(ml.al) as usize;
    let mut lit_pos = 0usize;
    for i in 0..nseq {
        let of_code = of.t[sof].sym as u32;
        let ml_code = ml.t[sml].sym as usize;
        let ll_code = ll.t[sll].sym as usize;
        if ml_code >= ML_BASE.len() || ll_code >= LL_BASE.len() || of_code > 31 {
            return Err("zstd: invalid sequence".into());
        }
        let of_val = (1u64 << of_code) + bits.read(of_code);
        let mlen = ML_BASE[ml_code] as usize + bits.read(ML_BITS[ml_code]) as usize;
        let llen = LL_BASE[ll_code] as usize + bits.read(LL_BITS[ll_code]) as usize;
        let offset;
        if of_val > 3 {
            offset = of_val - 3;
            st.reps = [offset, st.reps[0], st.reps[1]];
        } else {
            let idx = of_val + if llen == 0 { 1 } else { 0 };
            match idx {
                1 => offset = st.reps[0],
                2 => {
                    offset = st.reps[1];
                    st.reps = [offset, st.reps[0], st.reps[2]];
                }
                3 => {
                    offset = st.reps[2];
                    st.reps = [offset, st.reps[0], st.reps[1]];
                }
                _ => {
                    offset = st.reps[0].saturating_sub(1).max(1);
                    st.reps = [offset, st.reps[0], st.reps[1]];
                }
            }
        }
        if lit_pos + llen > lits.len() {
            return Err("zstd: literal overflow".into());
        }
        out.extend_from_slice(&lits[lit_pos..lit_pos + llen]);
        lit_pos += llen;
        let off = offset as usize;
        if off == 0 || off > out.len() {
            return Err("zstd: invalid offset".into());
        }
        let start = out.len() - off;
        for k in 0..mlen {
            let b = out[start + k];
            out.push(b);
        }
        if i + 1 < nseq {
            sll = ll.t[sll].base as usize + bits.read(ll.t[sll].nb as u32) as usize;
            sml = ml.t[sml].base as usize + bits.read(ml.t[sml].nb as u32) as usize;
            sof = of.t[sof].base as usize + bits.read(of.t[sof].nb as u32) as usize;
        }
    }
    out.extend_from_slice(&lits[lit_pos..]);
    st.ll = Some(ll);
    st.of = Some(of);
    st.ml = Some(ml);
    Ok(())
}

pub fn decode(d: &[u8]) -> R<Vec<u8>> {
    let mut out = vec![];
    let mut pos = 0;
    while pos + 4 <= d.len() {
        let magic = u32::from_le_bytes([d[pos], d[pos + 1], d[pos + 2], d[pos + 3]]);
        pos += 4;
        if magic & 0xFFFF_FFF0 == 0x184D_2A50 {
            if pos + 4 > d.len() {
                break;
            }
            let n = u32::from_le_bytes([d[pos], d[pos + 1], d[pos + 2], d[pos + 3]]) as usize;
            pos += 4 + n;
            continue;
        }
        if magic != 0xFD2F_B528 {
            return Err("zstd: invalid magic".into());
        }
        let fhd = *d.get(pos).ok_or("zstd: truncated")?;
        pos += 1;
        let single = fhd & 0x20 != 0;
        let checksum = fhd & 0x04 != 0;
        let did = [0, 1, 2, 4][(fhd & 3) as usize];
        let fcs = match fhd >> 6 {
            0 => {
                if single {
                    1
                } else {
                    0
                }
            }
            1 => 2,
            2 => 4,
            _ => 8,
        };
        if !single {
            pos += 1;
        }
        pos += did + fcs;
        let mut st = FrameState { huff: None, ll: None, of: None, ml: None, reps: [1, 4, 8] };
        let mut frame_out: Vec<u8> = vec![];
        loop {
            if pos + 3 > d.len() {
                return Err("zstd: truncated".into());
            }
            let h = d[pos] as usize | (d[pos + 1] as usize) << 8 | (d[pos + 2] as usize) << 16;
            pos += 3;
            let last = h & 1 != 0;
            let typ = (h >> 1) & 3;
            let size = h >> 3;
            match typ {
                0 => {
                    if pos + size > d.len() {
                        return Err("zstd: truncated".into());
                    }
                    frame_out.extend_from_slice(&d[pos..pos + size]);
                    pos += size;
                }
                1 => {
                    let b = *d.get(pos).ok_or("zstd: truncated")?;
                    frame_out.extend(std::iter::repeat(b).take(size));
                    pos += 1;
                }
                2 => {
                    if pos + size > d.len() {
                        return Err("zstd: truncated".into());
                    }
                    compressed_block(&d[pos..pos + size], &mut st, &mut frame_out)?;
                    pos += size;
                }
                _ => return Err("zstd: reserved block type".into()),
            }
            if last {
                break;
            }
        }
        if checksum {
            pos += 4;
        }
        out.extend(frame_out);
    }
    Ok(out)
}
