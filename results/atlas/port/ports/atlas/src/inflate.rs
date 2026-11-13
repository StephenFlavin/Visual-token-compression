// A small DEFLATE (RFC 1951) decoder plus gzip/zlib wrappers, used for archive scanning.

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit: u32,
    nbits: u32,
}

impl<'a> BitReader<'a> {
    fn need(&mut self, n: u32) -> Result<(), String> {
        while self.nbits < n {
            if self.pos >= self.data.len() {
                return Err("unexpected EOF".into());
            }
            self.bit |= (self.data[self.pos] as u32) << self.nbits;
            self.pos += 1;
            self.nbits += 8;
        }
        Ok(())
    }
    fn bits(&mut self, n: u32) -> Result<u32, String> {
        if n == 0 {
            return Ok(0);
        }
        self.need(n)?;
        let v = self.bit & ((1u32 << n) - 1);
        self.bit >>= n;
        self.nbits -= n;
        Ok(v)
    }
    fn align(&mut self) {
        self.bit = 0;
        self.nbits = 0;
    }
}

struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Huffman {
        let mut counts = [0u16; 16];
        for &l in lengths {
            counts[l as usize] += 1;
        }
        counts[0] = 0;
        let mut offs = [0u16; 16];
        for i in 1..16 {
            offs[i] = offs[i - 1] + counts[i - 1];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (s, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[offs[l as usize] as usize] = s as u16;
                offs[l as usize] += 1;
            }
        }
        Huffman { counts, symbols }
    }
    fn decode(&self, br: &mut BitReader) -> Result<u16, String> {
        let mut code: i32 = 0;
        let mut first: i32 = 0;
        let mut index: i32 = 0;
        for len in 1..16 {
            code |= br.bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err("invalid huffman code".into())
    }
}

const LBASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEXT: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DBASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577,
];
const DEXT: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

/// Inflates raw deflate data. Returns (output, bytes consumed).
pub fn inflate(data: &[u8]) -> Result<(Vec<u8>, usize), String> {
    let mut br = BitReader { data, pos: 0, bit: 0, nbits: 0 };
    let mut out: Vec<u8> = Vec::new();
    loop {
        let last = br.bits(1)?;
        let typ = br.bits(2)?;
        match typ {
            0 => {
                br.align();
                if br.pos + 4 > data.len() {
                    return Err("unexpected EOF".into());
                }
                let len = u16::from_le_bytes([data[br.pos], data[br.pos + 1]]) as usize;
                br.pos += 4;
                if br.pos + len > data.len() {
                    return Err("unexpected EOF".into());
                }
                out.extend_from_slice(&data[br.pos..br.pos + len]);
                br.pos += len;
            }
            1 | 2 => {
                let (lit, dist) = if typ == 1 {
                    let mut l = [0u8; 288];
                    for (i, v) in l.iter_mut().enumerate() {
                        *v = if i < 144 { 8 } else if i < 256 { 9 } else if i < 280 { 7 } else { 8 };
                    }
                    (Huffman::new(&l), Huffman::new(&[5u8; 30]))
                } else {
                    let hlit = br.bits(5)? as usize + 257;
                    let hdist = br.bits(5)? as usize + 1;
                    let hclen = br.bits(4)? as usize + 4;
                    const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                    let mut cl = [0u8; 19];
                    for &o in ORDER.iter().take(hclen) {
                        cl[o] = br.bits(3)? as u8;
                    }
                    let clh = Huffman::new(&cl);
                    let mut lens = vec![0u8; hlit + hdist];
                    let mut i = 0;
                    while i < hlit + hdist {
                        let sym = clh.decode(&mut br)?;
                        match sym {
                            0..=15 => {
                                lens[i] = sym as u8;
                                i += 1;
                            }
                            16 => {
                                if i == 0 {
                                    return Err("invalid repeat".into());
                                }
                                let prev = lens[i - 1];
                                let r = 3 + br.bits(2)? as usize;
                                for _ in 0..r {
                                    if i < lens.len() {
                                        lens[i] = prev;
                                        i += 1;
                                    }
                                }
                            }
                            17 => {
                                i += 3 + br.bits(3)? as usize;
                            }
                            _ => {
                                i += 11 + br.bits(7)? as usize;
                            }
                        }
                    }
                    if i > hlit + hdist {
                        return Err("invalid code lengths".into());
                    }
                    (Huffman::new(&lens[..hlit]), Huffman::new(&lens[hlit..]))
                };
                loop {
                    let sym = lit.decode(&mut br)?;
                    if sym < 256 {
                        out.push(sym as u8);
                    } else if sym == 256 {
                        break;
                    } else {
                        let s = (sym - 257) as usize;
                        if s >= 29 {
                            return Err("invalid length symbol".into());
                        }
                        let len = LBASE[s] as usize + br.bits(LEXT[s] as u32)? as usize;
                        let ds = dist.decode(&mut br)? as usize;
                        if ds >= 30 {
                            return Err("invalid distance symbol".into());
                        }
                        let d = DBASE[ds] as usize + br.bits(DEXT[ds] as u32)? as usize;
                        if d > out.len() {
                            return Err("invalid distance".into());
                        }
                        let start = out.len() - d;
                        for k in 0..len {
                            let b = out[start + k];
                            out.push(b);
                        }
                    }
                }
            }
            _ => return Err("invalid block type".into()),
        }
        if last == 1 {
            break;
        }
    }
    Ok((out, br.pos))
}

/// Decompresses gzip data (supports concatenated members).
pub fn gunzip(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = vec![];
    let mut pos = 0;
    let mut first = true;
    while pos < data.len() {
        let d = &data[pos..];
        if d.len() < 10 || d[0] != 0x1f || d[1] != 0x8b || d[2] != 8 {
            if first {
                return Err("gzip: invalid header".into());
            }
            break;
        }
        first = false;
        let flg = d[3];
        let mut p = 10;
        if flg & 4 != 0 {
            if p + 2 > d.len() {
                return Err("unexpected EOF".into());
            }
            let xlen = u16::from_le_bytes([d[p], d[p + 1]]) as usize;
            p += 2 + xlen;
        }
        if flg & 8 != 0 {
            while p < d.len() && d[p] != 0 {
                p += 1;
            }
            p += 1;
        }
        if flg & 16 != 0 {
            while p < d.len() && d[p] != 0 {
                p += 1;
            }
            p += 1;
        }
        if flg & 2 != 0 {
            p += 2;
        }
        if p > d.len() {
            return Err("unexpected EOF".into());
        }
        let (o, used) = inflate(&d[p..])?;
        out.extend(o);
        pos += p + used + 8;
    }
    Ok(out)
}

pub fn zlib_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 2 || data[0] & 0x0f != 8 {
        return Err("zlib: invalid header".into());
    }
    inflate(&data[2..]).map(|(o, _)| o)
}
