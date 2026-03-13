//! GGUF dequantization kernels — convert quantized blocks to BF16.

use crate::gguf::GgufQuantType;
use half::bf16;

/// Dequantize raw GGUF tensor bytes to a BF16 buffer.
///
/// This is the main entry point. Dispatches to per-type kernels.
/// Designed to be called with the GIL released for parallelism.
pub fn dequant_to_bf16(
    data: &[u8],
    quant_type: GgufQuantType,
    n_weights: usize,
) -> Result<Vec<bf16>, String> {
    match quant_type {
        GgufQuantType::F32 => dequant_f32(data, n_weights),
        GgufQuantType::F16 => dequant_f16(data, n_weights),
        GgufQuantType::BF16 => dequant_bf16_passthrough(data, n_weights),
        GgufQuantType::F64 => dequant_f64(data, n_weights),
        GgufQuantType::Q4_0 => dequant_q4_0(data, n_weights),
        GgufQuantType::Q4_1 => dequant_q4_1(data, n_weights),
        GgufQuantType::Q5_0 => dequant_q5_0(data, n_weights),
        GgufQuantType::Q5_1 => dequant_q5_1(data, n_weights),
        GgufQuantType::Q8_0 => dequant_q8_0(data, n_weights),
        GgufQuantType::Q8_1 => dequant_q8_1(data, n_weights),
        GgufQuantType::Q2K => dequant_q2k(data, n_weights),
        GgufQuantType::Q3K => dequant_q3k(data, n_weights),
        GgufQuantType::Q4K => dequant_q4k(data, n_weights),
        GgufQuantType::Q5K => dequant_q5k(data, n_weights),
        GgufQuantType::Q6K => dequant_q6k(data, n_weights),
        GgufQuantType::Q8K => dequant_q8k(data, n_weights),
        GgufQuantType::I8 => dequant_i8(data, n_weights),
        GgufQuantType::I16 => dequant_i16(data, n_weights),
        GgufQuantType::I32 => dequant_i32(data, n_weights),
        GgufQuantType::I64 => dequant_i64(data, n_weights),
        other => Err(format!("unsupported quant type for dequant: {}", other.name())),
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

#[inline(always)]
fn f16_to_f32(bytes: &[u8]) -> f32 {
    half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
}

#[inline(always)]
fn f32_from_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[inline(always)]
fn f64_from_le(bytes: &[u8]) -> f64 {
    f64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

#[inline(always)]
fn f32_to_bf16(v: f32) -> bf16 {
    bf16::from_f32(v)
}

fn check_size(data: &[u8], expected: usize, qt_name: &str) -> Result<(), String> {
    if data.len() < expected {
        return Err(format!(
            "{qt_name}: data too short: {} bytes, expected at least {expected}",
            data.len()
        ));
    }
    Ok(())
}

// ── Unquantized passthrough ─────────────────────────────────────────────────

fn dequant_f32(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    check_size(data, n * 4, "F32")?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        out.push(f32_to_bf16(f32_from_le(&data[off..off + 4])));
    }
    Ok(out)
}

fn dequant_f16(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    check_size(data, n * 2, "F16")?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 2;
        let f = f16_to_f32(&data[off..off + 2]);
        out.push(f32_to_bf16(f));
    }
    Ok(out)
}

fn dequant_bf16_passthrough(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    check_size(data, n * 2, "BF16")?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 2;
        out.push(bf16::from_le_bytes([data[off], data[off + 1]]));
    }
    Ok(out)
}

fn dequant_f64(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    check_size(data, n * 8, "F64")?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 8;
        out.push(f32_to_bf16(f64_from_le(&data[off..off + 8]) as f32));
    }
    Ok(out)
}

/// ── Integer types ──────────────────────────────────────────────────────────

fn dequant_i8(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    check_size(data, n, "I8")?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f32_to_bf16(data[i] as i8 as f32));
    }
    Ok(out)
}

fn dequant_i16(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    check_size(data, n * 2, "I16")?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 2;
        let v = i16::from_le_bytes([data[off], data[off + 1]]);
        out.push(f32_to_bf16(v as f32));
    }
    Ok(out)
}

fn dequant_i32(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    check_size(data, n * 4, "I32")?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        let v = i32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        out.push(f32_to_bf16(v as f32));
    }
    Ok(out)
}

fn dequant_i64(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    check_size(data, n * 8, "I64")?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 8;
        let v = i64::from_le_bytes([
            data[off], data[off + 1], data[off + 2], data[off + 3],
            data[off + 4], data[off + 5], data[off + 6], data[off + 7],
        ]);
        out.push(f32_to_bf16(v as f32));
    }
    Ok(out)
}

// ── Q8_0: 34 bytes/block, 32 weights ───────────────────────────────────────

fn dequant_q8_0(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize = 34;
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_size(data, n_blocks * TYPE_SIZE, "Q8_0")?;

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * TYPE_SIZE..];
        let scale = f16_to_f32(&block[0..2]);
        let quants = &block[2..34];
        let weights_in_block = BLOCK_SIZE.min(n - b * BLOCK_SIZE);
        for i in 0..weights_in_block {
            let q = quants[i] as i8;
            out.push(f32_to_bf16(q as f32 * scale));
        }
    }
    Ok(out)
}

// ── Q4_0: 18 bytes/block, 32 weights ───────────────────────────────────────

fn dequant_q4_0(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize = 18;
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_size(data, n_blocks * TYPE_SIZE, "Q4_0")?;

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * TYPE_SIZE..];
        let scale = f16_to_f32(&block[0..2]);
        let qs = &block[2..18];
        let weights_in_block = BLOCK_SIZE.min(n - b * BLOCK_SIZE);
        for i in 0..weights_in_block {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                (qs[byte_idx] & 0x0F) as i32
            } else {
                (qs[byte_idx] >> 4) as i32
            };
            out.push(f32_to_bf16((nibble - 8) as f32 * scale));
        }
    }
    Ok(out)
}

// ── Q4_1: 20 bytes/block, 32 weights ───────────────────────────────────────

fn dequant_q4_1(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize = 20;
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_size(data, n_blocks * TYPE_SIZE, "Q4_1")?;

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * TYPE_SIZE..];
        let scale = f16_to_f32(&block[0..2]);
        let min = f16_to_f32(&block[2..4]);
        let qs = &block[4..20];
        let weights_in_block = BLOCK_SIZE.min(n - b * BLOCK_SIZE);
        for i in 0..weights_in_block {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                (qs[byte_idx] & 0x0F) as f32
            } else {
                (qs[byte_idx] >> 4) as f32
            };
            out.push(f32_to_bf16(nibble * scale + min));
        }
    }
    Ok(out)
}

// ── Q5_0: 22 bytes/block, 32 weights ───────────────────────────────────────

fn dequant_q5_0(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize = 22;
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_size(data, n_blocks * TYPE_SIZE, "Q5_0")?;

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * TYPE_SIZE..];
        let scale = f16_to_f32(&block[0..2]);
        let qh = &block[2..6]; // 4 bytes = 32 high bits
        let qs = &block[6..22]; // 16 bytes = 32 nibbles
        let weights_in_block = BLOCK_SIZE.min(n - b * BLOCK_SIZE);
        for i in 0..weights_in_block {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                (qs[byte_idx] & 0x0F) as u32
            } else {
                (qs[byte_idx] >> 4) as u32
            };
            let hbit = ((qh[i / 8] >> (i % 8)) & 1) as u32;
            let val = nibble | (hbit << 4);
            out.push(f32_to_bf16((val as f32 - 16.0) * scale));
        }
    }
    Ok(out)
}

// ── Q5_1: 24 bytes/block, 32 weights ───────────────────────────────────────

fn dequant_q5_1(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize = 24;
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_size(data, n_blocks * TYPE_SIZE, "Q5_1")?;

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * TYPE_SIZE..];
        let scale = f16_to_f32(&block[0..2]);
        let min = f16_to_f32(&block[2..4]);
        let qh = &block[4..8]; // 4 bytes = 32 high bits
        let qs = &block[8..24]; // 16 bytes
        let weights_in_block = BLOCK_SIZE.min(n - b * BLOCK_SIZE);
        for i in 0..weights_in_block {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                (qs[byte_idx] & 0x0F) as u32
            } else {
                (qs[byte_idx] >> 4) as u32
            };
            let hbit = ((qh[i / 8] >> (i % 8)) & 1) as u32;
            let val = nibble | (hbit << 4);
            out.push(f32_to_bf16(val as f32 * scale + min));
        }
    }
    Ok(out)
}

// ── Q8_1: 36 bytes/block, 32 weights ───────────────────────────────────────
// Layout: f32 d (4 bytes) + f32 s (4 bytes, sum — unused for dequant) + i8 qs[32]
// Wait — checking ggml source more carefully:
// In ggml, block_q8_1 has: half d, half s, int8_t qs[QK8_1]
// where QK8_1=32. So: 2 + 2 + 32 = 36 bytes.
// But some ggml versions use float d, float s => 4+4+32=40.
// The type_size in our GgufQuantType is 36, so we follow that.
// Actually the ggml source I checked has:
//   typedef struct { ggml_half d; ggml_half s; int8_t qs[QK8_1]; } block_q8_1;
// So d and s are both f16 (half), total = 2+2+32 = 36.

fn dequant_q8_1(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize = 36;
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_size(data, n_blocks * TYPE_SIZE, "Q8_1")?;

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * TYPE_SIZE..];
        let d = f16_to_f32(&block[0..2]);
        // block[2..4] is 's' (sum), not needed for dequant
        let quants = &block[4..36];
        let weights_in_block = BLOCK_SIZE.min(n - b * BLOCK_SIZE);
        for i in 0..weights_in_block {
            let q = quants[i] as i8;
            out.push(f32_to_bf16(q as f32 * d));
        }
    }
    Ok(out)
}

// ── Q2_K: 84 bytes/super-block, 256 weights ────────────────────────────────
// struct block_q2_K {
//     uint8_t scales[16];  // 4-bit quantized scales and mins
//     uint8_t qs[64];      // 2-bit quants
//     ggml_half d;         // super-block scale
//     ggml_half dmin;      // super-block min scale
// };

fn dequant_q2k(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    const BLOCK_SIZE: usize = 256;
    const TYPE_SIZE: usize = 84;
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_size(data, n_blocks * TYPE_SIZE, "Q2_K")?;

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * TYPE_SIZE..];
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = f16_to_f32(&block[80..82]);
        let dmin = f16_to_f32(&block[82..84]);

        let weights_in_block = BLOCK_SIZE.min(n - b * BLOCK_SIZE);
        for i in 0..weights_in_block {
            let sub_block = i / 16;
            let sc = (scales[sub_block] & 0x0F) as f32;
            let mn = (scales[sub_block] >> 4) as f32;
            let q = ((qs[i / 4] >> (2 * (i % 4))) & 3) as f32;
            let w = q * (d * sc) - dmin * mn;
            out.push(f32_to_bf16(w));
        }
    }
    Ok(out)
}

// ── Q3_K: 110 bytes/super-block, 256 weights ───────────────────────────────
// struct block_q3_K {
//     uint8_t hmask[32];   // high bit of each weight
//     uint8_t qs[64];      // low 2 bits of each weight
//     uint8_t scales[12];  // 6-bit packed scales
//     ggml_half d;
// };

fn unpack_q3k_scales(scales_raw: &[u8]) -> [f32; 16] {
    // 12 bytes encode 16 scale values of 6 bits each.
    // Following the ggml dequantize_row_q3_K algorithm using u32 manipulation.
    let a0 = u32::from_le_bytes([scales_raw[0], scales_raw[1], scales_raw[2], scales_raw[3]]);
    let a1 = u32::from_le_bytes([scales_raw[4], scales_raw[5], scales_raw[6], scales_raw[7]]);
    let tmp = u32::from_le_bytes([scales_raw[8], scales_raw[9], scales_raw[10], scales_raw[11]]);

    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK1: u32 = 0x03030303;

    let r0 = (a0 & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
    let r1 = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let r2 = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    let r3 = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);

    let r0b = r0.to_le_bytes();
    let r1b = r1.to_le_bytes();
    let r2b = r2.to_le_bytes();
    let r3b = r3.to_le_bytes();

    let mut result = [0f32; 16];
    for j in 0..4 {
        result[j]      = r0b[j] as f32;
        result[j + 4]  = r1b[j] as f32;
        result[j + 8]  = r2b[j] as f32;
        result[j + 12] = r3b[j] as f32;
    }
    result
}

fn dequant_q3k(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    const BLOCK_SIZE: usize = 256;
    const TYPE_SIZE: usize = 110;
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_size(data, n_blocks * TYPE_SIZE, "Q3_K")?;

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * TYPE_SIZE..];
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales_raw = &block[96..108];
        let d = f16_to_f32(&block[108..110]);

        let unpacked_scales = unpack_q3k_scales(scales_raw);

        let weights_in_block = BLOCK_SIZE.min(n - b * BLOCK_SIZE);
        for i in 0..weights_in_block {
            let sub_block = i / 16;
            let low2 = ((qs[i / 4] >> (2 * (i % 4))) & 3) as i32;
            let hbit = ((hmask[i / 8] >> (i % 8)) & 1) as i32;
            let val3 = low2 | (hbit << 2); // 3-bit value 0..7
            let dl = d * (unpacked_scales[sub_block] - 32.0);
            let w = (val3 - 4) as f32 * dl;
            out.push(f32_to_bf16(w));
        }
    }
    Ok(out)
}

// ── K-quant scale unpacking (shared by Q4_K and Q5_K) ──────────────────────
// 12 bytes encode 8 scale + 8 min values, each 6 bits.
// Bytes 0..3: low 4 bits of scales 0..7 (nibbles)
// Bytes 4..7: low 4 bits of mins 0..7 (nibbles)
// Bytes 8..11: high 2 bits of scales[0..3]+mins[0..3] in byte 8,9
//              and scales[4..7]+mins[4..7] in byte 10,11.
//
// Actually the ggml layout for Q4_K/Q5_K scales:
// From ggml source dequantize_row_q4_K:
//   const uint8_t * sc = x[i].scales;
//   For j in 0..QK_K/64 (=4 sub-blocks of 64 weights each):
//     For 2 halves (32 weights each):
//       uint8_t sc_raw, m_raw  -- extracted from packed bytes
//
// Let me follow the actual ggml code pattern more carefully:
// x.scales[0..11] packs 8 (scale, min) pairs:
//   sc[0] = scales_raw[0] & 0x3F           (6 bits)
//   mn[0] = scales_raw[0+4] & 0x3F         (6 bits)
//   sc[1] = scales_raw[1] & 0x3F
//   mn[1] = scales_raw[1+4] & 0x3F
//   ...
// Wait, the actual layout from ggml get_scale_min_k4:
//   j < 4: sc = d[j] & 63, m = d[j+4] & 63
//   j >= 4: sc = (d[j+4] & 0xF) | ((d[j-4] >> 6) << 4)
//           m  = (d[j+4] >> 4)  | ((d[j-0] >> 6) << 4)
// Where d = x.scales (the 12-byte array), j = sub-block index (0..7)

fn unpack_q4k_scales_mins(scales_raw: &[u8]) -> ([f32; 8], [f32; 8]) {
    let d = scales_raw;
    let mut sc = [0f32; 8];
    let mut mn = [0f32; 8];

    for j in 0..4 {
        sc[j] = (d[j] & 63) as f32;
        mn[j] = (d[j + 4] & 63) as f32;
    }
    for j in 4..8 {
        sc[j] = ((d[j + 4] & 0x0F) | ((d[j - 4] >> 6) << 4)) as f32;
        mn[j] = ((d[j + 4] >> 4) | ((d[j] >> 6) << 4)) as f32;
    }
    (sc, mn)
}

// ── Q4_K: 144 bytes/super-block, 256 weights ───────────────────────────────
// struct block_q4_K {
//     ggml_half d;          // super-block scale
//     ggml_half dmin;       // super-block min
//     uint8_t scales[12];   // packed scales and mins for 8 sub-blocks
//     uint8_t qs[128];      // 4-bit quants
// };

fn dequant_q4k(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    const BLOCK_SIZE: usize = 256;
    const TYPE_SIZE: usize = 144;
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_size(data, n_blocks * TYPE_SIZE, "Q4_K")?;

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * TYPE_SIZE..];
        let d = f16_to_f32(&block[0..2]);
        let dmin = f16_to_f32(&block[2..4]);
        let scales_raw = &block[4..16];
        let qs = &block[16..144];

        let (sc, mn) = unpack_q4k_scales_mins(scales_raw);

        let weights_in_block = BLOCK_SIZE.min(n - b * BLOCK_SIZE);
        for i in 0..weights_in_block {
            let sub_block = i / 32; // 8 sub-blocks of 32 weights
            let nibble = if i % 2 == 0 {
                (qs[i / 2] & 0x0F) as f32
            } else {
                (qs[i / 2] >> 4) as f32
            };
            let w = nibble * (d * sc[sub_block]) - dmin * mn[sub_block];
            out.push(f32_to_bf16(w));
        }
    }
    Ok(out)
}

// ── Q5_K: 176 bytes/super-block, 256 weights ───────────────────────────────
// struct block_q5_K {
//     ggml_half d;
//     ggml_half dmin;
//     uint8_t scales[12];
//     uint8_t qh[32];       // 5th bit of each weight
//     uint8_t qs[128];      // low 4 bits
// };

fn dequant_q5k(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    const BLOCK_SIZE: usize = 256;
    const TYPE_SIZE: usize = 176;
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_size(data, n_blocks * TYPE_SIZE, "Q5_K")?;

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * TYPE_SIZE..];
        let d = f16_to_f32(&block[0..2]);
        let dmin = f16_to_f32(&block[2..4]);
        let scales_raw = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        let (sc, mn) = unpack_q4k_scales_mins(scales_raw);

        let weights_in_block = BLOCK_SIZE.min(n - b * BLOCK_SIZE);
        for i in 0..weights_in_block {
            let sub_block = i / 32;
            let nibble = if i % 2 == 0 {
                (qs[i / 2] & 0x0F) as u32
            } else {
                (qs[i / 2] >> 4) as u32
            };
            let hbit = ((qh[i / 8] >> (i % 8)) & 1) as u32;
            let val5 = nibble | (hbit << 4);
            let w = val5 as f32 * (d * sc[sub_block]) - dmin * mn[sub_block];
            out.push(f32_to_bf16(w));
        }
    }
    Ok(out)
}

// ── Q6_K: 210 bytes/super-block, 256 weights ───────────────────────────────
// struct block_q6_K {
//     uint8_t ql[128];     // low 4 bits of each weight
//     uint8_t qh[64];      // high 2 bits of each weight
//     int8_t scales[16];   // 8-bit scales for 16 sub-blocks of 16
//     ggml_half d;
// };

fn dequant_q6k(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    const BLOCK_SIZE: usize = 256;
    const TYPE_SIZE: usize = 210;
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_size(data, n_blocks * TYPE_SIZE, "Q6_K")?;

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * TYPE_SIZE..];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f32(&block[208..210]);

        let weights_in_block = BLOCK_SIZE.min(n - b * BLOCK_SIZE);
        // Temporary buffer for the full 256 weights
        let mut y = [0f32; 256];

        // Process two halves of 128 weights each
        for half in 0..2u32 {
            let ql_half = &ql[(half as usize * 64)..((half as usize + 1) * 64)];
            let qh_half = &qh[(half as usize * 32)..((half as usize + 1) * 32)];
            let sc = &scales[(half as usize * 8)..((half as usize + 1) * 8)];
            let base = half as usize * 128;

            for l in 0..32 {
                let is = l / 16; // 0 or 1, picks sub-sub-block
                let q1 = (ql_half[l] & 0xF) as u32 | ((((qh_half[l] >> 0) & 3) as u32) << 4);
                let q2 = (ql_half[l + 32] & 0xF) as u32 | ((((qh_half[l] >> 2) & 3) as u32) << 4);
                let q3 = (ql_half[l] >> 4) as u32 | ((((qh_half[l] >> 4) & 3) as u32) << 4);
                let q4 = (ql_half[l + 32] >> 4) as u32 | ((((qh_half[l] >> 6) & 3) as u32) << 4);

                y[base + l]      = d * (sc[is] as i8 as f32) * (q1 as i32 - 32) as f32;
                y[base + l + 32] = d * (sc[is + 2] as i8 as f32) * (q2 as i32 - 32) as f32;
                y[base + l + 64] = d * (sc[is + 4] as i8 as f32) * (q3 as i32 - 32) as f32;
                y[base + l + 96] = d * (sc[is + 6] as i8 as f32) * (q4 as i32 - 32) as f32;
            }
        }

        for i in 0..weights_in_block {
            out.push(f32_to_bf16(y[i]));
        }
    }
    Ok(out)
}

// ── Q8_K: 292 bytes/super-block, 256 weights ───────────────────────────────
// struct block_q8_K {
//     float d;              // super-block scale
//     int8_t qs[256];       // quants
//     int16_t bsums[16];   // not needed for dequant
// };

fn dequant_q8k(data: &[u8], n: usize) -> Result<Vec<bf16>, String> {
    const BLOCK_SIZE: usize = 256;
    const TYPE_SIZE: usize = 292;
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_size(data, n_blocks * TYPE_SIZE, "Q8_K")?;

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * TYPE_SIZE..];
        let d = f32_from_le(&block[0..4]);
        let quants = &block[4..260];
        // block[260..292] = bsums, not needed
        let weights_in_block = BLOCK_SIZE.min(n - b * BLOCK_SIZE);
        for i in 0..weights_in_block {
            let q = quants[i] as i8;
            out.push(f32_to_bf16(q as f32 * d));
        }
    }
    Ok(out)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "values differ: {a} vs {b} (tol={tol})"
        );
    }

    fn bf16_vec_to_f32(v: &[bf16]) -> Vec<f32> {
        v.iter().map(|x| x.to_f32()).collect()
    }

    // ── F32 passthrough ─────────────────────────────────────────────────────

    #[test]
    fn f32_dequant() {
        let values: Vec<f32> = vec![1.0, -2.5, 0.0, 3.14];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let result = dequant_to_bf16(&data, GgufQuantType::F32, 4).unwrap();
        let out = bf16_vec_to_f32(&result);
        assert_eq!(out.len(), 4);
        assert_close(out[0], 1.0, 0.01);
        assert_close(out[1], -2.5, 0.01);
        assert_close(out[2], 0.0, 0.01);
        assert_close(out[3], 3.14, 0.02);
    }

    // ── F16 passthrough ─────────────────────────────────────────────────────

    #[test]
    fn f16_dequant() {
        let values: Vec<f32> = vec![1.0, -0.5];
        let mut data = Vec::new();
        for v in &values {
            let h = half::f16::from_f32(*v);
            data.extend_from_slice(&h.to_le_bytes());
        }
        let result = dequant_to_bf16(&data, GgufQuantType::F16, 2).unwrap();
        let out = bf16_vec_to_f32(&result);
        assert_close(out[0], 1.0, 0.01);
        assert_close(out[1], -0.5, 0.01);
    }

    // ── BF16 passthrough ────────────────────────────────────────────────────

    #[test]
    fn bf16_passthrough() {
        let values: Vec<bf16> = vec![bf16::from_f32(2.0), bf16::from_f32(-1.0)];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let result = dequant_to_bf16(&data, GgufQuantType::BF16, 2).unwrap();
        assert_eq!(result[0].to_f32(), 2.0);
        assert_eq!(result[1].to_f32(), -1.0);
    }

    // ── Q8_0 ────────────────────────────────────────────────────────────────

    #[test]
    fn q8_0_dequant() {
        // Build one Q8_0 block: scale=0.5 (f16), 32 quants
        let scale = half::f16::from_f32(0.5);
        let mut block = Vec::new();
        block.extend_from_slice(&scale.to_le_bytes());
        // quants: [1, 2, -1, -2, 0, ...zeros...]
        let mut quants = vec![0i8; 32];
        quants[0] = 1;
        quants[1] = 2;
        quants[2] = -1;
        quants[3] = -2;
        for q in &quants {
            block.push(*q as u8);
        }
        assert_eq!(block.len(), 34);

        let result = dequant_to_bf16(&block, GgufQuantType::Q8_0, 32).unwrap();
        let out = bf16_vec_to_f32(&result);
        assert_eq!(out.len(), 32);
        assert_close(out[0], 0.5, 0.01);   // 1 * 0.5
        assert_close(out[1], 1.0, 0.01);   // 2 * 0.5
        assert_close(out[2], -0.5, 0.01);  // -1 * 0.5
        assert_close(out[3], -1.0, 0.01);  // -2 * 0.5
        assert_close(out[4], 0.0, 0.01);   // 0 * 0.5
    }

    // ── Q4_0 ────────────────────────────────────────────────────────────────

    #[test]
    fn q4_0_dequant() {
        // One Q4_0 block: scale=1.0, 16 bytes of nibbles
        let scale = half::f16::from_f32(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale.to_le_bytes());
        // Each byte encodes 2 weights: low nibble first, then high nibble
        // nibble value 8 => weight = (8-8)*1.0 = 0
        // nibble value 0 => weight = (0-8)*1.0 = -8
        // nibble value 15 => weight = (15-8)*1.0 = 7
        let mut qs = vec![0u8; 16];
        qs[0] = 0x80; // low=0 (w=-8), high=8 (w=0)
        qs[1] = 0xF1; // low=1 (w=-7), high=15 (w=7)
        block.extend_from_slice(&qs);
        assert_eq!(block.len(), 18);

        let result = dequant_to_bf16(&block, GgufQuantType::Q4_0, 32).unwrap();
        let out = bf16_vec_to_f32(&result);
        assert_close(out[0], -8.0, 0.1);  // nibble 0 - 8 = -8
        assert_close(out[1], 0.0, 0.1);   // nibble 8 - 8 = 0
        assert_close(out[2], -7.0, 0.1);  // nibble 1 - 8 = -7
        assert_close(out[3], 7.0, 0.1);   // nibble 15 - 8 = 7
    }

    // ── Q4_1 ────────────────────────────────────────────────────────────────

    #[test]
    fn q4_1_dequant() {
        let scale = half::f16::from_f32(2.0);
        let min = half::f16::from_f32(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale.to_le_bytes());
        block.extend_from_slice(&min.to_le_bytes());
        // nibble value 0 => weight = 0*2.0 + 1.0 = 1.0
        // nibble value 3 => weight = 3*2.0 + 1.0 = 7.0
        let mut qs = vec![0u8; 16];
        qs[0] = 0x30; // low=0, high=3
        block.extend_from_slice(&qs);
        assert_eq!(block.len(), 20);

        let result = dequant_to_bf16(&block, GgufQuantType::Q4_1, 32).unwrap();
        let out = bf16_vec_to_f32(&result);
        assert_close(out[0], 1.0, 0.1);  // 0*2 + 1
        assert_close(out[1], 7.0, 0.1);  // 3*2 + 1
    }

    // ── Q5_0 ────────────────────────────────────────────────────────────────

    #[test]
    fn q5_0_dequant() {
        let scale = half::f16::from_f32(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale.to_le_bytes());
        // qh: 4 bytes (32 high bits). Set bit 0 = 1, others 0.
        let mut qh = [0u8; 4];
        qh[0] = 0x01; // bit 0 set
        block.extend_from_slice(&qh);
        // qs: 16 bytes of nibble pairs
        let mut qs = vec![0u8; 16];
        qs[0] = 0x83; // low nibble=3, high nibble=8
        block.extend_from_slice(&qs);
        assert_eq!(block.len(), 22);

        let result = dequant_to_bf16(&block, GgufQuantType::Q5_0, 32).unwrap();
        let out = bf16_vec_to_f32(&result);
        // Weight 0: nibble=3, hbit=1 => val=3|16=19, w=(19-16)*1.0 = 3.0
        assert_close(out[0], 3.0, 0.1);
        // Weight 1: nibble=8, hbit=0 => val=8, w=(8-16)*1.0 = -8.0
        assert_close(out[1], -8.0, 0.1);
    }

    // ── Q5_1 ────────────────────────────────────────────────────────────────

    #[test]
    fn q5_1_dequant() {
        let scale = half::f16::from_f32(1.0);
        let min = half::f16::from_f32(0.5);
        let mut block = Vec::new();
        block.extend_from_slice(&scale.to_le_bytes());
        block.extend_from_slice(&min.to_le_bytes());
        let mut qh = [0u8; 4];
        qh[0] = 0x01; // bit 0 set
        block.extend_from_slice(&qh);
        let mut qs = vec![0u8; 16];
        qs[0] = 0x02; // low=2, high=0
        block.extend_from_slice(&qs);
        assert_eq!(block.len(), 24);

        let result = dequant_to_bf16(&block, GgufQuantType::Q5_1, 32).unwrap();
        let out = bf16_vec_to_f32(&result);
        // Weight 0: nibble=2, hbit=1 => val=2|16=18, w=18*1.0 + 0.5 = 18.5
        assert_close(out[0], 18.5, 0.1);
        // Weight 1: nibble=0, hbit=0 => val=0, w=0*1.0 + 0.5 = 0.5
        assert_close(out[1], 0.5, 0.1);
    }

    // ── Q8_1 ────────────────────────────────────────────────────────────────

    #[test]
    fn q8_1_dequant() {
        let d = half::f16::from_f32(0.25);
        let s = half::f16::from_f32(0.0); // sum, unused
        let mut block = Vec::new();
        block.extend_from_slice(&d.to_le_bytes());
        block.extend_from_slice(&s.to_le_bytes());
        let mut quants = vec![0i8; 32];
        quants[0] = 4;
        quants[1] = -8;
        for q in &quants {
            block.push(*q as u8);
        }
        assert_eq!(block.len(), 36);

        let result = dequant_to_bf16(&block, GgufQuantType::Q8_1, 32).unwrap();
        let out = bf16_vec_to_f32(&result);
        assert_close(out[0], 1.0, 0.01);   // 4 * 0.25
        assert_close(out[1], -2.0, 0.01);  // -8 * 0.25
    }

    // ── Q2_K ────────────────────────────────────────────────────────────────

    #[test]
    fn q2k_dequant_basic() {
        // Build a Q2_K block: 84 bytes
        let mut block = vec![0u8; 84];
        // scales[0..16]: each byte has low nibble = sc, high nibble = mn
        // Set sub-block 0: sc=2, mn=1 => scales[0] = 0x12
        block[0] = 0x12;
        // qs[16..80]: 2 bits per weight
        // Set weight 0 to quant value 3: qs[16] low 2 bits = 3
        block[16] = 0x03;
        // d at offset 80..82
        let d = half::f16::from_f32(1.0);
        block[80..82].copy_from_slice(&d.to_le_bytes());
        // dmin at offset 82..84
        let dmin = half::f16::from_f32(0.5);
        block[82..84].copy_from_slice(&dmin.to_le_bytes());

        let result = dequant_to_bf16(&block, GgufQuantType::Q2K, 256).unwrap();
        let out = bf16_vec_to_f32(&result);
        assert_eq!(out.len(), 256);
        // Weight 0: q=3, sc=2, mn=1, d=1.0, dmin=0.5
        // w = 3 * (1.0 * 2) - 0.5 * 1 = 6.0 - 0.5 = 5.5
        assert_close(out[0], 5.5, 0.1);
    }

    // ── Q4_K ────────────────────────────────────────────────────────────────

    #[test]
    fn q4k_dequant_basic() {
        let mut block = vec![0u8; 144];
        let d = half::f16::from_f32(1.0);
        let dmin = half::f16::from_f32(0.0);
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());
        // scales[4..16]: set sub-block 0 scale = 2 (6-bit)
        // For j<4: sc[j] = d[j] & 63. So scales_raw[0] = 2.
        block[4] = 2;
        // qs[16..144]: nibbles. Set weight 0 nibble = 5.
        block[16] = 0x05; // low nibble = 5, high nibble = 0
        // dmin=0, so w = nibble * (d * sc) = 5 * (1.0 * 2) = 10.0

        let result = dequant_to_bf16(&block, GgufQuantType::Q4K, 256).unwrap();
        let out = bf16_vec_to_f32(&result);
        assert_close(out[0], 10.0, 0.2);
    }

    // ── Q6_K ────────────────────────────────────────────────────────────────

    #[test]
    fn q6k_dequant_basic() {
        let mut block = vec![0u8; 210];
        // ql[0..128]: low 4 bits. Set weight 0 low4 = 10.
        block[0] = 0x0A; // low nibble = 10
        // qh[128..192]: high 2 bits. Set weight 0 high2 = 1.
        block[128] = 0x01; // bits 0..1 = 1
        // scales[192..208]: int8 scales. Set sub-block 0 = 2.
        block[192] = 2;
        // d at [208..210] = 0.5
        let d = half::f16::from_f32(0.5);
        block[208..210].copy_from_slice(&d.to_le_bytes());

        let result = dequant_to_bf16(&block, GgufQuantType::Q6K, 256).unwrap();
        let out = bf16_vec_to_f32(&result);
        // Weight 0: low4=10, high2=1, val6=10|(1<<4)=26
        // sc=2, d=0.5
        // w = (26-32) * 0.5 * 2 = -6 * 1.0 = -6.0
        assert_close(out[0], -6.0, 0.2);
    }

    // ── Q8_K ────────────────────────────────────────────────────────────────

    #[test]
    fn q8k_dequant_basic() {
        let mut block = vec![0u8; 292];
        // d at [0..4] = 0.1 (f32)
        let d: f32 = 0.1;
        block[0..4].copy_from_slice(&d.to_le_bytes());
        // qs[4..260]: set weight 0 = 10 (as i8)
        block[4] = 10u8;
        // bsums[260..292]: ignored

        let result = dequant_to_bf16(&block, GgufQuantType::Q8K, 256).unwrap();
        let out = bf16_vec_to_f32(&result);
        // Weight 0: 10 * 0.1 = 1.0
        assert_close(out[0], 1.0, 0.02);
    }

    // ── Error cases ─────────────────────────────────────────────────────────

    #[test]
    fn data_too_short_errors() {
        let result = dequant_to_bf16(&[0u8; 10], GgufQuantType::Q8_0, 32);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));
    }

    #[test]
    fn unsupported_type_errors() {
        let result = dequant_to_bf16(&[], GgufQuantType::IQ2XXS, 256);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unsupported"));
    }

    // ── Multi-block Q8_0 ────────────────────────────────────────────────────

    #[test]
    fn q8_0_two_blocks() {
        let scale1 = half::f16::from_f32(1.0);
        let scale2 = half::f16::from_f32(2.0);
        let mut data = Vec::new();
        // Block 1
        data.extend_from_slice(&scale1.to_le_bytes());
        let mut q1 = vec![0i8; 32];
        q1[0] = 3;
        for q in &q1 {
            data.push(*q as u8);
        }
        // Block 2
        data.extend_from_slice(&scale2.to_le_bytes());
        let mut q2 = vec![0i8; 32];
        q2[0] = 5;
        for q in &q2 {
            data.push(*q as u8);
        }

        let result = dequant_to_bf16(&data, GgufQuantType::Q8_0, 64).unwrap();
        let out = bf16_vec_to_f32(&result);
        assert_eq!(out.len(), 64);
        assert_close(out[0], 3.0, 0.01);   // 3 * 1.0
        assert_close(out[32], 10.0, 0.1);  // 5 * 2.0
    }

    // ── Q3_K basic ──────────────────────────────────────────────────────────

    #[test]
    fn q3k_dequant_basic() {
        let mut block = vec![0u8; 110];
        // hmask[0..32]: set bit 0 of hmask[0] = 1
        block[0] = 0x01;
        // qs[32..96]: set weight 0 low2 = 2 => qs[32] low 2 bits = 2
        block[32] = 0x02;
        // scales[96..108]: all zero => after unpack all scales = -32
        // Actually need to set scales so they're nonzero.
        // For Q3_K, scales are encoded as (value + 32). So to get scale=1,
        // we need encoded value = 33.
        // Low 4 bits of scale[0] = 33 & 0xF = 1
        // High 2 bits of scale[0] = (33 >> 4) & 3 = 2
        // scales_raw[0] low nibble: scale[0] low4 = 1 => scales_raw[0] = 0x01
        // scales_raw[8] low 2 bits: scale[0] high2 = 2 => scales_raw[8] = 0x02
        block[96] = 0x01; // low4 of scale[0] = 1
        block[104] = 0x02; // high2 of scale[0] = 2
        // unpacked: low4(1) + high2(2)*16 = 1+32 = 33, minus 32 = 1.0

        // d at [108..110] = 0.5
        let d = half::f16::from_f32(0.5);
        block[108..110].copy_from_slice(&d.to_le_bytes());

        let result = dequant_to_bf16(&block, GgufQuantType::Q3K, 256).unwrap();
        let out = bf16_vec_to_f32(&result);
        // Weight 0: low2=2, hbit=1, val3 = 2|(1<<2) = 6
        // w = (6-4) * 0.5 * 1.0 = 2 * 0.5 = 1.0
        assert_close(out[0], 1.0, 0.1);
    }

    // ── Q5_K basic ──────────────────────────────────────────────────────────

    #[test]
    fn q5k_dequant_basic() {
        let mut block = vec![0u8; 176];
        let d = half::f16::from_f32(1.0);
        let dmin = half::f16::from_f32(0.0);
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());
        // scales[4..16]: sub-block 0 scale = 1
        block[4] = 1;
        // qh[16..48]: set bit 0 = 1
        block[16] = 0x01;
        // qs[48..176]: nibble 0 = 3
        block[48] = 0x03;

        let result = dequant_to_bf16(&block, GgufQuantType::Q5K, 256).unwrap();
        let out = bf16_vec_to_f32(&result);
        // Weight 0: nibble=3, hbit=1, val5=3|16=19
        // w = 19 * (1.0 * 1) - 0 = 19.0
        assert_close(out[0], 19.0, 0.5);
    }
}
