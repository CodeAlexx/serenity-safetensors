//! GGUF format parser — reads tensor index and metadata from llama.cpp GGUF files.

use crate::format_detect::ModelFormat;
use crate::probe::ModelInfo;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

// ── GGUF magic ──────────────────────────────────────────────────────────────

const GGUF_MAGIC_LE: u32 = 0x46554747; // "GGUF" as little-endian u32

// ── Quantization types ──────────────────────────────────────────────────────

/// GGUF quantization types (from llama.cpp ggml-common.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgufQuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4, 5 removed (old Q4_2 / Q4_3)
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
}

impl GgufQuantType {
    pub fn from_u32(v: u32) -> Result<Self, String> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::IQ2XXS),
            17 => Ok(Self::IQ2XS),
            18 => Ok(Self::IQ3XXS),
            19 => Ok(Self::IQ1S),
            20 => Ok(Self::IQ4NL),
            21 => Ok(Self::IQ3S),
            22 => Ok(Self::IQ2S),
            23 => Ok(Self::IQ4XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1M),
            30 => Ok(Self::BF16),
            other => Err(format!("unknown GGUF quant type: {other}")),
        }
    }

    /// Block size in weights (number of weights per quant block).
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::IQ2XXS | Self::IQ2XS | Self::IQ2S | Self::IQ3XXS | Self::IQ3S => 256,
            Self::IQ4NL | Self::IQ4XS | Self::IQ1S | Self::IQ1M => 256,
        }
    }

    /// Bytes per block of `block_size()` weights.
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,  // 2 (f16 scale) + 16 (32 nibbles)
            Self::Q4_1 => 20,  // 2 (f16 scale) + 2 (f16 min) + 16
            Self::Q5_0 => 22,  // 2 (f16 scale) + 4 (hmask) + 16
            Self::Q5_1 => 24,  // 2 (f16 scale) + 2 (f16 min) + 4 (hmask) + 16
            Self::Q8_0 => 34,  // 2 (f16 scale) + 32 (i8 quants)
            Self::Q8_1 => 36,  // half d(2) + half s(2) + i8 qs[32]
            Self::Q2K => 84,  // scales(16) + qs(64) + d(2) + dmin(2)
            Self::Q3K => 110, // hmask(32) + qs(64) + scales(12) + d(2)
            Self::Q4K => 144, // d(2) + dmin(2) + scales(12) + qs(128)
            Self::Q5K => 176, // d(2) + dmin(2) + scales(12) + qh(32) + qs(128)
            Self::Q6K => 210, // ql(128) + qh(64) + scales(16) + d(2)
            Self::Q8K => 292, // d(4) + qs(256) + bsums(32)
            // IQ types: unsupported for now, return 0
            Self::IQ2XXS | Self::IQ2XS | Self::IQ2S => 0,
            Self::IQ3XXS | Self::IQ3S => 0,
            Self::IQ4NL | Self::IQ4XS => 0,
            Self::IQ1S | Self::IQ1M => 0,
        }
    }

    /// Compute total bytes for `n_weights` of this type.
    pub fn compute_byte_size(&self, n_weights: usize) -> usize {
        let bs = self.block_size();
        let ts = self.type_size();
        if bs == 0 || ts == 0 {
            return 0;
        }
        let n_blocks = (n_weights + bs - 1) / bs;
        n_blocks * ts
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(
            self,
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 | Self::I8 | Self::I16 | Self::I32 | Self::I64
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ2S => "IQ2_S",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ3S => "IQ3_S",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ4XS => "IQ4_XS",
            Self::IQ1S => "IQ1_S",
            Self::IQ1M => "IQ1_M",
        }
    }
}

// ── Metadata values ─────────────────────────────────────────────────────────

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Return a string representation for probe metadata export.
    pub fn to_string_lossy(&self) -> String {
        match self {
            GgufValue::U8(v) => v.to_string(),
            GgufValue::I8(v) => v.to_string(),
            GgufValue::U16(v) => v.to_string(),
            GgufValue::I16(v) => v.to_string(),
            GgufValue::U32(v) => v.to_string(),
            GgufValue::I32(v) => v.to_string(),
            GgufValue::U64(v) => v.to_string(),
            GgufValue::I64(v) => v.to_string(),
            GgufValue::F32(v) => v.to_string(),
            GgufValue::F64(v) => v.to_string(),
            GgufValue::Bool(v) => v.to_string(),
            GgufValue::String(v) => v.clone(),
            GgufValue::Array(v) => format!("[{} elements]", v.len()),
        }
    }
}

// ── Tensor info ─────────────────────────────────────────────────────────────

/// Parsed tensor metadata from the GGUF header.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub quant_type: GgufQuantType,
    pub offset: u64,       // relative to data section start
    pub byte_size: usize,
    pub param_count: usize, // product of shape dims
}

// ── Main GGUF index ─────────────────────────────────────────────────────────

/// Parsed GGUF file index with mmap backing.
#[derive(Debug)]
pub struct GgufIndex {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_offset: usize,
    pub alignment: usize,
    mmap: Mmap,
}

// ── Reader helpers ──────────────────────────────────────────────────────────

/// Cursor over a byte slice for sequential reads.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err(format!(
                "unexpected EOF at offset {}: need {} bytes, {} available",
                self.pos,
                n,
                self.remaining()
            ));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8, String> {
        Ok(self.read_bytes(1)?[0] as i8)
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> Result<i16, String> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64(&mut self) -> Result<f64, String> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u64()? as usize;
        if len > 1_000_000 {
            return Err(format!("unreasonable string length: {len}"));
        }
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| format!("invalid UTF-8 in GGUF string: {e}"))
    }

    fn read_bool(&mut self) -> Result<bool, String> {
        Ok(self.read_u8()? != 0)
    }

    /// Read a typed value given a GGUF value type ID.
    fn read_value(&mut self, type_id: u32) -> Result<GgufValue, String> {
        match type_id {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_i8()?)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_i16()?)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_i32()?)),
            6 => Ok(GgufValue::F32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_bool()?)),
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                // Array: element type (u32) + count (u64) + elements
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                if count > 10_000_000 {
                    return Err(format!("unreasonable array count: {count}"));
                }
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array(items))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            other => Err(format!("unknown GGUF value type: {other}")),
        }
    }
}

// ── GgufIndex implementation ────────────────────────────────────────────────

impl GgufIndex {
    /// Open and parse a GGUF file.
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| format!("mmap failed for {}: {e}", path.display()))?
        };

        if mmap.len() < 24 {
            return Err(format!("file too small for GGUF header: {}", path.display()));
        }

        let mut r = Reader::new(&mmap);

        // 1. Validate magic
        let magic = r.read_u32()?;
        if magic != GGUF_MAGIC_LE {
            if magic == GGUF_MAGIC_LE.swap_bytes() {
                return Err("big-endian GGUF files are not supported".to_string());
            }
            return Err(format!(
                "invalid GGUF magic: 0x{magic:08X} (expected 0x{GGUF_MAGIC_LE:08X})"
            ));
        }

        // 2. Version
        let version = r.read_u32()?;
        if version < 2 || version > 3 {
            return Err(format!("unsupported GGUF version: {version} (need 2 or 3)"));
        }

        // 3. Counts
        let tensor_count = r.read_u64()? as usize;
        let metadata_kv_count = r.read_u64()? as usize;

        if tensor_count > 100_000 {
            return Err(format!("unreasonable tensor count: {tensor_count}"));
        }
        if metadata_kv_count > 100_000 {
            return Err(format!("unreasonable metadata count: {metadata_kv_count}"));
        }

        // 4. Parse metadata KV pairs
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = r.read_string()?;
            let value_type = r.read_u32()?;
            let value = r.read_value(value_type)?;
            metadata.insert(key, value);
        }

        // 5. Read alignment from metadata
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(a)) => *a as usize,
            Some(GgufValue::U64(a)) => *a as usize,
            Some(GgufValue::I32(a)) if *a > 0 => *a as usize,
            _ => 32,
        };

        // 6. Parse tensor infos
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.read_string()?;
            let n_dims = r.read_u32()? as usize;
            if n_dims > 8 {
                return Err(format!("unreasonable dimension count for tensor {name}: {n_dims}"));
            }
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(r.read_u64()? as usize);
            }
            let quant_type_id = r.read_u32()?;
            let quant_type = GgufQuantType::from_u32(quant_type_id)?;
            let offset = r.read_u64()?;

            let param_count: usize = if shape.is_empty() {
                0
            } else {
                shape.iter().product()
            };
            let byte_size = quant_type.compute_byte_size(param_count);

            tensors.push(GgufTensorInfo {
                name,
                shape,
                quant_type,
                offset,
                byte_size,
                param_count,
            });
        }

        // 7. Compute data offset — pad current position up to alignment boundary
        let data_offset = (r.pos + alignment - 1) / alignment * alignment;

        // 8. Validate tensor offsets fit within file
        for t in &tensors {
            let abs_start = data_offset + t.offset as usize;
            let abs_end = abs_start + t.byte_size;
            if abs_end > mmap.len() {
                return Err(format!(
                    "tensor '{}' extends past file end: offset {} + {} bytes = {}, file size = {}",
                    t.name, abs_start, t.byte_size, abs_end, mmap.len()
                ));
            }
        }

        Ok(GgufIndex {
            version,
            metadata,
            tensors,
            data_offset,
            alignment,
            mmap,
        })
    }

    /// Get raw bytes for a tensor (zero-copy slice of mmap).
    pub fn tensor_data(&self, name: &str) -> Result<&[u8], String> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("tensor not found: {name}"))?;
        let abs_offset = self.data_offset + info.offset as usize;
        let end = abs_offset + info.byte_size;
        if end > self.mmap.len() {
            return Err(format!("tensor {name} extends past file end"));
        }
        Ok(&self.mmap[abs_offset..end])
    }

    /// Get raw bytes for a tensor by index (zero-copy slice of mmap).
    pub fn tensor_data_by_index(&self, idx: usize) -> Result<&[u8], String> {
        let info = self
            .tensors
            .get(idx)
            .ok_or_else(|| format!("tensor index out of range: {idx}"))?;
        let abs_offset = self.data_offset + info.offset as usize;
        let end = abs_offset + info.byte_size;
        if end > self.mmap.len() {
            return Err(format!("tensor {} extends past file end", info.name));
        }
        Ok(&self.mmap[abs_offset..end])
    }

    /// Build a `ModelInfo` for `probe_model`.
    pub fn to_model_info(&self, path: &Path) -> ModelInfo {
        let file_bytes = self.mmap.len() as u64;

        let tensor_names: Vec<String> = {
            let mut names: Vec<String> = self.tensors.iter().map(|t| t.name.clone()).collect();
            names.sort();
            names
        };

        let mut tensor_shapes = HashMap::new();
        let mut tensor_dtypes = HashMap::new();
        let mut quant_types = HashMap::new();
        let mut param_count: u64 = 0;

        for t in &self.tensors {
            tensor_shapes.insert(t.name.clone(), t.shape.clone());
            tensor_dtypes.insert(t.name.clone(), t.quant_type.name().to_string());
            quant_types.insert(t.name.clone(), t.quant_type.name().to_string());
            param_count += t.param_count as u64;
        }

        let mut metadata = HashMap::new();
        for (k, v) in &self.metadata {
            metadata.insert(k.clone(), v.to_string_lossy());
        }

        ModelInfo {
            format: ModelFormat::Gguf,
            path: path.to_path_buf(),
            tensor_count: self.tensors.len(),
            tensor_names,
            tensor_shapes,
            tensor_dtypes,
            metadata,
            quant_types: Some(quant_types),
            total_file_bytes: file_bytes,
            param_count,
        }
    }
}

// ── Helper: build a minimal GGUF file in memory (for tests) ─────────────────

#[cfg(test)]
pub(crate) fn build_test_gguf(
    version: u32,
    metadata: &[(&str, GgufValue)],
    tensors: &[(&str, &[usize], GgufQuantType, &[u8])],
) -> Vec<u8> {
    let mut buf = Vec::new();
    let alignment: usize = 32;

    // Magic
    buf.extend_from_slice(&GGUF_MAGIC_LE.to_le_bytes());
    // Version
    buf.extend_from_slice(&version.to_le_bytes());
    // Tensor count
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    // Metadata KV count
    buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    // Write metadata
    for (key, val) in metadata {
        // Key: u64 len + bytes
        let key_bytes = key.as_bytes();
        buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(key_bytes);
        // Value type + value
        write_test_value(&mut buf, val);
    }

    // Compute where tensor data will start
    // First, figure out how many bytes the tensor info section takes
    let mut tensor_info_size = 0usize;
    for (name, shape, _, _) in tensors {
        tensor_info_size += 8; // name length (u64)
        tensor_info_size += name.len(); // name bytes
        tensor_info_size += 4; // n_dims (u32)
        tensor_info_size += shape.len() * 8; // dims (u64 each)
        tensor_info_size += 4; // quant type (u32)
        tensor_info_size += 8; // offset (u64)
    }
    let header_end = buf.len() + tensor_info_size;
    let data_start = (header_end + alignment - 1) / alignment * alignment;

    // Compute per-tensor offsets (relative to data_start)
    let mut offsets = Vec::new();
    let mut cursor: usize = 0;
    for (_, _, _, data) in tensors {
        offsets.push(cursor as u64);
        cursor += data.len();
    }

    // Write tensor infos
    for (i, (name, shape, qt, _)) in tensors.iter().enumerate() {
        let name_bytes = name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &dim in *shape {
            buf.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        buf.extend_from_slice(&(*qt as u32).to_le_bytes());
        buf.extend_from_slice(&offsets[i].to_le_bytes());
    }

    // Pad to alignment
    while buf.len() < data_start {
        buf.push(0);
    }

    // Write tensor data
    for (_, _, _, data) in tensors {
        buf.extend_from_slice(data);
    }

    buf
}

#[cfg(test)]
fn write_test_value(buf: &mut Vec<u8>, val: &GgufValue) {
    match val {
        GgufValue::U8(v) => {
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.push(*v);
        }
        GgufValue::I8(v) => {
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.push(*v as u8);
        }
        GgufValue::U16(v) => {
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I16(v) => {
            buf.extend_from_slice(&3u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::U32(v) => {
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I32(v) => {
            buf.extend_from_slice(&5u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::F32(v) => {
            buf.extend_from_slice(&6u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::Bool(v) => {
            buf.extend_from_slice(&7u32.to_le_bytes());
            buf.push(if *v { 1 } else { 0 });
        }
        GgufValue::String(v) => {
            buf.extend_from_slice(&8u32.to_le_bytes());
            let bytes = v.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
        GgufValue::Array(items) => {
            buf.extend_from_slice(&9u32.to_le_bytes());
            // Determine element type from first item (or default to U8)
            let elem_type = if items.is_empty() {
                0u32
            } else {
                match &items[0] {
                    GgufValue::U8(_) => 0,
                    GgufValue::I8(_) => 1,
                    GgufValue::U32(_) => 4,
                    GgufValue::String(_) => 8,
                    _ => 0,
                }
            };
            buf.extend_from_slice(&elem_type.to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                write_test_value_no_type(buf, item);
            }
        }
        GgufValue::U64(v) => {
            buf.extend_from_slice(&10u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I64(v) => {
            buf.extend_from_slice(&11u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::F64(v) => {
            buf.extend_from_slice(&12u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
}

#[cfg(test)]
fn write_test_value_no_type(buf: &mut Vec<u8>, val: &GgufValue) {
    match val {
        GgufValue::U8(v) => buf.push(*v),
        GgufValue::I8(v) => buf.push(*v as u8),
        GgufValue::U16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::U32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Bool(v) => buf.push(if *v { 1 } else { 0 }),
        GgufValue::String(v) => {
            let bytes = v.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
        GgufValue::U64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Array(_) => {} // nested arrays not needed for tests
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn temp_gguf(name: &str) -> PathBuf {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("gguf_test_{name}_{ts}.gguf"))
    }

    fn write_temp_gguf(name: &str, data: &[u8]) -> PathBuf {
        let path = temp_gguf(name);
        let mut f = File::create(&path).unwrap();
        f.write_all(data).unwrap();
        f.sync_all().unwrap();
        path
    }

    #[test]
    fn parse_minimal_gguf_v3() {
        // Minimal GGUF v3 with 1 metadata key and 1 F32 tensor (shape [4])
        let tensor_data = vec![0u8; 16]; // 4 x f32 = 16 bytes
        let data = build_test_gguf(
            3,
            &[("general.name", GgufValue::String("test_model".to_string()))],
            &[("weight", &[4], GgufQuantType::F32, &tensor_data)],
        );

        let path = write_temp_gguf("minimal_v3", &data);
        let idx = GgufIndex::open(&path).unwrap();

        assert_eq!(idx.version, 3);
        assert_eq!(idx.tensors.len(), 1);
        assert_eq!(idx.tensors[0].name, "weight");
        assert_eq!(idx.tensors[0].shape, vec![4]);
        assert_eq!(idx.tensors[0].quant_type, GgufQuantType::F32);
        assert_eq!(idx.tensors[0].param_count, 4);
        assert_eq!(idx.tensors[0].byte_size, 16);

        // Check metadata
        match idx.metadata.get("general.name") {
            Some(GgufValue::String(s)) => assert_eq!(s, "test_model"),
            other => panic!("expected String, got: {other:?}"),
        }

        // Check tensor data access
        let raw = idx.tensor_data("weight").unwrap();
        assert_eq!(raw.len(), 16);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn parse_gguf_v2() {
        let tensor_data = vec![0u8; 8]; // 4 x f16 = 8 bytes
        let data = build_test_gguf(
            2,
            &[],
            &[("bias", &[4], GgufQuantType::F16, &tensor_data)],
        );

        let path = write_temp_gguf("v2", &data);
        let idx = GgufIndex::open(&path).unwrap();
        assert_eq!(idx.version, 2);
        assert_eq!(idx.tensors.len(), 1);
        assert_eq!(idx.tensors[0].quant_type, GgufQuantType::F16);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn invalid_magic_rejected() {
        let path = write_temp_gguf("bad_magic", &[0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        let result = GgufIndex::open(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid GGUF magic"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn be_magic_rejected() {
        // Big-endian GGUF magic (reversed)
        let path = write_temp_gguf("be_magic", &[0x46, 0x55, 0x47, 0x47, 0x03, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        let result = GgufIndex::open(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("big-endian"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn unsupported_version_rejected() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC_LE.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes()); // version 1
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata count

        let path = write_temp_gguf("bad_version", &data);
        let result = GgufIndex::open(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unsupported GGUF version"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn multiple_tensors_and_metadata() {
        let t1_data = vec![0u8; 34]; // Q8_0: 1 block = 34 bytes for 32 weights
        let t2_data = vec![0u8; 8];  // F16: 4 weights = 8 bytes

        let data = build_test_gguf(
            3,
            &[
                ("general.name", GgufValue::String("multi".to_string())),
                ("general.architecture", GgufValue::String("llama".to_string())),
                ("llama.context_length", GgufValue::U32(4096)),
            ],
            &[
                ("blk.0.attn.weight", &[32], GgufQuantType::Q8_0, &t1_data),
                ("output.bias", &[4], GgufQuantType::F16, &t2_data),
            ],
        );

        let path = write_temp_gguf("multi", &data);
        let idx = GgufIndex::open(&path).unwrap();

        assert_eq!(idx.tensors.len(), 2);
        assert_eq!(idx.tensors[0].name, "blk.0.attn.weight");
        assert_eq!(idx.tensors[0].quant_type, GgufQuantType::Q8_0);
        assert_eq!(idx.tensors[0].byte_size, 34);
        assert_eq!(idx.tensors[1].name, "output.bias");
        assert_eq!(idx.tensors[1].quant_type, GgufQuantType::F16);

        assert_eq!(idx.metadata.len(), 3);

        // Verify tensor data can be read
        let raw0 = idx.tensor_data("blk.0.attn.weight").unwrap();
        assert_eq!(raw0.len(), 34);
        let raw1 = idx.tensor_data("output.bias").unwrap();
        assert_eq!(raw1.len(), 8);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn tensor_not_found() {
        let tensor_data = vec![0u8; 16];
        let data = build_test_gguf(3, &[], &[("w", &[4], GgufQuantType::F32, &tensor_data)]);
        let path = write_temp_gguf("notfound", &data);
        let idx = GgufIndex::open(&path).unwrap();

        let result = idx.tensor_data("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("tensor not found"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn to_model_info_builds_correctly() {
        let tensor_data = vec![0u8; 34];
        let data = build_test_gguf(
            3,
            &[("general.name", GgufValue::String("test".to_string()))],
            &[("layer.weight", &[8, 4], GgufQuantType::Q8_0, &tensor_data)],
        );
        let path = write_temp_gguf("model_info", &data);
        let idx = GgufIndex::open(&path).unwrap();
        let info = idx.to_model_info(&path);

        assert_eq!(info.format, ModelFormat::Gguf);
        assert_eq!(info.tensor_count, 1);
        assert_eq!(info.tensor_names, vec!["layer.weight".to_string()]);
        assert_eq!(info.tensor_shapes["layer.weight"], vec![8, 4]);
        assert_eq!(info.tensor_dtypes["layer.weight"], "Q8_0");
        assert!(info.quant_types.is_some());
        assert_eq!(info.quant_types.as_ref().unwrap()["layer.weight"], "Q8_0");
        assert_eq!(info.param_count, 32);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn quant_type_roundtrip() {
        for qt_val in [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 30] {
            let qt = GgufQuantType::from_u32(qt_val).unwrap();
            assert!(!qt.name().is_empty());
            assert!(qt.block_size() > 0);
            assert!(qt.type_size() > 0);
        }
    }

    #[test]
    fn quant_type_unknown_rejected() {
        assert!(GgufQuantType::from_u32(4).is_err()); // removed type
        assert!(GgufQuantType::from_u32(5).is_err()); // removed type
        assert!(GgufQuantType::from_u32(99).is_err());
    }

    #[test]
    fn quant_type_byte_size_computation() {
        // Q8_0: 32 weights per block, 34 bytes per block
        assert_eq!(GgufQuantType::Q8_0.compute_byte_size(32), 34);
        assert_eq!(GgufQuantType::Q8_0.compute_byte_size(64), 68);
        assert_eq!(GgufQuantType::Q8_0.compute_byte_size(33), 68); // rounds up

        // F32: 1 weight per "block", 4 bytes each
        assert_eq!(GgufQuantType::F32.compute_byte_size(10), 40);

        // Q4_0: 32 weights per block, 18 bytes per block
        assert_eq!(GgufQuantType::Q4_0.compute_byte_size(32), 18);
        assert_eq!(GgufQuantType::Q4_0.compute_byte_size(256), 144);

        // Q4K: 256 weights per block, 144 bytes per block
        assert_eq!(GgufQuantType::Q4K.compute_byte_size(256), 144);
        assert_eq!(GgufQuantType::Q4K.compute_byte_size(512), 288);
    }

    #[test]
    fn quant_type_is_quantized() {
        assert!(!GgufQuantType::F32.is_quantized());
        assert!(!GgufQuantType::F16.is_quantized());
        assert!(!GgufQuantType::BF16.is_quantized());
        assert!(!GgufQuantType::I8.is_quantized());
        assert!(GgufQuantType::Q4_0.is_quantized());
        assert!(GgufQuantType::Q8_0.is_quantized());
        assert!(GgufQuantType::Q4K.is_quantized());
        assert!(GgufQuantType::Q6K.is_quantized());
    }

    #[test]
    fn metadata_value_types() {
        let tensor_data = vec![0u8; 4]; // 1 x F32
        let data = build_test_gguf(
            3,
            &[
                ("key.u8", GgufValue::U8(42)),
                ("key.i32", GgufValue::I32(-7)),
                ("key.f32", GgufValue::F32(3.14)),
                ("key.bool", GgufValue::Bool(true)),
                ("key.str", GgufValue::String("hello".to_string())),
                ("key.u64", GgufValue::U64(123456789)),
            ],
            &[("t", &[1], GgufQuantType::F32, &tensor_data)],
        );

        let path = write_temp_gguf("meta_types", &data);
        let idx = GgufIndex::open(&path).unwrap();

        match &idx.metadata["key.u8"] {
            GgufValue::U8(v) => assert_eq!(*v, 42),
            other => panic!("expected U8, got {other:?}"),
        }
        match &idx.metadata["key.i32"] {
            GgufValue::I32(v) => assert_eq!(*v, -7),
            other => panic!("expected I32, got {other:?}"),
        }
        match &idx.metadata["key.f32"] {
            GgufValue::F32(v) => assert!((v - 3.14).abs() < 0.001),
            other => panic!("expected F32, got {other:?}"),
        }
        match &idx.metadata["key.bool"] {
            GgufValue::Bool(v) => assert!(*v),
            other => panic!("expected Bool, got {other:?}"),
        }
        match &idx.metadata["key.str"] {
            GgufValue::String(v) => assert_eq!(v, "hello"),
            other => panic!("expected String, got {other:?}"),
        }
        match &idx.metadata["key.u64"] {
            GgufValue::U64(v) => assert_eq!(*v, 123456789),
            other => panic!("expected U64, got {other:?}"),
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn file_too_small() {
        let path = write_temp_gguf("tiny", &[0x47, 0x47, 0x55, 0x46]);
        let result = GgufIndex::open(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too small"));
        std::fs::remove_file(path).ok();
    }
}
