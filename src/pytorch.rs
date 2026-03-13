//! PyTorch checkpoint (.pt / .pth / .bin) pickle scanner.
//!
//! This is NOT a general-purpose pickle VM.  It is a focused scanner that parses
//! just enough pickle opcodes to locate `_rebuild_tensor_v2` / `_rebuild_tensor_v3`
//! calls and extract tensor metadata (name, shape, dtype, storage location) from
//! PyTorch state-dict ZIP archives.
//!
//! Security: only a small allowlist of GLOBAL references is "understood"; everything
//! else is recorded as `Opaque` and never executed.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

// ── Public types ────────────────────────────────────────────────────────────

/// Tensor metadata extracted from a pickle stream.
#[derive(Debug, Clone)]
pub struct PickleTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub storage_key: String,
    pub storage_offset: usize,
    pub byte_size: usize,
    pub numel: usize,
}

/// Complete index of tensors inside a PyTorch ZIP checkpoint.
#[derive(Debug)]
pub struct PickleIndex {
    pub tensors: Vec<PickleTensorInfo>,
    pub zip_path: PathBuf,
    /// storage_key → filename inside the ZIP (e.g. "0" → "archive/data/0")
    pub storage_files: HashMap<String, String>,
}

// ── Pickle value representation ─────────────────────────────────────────────

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum PV {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<PV>),
    List(Vec<PV>),
    Dict(Vec<(PV, PV)>),
    Global { module: String, name: String },
    Storage { key: String, dtype: String },
    TensorRebuild {
        storage_key: String,
        dtype: String,
        shape: Vec<usize>,
        storage_offset: usize,
    },
    OrderedDict(Vec<(PV, PV)>),
    /// Unrecognised / opaque — we don't care about the value.
    Opaque,
    /// Internal: stack mark sentinel.
    Mark,
}

impl PV {
    fn as_int(&self) -> Option<i64> {
        match self {
            PV::Int(v) => Some(*v),
            _ => Option::None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            PV::Str(s) => Some(s.as_str()),
            _ => Option::None,
        }
    }
    fn into_tuple(self) -> Vec<PV> {
        match self {
            PV::Tuple(v) => v,
            _ => vec![self],
        }
    }
    fn as_usize_vec(&self) -> Option<Vec<usize>> {
        match self {
            PV::Tuple(items) | PV::List(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(item.as_int()? as usize);
                }
                Some(out)
            }
            _ => Option::None,
        }
    }
    /// Retrieve the entries of a dict-like value (Dict or OrderedDict).
    fn dict_entries(&self) -> Option<&[(PV, PV)]> {
        match self {
            PV::Dict(entries) | PV::OrderedDict(entries) => Some(entries),
            _ => Option::None,
        }
    }
}

// ── dtype helpers ───────────────────────────────────────────────────────────

fn storage_type_to_dtype(name: &str) -> &'static str {
    match name {
        "FloatStorage" | "float" | "float32" | "Float" => "float32",
        "HalfStorage" | "half" | "float16" | "Half" => "float16",
        "BFloat16Storage" | "bfloat16" | "BFloat16" => "bfloat16",
        "DoubleStorage" | "double" | "float64" | "Double" => "float64",
        "ByteStorage" | "uint8" | "Byte" => "uint8",
        "CharStorage" | "int8" | "Char" => "int8",
        "ShortStorage" | "int16" | "Short" => "int16",
        "IntStorage" | "int32" | "Int" => "int32",
        "LongStorage" | "int64" | "Long" => "int64",
        _ => "float32",
    }
}

fn dtype_byte_size(dtype: &str) -> usize {
    match dtype {
        "float64" | "int64" => 8,
        "float32" | "int32" => 4,
        "float16" | "bfloat16" | "int16" => 2,
        "uint8" | "int8" => 1,
        _ => 4,
    }
}

// ── Security allowlist ──────────────────────────────────────────────────────

fn is_allowed_global(module: &str, name: &str) -> bool {
    matches!(
        (module, name),
        ("torch._utils", "_rebuild_tensor_v2")
            | ("torch._utils", "_rebuild_tensor_v3")
            | ("torch._utils", "_rebuild_tensor")
            | ("collections", "OrderedDict")
            | ("torch", "FloatStorage")
            | ("torch", "HalfStorage")
            | ("torch", "BFloat16Storage")
            | ("torch", "DoubleStorage")
            | ("torch", "ByteStorage")
            | ("torch", "CharStorage")
            | ("torch", "IntStorage")
            | ("torch", "LongStorage")
            | ("torch", "ShortStorage")
            | ("torch.storage", "_load_from_bytes")
            | ("_codecs", "encode")
            | ("torch", "Size")
            | ("torch", "_utils")
            // Typed-storage constructors used by newer PyTorch
            | ("torch", "FloatTensor")
            | ("torch", "HalfTensor")
            | ("torch", "BFloat16Tensor")
            | ("torch", "DoubleTensor")
            | ("torch", "ByteTensor")
            | ("torch", "CharTensor")
            | ("torch", "IntTensor")
            | ("torch", "LongTensor")
            | ("torch", "ShortTensor")
            // Typed storage via torch.storage module
            | ("torch.storage", "TypedStorage")
            | ("torch.storage", "UntypedStorage")
            // Persistent-id storage references
            | ("torch._utils", "_rebuild_parameter")
            | ("torch._utils", "_rebuild_parameter_with_state")
            | ("torch.nn.parameter", "Parameter")
    )
}

fn is_dangerous_global(module: &str, _name: &str) -> bool {
    let dangerous_prefixes = [
        "os.", "sys.", "builtins.", "subprocess.", "shutil.", "importlib.",
    ];
    let dangerous_modules = ["os", "sys", "builtins", "subprocess", "shutil"];
    if dangerous_modules.contains(&module) {
        return true;
    }
    for prefix in &dangerous_prefixes {
        if module.starts_with(prefix) {
            return true;
        }
    }
    false
}

// ── Pickle opcode scanner ───────────────────────────────────────────────────

struct PickleScanner {
    data: Vec<u8>,
    pos: usize,
    stack: Vec<PV>,
    memo: HashMap<u32, PV>,
    next_memo_idx: u32,
    /// Collected tensor rebuild results (unused but kept for future extensibility).
    #[allow(dead_code)]
    tensors: Vec<PV>,
}

impl PickleScanner {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            pos: 0,
            stack: Vec::with_capacity(256),
            memo: HashMap::new(),
            next_memo_idx: 0,
            tensors: Vec::new(),
        }
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        if self.pos >= self.data.len() {
            return Err("unexpected end of pickle stream".into());
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&[u8], String> {
        if self.pos + n > self.data.len() {
            return Err(format!(
                "unexpected end of pickle stream: need {n} bytes at pos {}",
                self.pos
            ));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u16_le(&mut self) -> Result<u16, String> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i32_le(&mut self) -> Result<i32, String> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u32_le(&mut self) -> Result<u32, String> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64_le(&mut self) -> Result<u64, String> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_f64_be(&mut self) -> Result<f64, String> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_line(&mut self) -> Result<String, String> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err("unterminated line in pickle".into());
        }
        let line = String::from_utf8_lossy(&self.data[start..self.pos]).to_string();
        self.pos += 1; // skip newline
        Ok(line)
    }

    fn pop(&mut self) -> PV {
        self.stack.pop().unwrap_or(PV::Opaque)
    }

    fn pop_to_mark(&mut self) -> Vec<PV> {
        let mut items = Vec::new();
        while let Some(val) = self.stack.pop() {
            if matches!(val, PV::Mark) {
                break;
            }
            items.push(val);
        }
        items.reverse();
        items
    }

    fn top(&self) -> &PV {
        self.stack.last().unwrap_or(&PV::Opaque)
    }

    /// Execute the pickle byte-stream, building up tensor metadata.
    fn scan(&mut self) -> Result<PV, String> {
        loop {
            if self.pos >= self.data.len() {
                break;
            }
            let op = self.read_u8()?;
            match op {
                // PROTO
                0x80 => {
                    let _version = self.read_u8()?;
                }
                // FRAME
                0x95 => {
                    let _frame_len = self.read_u64_le()?;
                }
                // STOP
                0x2E => break,

                // ── Stack manipulation ──
                // MARK
                0x28 => self.stack.push(PV::Mark),
                // POP
                0x30 => {
                    self.pop();
                }
                // POP_MARK
                0x31 => {
                    self.pop_to_mark();
                }
                // DUP
                0x32 => {
                    let top = self.top().clone();
                    self.stack.push(top);
                }

                // ── Containers ──
                // EMPTY_DICT
                0x7D => self.stack.push(PV::Dict(Vec::new())),
                // EMPTY_LIST
                0x5D => self.stack.push(PV::List(Vec::new())),
                // EMPTY_TUPLE
                0x29 => self.stack.push(PV::Tuple(Vec::new())),
                // TUPLE (items from mark)
                0x74 => {
                    let items = self.pop_to_mark();
                    self.stack.push(PV::Tuple(items));
                }
                // TUPLE1
                0x85 => {
                    let a = self.pop();
                    self.stack.push(PV::Tuple(vec![a]));
                }
                // TUPLE2
                0x86 => {
                    let b = self.pop();
                    let a = self.pop();
                    self.stack.push(PV::Tuple(vec![a, b]));
                }
                // TUPLE3
                0x87 => {
                    let c = self.pop();
                    let b = self.pop();
                    let a = self.pop();
                    self.stack.push(PV::Tuple(vec![a, b, c]));
                }
                // LIST (items from mark)
                0x6C => {
                    let items = self.pop_to_mark();
                    self.stack.push(PV::List(items));
                }
                // DICT (interleaved key/value from mark)
                0x64 => {
                    let items = self.pop_to_mark();
                    let mut entries = Vec::new();
                    let mut i = 0;
                    while i + 1 < items.len() {
                        entries.push((items[i].clone(), items[i + 1].clone()));
                        i += 2;
                    }
                    self.stack.push(PV::Dict(entries));
                }

                // ── Data atoms ──
                // NONE
                0x4E => self.stack.push(PV::None),
                // NEWTRUE
                0x88 => self.stack.push(PV::Bool(true)),
                // NEWFALSE
                0x89 => self.stack.push(PV::Bool(false)),
                // INT (text line, protocol 0)
                0x49 => {
                    let line = self.read_line()?;
                    let v = line.trim().parse::<i64>().unwrap_or(0);
                    self.stack.push(PV::Int(v));
                }
                // BININT (4 bytes LE signed)
                0x4A => {
                    let v = self.read_i32_le()?;
                    self.stack.push(PV::Int(v as i64));
                }
                // BININT1 (1 byte unsigned)
                0x4B => {
                    let v = self.read_u8()?;
                    self.stack.push(PV::Int(v as i64));
                }
                // BININT2 (2 bytes LE unsigned)
                0x4D => {
                    let v = self.read_u16_le()?;
                    self.stack.push(PV::Int(v as i64));
                }
                // LONG1 (1-byte length + n bytes)
                0x8A => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let v = long_from_bytes(bytes);
                    self.stack.push(PV::Int(v));
                }
                // LONG4 (4-byte length + n bytes)
                0x8B => {
                    let n = self.read_i32_le()? as usize;
                    if n > 256 {
                        // Absurdly large long — skip
                        self.read_bytes(n)?;
                        self.stack.push(PV::Opaque);
                    } else {
                        let bytes = self.read_bytes(n)?;
                        let v = long_from_bytes(bytes);
                        self.stack.push(PV::Int(v));
                    }
                }
                // BINFLOAT (8 bytes BE double)
                0x47 => {
                    let v = self.read_f64_be()?;
                    self.stack.push(PV::Float(v));
                }
                // FLOAT (text line, protocol 0)
                0x46 => {
                    let line = self.read_line()?;
                    let v = line.trim().parse::<f64>().unwrap_or(0.0);
                    self.stack.push(PV::Float(v));
                }

                // ── Strings ──
                // SHORT_BINUNICODE (1-byte len)
                0x8C => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let s = String::from_utf8_lossy(bytes).to_string();
                    self.stack.push(PV::Str(s));
                }
                // BINUNICODE (4-byte LE len)
                0x58 => {
                    let n = self.read_u32_le()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let s = String::from_utf8_lossy(bytes).to_string();
                    self.stack.push(PV::Str(s));
                }
                // SHORT_BINSTRING (1-byte len, protocol 1)
                0x55 => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let s = String::from_utf8_lossy(bytes).to_string();
                    self.stack.push(PV::Str(s));
                }
                // BINSTRING (4-byte LE len)
                0x54 => {
                    let n = self.read_u32_le()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let s = String::from_utf8_lossy(bytes).to_string();
                    self.stack.push(PV::Str(s));
                }
                // STRING (quoted text, protocol 0)
                0x53 => {
                    let line = self.read_line()?;
                    // Strip quotes
                    let s = line.trim().trim_matches('\'').trim_matches('"').to_string();
                    self.stack.push(PV::Str(s));
                }
                // UNICODE (text line, protocol 0)
                0x56 => {
                    let line = self.read_line()?;
                    self.stack.push(PV::Str(line));
                }

                // ── Bytes ──
                // SHORT_BINBYTES (1-byte len)
                0x43 => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_bytes(n)?.to_vec();
                    self.stack.push(PV::Bytes(bytes));
                }
                // BINBYTES (4-byte LE len)
                0x42 => {
                    let n = self.read_u32_le()? as usize;
                    let bytes = self.read_bytes(n)?.to_vec();
                    self.stack.push(PV::Bytes(bytes));
                }
                // BINBYTES8 (8-byte LE len)
                0x8E => {
                    let n = self.read_u64_le()? as usize;
                    if n > 100_000_000 {
                        // Refuse to allocate huge blobs — skip
                        return Err(format!("BINBYTES8 too large: {n}"));
                    }
                    let bytes = self.read_bytes(n)?.to_vec();
                    self.stack.push(PV::Bytes(bytes));
                }

                // ── Collection mutation ──
                // APPEND
                0x61 => {
                    let val = self.pop();
                    if let Some(PV::List(list)) = self.stack.last_mut() {
                        list.push(val);
                    }
                }
                // APPENDS
                0x65 => {
                    let items = self.pop_to_mark();
                    if let Some(PV::List(list)) = self.stack.last_mut() {
                        list.extend(items);
                    }
                }
                // SETITEM
                0x73 => {
                    let val = self.pop();
                    let key = self.pop();
                    if let Some(PV::Dict(entries) | PV::OrderedDict(entries)) =
                        self.stack.last_mut()
                    {
                        entries.push((key, val));
                    }
                }
                // SETITEMS
                0x75 => {
                    let items = self.pop_to_mark();
                    let mut pairs = Vec::new();
                    let mut i = 0;
                    while i + 1 < items.len() {
                        pairs.push((items[i].clone(), items[i + 1].clone()));
                        i += 2;
                    }
                    if let Some(PV::Dict(entries) | PV::OrderedDict(entries)) =
                        self.stack.last_mut()
                    {
                        entries.extend(pairs);
                    }
                }

                // ── GLOBAL (two newline-delimited strings) ──
                0x63 => {
                    let module = self.read_line()?;
                    let name = self.read_line()?;
                    if is_dangerous_global(&module, &name) {
                        // Security: record but don't execute
                        self.stack.push(PV::Opaque);
                    } else {
                        self.stack.push(PV::Global {
                            module: module.clone(),
                            name: name.clone(),
                        });
                    }
                }
                // STACK_GLOBAL
                0x93 => {
                    let name_val = self.pop();
                    let module_val = self.pop();
                    let module = match &module_val {
                        PV::Str(s) => s.clone(),
                        _ => String::new(),
                    };
                    let name = match &name_val {
                        PV::Str(s) => s.clone(),
                        _ => String::new(),
                    };
                    if is_dangerous_global(&module, &name) {
                        self.stack.push(PV::Opaque);
                    } else {
                        self.stack.push(PV::Global { module, name });
                    }
                }
                // INST (like GLOBAL but also consumes mark args)
                0x69 => {
                    let module = self.read_line()?;
                    let name = self.read_line()?;
                    let _args = self.pop_to_mark();
                    if is_dangerous_global(&module, &name) {
                        self.stack.push(PV::Opaque);
                    } else {
                        self.stack.push(PV::Global { module, name });
                    }
                }

                // ── Object construction ──
                // REDUCE: pop args + callable, call
                0x52 => {
                    let args = self.pop();
                    let callable = self.pop();
                    let result = self.apply_reduce(callable, args);
                    self.stack.push(result);
                }
                // NEWOBJ: pop args + cls
                0x81 => {
                    let args = self.pop();
                    let cls = self.pop();
                    let result = self.apply_reduce(cls, args);
                    self.stack.push(result);
                }
                // NEWOBJ_EX
                0x92 => {
                    let _kwargs = self.pop();
                    let args = self.pop();
                    let cls = self.pop();
                    let result = self.apply_reduce(cls, args);
                    self.stack.push(result);
                }
                // BUILD: pop state, apply to top
                0x62 => {
                    let state = self.pop();
                    // For OrderedDict, BUILD with a list of (k,v) tuples populates it
                    if let Some(PV::OrderedDict(entries)) = self.stack.last_mut() {
                        if let PV::List(items) | PV::Tuple(items) = state {
                            for item in items {
                                if let PV::Tuple(pair) = item {
                                    if pair.len() == 2 {
                                        entries.push((pair[0].clone(), pair[1].clone()));
                                    }
                                }
                            }
                        }
                    }
                    // Otherwise BUILD is a no-op for our purposes
                }

                // ── Memo ──
                // BINPUT (1-byte index)
                0x71 => {
                    let idx = self.read_u8()? as u32;
                    let val = self.top().clone();
                    self.memo.insert(idx, val);
                }
                // LONG_BINPUT (4-byte index)
                0x72 => {
                    let idx = self.read_u32_le()?;
                    let val = self.top().clone();
                    self.memo.insert(idx, val);
                }
                // MEMOIZE — memo the top under next sequential index
                0x94 => {
                    let idx = self.next_memo_idx;
                    let val = self.top().clone();
                    self.memo.insert(idx, val);
                    self.next_memo_idx += 1;
                }
                // BINGET (1-byte index)
                0x68 => {
                    let idx = self.read_u8()? as u32;
                    let val = self.memo.get(&idx).cloned().unwrap_or(PV::Opaque);
                    self.stack.push(val);
                }
                // LONG_BINGET (4-byte index)
                0x6A => {
                    let idx = self.read_u32_le()?;
                    let val = self.memo.get(&idx).cloned().unwrap_or(PV::Opaque);
                    self.stack.push(val);
                }

                // ── Protocol 0/1 compat ──
                // PUT (text line memo)
                0x70 => {
                    let line = self.read_line()?;
                    if let Ok(idx) = line.trim().parse::<u32>() {
                        let val = self.top().clone();
                        self.memo.insert(idx, val);
                    }
                }
                // GET (text line memo)
                0x67 => {
                    let line = self.read_line()?;
                    if let Ok(idx) = line.trim().parse::<u32>() {
                        let val = self.memo.get(&idx).cloned().unwrap_or(PV::Opaque);
                        self.stack.push(val);
                    } else {
                        self.stack.push(PV::Opaque);
                    }
                }
                // LONG (text line, protocol 0 — e.g. "123L\n")
                0x4C => {
                    let line = self.read_line()?;
                    let trimmed = line.trim().trim_end_matches('L');
                    let v = trimmed.parse::<i64>().unwrap_or(0);
                    self.stack.push(PV::Int(v));
                }

                // ── Persistent ID (PyTorch uses this for storage references) ──
                // BINPERSID
                0x51 => {
                    let pid = self.pop();
                    let result = self.resolve_persistent_id(pid);
                    self.stack.push(result);
                }
                // PERSID (text line)
                0x50 => {
                    let _line = self.read_line()?;
                    self.stack.push(PV::Opaque);
                }

                // Unknown opcode — we can't know its argument size, so parsing
                // is now corrupt.  Return an error rather than continuing with
                // garbage data on the stack.
                _ => {
                    return Err(format!(
                        "unknown pickle opcode 0x{:02x} at position {}",
                        op,
                        self.pos - 1,
                    ));
                }
            }
        }

        // Return whatever is on top of the stack
        Ok(self.pop())
    }

    /// Handle REDUCE / NEWOBJ — recognise tensor rebuilds and OrderedDict.
    fn apply_reduce(&mut self, callable: PV, args: PV) -> PV {
        match &callable {
            PV::Global { module, name } => {
                if !is_allowed_global(module, name) {
                    return PV::Opaque;
                }

                match (module.as_str(), name.as_str()) {
                    // _rebuild_tensor_v2(storage, storage_offset, size, stride)
                    // _rebuild_tensor_v3(storage, storage_offset, size, stride, requires_grad, ...)
                    ("torch._utils", "_rebuild_tensor_v2")
                    | ("torch._utils", "_rebuild_tensor_v3")
                    | ("torch._utils", "_rebuild_tensor") => {
                        let items = args.into_tuple();
                        // items: [storage, storage_offset, shape, stride, ...]
                        if items.len() >= 4 {
                            let storage = &items[0];
                            let offset = items[1].as_int().unwrap_or(0) as usize;
                            let shape = items[2].as_usize_vec().unwrap_or_default();

                            let (key, dtype) = match storage {
                                PV::Storage { key, dtype } => {
                                    (key.clone(), dtype.clone())
                                }
                                _ => (String::new(), "float32".into()),
                            };

                            PV::TensorRebuild {
                                storage_key: key,
                                dtype,
                                shape,
                                storage_offset: offset,
                            }
                        } else {
                            PV::Opaque
                        }
                    }
                    ("collections", "OrderedDict") => {
                        // args is typically an empty tuple
                        PV::OrderedDict(Vec::new())
                    }
                    ("torch", name) if name.ends_with("Storage") || name.ends_with("Tensor") => {
                        // e.g. torch.FloatStorage(...)
                        let dtype = storage_type_to_dtype(
                            name.trim_end_matches("Storage").trim_end_matches("Tensor"),
                        );
                        PV::Storage {
                            key: String::new(),
                            dtype: dtype.to_string(),
                        }
                    }
                    ("_codecs", "encode") => {
                        // Used for encoding bytes — return as bytes/opaque
                        let items = args.into_tuple();
                        if !items.is_empty() {
                            if let PV::Str(s) = &items[0] {
                                PV::Bytes(s.as_bytes().to_vec())
                            } else {
                                items.into_iter().next().unwrap_or(PV::Opaque)
                            }
                        } else {
                            PV::Opaque
                        }
                    }
                    ("torch.storage", "_load_from_bytes") => PV::Opaque,
                    ("torch", "Size") => {
                        // torch.Size is just a tuple of ints
                        args
                    }
                    _ => PV::Opaque,
                }
            }
            _ => PV::Opaque,
        }
    }

    /// Handle BINPERSID — PyTorch stores tensor data references as persistent IDs.
    ///
    /// The persistent ID tuple is typically:
    /// ("storage", storage_type_global, storage_key_str, device_str, numel)
    /// e.g. ("storage", <torch.FloatStorage>, "0", "cpu", 1024)
    fn resolve_persistent_id(&self, pid: PV) -> PV {
        if let PV::Tuple(items) = &pid {
            if items.len() >= 5 {
                if let Some(tag) = items[0].as_str() {
                    if tag == "storage" {
                        let dtype = match &items[1] {
                            PV::Global { name, .. } => {
                                storage_type_to_dtype(name).to_string()
                            }
                            PV::Str(s) => storage_type_to_dtype(s).to_string(),
                            _ => "float32".to_string(),
                        };
                        let key = match &items[2] {
                            PV::Str(s) => s.clone(),
                            PV::Int(i) => i.to_string(),
                            _ => String::new(),
                        };
                        return PV::Storage { key, dtype };
                    }
                }
            }
        }
        PV::Opaque
    }
}

/// Decode a pickle LONG from raw bytes (little-endian, two's complement).
fn long_from_bytes(bytes: &[u8]) -> i64 {
    if bytes.is_empty() {
        return 0;
    }
    let mut result: i64 = 0;
    for (i, &b) in bytes.iter().enumerate().take(8) {
        result |= (b as i64) << (i * 8);
    }
    // Sign-extend if the top bit of the last byte is set
    if bytes.len() <= 8 && (bytes[bytes.len() - 1] & 0x80) != 0 {
        for i in bytes.len()..8 {
            result |= 0xFFi64 << (i * 8);
        }
    }
    result
}

// ── Public API ──────────────────────────────────────────────────────────────

impl PickleIndex {
    /// Open a PyTorch ZIP checkpoint and extract its tensor index.
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let mut archive =
            zip::ZipArchive::new(file).map_err(|e| format!("invalid ZIP: {e}"))?;

        // Find the pickle file
        let pkl_name = find_pickle_entry(&mut archive)?;

        // Read pickle bytes
        let mut pkl_data = Vec::new();
        {
            let mut pkl_entry = archive
                .by_name(&pkl_name)
                .map_err(|e| format!("cannot read {pkl_name}: {e}"))?;
            pkl_entry
                .read_to_end(&mut pkl_data)
                .map_err(|e| format!("cannot read pickle data: {e}"))?;
        }

        // Build storage file map from ZIP entries
        let mut storage_files: HashMap<String, String> = HashMap::new();
        for i in 0..archive.len() {
            if let Ok(entry) = archive.by_index(i) {
                let entry_name = entry.name().to_string();
                // Match patterns like "archive/data/0" or "data/0"
                if let Some(key) = extract_storage_key(&entry_name) {
                    storage_files.insert(key, entry_name);
                }
            }
        }

        // Parse pickle
        let mut scanner = PickleScanner::new(pkl_data);
        let root = scanner.scan()?;

        // Extract state dict
        let state_dict = unwrap_state_dict(root);

        // Build tensor list
        let mut tensors = Vec::new();
        if let Some(entries) = state_dict.dict_entries() {
            for (key, val) in entries {
                if let (Some(name), PV::TensorRebuild { storage_key, dtype, shape, storage_offset }) =
                    (key.as_str(), val)
                {
                    let numel: usize = if shape.is_empty() {
                        0
                    } else {
                        shape.iter().product()
                    };
                    let byte_size = numel * dtype_byte_size(dtype);
                    tensors.push(PickleTensorInfo {
                        name: name.to_string(),
                        shape: shape.clone(),
                        dtype: dtype.clone(),
                        storage_key: storage_key.clone(),
                        storage_offset: *storage_offset,
                        byte_size,
                        numel,
                    });
                }
            }
        }

        Ok(PickleIndex {
            tensors,
            zip_path: path.to_path_buf(),
            storage_files,
        })
    }

    /// Read raw tensor bytes from the ZIP for a given tensor.
    pub fn read_tensor_bytes(&self, info: &PickleTensorInfo) -> Result<Vec<u8>, String> {
        let zip_entry_name = self
            .storage_files
            .get(&info.storage_key)
            .ok_or_else(|| {
                format!(
                    "storage key '{}' not found in ZIP for tensor '{}'",
                    info.storage_key, info.name
                )
            })?;

        let file = std::fs::File::open(&self.zip_path)
            .map_err(|e| format!("cannot open {}: {e}", self.zip_path.display()))?;
        let mut archive =
            zip::ZipArchive::new(file).map_err(|e| format!("invalid ZIP: {e}"))?;

        let mut entry = archive
            .by_name(zip_entry_name)
            .map_err(|e| format!("cannot read {zip_entry_name}: {e}"))?;

        let mut buf = Vec::new();
        entry
            .read_to_end(&mut buf)
            .map_err(|e| format!("cannot read storage data: {e}"))?;

        // Apply offset
        let elem_size = dtype_byte_size(&info.dtype);
        let byte_offset = info.storage_offset * elem_size;
        if byte_offset + info.byte_size > buf.len() {
            return Err(format!(
                "tensor '{}' data out of range: offset {} + size {} > storage size {}",
                info.name, byte_offset, info.byte_size, buf.len()
            ));
        }

        Ok(buf[byte_offset..byte_offset + info.byte_size].to_vec())
    }
}

/// Find the main pickle file within a ZIP archive.
fn find_pickle_entry<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
) -> Result<String, String> {
    let candidates = [
        "archive/data.pkl",
        "data.pkl",
    ];

    for name in &candidates {
        for i in 0..archive.len() {
            if let Ok(entry) = archive.by_index_raw(i) {
                if entry.name() == *name {
                    return Ok(name.to_string());
                }
            }
        }
    }

    // Fallback: find any .pkl file
    for i in 0..archive.len() {
        if let Ok(entry) = archive.by_index_raw(i) {
            let name = entry.name().to_string();
            if name.ends_with(".pkl") {
                return Ok(name);
            }
        }
    }

    Err("no .pkl file found in ZIP archive".into())
}

/// Extract storage key from a ZIP entry path.
/// "archive/data/0" → Some("0"), "data/1" → Some("1")
fn extract_storage_key(entry_name: &str) -> Option<String> {
    // Skip pickle and other non-data files
    if entry_name.ends_with(".pkl") || entry_name.ends_with("/") {
        return None;
    }

    // Match "archive/data/KEY" or "data/KEY"
    let parts: Vec<&str> = entry_name.rsplitn(2, '/').collect();
    if parts.len() == 2 {
        let parent = parts[1];
        let key = parts[0];
        if parent.ends_with("data") || parent.ends_with("data/") {
            return Some(key.to_string());
        }
        // Also match "archive/data"
        if parent.ends_with("/data") {
            return Some(key.to_string());
        }
    }
    None
}

/// Unwrap nested state dicts: {"state_dict": {...}} → inner dict.
fn unwrap_state_dict(root: PV) -> PV {
    let wrapper_keys = ["state_dict", "model_state_dict", "model"];

    if let Some(entries) = root.dict_entries() {
        // If the dict has a single key that maps to another dict, unwrap it
        for wrapper_key in &wrapper_keys {
            for (k, v) in entries {
                if let Some(key_str) = k.as_str() {
                    if key_str == *wrapper_key {
                        if v.dict_entries().is_some() {
                            return v.clone();
                        }
                    }
                }
            }
        }
    }

    root
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── long_from_bytes ──

    #[test]
    fn long_from_bytes_empty() {
        assert_eq!(long_from_bytes(&[]), 0);
    }

    #[test]
    fn long_from_bytes_positive() {
        // 255 = 0xFF stored as unsigned, but 1-byte → top bit set → -1 in two's complement
        assert_eq!(long_from_bytes(&[0xFF]), -1);
        // 1 in one byte
        assert_eq!(long_from_bytes(&[1]), 1);
        // 256 in two bytes LE
        assert_eq!(long_from_bytes(&[0, 1]), 256);
        // 127 (no sign extension)
        assert_eq!(long_from_bytes(&[127]), 127);
        // 128 → -128 (sign bit set in one byte)
        assert_eq!(long_from_bytes(&[128]), -128);
    }

    // ── dtype helpers ──

    #[test]
    fn test_storage_type_to_dtype() {
        assert_eq!(storage_type_to_dtype("FloatStorage"), "float32");
        assert_eq!(storage_type_to_dtype("HalfStorage"), "float16");
        assert_eq!(storage_type_to_dtype("BFloat16Storage"), "bfloat16");
        assert_eq!(storage_type_to_dtype("LongStorage"), "int64");
        assert_eq!(storage_type_to_dtype("ByteStorage"), "uint8");
        assert_eq!(storage_type_to_dtype("unknown"), "float32");
    }

    #[test]
    fn test_storage_type_to_dtype_trimmed_names() {
        // After suffix trimming (e.g. "HalfStorage" → "Half"), these must still resolve
        assert_eq!(storage_type_to_dtype("Float"), "float32");
        assert_eq!(storage_type_to_dtype("Half"), "float16");
        assert_eq!(storage_type_to_dtype("BFloat16"), "bfloat16");
        assert_eq!(storage_type_to_dtype("Double"), "float64");
        assert_eq!(storage_type_to_dtype("Byte"), "uint8");
        assert_eq!(storage_type_to_dtype("Char"), "int8");
        assert_eq!(storage_type_to_dtype("Short"), "int16");
        assert_eq!(storage_type_to_dtype("Int"), "int32");
        assert_eq!(storage_type_to_dtype("Long"), "int64");
    }

    #[test]
    fn test_dtype_byte_size() {
        assert_eq!(dtype_byte_size("float64"), 8);
        assert_eq!(dtype_byte_size("float32"), 4);
        assert_eq!(dtype_byte_size("float16"), 2);
        assert_eq!(dtype_byte_size("bfloat16"), 2);
        assert_eq!(dtype_byte_size("uint8"), 1);
    }

    // ── Security ──

    #[test]
    fn test_allowed_globals() {
        assert!(is_allowed_global("torch._utils", "_rebuild_tensor_v2"));
        assert!(is_allowed_global("collections", "OrderedDict"));
        assert!(is_allowed_global("torch", "FloatStorage"));
        assert!(!is_allowed_global("os", "system"));
        assert!(!is_allowed_global("subprocess", "call"));
    }

    #[test]
    fn test_dangerous_globals() {
        assert!(is_dangerous_global("os", "system"));
        assert!(is_dangerous_global("subprocess", "call"));
        assert!(is_dangerous_global("builtins", "eval"));
        assert!(is_dangerous_global("sys", "exit"));
        assert!(!is_dangerous_global("torch", "FloatStorage"));
    }

    // ── extract_storage_key ──

    #[test]
    fn test_extract_storage_key() {
        assert_eq!(
            extract_storage_key("archive/data/0"),
            Some("0".into())
        );
        assert_eq!(
            extract_storage_key("archive/data/123"),
            Some("123".into())
        );
        assert_eq!(extract_storage_key("data/0"), Some("0".into()));
        assert_eq!(extract_storage_key("archive/data.pkl"), None);
        assert_eq!(extract_storage_key("archive/data/"), None);
    }

    // ── Minimal pickle parsing ──

    #[test]
    fn scan_empty_dict() {
        // Protocol 2: PROTO 2, EMPTY_DICT, STOP
        let data = vec![0x80, 0x02, 0x7D, 0x2E];
        let mut scanner = PickleScanner::new(data);
        let result = scanner.scan().unwrap();
        assert!(matches!(result, PV::Dict(ref entries) if entries.is_empty()));
    }

    #[test]
    fn scan_dict_with_string_keys() {
        // Build: {}, MARK, "key1", 42, SETITEMS
        // PROTO 2, EMPTY_DICT, MARK, SHORT_BINUNICODE "key1", BININT1 42, SETITEMS, STOP
        let mut data = vec![0x80, 0x02]; // PROTO 2
        data.push(0x7D); // EMPTY_DICT
        data.push(0x28); // MARK
        data.push(0x8C); // SHORT_BINUNICODE
        data.push(4); // length
        data.extend_from_slice(b"key1");
        data.push(0x4B); // BININT1
        data.push(42);
        data.push(0x75); // SETITEMS
        data.push(0x2E); // STOP

        let mut scanner = PickleScanner::new(data);
        let result = scanner.scan().unwrap();
        if let PV::Dict(entries) = &result {
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].0.as_str(), Some("key1"));
            assert_eq!(entries[0].1.as_int(), Some(42));
        } else {
            panic!("expected Dict, got {:?}", result);
        }
    }

    #[test]
    fn scan_tuple() {
        // PROTO 2, MARK, BININT1 1, BININT1 2, BININT1 3, TUPLE, STOP
        let data = vec![0x80, 0x02, 0x28, 0x4B, 1, 0x4B, 2, 0x4B, 3, 0x74, 0x2E];
        let mut scanner = PickleScanner::new(data);
        let result = scanner.scan().unwrap();
        if let PV::Tuple(items) = &result {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0].as_int(), Some(1));
            assert_eq!(items[1].as_int(), Some(2));
            assert_eq!(items[2].as_int(), Some(3));
        } else {
            panic!("expected Tuple, got {:?}", result);
        }
    }

    #[test]
    fn scan_security_blocks_dangerous_globals() {
        // PROTO 2, GLOBAL "os\nsystem\n", SHORT_BINUNICODE "rm -rf /", TUPLE1, REDUCE, STOP
        let mut data = vec![0x80, 0x02];
        data.push(0x63); // GLOBAL
        data.extend_from_slice(b"os\nsystem\n");
        data.push(0x8C); // SHORT_BINUNICODE
        data.push(8);
        data.extend_from_slice(b"rm -rf /");
        data.push(0x85); // TUPLE1
        data.push(0x52); // REDUCE
        data.push(0x2E); // STOP

        let mut scanner = PickleScanner::new(data);
        let result = scanner.scan().unwrap();
        // Should be Opaque (the REDUCE result), not something executable
        assert!(matches!(result, PV::Opaque));
    }

    #[test]
    fn scan_memo_roundtrip() {
        // PROTO 2, BININT1 99, BINPUT 0, POP, BINGET 0, STOP
        let data = vec![0x80, 0x02, 0x4B, 99, 0x71, 0, 0x30, 0x68, 0, 0x2E];
        let mut scanner = PickleScanner::new(data);
        let result = scanner.scan().unwrap();
        assert_eq!(result.as_int(), Some(99));
    }

    // ── ZIP-based integration test ──

    #[test]
    fn test_minimal_pytorch_zip() {
        // Create a minimal .pt file: a ZIP containing data.pkl + data/0
        // The pickle encodes: {"weight": _rebuild_tensor_v2(storage, 0, (2, 3), (3, 1))}
        //
        // We use a realistic protocol-2 pickle that uses BINPERSID for storage.

        let pkl_bytes = build_test_pickle();
        let storage_bytes = vec![0u8; 2 * 3 * 4]; // 6 float32s = 24 bytes

        let tmp = std::env::temp_dir().join(format!(
            "test_pytorch_zip_{}.pt",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        // Write ZIP
        {
            let file = std::fs::File::create(&tmp).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("archive/data.pkl", options).unwrap();
            zip.write_all(&pkl_bytes).unwrap();
            zip.start_file("archive/data/0", options).unwrap();
            zip.write_all(&storage_bytes).unwrap();
            zip.finish().unwrap();
        }

        // Parse
        let index = PickleIndex::open(&tmp).unwrap();
        assert_eq!(index.tensors.len(), 1);
        assert_eq!(index.tensors[0].name, "weight");
        assert_eq!(index.tensors[0].shape, vec![2, 3]);
        assert_eq!(index.tensors[0].dtype, "float32");
        assert_eq!(index.tensors[0].numel, 6);
        assert_eq!(index.tensors[0].byte_size, 24);
        assert!(index.storage_files.contains_key("0"));

        // Read bytes
        let bytes = index.read_tensor_bytes(&index.tensors[0]).unwrap();
        assert_eq!(bytes.len(), 24);

        std::fs::remove_file(tmp).ok();
    }

    /// Build a minimal pickle bytestream that encodes:
    /// OrderedDict([("weight", _rebuild_tensor_v2(PersistentStorage("0", float32), 0, (2,3), (3,1)))])
    ///
    /// Uses BINPERSID for storage references (the way real PyTorch saves work).
    fn build_test_pickle() -> Vec<u8> {
        let mut p = Vec::new();

        // Protocol 2 header
        p.push(0x80); // PROTO
        p.push(0x02); // version 2

        // Push GLOBAL collections.OrderedDict
        p.push(0x63); // GLOBAL
        p.extend_from_slice(b"collections\nOrderedDict\n");

        // Empty tuple args for OrderedDict()
        p.push(0x29); // EMPTY_TUPLE

        // REDUCE → creates OrderedDict
        p.push(0x52); // REDUCE

        // Now push the entries via SETITEM
        // Key: "weight"
        p.push(0x8C); // SHORT_BINUNICODE
        p.push(6);
        p.extend_from_slice(b"weight");

        // Value: _rebuild_tensor_v2(storage, 0, (2,3), (3,1))
        // First, push the GLOBAL for _rebuild_tensor_v2
        p.push(0x63); // GLOBAL
        p.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n");

        // Now build args tuple: (storage, 0, (2,3), (3,1))
        // storage via BINPERSID
        p.push(0x28); // MARK (for outer tuple)

        // storage persistent id: ("storage", torch.FloatStorage, "0", "cpu", 6)
        p.push(0x28); // MARK (for persistent id tuple)
        p.push(0x8C); // SHORT_BINUNICODE "storage"
        p.push(7);
        p.extend_from_slice(b"storage");
        p.push(0x63); // GLOBAL torch.FloatStorage
        p.extend_from_slice(b"torch\nFloatStorage\n");
        p.push(0x8C); // SHORT_BINUNICODE "0"
        p.push(1);
        p.extend_from_slice(b"0");
        p.push(0x8C); // SHORT_BINUNICODE "cpu"
        p.push(3);
        p.extend_from_slice(b"cpu");
        p.push(0x4B); // BININT1 6
        p.push(6);
        p.push(0x74); // TUPLE → persistent id tuple
        p.push(0x51); // BINPERSID → resolves to Storage { key: "0", dtype: "float32" }

        // storage_offset = 0
        p.push(0x4B); // BININT1 0
        p.push(0);

        // shape = (2, 3)
        p.push(0x4B); p.push(2); // BININT1 2
        p.push(0x4B); p.push(3); // BININT1 3
        p.push(0x86); // TUPLE2

        // stride = (3, 1)
        p.push(0x4B); p.push(3); // BININT1 3
        p.push(0x4B); p.push(1); // BININT1 1
        p.push(0x86); // TUPLE2

        p.push(0x74); // TUPLE (args for _rebuild_tensor_v2)

        // REDUCE → TensorRebuild
        p.push(0x52); // REDUCE

        // SETITEM on the OrderedDict
        p.push(0x73); // SETITEM

        // STOP
        p.push(0x2E);

        p
    }

    #[test]
    fn test_unwrap_state_dict() {
        // Wrapping: {"state_dict": {"a": 1}}
        let inner = PV::Dict(vec![(PV::Str("a".into()), PV::Int(1))]);
        let outer = PV::Dict(vec![(PV::Str("state_dict".into()), inner.clone())]);
        let unwrapped = unwrap_state_dict(outer);
        if let PV::Dict(entries) = &unwrapped {
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].0.as_str(), Some("a"));
        } else {
            panic!("expected dict");
        }

        // No wrapping: {"a": 1} stays as-is
        let flat = PV::Dict(vec![(PV::Str("a".into()), PV::Int(1))]);
        let result = unwrap_state_dict(flat.clone());
        assert!(result.dict_entries().is_some());
    }

    use std::io::Write;

    #[test]
    fn test_multiple_tensors_in_zip() {
        // Two tensors: "layer.weight" (3,4) float16 and "layer.bias" (4,) float32
        let pkl_bytes = build_two_tensor_pickle();
        let storage_0 = vec![0u8; 3 * 4 * 2]; // 12 float16s = 24 bytes
        let storage_1 = vec![0u8; 4 * 4]; // 4 float32s = 16 bytes

        let tmp = std::env::temp_dir().join(format!(
            "test_multi_tensor_{}.pt",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        {
            let file = std::fs::File::create(&tmp).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("archive/data.pkl", options).unwrap();
            zip.write_all(&pkl_bytes).unwrap();
            zip.start_file("archive/data/0", options).unwrap();
            zip.write_all(&storage_0).unwrap();
            zip.start_file("archive/data/1", options).unwrap();
            zip.write_all(&storage_1).unwrap();
            zip.finish().unwrap();
        }

        let index = PickleIndex::open(&tmp).unwrap();
        assert_eq!(index.tensors.len(), 2);

        // Find by name
        let weight = index.tensors.iter().find(|t| t.name == "layer.weight").unwrap();
        assert_eq!(weight.shape, vec![3, 4]);
        assert_eq!(weight.dtype, "float16");
        assert_eq!(weight.numel, 12);
        assert_eq!(weight.byte_size, 24);

        let bias = index.tensors.iter().find(|t| t.name == "layer.bias").unwrap();
        assert_eq!(bias.shape, vec![4]);
        assert_eq!(bias.dtype, "float32");
        assert_eq!(bias.numel, 4);
        assert_eq!(bias.byte_size, 16);

        std::fs::remove_file(tmp).ok();
    }

    fn build_two_tensor_pickle() -> Vec<u8> {
        let mut p = Vec::new();

        p.push(0x80); p.push(0x02); // PROTO 2

        // OrderedDict()
        p.push(0x63);
        p.extend_from_slice(b"collections\nOrderedDict\n");
        p.push(0x29); // EMPTY_TUPLE
        p.push(0x52); // REDUCE

        // -- tensor 1: "layer.weight" float16 (3,4) from storage "0" --
        p.push(0x8C); p.push(12); p.extend_from_slice(b"layer.weight");
        p.push(0x63); p.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n");
        p.push(0x28); // MARK for args tuple
        // persistent storage
        p.push(0x28); // MARK for pid tuple
        p.push(0x8C); p.push(7); p.extend_from_slice(b"storage");
        p.push(0x63); p.extend_from_slice(b"torch\nHalfStorage\n");
        p.push(0x8C); p.push(1); p.extend_from_slice(b"0");
        p.push(0x8C); p.push(3); p.extend_from_slice(b"cpu");
        p.push(0x4B); p.push(12); // numel
        p.push(0x74); // TUPLE
        p.push(0x51); // BINPERSID
        p.push(0x4B); p.push(0); // offset
        p.push(0x4B); p.push(3); p.push(0x4B); p.push(4); p.push(0x86); // shape (3,4)
        p.push(0x4B); p.push(4); p.push(0x4B); p.push(1); p.push(0x86); // stride (4,1)
        p.push(0x74); // args TUPLE
        p.push(0x52); // REDUCE
        p.push(0x73); // SETITEM

        // -- tensor 2: "layer.bias" float32 (4,) from storage "1" --
        p.push(0x8C); p.push(10); p.extend_from_slice(b"layer.bias");
        p.push(0x63); p.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n");
        p.push(0x28); // MARK for args tuple
        p.push(0x28); // MARK for pid tuple
        p.push(0x8C); p.push(7); p.extend_from_slice(b"storage");
        p.push(0x63); p.extend_from_slice(b"torch\nFloatStorage\n");
        p.push(0x8C); p.push(1); p.extend_from_slice(b"1");
        p.push(0x8C); p.push(3); p.extend_from_slice(b"cpu");
        p.push(0x4B); p.push(4); // numel
        p.push(0x74); // TUPLE
        p.push(0x51); // BINPERSID
        p.push(0x4B); p.push(0); // offset
        p.push(0x4B); p.push(4); p.push(0x85); // shape (4,) via TUPLE1
        p.push(0x4B); p.push(1); p.push(0x85); // stride (1,) via TUPLE1
        p.push(0x74); // args TUPLE
        p.push(0x52); // REDUCE
        p.push(0x73); // SETITEM

        p.push(0x2E); // STOP
        p
    }
}
