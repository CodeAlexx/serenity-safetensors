use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};
use safetensors::tensor::{Dtype as StDtype, View};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

// ── Error types ──────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
enum SsError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Other(String),
}

impl From<SsError> for PyErr {
    fn from(e: SsError) -> PyErr {
        match e {
            SsError::Io(e) => PyIOError::new_err(e.to_string()),
            SsError::SafeTensors(e) => PyRuntimeError::new_err(e.to_string()),
            SsError::Json(e) => PyRuntimeError::new_err(e.to_string()),
            SsError::Other(e) => PyRuntimeError::new_err(e),
        }
    }
}

// ── Dtype conversion ─────────────────────────────────────────────────────────

fn dtype_from_torch_str(s: &str) -> Result<StDtype, SsError> {
    match s {
        "torch.float64" | "float64" => Ok(StDtype::F64),
        "torch.float32" | "float32" => Ok(StDtype::F32),
        "torch.float16" | "float16" => Ok(StDtype::F16),
        "torch.bfloat16" | "bfloat16" => Ok(StDtype::BF16),
        "torch.float8_e4m3fn" | "float8_e4m3fn" | "float8_e4m3" => Ok(StDtype::F8_E4M3),
        "torch.float8_e5m2" | "float8_e5m2" => Ok(StDtype::F8_E5M2),
        "torch.int64" | "int64" => Ok(StDtype::I64),
        "torch.int32" | "int32" => Ok(StDtype::I32),
        "torch.int16" | "int16" => Ok(StDtype::I16),
        "torch.int8" | "int8" => Ok(StDtype::I8),
        "torch.uint8" | "uint8" => Ok(StDtype::U8),
        "torch.bool" | "bool" => Ok(StDtype::BOOL),
        other => Err(SsError::Other(format!("Unknown dtype: {other}"))),
    }
}

fn dtype_element_size(dtype: StDtype) -> usize {
    match dtype {
        StDtype::F64 | StDtype::I64 | StDtype::U64 => 8,
        StDtype::F32 | StDtype::I32 | StDtype::U32 => 4,
        StDtype::F16 | StDtype::BF16 | StDtype::I16 | StDtype::U16 => 2,
        StDtype::F8_E4M3 | StDtype::F8_E5M2 | StDtype::I8 | StDtype::U8 | StDtype::BOOL => 1,
        _ => 4,
    }
}

fn dtype_to_safetensors_str(dtype: StDtype) -> &'static str {
    match dtype {
        StDtype::BOOL => "BOOL",
        StDtype::U8 => "U8",
        StDtype::I8 => "I8",
        StDtype::F8_E5M2 => "F8_E5M2",
        StDtype::F8_E4M3 => "F8_E4M3",
        StDtype::I16 => "I16",
        StDtype::U16 => "U16",
        StDtype::F16 => "F16",
        StDtype::BF16 => "BF16",
        StDtype::I32 => "I32",
        StDtype::U32 => "U32",
        StDtype::F32 => "F32",
        StDtype::F64 => "F64",
        StDtype::I64 => "I64",
        StDtype::U64 => "U64",
        _ => "F32",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorLayoutEntry {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    nbytes: usize,
    data_offsets: (usize, usize),
    absolute_offsets: (usize, usize),
}

// ── Header-only file read ───────────────────────────────────────────────────

fn read_header_with_len(
    path: &str,
) -> Result<(HashMap<String, String>, serde_json::Value, usize), SsError> {
    use std::io::Read;
    let mut file = fs::File::open(path)?;

    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    if header_len > 100 * 1024 * 1024 {
        return Err(SsError::Other(format!(
            "Header length {header_len} seems unreasonable"
        )));
    }

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;

    let header: serde_json::Value = serde_json::from_slice(&header_buf)?;

    let mut metadata = HashMap::new();
    if let Some(md) = header.get("__metadata__") {
        if let Some(obj) = md.as_object() {
            for (k, v) in obj {
                if let Some(s) = v.as_str() {
                    metadata.insert(k.clone(), s.to_string());
                }
            }
        }
    }
    Ok((metadata, header, header_len))
}

fn collect_tensor_layout(
    path: &str,
) -> Result<(HashMap<String, String>, Vec<TensorLayoutEntry>), SsError> {
    let (metadata, header, header_len) = read_header_with_len(path)?;
    let data_base = 8 + header_len;

    let mut entries = Vec::new();
    if let Some(obj) = header.as_object() {
        for (name, tensor_info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = tensor_info
                .get("dtype")
                .and_then(|d| d.as_str())
                .unwrap_or("F32")
                .to_string();
            let shape: Vec<usize> = tensor_info
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            let data_offsets = tensor_info
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .map(|arr| {
                    let vals: Vec<usize> = arr
                        .iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect();
                    (
                        vals.first().copied().unwrap_or(0),
                        vals.get(1).copied().unwrap_or(0),
                    )
                })
                .unwrap_or((0, 0));
            let nbytes = data_offsets.1.saturating_sub(data_offsets.0);

            entries.push(TensorLayoutEntry {
                name: name.clone(),
                dtype,
                shape,
                nbytes,
                data_offsets,
                absolute_offsets: (data_base + data_offsets.0, data_base + data_offsets.1),
            });
        }
    }
    entries.sort_by_key(|entry| entry.data_offsets.0);

    Ok((metadata, entries))
}

fn read_shard_index(path: &str) -> Result<serde_json::Value, SsError> {
    let raw = fs::read_to_string(path)?;
    let parsed: serde_json::Value = serde_json::from_str(&raw)?;
    Ok(parsed)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ShardIndexSummary {
    metadata_strings: HashMap<String, String>,
    metadata_numbers: HashMap<String, u64>,
    weight_map: HashMap<String, String>,
    shards: HashMap<String, Vec<String>>,
}

fn summarize_shard_index(parsed: &serde_json::Value) -> ShardIndexSummary {
    let mut metadata_strings = HashMap::new();
    let mut metadata_numbers = HashMap::new();
    if let Some(metadata) = parsed.get("metadata").and_then(|value| value.as_object()) {
        for (key, value) in metadata {
            if let Some(as_str) = value.as_str() {
                metadata_strings.insert(key.clone(), as_str.to_string());
            } else if let Some(as_u64) = value.as_u64() {
                metadata_numbers.insert(key.clone(), as_u64);
            }
        }
    }

    let mut weight_map = HashMap::new();
    let mut shards: HashMap<String, Vec<String>> = HashMap::new();
    if let Some(weights) = parsed.get("weight_map").and_then(|value| value.as_object()) {
        for (tensor_name, shard_name) in weights {
            if let Some(shard) = shard_name.as_str() {
                weight_map.insert(tensor_name.clone(), shard.to_string());
                shards
                    .entry(shard.to_string())
                    .or_default()
                    .push(tensor_name.clone());
            }
        }
    }
    for tensors in shards.values_mut() {
        tensors.sort();
    }

    ShardIndexSummary {
        metadata_strings,
        metadata_numbers,
        weight_map,
        shards,
    }
}

// ── Streaming write helpers ─────────────────────────────────────────────────

struct PythonTensorView {
    dtype: StDtype,
    shape: Vec<usize>,
    data_ptr: usize,
    nbytes: usize,
    tensor: Py<PyAny>,
}

impl View for PythonTensorView {
    fn dtype(&self) -> StDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        let ptr = if self.nbytes == 0 {
            std::ptr::NonNull::<u8>::dangling().as_ptr()
        } else {
            self.data_ptr as *const u8
        };
        let _keep_alive = &self.tensor;
        unsafe { Cow::Borrowed(std::slice::from_raw_parts(ptr, self.nbytes)) }
    }

    fn data_len(&self) -> usize {
        self.nbytes
    }
}

fn metadata_from_py_dict(
    metadata: Option<&Bound<'_, PyDict>>,
) -> Result<Option<HashMap<String, String>>, SsError> {
    Ok(metadata.map(|m| {
            m.iter()
                .filter_map(|(k, v)| {
                    let key = k.extract::<String>().ok()?;
                    let val = v.extract::<String>().ok()?;
                    Some((key, val))
                })
                .collect()
        }))
}

fn prepare_python_tensor_views(
    state_dict: &Bound<'_, PyDict>,
) -> Result<Vec<(String, PythonTensorView)>, SsError> {
    let mut views = Vec::with_capacity(state_dict.len());

    for (key, value) in state_dict.iter() {
        let name: String = key.extract().map_err(|e| SsError::Other(e.to_string()))?;

        let dtype_str: String = value
            .getattr("dtype")
            .and_then(|d| d.str().map(|s| s.to_string()))
            .map_err(|e| SsError::Other(format!("Cannot get dtype for {name}: {e}")))?;
        let st_dtype = dtype_from_torch_str(&dtype_str)?;

        let shape: Vec<usize> = value
            .getattr("shape")
            .and_then(|s| s.extract())
            .map_err(|e| SsError::Other(format!("Cannot get shape for {name}: {e}")))?;

        let cpu_tensor = value
            .call_method0("detach")
            .and_then(|t| t.call_method0("cpu"))
            .and_then(|t| t.call_method0("contiguous"))
            .map_err(|e| SsError::Other(format!("Cannot prepare tensor {name}: {e}")))?;

        let numel: usize = shape.iter().product();
        let nbytes = numel * dtype_element_size(st_dtype);

        let storage = cpu_tensor
            .call_method0("untyped_storage")
            .map_err(|e| SsError::Other(format!("untyped_storage failed for {name}: {e}")))?;
        let data_ptr = storage
            .call_method0("data_ptr")
            .map_err(|e| SsError::Other(format!("data_ptr failed for {name}: {e}")))?
            .extract::<usize>()
            .map_err(|e| SsError::Other(format!("ptr extract failed for {name}: {e}")))?;

        if data_ptr == 0 && nbytes > 0 {
            return Err(SsError::Other(format!("Null data_ptr for {name}")));
        }

        views.push((
            name,
            PythonTensorView {
                dtype: st_dtype,
                shape,
                data_ptr,
                nbytes,
                tensor: cpu_tensor.unbind(),
            },
        ));
    }

    Ok(views)
}

fn prepare_write_plan<V: View>(
    mut tensors: Vec<(String, V)>,
    metadata: &Option<HashMap<String, String>>,
) -> Result<(u64, Vec<u8>, Vec<(String, V)>, usize), SsError> {
    tensors.sort_by(|(lname, left), (rname, right)| {
        right.dtype().cmp(&left.dtype()).then(lname.cmp(rname))
    });

    let mut header = serde_json::Map::new();
    if let Some(metadata) = metadata {
        header.insert(
            "__metadata__".to_string(),
            serde_json::to_value(metadata)?,
        );
    }

    let mut offset = 0usize;
    for (name, tensor) in &tensors {
        let tensor_len = tensor.data_len();
        let info = serde_json::json!({
            "dtype": dtype_to_safetensors_str(tensor.dtype()),
            "shape": tensor.shape(),
            "data_offsets": [offset, offset + tensor_len],
        });
        header.insert(name.clone(), info);
        offset += tensor_len;
    }

    let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))?;
    let extra = (8 - header_bytes.len() % 8) % 8;
    header_bytes.extend(std::iter::repeat(b' ').take(extra));
    let header_len = header_bytes.len() as u64;

    Ok((header_len, header_bytes, tensors, offset))
}

fn write_standard_streaming<V: View>(
    path: &Path,
    header_len: u64,
    header_bytes: &[u8],
    tensors: &[(String, V)],
) -> Result<(), SsError> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(header_bytes)?;
    for (_, tensor) in tensors {
        let data = tensor.data();
        writer.write_all(data.as_ref())?;
    }
    writer.flush()?;
    Ok(())
}

// ── O_DIRECT chunked write ──────────────────────────────────────────────────

#[cfg(unix)]
fn write_direct_streaming<V: View>(
    path: &Path,
    header_len: u64,
    header_bytes: &[u8],
    tensors: &[(String, V)],
    total_tensor_bytes: usize,
) -> Result<(), SsError> {
    let c_path = std::ffi::CString::new(path.to_str().unwrap_or(""))
        .map_err(|e| SsError::Other(e.to_string()))?;

    let fd = unsafe {
        libc::open(
            c_path.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC | libc::O_DIRECT,
            0o644,
        )
    };

    if fd < 0 {
        return write_standard_streaming(path, header_len, header_bytes, tensors);
    }

    const ALIGN: usize = 4096;
    const CHUNK_SIZE: usize = 4 * 1024 * 1024;

    let alloc_size = (CHUNK_SIZE + ALIGN - 1) & !(ALIGN - 1);
    let layout = std::alloc::Layout::from_size_align(alloc_size, ALIGN)
        .map_err(|e| SsError::Other(e.to_string()))?;

    unsafe {
        let buf = std::alloc::alloc_zeroed(layout);
        if buf.is_null() {
            libc::close(fd);
            return Err(SsError::Other("Failed to allocate aligned buffer".into()));
        }

        let mut buffered = 0usize;
        let mut exact_len = 0usize;

        let flush_buffer = |buffered: &mut usize| -> Result<(), SsError> {
            if *buffered == 0 {
                return Ok(());
            }
            let write_len = (*buffered + ALIGN - 1) & !(ALIGN - 1);
            if write_len > *buffered {
                std::ptr::write_bytes(buf.add(*buffered), 0, write_len - *buffered);
            }
            let written = libc::write(fd, buf as *const libc::c_void, write_len);
            if written < 0 {
                return Err(SsError::Io(std::io::Error::last_os_error()));
            }
            *buffered = 0;
            Ok(())
        };

        let copy_segment = |segment: &[u8],
                            buffered: &mut usize,
                            exact_len: &mut usize|
         -> Result<(), SsError> {
            let mut segment_offset = 0usize;
            *exact_len += segment.len();
            while segment_offset < segment.len() {
                let space = CHUNK_SIZE - *buffered;
                let take = (segment.len() - segment_offset).min(space);
                std::ptr::copy_nonoverlapping(
                    segment.as_ptr().add(segment_offset),
                    buf.add(*buffered),
                    take,
                );
                *buffered += take;
                segment_offset += take;
                if *buffered == CHUNK_SIZE {
                    flush_buffer(buffered)?;
                }
            }
            Ok(())
        };

        let header_len_bytes = header_len.to_le_bytes();
        let write_result = (|| -> Result<(), SsError> {
            copy_segment(&header_len_bytes, &mut buffered, &mut exact_len)?;
            copy_segment(header_bytes, &mut buffered, &mut exact_len)?;
            for (_, tensor) in tensors {
                let data = tensor.data();
                copy_segment(data.as_ref(), &mut buffered, &mut exact_len)?;
            }
            flush_buffer(&mut buffered)?;
            let expected_len = 8 + header_bytes.len() + total_tensor_bytes;
            if exact_len != expected_len {
                return Err(SsError::Other(format!(
                    "Streaming write length mismatch: expected {expected_len}, wrote {exact_len}"
                )));
            }
            libc::ftruncate(fd, expected_len as libc::off_t);
            Ok(())
        })();

        let close_result = libc::close(fd);
        std::alloc::dealloc(buf, layout);
        if let Err(err) = write_result {
            return Err(err);
        }
        if close_result < 0 {
            return Err(SsError::Io(std::io::Error::last_os_error()));
        }
    }

    Ok(())
}

#[cfg(not(unix))]
fn write_direct_streaming<V: View>(
    path: &Path,
    header_len: u64,
    header_bytes: &[u8],
    tensors: &[(String, V)],
    _total_tensor_bytes: usize,
) -> Result<(), SsError> {
    write_standard_streaming(path, header_len, header_bytes, tensors)
}

// ── Load via Python mmap (lazy, zero-copy views) ────────────────────────────
//
// Strategy: use Python's mmap module so the mmap object lives in Python's heap.
// torch.frombuffer() on a mmap slice creates a view — no data copy.
// The mmap stays alive as long as any tensor references it (Python refcount).
// OS pages in data on demand. Same behavior as safetensors.torch.load_file.

/// Internal: parse header from a Python mmap/buffer to get tensor layout.
/// Returns list of (name, dtype_str, shape, start_offset, end_offset).
fn parse_tensor_layout(
    py: Python<'_>,
    mmap_obj: &Bound<'_, PyAny>,
) -> PyResult<(Vec<(String, String, Vec<usize>, usize, usize)>, usize)> {
    // Read first 8 bytes to get header length
    let sl = PySlice::new_bound(py, 0, 8, 1);
    let header_len_bytes = mmap_obj.call_method1("__getitem__", (sl,))?;
    let header_len_bytes: Vec<u8> = header_len_bytes.extract()?;
    let header_len = u64::from_le_bytes(header_len_bytes.try_into().map_err(|_| {
        PyRuntimeError::new_err("Failed to read header length")
    })?) as usize;

    // Read header JSON
    let data_start = 8 + header_len;
    let sl = PySlice::new_bound(py, 8, data_start as isize, 1);
    let header_slice = mmap_obj.call_method1("__getitem__", (sl,))?;
    let header_bytes: Vec<u8> = header_slice.extract()?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("Invalid header JSON: {e}")))?;

    let mut tensors = Vec::new();
    if let Some(obj) = header.as_object() {
        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = info
                .get("dtype")
                .and_then(|d| d.as_str())
                .unwrap_or("F32")
                .to_string();
            let shape: Vec<usize> = info
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
                .unwrap_or_default();
            let offsets = info
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .map(|arr| {
                    let vals: Vec<usize> = arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect();
                    (vals.get(0).copied().unwrap_or(0), vals.get(1).copied().unwrap_or(0))
                })
                .unwrap_or((0, 0));

            tensors.push((name.clone(), dtype, shape, offsets.0, offsets.1));
        }
    }

    Ok((tensors, data_start))
}

/// Convert safetensors dtype string (from header) to torch dtype attribute name.
fn st_dtype_to_torch(dtype: &str) -> &str {
    match dtype {
        "F64" => "float64",
        "F32" => "float32",
        "F16" => "float16",
        "BF16" => "bfloat16",
        "F8_E4M3" => "float8_e4m3fn",
        "F8_E5M2" => "float8_e5m2",
        "I64" => "int64",
        "I32" => "int32",
        "I16" => "int16",
        "I8" => "int8",
        "U8" => "uint8",
        "BOOL" => "bool",
        _ => "float32",
    }
}

/// Create a tensor from a memoryview slice — true zero-copy.
/// memoryview slicing doesn't copy data (unlike mmap.__getitem__).
fn memview_slice_to_tensor(
    py: Python<'_>,
    torch: &Bound<'_, PyModule>,
    memview: &Bound<'_, PyAny>,
    dtype_str: &str,
    shape: &[usize],
    byte_start: usize,
    byte_end: usize,
    device: &str,
) -> PyResult<PyObject> {
    let torch_dtype = st_dtype_to_torch(dtype_str);

    // memoryview[start:end] is zero-copy — just adjusts pointer + length
    let sl = PySlice::new_bound(py, byte_start as isize, byte_end as isize, 1);
    let buf_slice = memview.call_method1("__getitem__", (sl,))?;

    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", torch.getattr(torch_dtype)?)?;

    let tensor = torch.call_method("frombuffer", (&buf_slice,), Some(&kwargs))?;
    let shape_tuple = PyTuple::new_bound(py, shape.iter().map(|&s| s as i64));
    let tensor = tensor.call_method1("reshape", (shape_tuple,))?;

    if device != "cpu" {
        let tensor = tensor.call_method1("to", (device,))?;
        Ok(tensor.into())
    } else {
        Ok(tensor.into())
    }
}

// ── Python-exposed functions ────────────────────────────────────────────────

/// Save a state_dict using O_DIRECT — no page cache pollution.
#[pyfunction]
#[pyo3(signature = (state_dict, path, metadata=None))]
fn save_file_direct(
    state_dict: &Bound<'_, PyDict>,
    path: &str,
    metadata: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let views = prepare_python_tensor_views(state_dict)?;
    let metadata = metadata_from_py_dict(metadata)?;
    let (header_len, header_bytes, views, total_tensor_bytes) =
        prepare_write_plan(views, &metadata)?;
    write_direct_streaming(
        Path::new(path),
        header_len,
        &header_bytes,
        &views,
        total_tensor_bytes,
    )?;
    Ok(())
}

/// Save a state_dict (normal write). Drop-in for safetensors.torch.save_file.
#[pyfunction]
#[pyo3(signature = (state_dict, path, metadata=None))]
fn save_file(
    state_dict: &Bound<'_, PyDict>,
    path: &str,
    metadata: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let views = prepare_python_tensor_views(state_dict)?;
    let metadata = metadata_from_py_dict(metadata)?;
    let (header_len, header_bytes, views, _) = prepare_write_plan(views, &metadata)?;
    write_standard_streaming(Path::new(path), header_len, &header_bytes, &views)?;
    Ok(())
}

/// Load all tensors from a safetensors file. Returns (dict, mmap_handle).
/// Tensors are lazy mmap views — no data copied until accessed.
/// The mmap_handle MUST be kept alive as long as tensors are in use.
/// For convenience, use the Python wrapper which handles this automatically.
#[pyfunction]
#[pyo3(signature = (path, device="cpu"))]
fn _load_file_raw(py: Python<'_>, path: &str, device: &str) -> PyResult<PyObject> {
    let mmap_mod = py.import_bound("mmap")?;
    let builtins = py.import_bound("builtins")?;
    let torch = py.import_bound("torch")?;

    // Open file and create read-only Python mmap
    let f = builtins.call_method1("open", (path, "rb"))?;
    let fileno = f.call_method0("fileno")?;
    let mmap_kwargs = PyDict::new_bound(py);
    mmap_kwargs.set_item("access", mmap_mod.getattr("ACCESS_READ")?)?;
    let py_mmap = mmap_mod.call_method("mmap", (fileno, 0i64), Some(&mmap_kwargs))?;
    f.call_method0("close")?;

    // Parse tensor layout from header
    let (tensors, data_start) = parse_tensor_layout(py, &py_mmap)?;

    // Create memoryview — slicing this is zero-copy (unlike mmap.__getitem__)
    let memview = builtins.call_method1("memoryview", (&py_mmap,))?;

    let result = PyDict::new_bound(py);

    for (name, dtype, shape, start, end) in &tensors {
        let tensor = memview_slice_to_tensor(
            py,
            &torch,
            &memview,
            dtype,
            shape,
            data_start + start,
            data_start + end,
            device,
        )?;
        result.set_item(name, tensor)?;
    }

    // Return tuple: (tensor_dict, mmap_handle)
    // Python wrapper stores mmap_handle to keep it alive
    let ret = PyTuple::new_bound(py, &[result.as_any(), py_mmap.as_any()]);
    Ok(ret.into())
}

/// Load selected tensors by name. Returns (dict, mmap_handle).
#[pyfunction]
#[pyo3(signature = (path, names, device="cpu"))]
fn _load_selective_raw(
    py: Python<'_>,
    path: &str,
    names: Vec<String>,
    device: &str,
) -> PyResult<PyObject> {
    let mmap_mod = py.import_bound("mmap")?;
    let builtins = py.import_bound("builtins")?;
    let torch = py.import_bound("torch")?;

    let f = builtins.call_method1("open", (path, "rb"))?;
    let fileno = f.call_method0("fileno")?;
    let mmap_kwargs = PyDict::new_bound(py);
    mmap_kwargs.set_item("access", mmap_mod.getattr("ACCESS_READ")?)?;
    let py_mmap = mmap_mod.call_method("mmap", (fileno, 0i64), Some(&mmap_kwargs))?;
    f.call_method0("close")?;

    let (tensors, data_start) = parse_tensor_layout(py, &py_mmap)?;
    let memview = builtins.call_method1("memoryview", (&py_mmap,))?;
    let name_set: std::collections::HashSet<&str> = names.iter().map(|s| s.as_str()).collect();

    let result = PyDict::new_bound(py);

    for (name, dtype, shape, start, end) in &tensors {
        if name_set.contains(name.as_str()) {
            let tensor = memview_slice_to_tensor(
                py, &torch, &memview, dtype, shape,
                data_start + start, data_start + end, device,
            )?;
            result.set_item(name, tensor)?;
        }
    }

    let ret = PyTuple::new_bound(py, &[result.as_any(), py_mmap.as_any()]);
    Ok(ret.into())
}

/// Load tensors matching a prefix. Returns (dict, mmap_handle).
#[pyfunction]
#[pyo3(signature = (path, prefix, device="cpu"))]
fn _load_by_prefix_raw(
    py: Python<'_>,
    path: &str,
    prefix: &str,
    device: &str,
) -> PyResult<PyObject> {
    let mmap_mod = py.import_bound("mmap")?;
    let builtins = py.import_bound("builtins")?;
    let torch = py.import_bound("torch")?;

    let f = builtins.call_method1("open", (path, "rb"))?;
    let fileno = f.call_method0("fileno")?;
    let mmap_kwargs = PyDict::new_bound(py);
    mmap_kwargs.set_item("access", mmap_mod.getattr("ACCESS_READ")?)?;
    let py_mmap = mmap_mod.call_method("mmap", (fileno, 0i64), Some(&mmap_kwargs))?;
    f.call_method0("close")?;

    let (tensors, data_start) = parse_tensor_layout(py, &py_mmap)?;
    let memview = builtins.call_method1("memoryview", (&py_mmap,))?;

    let result = PyDict::new_bound(py);

    for (name, dtype, shape, start, end) in &tensors {
        if name.starts_with(prefix) {
            let tensor = memview_slice_to_tensor(
                py, &torch, &memview, dtype, shape,
                data_start + start, data_start + end, device,
            )?;
            result.set_item(name, tensor)?;
        }
    }

    let ret = PyTuple::new_bound(py, &[result.as_any(), py_mmap.as_any()]);
    Ok(ret.into())
}

/// List tensor names and info without loading data. Header-only read.
#[pyfunction]
fn file_metadata(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let (user_metadata, entries) = collect_tensor_layout(path)?;

    let result = PyDict::new_bound(py);

    let meta_dict = PyDict::new_bound(py);
    for (k, v) in &user_metadata {
        meta_dict.set_item(k, v)?;
    }
    result.set_item("metadata", meta_dict)?;

    let tensors_dict = PyDict::new_bound(py);
    for entry in &entries {
        let info = PyDict::new_bound(py);
        let shape: Vec<u64> = entry.shape.iter().map(|&n| n as u64).collect();
        let data_offsets = (entry.data_offsets.0 as u64, entry.data_offsets.1 as u64);
        let absolute_offsets = (
            entry.absolute_offsets.0 as u64,
            entry.absolute_offsets.1 as u64,
        );
        info.set_item("dtype", &entry.dtype)?;
        info.set_item("shape", &shape)?;
        info.set_item("nbytes", entry.nbytes as u64)?;
        info.set_item("data_offsets", data_offsets)?;
        info.set_item("absolute_offsets", absolute_offsets)?;
        tensors_dict.set_item(&entry.name, info)?;
    }
    result.set_item("tensors", tensors_dict)?;

    Ok(result.into())
}

/// Return tensor layout with absolute offsets for Stagehand-style consumers.
#[pyfunction]
fn tensor_layout(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let (user_metadata, entries) = collect_tensor_layout(path)?;
    let result = PyDict::new_bound(py);

    let meta_dict = PyDict::new_bound(py);
    for (k, v) in &user_metadata {
        meta_dict.set_item(k, v)?;
    }
    result.set_item("metadata", meta_dict)?;

    let tensors = PyDict::new_bound(py);
    for entry in &entries {
        let item = PyDict::new_bound(py);
        let shape: Vec<u64> = entry.shape.iter().map(|&n| n as u64).collect();
        item.set_item("dtype", &entry.dtype)?;
        item.set_item("shape", shape)?;
        item.set_item("nbytes", entry.nbytes as u64)?;
        item.set_item(
            "data_offsets",
            (entry.data_offsets.0 as u64, entry.data_offsets.1 as u64),
        )?;
        item.set_item(
            "absolute_offsets",
            (
                entry.absolute_offsets.0 as u64,
                entry.absolute_offsets.1 as u64,
            ),
        )?;
        tensors.set_item(&entry.name, item)?;
    }
    result.set_item("tensors", tensors)?;
    Ok(result.into())
}

/// Build training metadata dict for checkpoint saves.
#[pyfunction]
#[pyo3(signature = (step, lr=None, loss=None, epoch=None, extra=None))]
fn training_metadata(
    py: Python<'_>,
    step: u64,
    lr: Option<f64>,
    loss: Option<f64>,
    epoch: Option<u64>,
    extra: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("step", step.to_string())?;
    if let Some(lr) = lr {
        dict.set_item("lr", format!("{lr:.8e}"))?;
    }
    if let Some(loss) = loss {
        dict.set_item("loss", format!("{loss:.6}"))?;
    }
    if let Some(epoch) = epoch {
        dict.set_item("epoch", epoch.to_string())?;
    }
    dict.set_item("format", "pt")?;
    if let Some(extra) = extra {
        for (k, v) in extra.iter() {
            let key: String = k.extract()?;
            let val: String = v.str()?.to_string();
            dict.set_item(key, val)?;
        }
    }
    Ok(dict.into())
}

/// List tensor names in a file without loading any data. Fast header-only read.
#[pyfunction]
fn tensor_names(path: &str) -> PyResult<Vec<String>> {
    let (_, entries) = collect_tensor_layout(path)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(entries.into_iter().map(|entry| entry.name).collect())
}

/// Parse a diffusers-style safetensors shard index JSON.
#[pyfunction]
fn shard_index(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let parsed = read_shard_index(path)?;
    let summary = summarize_shard_index(&parsed);
    let result = PyDict::new_bound(py);

    let metadata_dict = PyDict::new_bound(py);
    for (k, v) in &summary.metadata_strings {
        metadata_dict.set_item(k, v)?;
    }
    for (k, v) in &summary.metadata_numbers {
        metadata_dict.set_item(k, v)?;
    }
    result.set_item("metadata", metadata_dict)?;

    let weight_map_dict = PyDict::new_bound(py);
    let shards_dict = PyDict::new_bound(py);
    for (tensor_name, shard_name) in &summary.weight_map {
        weight_map_dict.set_item(tensor_name, shard_name)?;
    }
    for (shard, tensors) in &summary.shards {
        shards_dict.set_item(shard, tensors)?;
    }
    result.set_item("weight_map", weight_map_dict)?;
    result.set_item("shards", shards_dict)?;

    Ok(result.into())
}

// ── Module definition ───────────────────────────────────────────────────────

#[pymodule]
fn serenity_safetensors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save_file_direct, m)?)?;
    m.add_function(wrap_pyfunction!(save_file, m)?)?;
    m.add_function(wrap_pyfunction!(_load_file_raw, m)?)?;
    m.add_function(wrap_pyfunction!(_load_selective_raw, m)?)?;
    m.add_function(wrap_pyfunction!(_load_by_prefix_raw, m)?)?;
    m.add_function(wrap_pyfunction!(file_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_layout, m)?)?;
    m.add_function(wrap_pyfunction!(training_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_names, m)?)?;
    m.add_function(wrap_pyfunction!(shard_index, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use safetensors::{serialize, SafeTensors};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Clone)]
    struct ByteView {
        dtype: StDtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for ByteView {
        fn dtype(&self) -> StDtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    fn temp_path(name: &str, suffix: &str) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let pid = std::process::id();
        std::env::temp_dir()
            .join(format!("serenity_safetensors_{name}_{pid}_{nanos}.{suffix}"))
            .to_string_lossy()
            .into_owned()
    }

    fn write_test_safetensors(path: &str) {
        let weight_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bias_data: Vec<u8> = [5i8, 6i8, 7i8]
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let metadata = HashMap::from([
            ("family".to_string(), "ltx2".to_string()),
            ("step".to_string(), "3".to_string()),
        ]);
        let views = vec![
            (
                "weight".to_string(),
                TensorView::new(StDtype::F32, vec![2, 2], &weight_data).expect("weight view"),
            ),
            (
                "bias".to_string(),
                TensorView::new(StDtype::I8, vec![3], &bias_data).expect("bias view"),
            ),
        ];
        let serialized = serialize(views, &Some(metadata)).expect("serialize");
        fs::write(path, serialized).expect("write test safetensors");
    }

    #[test]
    fn dtype_aliases_cover_fp8_and_bfloat16() {
        assert_eq!(dtype_from_torch_str("torch.bfloat16").expect("bf16"), StDtype::BF16);
        assert_eq!(
            dtype_from_torch_str("float8_e4m3fn").expect("fp8 e4m3"),
            StDtype::F8_E4M3
        );
        assert_eq!(
            dtype_from_torch_str("torch.float8_e5m2").expect("fp8 e5m2"),
            StDtype::F8_E5M2
        );
    }

    #[test]
    fn collect_tensor_layout_reports_offsets_and_metadata() {
        let path = temp_path("layout", "safetensors");
        write_test_safetensors(&path);

        let (metadata, entries) = collect_tensor_layout(&path).expect("layout");
        assert_eq!(metadata.get("family").map(String::as_str), Some("ltx2"));
        assert_eq!(entries.len(), 2);

        let weight = entries.iter().find(|entry| entry.name == "weight").expect("weight entry");
        assert_eq!(weight.dtype, "F32");
        assert_eq!(weight.shape, vec![2, 2]);
        assert_eq!(weight.nbytes, 16);
        assert!(weight.absolute_offsets.0 >= 8);
        assert_eq!(
            weight.absolute_offsets.1 - weight.absolute_offsets.0,
            weight.data_offsets.1 - weight.data_offsets.0,
        );

        fs::remove_file(path).ok();
    }

    #[test]
    fn tensor_names_roundtrip() {
        let path = temp_path("metadata", "safetensors");
        write_test_safetensors(&path);

        let names = tensor_names(&path).expect("tensor names");
        assert_eq!(names, vec!["weight".to_string(), "bias".to_string()]);

        fs::remove_file(path).ok();
    }

    #[test]
    fn shard_index_groups_weight_map_by_file() {
        let path = temp_path("index", "json");
        let payload = serde_json::json!({
            "metadata": {"total_size": 1234},
            "weight_map": {
                "a.weight": "model-00001-of-00002.safetensors",
                "b.weight": "model-00002-of-00002.safetensors",
                "b.bias": "model-00002-of-00002.safetensors"
            }
        });
        fs::write(&path, serde_json::to_vec(&payload).expect("json bytes")).expect("write index");

        let parsed = read_shard_index(&path).expect("read shard index");
        let summary = summarize_shard_index(&parsed);
        assert_eq!(
            summary.metadata_numbers.get("total_size").copied(),
            Some(1234)
        );
        assert_eq!(
            summary
                .weight_map
                .get("a.weight")
                .map(String::as_str),
            Some("model-00001-of-00002.safetensors")
        );
        assert_eq!(
            summary
                .shards
                .get("model-00002-of-00002.safetensors")
                .cloned()
                .unwrap_or_default(),
            vec!["b.bias".to_string(), "b.weight".to_string()]
        );

        fs::remove_file(path).ok();
    }

    #[test]
    fn streaming_writer_roundtrips_metadata_and_layout() {
        let path = temp_path("streaming", "safetensors");
        let metadata = Some(HashMap::from([
            ("family".to_string(), "ltx2".to_string()),
            ("quant".to_string(), "eriquant".to_string()),
        ]));
        let tensors = vec![
            (
                "linear.weight".to_string(),
                ByteView {
                    dtype: StDtype::F32,
                    shape: vec![2, 2],
                    data: [1.0f32, 2.0, 3.0, 4.0]
                        .into_iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                },
            ),
            (
                "linear.bias".to_string(),
                ByteView {
                    dtype: StDtype::I16,
                    shape: vec![2],
                    data: [7i16, 8i16]
                        .into_iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                },
            ),
        ];

        let (header_len, header_bytes, tensors, _) =
            prepare_write_plan(tensors.clone(), &metadata).expect("prepare");
        write_standard_streaming(Path::new(&path), header_len, &header_bytes, &tensors)
            .expect("write standard");

        let file_bytes = fs::read(&path).expect("read file");
        let parsed = SafeTensors::deserialize(&file_bytes).expect("deserialize");
        assert_eq!(parsed.tensor("linear.weight").expect("weight").shape(), &[2, 2]);

        let (layout_metadata, entries) = collect_tensor_layout(&path).expect("layout");
        assert_eq!(layout_metadata.get("family").map(String::as_str), Some("ltx2"));
        assert_eq!(layout_metadata.get("quant").map(String::as_str), Some("eriquant"));
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().all(|entry| entry.absolute_offsets.1 > entry.absolute_offsets.0));

        fs::remove_file(path).ok();
    }

    #[test]
    fn direct_streaming_writer_roundtrips() {
        let path = temp_path("direct", "safetensors");
        let tensors = vec![(
            "tensor".to_string(),
            ByteView {
                dtype: StDtype::BF16,
                shape: vec![4],
                data: [1u16, 2u16, 3u16, 4u16]
                    .into_iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            },
        )];
        let metadata = Some(HashMap::from([("step".to_string(), "42".to_string())]));

        let (header_len, header_bytes, tensors, total_tensor_bytes) =
            prepare_write_plan(tensors, &metadata).expect("prepare");
        write_direct_streaming(
            Path::new(&path),
            header_len,
            &header_bytes,
            &tensors,
            total_tensor_bytes,
        )
        .expect("write direct");

        let file_bytes = fs::read(&path).expect("read file");
        let parsed = SafeTensors::deserialize(&file_bytes).expect("deserialize");
        assert_eq!(parsed.tensor("tensor").expect("tensor").dtype(), StDtype::BF16);
        let (layout_metadata, _) = collect_tensor_layout(&path).expect("layout");
        assert_eq!(layout_metadata.get("step").map(String::as_str), Some("42"));

        fs::remove_file(path).ok();
    }
}
