use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use safetensors::tensor::{Dtype as StDtype, SafeTensors, TensorView};
use std::collections::HashMap;
use std::fs;
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
// FIX #4: Added F8_E4M3 and F8_E5M2 for FP8 support

fn torch_dtype_str(dtype: StDtype) -> &'static str {
    match dtype {
        StDtype::F64 => "float64",
        StDtype::F32 => "float32",
        StDtype::F16 => "float16",
        StDtype::BF16 => "bfloat16",
        StDtype::F8_E4M3 => "float8_e4m3fn",
        StDtype::F8_E5M2 => "float8_e5m2",
        StDtype::I64 => "int64",
        StDtype::I32 => "int32",
        StDtype::I16 => "int16",
        StDtype::I8 => "int8",
        StDtype::U8 => "uint8",
        StDtype::BOOL => "bool",
        _ => "float32",
    }
}

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
        StDtype::F64 | StDtype::I64 => 8,
        StDtype::F32 | StDtype::I32 => 4,
        StDtype::F16 | StDtype::BF16 | StDtype::I16 => 2,
        StDtype::F8_E4M3 | StDtype::F8_E5M2 | StDtype::I8 | StDtype::U8 | StDtype::BOOL => 1,
        _ => 4,
    }
}

// ── Mmap-based file loading ──────────────────────────────────────────────────
// FIX #1: Use mmap instead of fs::read for loading. No heap copy of multi-GB files.

/// Memory-map a safetensors file. Returns the mmap and parsed SafeTensors.
/// The mmap stays alive as long as the caller holds it — tensors reference
/// directly into the mapped pages. OS handles paging, no userspace copy.
fn mmap_safetensors(path: &str) -> Result<memmap2::Mmap, SsError> {
    let file = fs::File::open(path)?;
    // SAFETY: We don't modify the file while mapped. The file is opened read-only.
    // If the file is modified externally while mapped, that's UB, but that's
    // the caller's problem (same as safetensors' own mmap loading).
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    Ok(mmap)
}

// ── Header-only read ─────────────────────────────────────────────────────────
// FIX #5: Read only the 8-byte length prefix + header bytes, not the entire file.

fn read_header(path: &str) -> Result<(HashMap<String, String>, serde_json::Value), SsError> {
    use std::io::Read;

    let mut file = fs::File::open(path)?;

    // Read 8-byte header length
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    // Sanity check: header shouldn't be > 100MB
    if header_len > 100 * 1024 * 1024 {
        return Err(SsError::Other(format!(
            "Header length {header_len} seems unreasonable"
        )));
    }

    // Read just the header
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;

    let header: serde_json::Value = serde_json::from_slice(&header_buf)?;

    // Extract user metadata
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

    Ok((metadata, header))
}

// ── O_DIRECT chunked write ───────────────────────────────────────────────────
// FIX #3: Chunked writes (4MB at a time) instead of single blocking write.

#[cfg(unix)]
fn write_direct(path: &Path, data: &[u8]) -> Result<(), SsError> {
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
        // O_DIRECT not supported (tmpfs, NFS, etc.) — fall back to normal write
        fs::write(path, data)?;
        return Ok(());
    }

    const ALIGN: usize = 4096;
    const CHUNK_SIZE: usize = 4 * 1024 * 1024; // 4MB chunks

    // Allocate one aligned buffer, reuse for all chunks
    let alloc_size = (CHUNK_SIZE + ALIGN - 1) & !(ALIGN - 1);
    let layout = std::alloc::Layout::from_size_align(alloc_size, ALIGN)
        .map_err(|e| SsError::Other(e.to_string()))?;

    unsafe {
        let buf = std::alloc::alloc_zeroed(layout);
        if buf.is_null() {
            libc::close(fd);
            return Err(SsError::Other("Failed to allocate aligned buffer".into()));
        }

        let mut offset = 0usize;
        while offset < data.len() {
            let remaining = data.len() - offset;
            let this_chunk = remaining.min(CHUNK_SIZE);
            // Pad to alignment for O_DIRECT
            let write_len = (this_chunk + ALIGN - 1) & !(ALIGN - 1);

            // Zero the buffer (important for padding bytes)
            std::ptr::write_bytes(buf, 0, alloc_size);
            // Copy this chunk
            std::ptr::copy_nonoverlapping(data.as_ptr().add(offset), buf, this_chunk);

            let written = libc::write(fd, buf as *const libc::c_void, write_len);
            if written < 0 {
                let err = std::io::Error::last_os_error();
                std::alloc::dealloc(buf, layout);
                libc::close(fd);
                return Err(SsError::Io(err));
            }

            offset += this_chunk;
        }

        // Truncate to exact size (last chunk may have been padded)
        libc::ftruncate(fd, data.len() as libc::off_t);
        libc::close(fd);
        std::alloc::dealloc(buf, layout);
    }

    Ok(())
}

#[cfg(not(unix))]
fn write_direct(path: &Path, data: &[u8]) -> Result<(), SsError> {
    fs::write(path, data)?;
    Ok(())
}

// ── Serialize state_dict ─────────────────────────────────────────────────────
// FIX #2: Use buffer protocol via ctypes to get raw pointer instead of numpy roundtrip.
// Falls back to numpy if ctypes approach fails (e.g., non-contiguous tensor).

struct TensorData {
    name: String,
    dtype: StDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn serialize_state_dict(
    py: Python<'_>,
    state_dict: &Bound<'_, PyDict>,
    metadata: Option<&Bound<'_, PyDict>>,
) -> Result<Vec<u8>, SsError> {
    let meta_map: Option<HashMap<String, String>> = metadata.map(|m| {
        m.iter()
            .filter_map(|(k, v)| {
                let key = k.extract::<String>().ok()?;
                let val = v.extract::<String>().ok()?;
                Some((key, val))
            })
            .collect()
    });

    let mut tensor_data_vec: Vec<TensorData> = Vec::new();

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

        // Ensure contiguous CPU tensor
        let cpu_tensor = value
            .call_method0("detach")
            .and_then(|t| t.call_method0("cpu"))
            .and_then(|t| t.call_method0("contiguous"))
            .map_err(|e| SsError::Other(format!("Cannot prepare tensor {name}: {e}")))?;

        // Calculate expected byte count
        let numel: usize = shape.iter().product();
        let nbytes = numel * dtype_element_size(st_dtype);

        // Try fast path: data_ptr() via untyped_storage to avoid copy
        let data = match extract_tensor_bytes_fast(py, &cpu_tensor, nbytes) {
            Ok(d) => d,
            Err(_) => {
                // Fallback: numpy roundtrip (handles edge cases)
                extract_tensor_bytes_numpy(py, &cpu_tensor, &name)?
            }
        };

        tensor_data_vec.push(TensorData {
            name,
            dtype: st_dtype,
            shape,
            data,
        });
    }

    let views: Vec<(String, TensorView<'_>)> = tensor_data_vec
        .iter()
        .map(|td| {
            (
                td.name.clone(),
                TensorView::new(td.dtype, td.shape.clone(), &td.data)
                    .expect("Invalid tensor view"),
            )
        })
        .collect();

    let serialized = safetensors::tensor::serialize(views, &meta_map)?;
    Ok(serialized)
}

/// Fast path: read raw bytes directly from tensor storage via data_ptr + ctypes.
/// Zero-copy read from PyTorch's memory — we only copy into our Vec<u8>.
fn extract_tensor_bytes_fast(
    py: Python<'_>,
    tensor: &Bound<'_, PyAny>,
    nbytes: usize,
) -> Result<Vec<u8>, SsError> {
    let ctypes = py
        .import_bound("ctypes")
        .map_err(|e| SsError::Other(e.to_string()))?;

    // Get raw data pointer: tensor.untyped_storage().data_ptr()
    let storage = tensor
        .call_method0("untyped_storage")
        .map_err(|e| SsError::Other(format!("untyped_storage failed: {e}")))?;
    let data_ptr = storage
        .call_method0("data_ptr")
        .map_err(|e| SsError::Other(format!("data_ptr failed: {e}")))?;
    let ptr_val: usize = data_ptr
        .extract()
        .map_err(|e| SsError::Other(format!("ptr extract failed: {e}")))?;

    if ptr_val == 0 {
        return Err(SsError::Other("Null data_ptr".into()));
    }

    // Use ctypes to read nbytes from the pointer
    // ctypes.string_at(ptr, nbytes) -> bytes
    let bytes_obj = ctypes
        .call_method1("string_at", (ptr_val, nbytes))
        .map_err(|e| SsError::Other(format!("ctypes.string_at failed: {e}")))?;

    let data: Vec<u8> = bytes_obj
        .extract()
        .map_err(|e| SsError::Other(format!("bytes extract failed: {e}")))?;

    Ok(data)
}

/// Fallback: numpy roundtrip for tensors where fast path fails.
fn extract_tensor_bytes_numpy(
    py: Python<'_>,
    tensor: &Bound<'_, PyAny>,
    name: &str,
) -> Result<Vec<u8>, SsError> {
    let numpy = py
        .import_bound("numpy")
        .map_err(|e| SsError::Other(e.to_string()))?;

    let np_array = numpy
        .call_method1("asarray", (tensor,))
        .map_err(|e| SsError::Other(format!("numpy asarray failed for {name}: {e}")))?;

    let bytes_obj = np_array
        .call_method0("tobytes")
        .map_err(|e| SsError::Other(format!("tobytes failed for {name}: {e}")))?;

    let data: Vec<u8> = bytes_obj
        .extract()
        .map_err(|e| SsError::Other(format!("bytes extract failed for {name}: {e}")))?;

    Ok(data)
}

// ── Tensor reconstruction ────────────────────────────────────────────────────

fn bytes_to_torch_tensor(
    py: Python<'_>,
    torch: &Bound<'_, PyModule>,
    view: &safetensors::tensor::TensorView<'_>,
    device: &str,
) -> PyResult<PyObject> {
    let dtype_str = torch_dtype_str(view.dtype());
    let shape: Vec<usize> = view.shape().to_vec();

    let bytes = PyBytes::new_bound(py, view.data());
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", torch.getattr(dtype_str)?)?;

    let tensor = torch.call_method("frombuffer", (bytes,), Some(&kwargs))?;
    let tensor = tensor.call_method1("reshape", (shape,))?;
    // clone() to own the memory — frombuffer shares the mmap/bytes backing
    let tensor = tensor.call_method0("clone")?;

    if device != "cpu" {
        let tensor = tensor.call_method1("to", (device,))?;
        Ok(tensor.into())
    } else {
        Ok(tensor.into())
    }
}

// ── Python-exposed functions ─────────────────────────────────────────────────

/// Save a state_dict using O_DIRECT — no page cache pollution.
/// Writes in 4MB aligned chunks. This is the key function for training checkpoints.
#[pyfunction]
#[pyo3(signature = (state_dict, path, metadata=None))]
fn save_file_direct(
    py: Python<'_>,
    state_dict: &Bound<'_, PyDict>,
    path: &str,
    metadata: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let data = serialize_state_dict(py, state_dict, metadata)?;
    write_direct(Path::new(path), &data)?;
    Ok(())
}

/// Save a state_dict (normal write). Drop-in for safetensors.torch.save_file.
#[pyfunction]
#[pyo3(signature = (state_dict, path, metadata=None))]
fn save_file(
    py: Python<'_>,
    state_dict: &Bound<'_, PyDict>,
    path: &str,
    metadata: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let data = serialize_state_dict(py, state_dict, metadata)?;
    fs::write(path, data)?;
    Ok(())
}

/// Load all tensors from a safetensors file via mmap. Returns dict of name -> torch.Tensor.
/// Drop-in replacement for safetensors.torch.load_file.
#[pyfunction]
#[pyo3(signature = (path, device="cpu"))]
fn load_file(py: Python<'_>, path: &str, device: &str) -> PyResult<PyObject> {
    let mmap = mmap_safetensors(path)?;
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let torch = py.import_bound("torch")?;
    let result = PyDict::new_bound(py);

    for (name, view) in st.tensors() {
        let tensor = bytes_to_torch_tensor(py, &torch, &view, device)?;
        result.set_item(name, tensor)?;
    }

    Ok(result.into())
}

/// Load selected tensors by name. Only materializes the tensors you ask for.
#[pyfunction]
#[pyo3(signature = (path, names, device="cpu"))]
fn load_selective(
    py: Python<'_>,
    path: &str,
    names: Vec<String>,
    device: &str,
) -> PyResult<PyObject> {
    let mmap = mmap_safetensors(path)?;
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let torch = py.import_bound("torch")?;
    let result = PyDict::new_bound(py);

    for name in &names {
        if let Ok(view) = st.tensor(name) {
            let tensor = bytes_to_torch_tensor(py, &torch, &view, device)?;
            result.set_item(name, tensor)?;
        }
    }

    Ok(result.into())
}

/// Load tensors whose names start with a given prefix.
#[pyfunction]
#[pyo3(signature = (path, prefix, device="cpu"))]
fn load_by_prefix(
    py: Python<'_>,
    path: &str,
    prefix: &str,
    device: &str,
) -> PyResult<PyObject> {
    let mmap = mmap_safetensors(path)?;
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let torch = py.import_bound("torch")?;
    let result = PyDict::new_bound(py);

    for (name, view) in st.tensors() {
        if name.starts_with(prefix) {
            let tensor = bytes_to_torch_tensor(py, &torch, &view, device)?;
            result.set_item(&name, tensor)?;
        }
    }

    Ok(result.into())
}

/// Get file metadata + tensor info without loading tensor data.
/// Only reads the 8-byte length prefix + header bytes from disk.
#[pyfunction]
fn file_metadata(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let (user_metadata, header) = read_header(path)?;

    let result = PyDict::new_bound(py);

    // User metadata
    let meta_dict = PyDict::new_bound(py);
    for (k, v) in &user_metadata {
        meta_dict.set_item(k, v)?;
    }
    result.set_item("metadata", meta_dict)?;

    // Tensor info from header (no tensor data touched)
    let tensors_dict = PyDict::new_bound(py);
    if let Some(obj) = header.as_object() {
        for (name, tensor_info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let info = PyDict::new_bound(py);

            if let Some(dtype_str) = tensor_info.get("dtype").and_then(|d| d.as_str()) {
                info.set_item("dtype", dtype_str)?;

                // Parse shape
                if let Some(shape_arr) = tensor_info.get("shape").and_then(|s| s.as_array()) {
                    let shape: Vec<u64> = shape_arr
                        .iter()
                        .filter_map(|v| v.as_u64())
                        .collect();

                    // Calculate nbytes from dtype + shape
                    let numel: u64 = shape.iter().product();
                    let elem_size = match dtype_str {
                        "F64" | "I64" => 8u64,
                        "F32" | "I32" => 4,
                        "F16" | "BF16" | "I16" => 2,
                        "F8_E4M3" | "F8_E5M2" | "I8" | "U8" | "BOOL" => 1,
                        _ => 4,
                    };

                    info.set_item("shape", &shape)?;
                    info.set_item("nbytes", numel * elem_size)?;
                }

                // Also include offsets if available
                if let Some(offsets) = tensor_info.get("data_offsets").and_then(|o| o.as_array()) {
                    let offs: Vec<u64> = offsets.iter().filter_map(|v| v.as_u64()).collect();
                    if offs.len() == 2 {
                        info.set_item("data_offsets", (&offs[0], &offs[1]))?;
                    }
                }
            }

            tensors_dict.set_item(name, info)?;
        }
    }
    result.set_item("tensors", tensors_dict)?;

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

// ── Module definition ────────────────────────────────────────────────────────

#[pymodule]
fn serenity_safetensors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save_file_direct, m)?)?;
    m.add_function(wrap_pyfunction!(save_file, m)?)?;
    m.add_function(wrap_pyfunction!(load_file, m)?)?;
    m.add_function(wrap_pyfunction!(load_selective, m)?)?;
    m.add_function(wrap_pyfunction!(load_by_prefix, m)?)?;
    m.add_function(wrap_pyfunction!(file_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(training_metadata, m)?)?;

    let torch_mod = PyModule::new_bound(m.py(), "torch")?;
    torch_mod.add_function(wrap_pyfunction!(save_file_direct, &torch_mod)?)?;
    torch_mod.add_function(wrap_pyfunction!(save_file, &torch_mod)?)?;
    torch_mod.add_function(wrap_pyfunction!(load_file, &torch_mod)?)?;
    torch_mod.add_function(wrap_pyfunction!(load_selective, &torch_mod)?)?;
    torch_mod.add_function(wrap_pyfunction!(load_by_prefix, &torch_mod)?)?;
    torch_mod.add_function(wrap_pyfunction!(file_metadata, &torch_mod)?)?;
    torch_mod.add_function(wrap_pyfunction!(training_metadata, &torch_mod)?)?;
    m.add_submodule(&torch_mod)?;

    Ok(())
}
