use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};
use safetensors::tensor::{Dtype as StDtype, TensorView};
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

// ── Header-only file read ───────────────────────────────────────────────────

fn read_header(path: &str) -> Result<(HashMap<String, String>, serde_json::Value), SsError> {
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
    Ok((metadata, header))
}

// ── O_DIRECT chunked write ──────────────────────────────────────────────────

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
        fs::write(path, data)?;
        return Ok(());
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

        let mut offset = 0usize;
        while offset < data.len() {
            let remaining = data.len() - offset;
            let this_chunk = remaining.min(CHUNK_SIZE);
            let write_len = (this_chunk + ALIGN - 1) & !(ALIGN - 1);

            std::ptr::write_bytes(buf, 0, alloc_size);
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

// ── Serialize state_dict ────────────────────────────────────────────────────

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

        let cpu_tensor = value
            .call_method0("detach")
            .and_then(|t| t.call_method0("cpu"))
            .and_then(|t| t.call_method0("contiguous"))
            .map_err(|e| SsError::Other(format!("Cannot prepare tensor {name}: {e}")))?;

        let numel: usize = shape.iter().product();
        let nbytes = numel * dtype_element_size(st_dtype);

        let data = match extract_tensor_bytes_fast(py, &cpu_tensor, nbytes) {
            Ok(d) => d,
            Err(_) => extract_tensor_bytes_numpy(py, &cpu_tensor, &name)?,
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

fn extract_tensor_bytes_fast(
    py: Python<'_>,
    tensor: &Bound<'_, PyAny>,
    nbytes: usize,
) -> Result<Vec<u8>, SsError> {
    let ctypes = py
        .import_bound("ctypes")
        .map_err(|e| SsError::Other(e.to_string()))?;

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

    let bytes_obj = ctypes
        .call_method1("string_at", (ptr_val, nbytes))
        .map_err(|e| SsError::Other(format!("ctypes.string_at failed: {e}")))?;

    let data: Vec<u8> = bytes_obj
        .extract()
        .map_err(|e| SsError::Other(format!("bytes extract failed: {e}")))?;

    Ok(data)
}

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
    let (user_metadata, header) = read_header(path)?;

    let result = PyDict::new_bound(py);

    let meta_dict = PyDict::new_bound(py);
    for (k, v) in &user_metadata {
        meta_dict.set_item(k, v)?;
    }
    result.set_item("metadata", meta_dict)?;

    let tensors_dict = PyDict::new_bound(py);
    if let Some(obj) = header.as_object() {
        for (name, tensor_info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let info = PyDict::new_bound(py);

            if let Some(dtype_str) = tensor_info.get("dtype").and_then(|d| d.as_str()) {
                info.set_item("dtype", dtype_str)?;

                if let Some(shape_arr) = tensor_info.get("shape").and_then(|s| s.as_array()) {
                    let shape: Vec<u64> = shape_arr
                        .iter()
                        .filter_map(|v| v.as_u64())
                        .collect();

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

/// List tensor names in a file without loading any data. Fast header-only read.
#[pyfunction]
fn tensor_names(path: &str) -> PyResult<Vec<String>> {
    let (_, header) = read_header(path)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let mut names = Vec::new();
    if let Some(obj) = header.as_object() {
        for name in obj.keys() {
            if name != "__metadata__" {
                names.push(name.clone());
            }
        }
    }
    Ok(names)
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
    m.add_function(wrap_pyfunction!(training_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_names, m)?)?;
    Ok(())
}
