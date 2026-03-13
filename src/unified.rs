//! Unified `load_model` — single entry point that dispatches to format-specific loaders.
//!
//! Supports: safetensors, GGUF, PyTorch ZIP, diffusers directories.
//! Returns a Python dict mapping tensor names to `torch.Tensor` (dense) or
//! `QuantizedTensor` (GGUF quantized types).

use std::path::Path;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyTuple};

use crate::diffusers::{DiffusersLayout, WeightSource};
use crate::format_detect::{detect_format, ModelFormat};
use crate::gguf::{GgufIndex, GgufQuantType};
use crate::gguf_dequant;
use crate::quantized_tensor::QuantizedTensor;
use crate::pytorch::PickleIndex;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Open a read-only Python mmap for a file path.
fn open_py_mmap<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyAny>> {
    let mmap_mod = py.import_bound("mmap")?;
    let builtins = py.import_bound("builtins")?;
    let f = builtins.call_method1("open", (path, "rb"))?;
    let fileno = f.call_method0("fileno")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("access", mmap_mod.getattr("ACCESS_READ")?)?;
    let py_mmap = mmap_mod.call_method("mmap", (fileno, 0i64), Some(&kwargs))?;
    f.call_method0("close")?;
    Ok(py_mmap)
}

/// Parse tensor layout from a safetensors mmap and load all tensors into `result`.
/// Returns the mmap object (caller must keep alive).
fn load_safetensors_into_dict<'py>(
    py: Python<'py>,
    path: &str,
    strip_prefix: Option<&str>,
    result: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let builtins = py.import_bound("builtins")?;
    let torch = py.import_bound("torch")?;
    let py_mmap = open_py_mmap(py, path)?;

    // Parse header
    let sl = PySlice::new_bound(py, 0, 8, 1);
    let header_len_bytes: Vec<u8> = py_mmap.call_method1("__getitem__", (sl,))?.extract()?;
    let header_len = u64::from_le_bytes(
        header_len_bytes
            .try_into()
            .map_err(|_| PyRuntimeError::new_err("Failed to read header length"))?,
    ) as usize;

    let data_start = 8 + header_len;
    let sl = PySlice::new_bound(py, 8, data_start as isize, 1);
    let header_bytes: Vec<u8> = py_mmap.call_method1("__getitem__", (sl,))?.extract()?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("Invalid header JSON: {e}")))?;

    let memview = builtins.call_method1("memoryview", (&py_mmap,))?;

    if let Some(obj) = header.as_object() {
        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype_str = info
                .get("dtype")
                .and_then(|d| d.as_str())
                .unwrap_or("F32");
            let shape: Vec<usize> = info
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            let offsets = info
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .map(|arr| {
                    let vals: Vec<usize> = arr
                        .iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect();
                    (vals.first().copied().unwrap_or(0), vals.get(1).copied().unwrap_or(0))
                })
                .unwrap_or((0, 0));

            let byte_start = data_start + offsets.0;
            let byte_end = data_start + offsets.1;

            let torch_dtype = st_dtype_to_torch(dtype_str);
            let sl = PySlice::new_bound(py, byte_start as isize, byte_end as isize, 1);
            let buf_slice = memview.call_method1("__getitem__", (sl,))?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("dtype", torch.getattr(torch_dtype)?)?;
            let tensor = torch.call_method("frombuffer", (&buf_slice,), Some(&kwargs))?;
            let shape_tuple = PyTuple::new_bound(py, shape.iter().map(|&s| s as i64));
            let tensor = tensor.call_method1("reshape", (shape_tuple,))?;

            let output_name = maybe_strip_prefix(name, strip_prefix);
            result.set_item(output_name, tensor)?;
        }
    }

    Ok(py_mmap)
}

/// Convert safetensors dtype string to torch dtype attribute name.
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

/// Optionally strip a prefix from a tensor name.
fn maybe_strip_prefix(name: &str, prefix: Option<&str>) -> String {
    match prefix {
        Some(p) => {
            if let Some(stripped) = name.strip_prefix(p) {
                stripped.to_string()
            } else {
                name.to_string()
            }
        }
        None => name.to_string(),
    }
}

/// Convert PyTorch dtype string to torch attribute name.
fn pytorch_dtype_to_torch(dtype: &str) -> &str {
    match dtype {
        "float32" => "float32",
        "float16" => "float16",
        "bfloat16" => "bfloat16",
        "float64" => "float64",
        "int64" => "int64",
        "int32" => "int32",
        "int16" => "int16",
        "int8" => "int8",
        "uint8" => "uint8",
        _ => "float32",
    }
}

// ── Format-specific loaders ─────────────────────────────────────────────────

/// Load safetensors (single file).
fn load_safetensors_model(py: Python, path: &str, strip_prefix: Option<&str>) -> PyResult<PyObject> {
    let result = PyDict::new_bound(py);
    let py_mmap = load_safetensors_into_dict(py, path, strip_prefix, &result)?;

    let info = PyDict::new_bound(py);
    info.set_item("format", "safetensors")?;
    info.set_item("path", path)?;

    let mmap_list = PyList::new_bound(py, &[py_mmap.unbind()]);
    let ret = PyTuple::new_bound(py, &[result.as_any(), info.as_any(), mmap_list.as_any()]);
    Ok(ret.into())
}

/// Load GGUF file. Dense types (F16/F32/BF16/F64) get dequanted eagerly to BF16.
/// Quantized types become QuantizedTensor with raw bytes reference.
fn load_gguf_model(py: Python, path: &str, strip_prefix: Option<&str>) -> PyResult<PyObject> {
    let idx = GgufIndex::open(Path::new(path))
        .map_err(|e| PyValueError::new_err(e))?;

    let torch = py.import_bound("torch")?;
    let builtins = py.import_bound("builtins")?;

    // Open a Python mmap so QuantizedTensor can hold memoryview slices
    let py_mmap = open_py_mmap(py, path)?;
    let memview = builtins.call_method1("memoryview", (&py_mmap,))?;

    let result = PyDict::new_bound(py);

    for (i, tinfo) in idx.tensors.iter().enumerate() {
        let output_name = maybe_strip_prefix(&tinfo.name, strip_prefix);
        let abs_offset = idx.data_offset + tinfo.offset as usize;
        let abs_end = abs_offset + tinfo.byte_size;

        if is_dense_gguf_type(tinfo.quant_type) {
            // Dequant eagerly to BF16 torch.Tensor
            let raw_data = idx.tensor_data_by_index(i)
                .map_err(|e| PyRuntimeError::new_err(e))?;

            let quant_type = tinfo.quant_type;
            let param_count = tinfo.param_count;

            let bf16_bytes = py
                .allow_threads(move || {
                    let bf16_vec = gguf_dequant::dequant_to_bf16(raw_data, quant_type, param_count)?;
                    let raw: &[u8] = bytemuck::cast_slice(&bf16_vec);
                    Ok::<Vec<u8>, String>(raw.to_vec())
                })
                .map_err(|e| PyValueError::new_err(e))?;

            let bf16_dtype = torch.getattr("bfloat16")?;
            let py_bytes = pyo3::types::PyBytes::new_bound(py, &bf16_bytes);
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("dtype", bf16_dtype)?;
            let tensor = torch.call_method("frombuffer", (py_bytes,), Some(&kwargs))?;
            let tensor = tensor.call_method0("clone")?;
            let shape_tuple = PyTuple::new_bound(py, &tinfo.shape);
            let tensor = tensor.call_method1("reshape", (shape_tuple,))?;

            result.set_item(output_name, tensor)?;
        } else {
            // Quantized: wrap in QuantizedTensor with memoryview slice
            let sl = PySlice::new_bound(py, abs_offset as isize, abs_end as isize, 1);
            let data_slice = memview.call_method1("__getitem__", (sl,))?;

            let qt = QuantizedTensor {
                name: output_name.clone(),
                shape: tinfo.shape.clone(),
                quant_type: tinfo.quant_type,
                param_count: tinfo.param_count,
                byte_size: tinfo.byte_size,
                data: data_slice.unbind(),
            };
            result.set_item(output_name, qt.into_py(py))?;
        }
    }

    let info = PyDict::new_bound(py);
    info.set_item("format", "gguf")?;
    info.set_item("path", path)?;
    info.set_item("version", idx.version)?;

    // Export metadata
    let meta = PyDict::new_bound(py);
    for (k, v) in &idx.metadata {
        meta.set_item(k, v.to_string_lossy())?;
    }
    info.set_item("metadata", meta)?;

    let mmap_list = PyList::new_bound(py, &[py_mmap.unbind()]);
    let ret = PyTuple::new_bound(py, &[result.as_any(), info.as_any(), mmap_list.as_any()]);
    Ok(ret.into())
}

/// Check if a GGUF type should be treated as dense (dequant to BF16).
fn is_dense_gguf_type(qt: GgufQuantType) -> bool {
    matches!(
        qt,
        GgufQuantType::F32
            | GgufQuantType::F16
            | GgufQuantType::BF16
            | GgufQuantType::F64
            | GgufQuantType::I8
            | GgufQuantType::I16
            | GgufQuantType::I32
            | GgufQuantType::I64
    )
}

/// Load PyTorch ZIP checkpoint.
fn load_pytorch_model(py: Python, path: &str, strip_prefix: Option<&str>) -> PyResult<PyObject> {
    let index = PickleIndex::open(Path::new(path))
        .map_err(|e| PyRuntimeError::new_err(e))?;

    let torch = py.import_bound("torch")?;
    let result = PyDict::new_bound(py);

    for tinfo in &index.tensors {
        let raw_bytes = index
            .read_tensor_bytes(tinfo)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let torch_dtype = pytorch_dtype_to_torch(&tinfo.dtype);
        let actual_dtype = torch.getattr(torch_dtype)?;

        let py_bytes = pyo3::types::PyBytes::new_bound(py, &raw_bytes);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", actual_dtype)?;
        let tensor = torch.call_method("frombuffer", (py_bytes,), Some(&kwargs))?;
        let tensor = tensor.call_method0("clone")?;

        let shape: Vec<i64> = tinfo.shape.iter().map(|&s| s as i64).collect();
        let py_shape = PyTuple::new_bound(py, &shape);
        let tensor = tensor.call_method1("reshape", (py_shape,))?;

        let output_name = maybe_strip_prefix(&tinfo.name, strip_prefix);
        result.set_item(output_name, tensor)?;
    }

    let info = PyDict::new_bound(py);
    info.set_item("format", "pytorch_zip")?;
    info.set_item("path", path)?;
    info.set_item("tensor_count", index.tensors.len())?;

    // No mmap handles for pytorch (data is copied via read_tensor_bytes)
    let mmap_list = PyList::new_bound(py, Vec::<PyObject>::new());
    let ret = PyTuple::new_bound(py, &[result.as_any(), info.as_any(), mmap_list.as_any()]);
    Ok(ret.into())
}

/// Load diffusers directory by iterating components and loading their weight files.
fn load_diffusers_model(py: Python, path: &str, strip_prefix: Option<&str>) -> PyResult<PyObject> {
    let layout = DiffusersLayout::open(Path::new(path))
        .map_err(|e| PyValueError::new_err(e))?;

    let result = PyDict::new_bound(py);
    let mut mmap_handles: Vec<PyObject> = Vec::new();

    for comp in &layout.components {
        let prefix = format!("{}/", comp.name);

        match &comp.weight_source {
            WeightSource::SingleSafetensors(st_path) => {
                let st_path_str = st_path.to_string_lossy().to_string();
                let comp_dict = PyDict::new_bound(py);
                let py_mmap = load_safetensors_into_dict(py, &st_path_str, None, &comp_dict)?;
                mmap_handles.push(py_mmap.unbind());

                // Re-key with component prefix, then apply strip_prefix
                for (key, value) in comp_dict.iter() {
                    let key_str: String = key.extract()?;
                    let prefixed = format!("{prefix}{key_str}");
                    let output_name = maybe_strip_prefix(&prefixed, strip_prefix);
                    result.set_item(output_name, value)?;
                }
            }
            WeightSource::ShardedSafetensors { shard_paths, .. } => {
                for shard_path in shard_paths {
                    if shard_path.is_file() {
                        let shard_str = shard_path.to_string_lossy().to_string();
                        let comp_dict = PyDict::new_bound(py);
                        let py_mmap = load_safetensors_into_dict(py, &shard_str, None, &comp_dict)?;
                        mmap_handles.push(py_mmap.unbind());

                        for (key, value) in comp_dict.iter() {
                            let key_str: String = key.extract()?;
                            let prefixed = format!("{prefix}{key_str}");
                            let output_name = maybe_strip_prefix(&prefixed, strip_prefix);
                            result.set_item(output_name, value)?;
                        }
                    }
                }
            }
            WeightSource::SinglePytorch(_)
            | WeightSource::ShardedPytorch { .. }
            | WeightSource::None => {
                // Skip components without safetensors weights
            }
        }
    }

    let info = PyDict::new_bound(py);
    info.set_item("format", "diffusers")?;
    info.set_item("path", path)?;
    if let Some(cls) = layout.model_index.get("_class_name").and_then(|v| v.as_str()) {
        info.set_item("pipeline_class", cls)?;
    }
    let comp_names: Vec<&str> = layout.components.iter().map(|c| c.name.as_str()).collect();
    info.set_item("components", comp_names)?;

    let mmap_list = PyList::new_bound(py, mmap_handles);
    let ret = PyTuple::new_bound(py, &[result.as_any(), info.as_any(), mmap_list.as_any()]);
    Ok(ret.into())
}

// ── Public entry point ──────────────────────────────────────────────────────

/// Load model from any supported format.
///
/// Returns `(tensors_dict, info_dict, mmap_handles)` where:
/// - `tensors_dict`: maps names to `torch.Tensor` (dense) or `QuantizedTensor` (GGUF quant)
/// - `info_dict`: format metadata (format name, path, etc.)
/// - `mmap_handles`: list of mmap objects that must stay alive while tensors are in use
#[pyfunction]
#[pyo3(signature = (path, strip_prefix=None))]
pub fn load_model(py: Python, path: &str, strip_prefix: Option<&str>) -> PyResult<PyObject> {
    let p = Path::new(path);
    let format = detect_format(p)
        .map_err(|e| PyValueError::new_err(e))?;

    match format {
        ModelFormat::Safetensors => load_safetensors_model(py, path, strip_prefix),
        ModelFormat::Gguf => load_gguf_model(py, path, strip_prefix),
        ModelFormat::PyTorchZip => load_pytorch_model(py, path, strip_prefix),
        ModelFormat::PyTorchLegacy => Err(PyValueError::new_err(
            "Legacy pickle (.ckpt) loading not yet supported. Convert to safetensors first.",
        )),
        ModelFormat::Diffusers => load_diffusers_model(py, path, strip_prefix),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maybe_strip_prefix() {
        assert_eq!(maybe_strip_prefix("model.layer.weight", Some("model.")), "layer.weight");
        assert_eq!(maybe_strip_prefix("model.layer.weight", None), "model.layer.weight");
        assert_eq!(maybe_strip_prefix("other.weight", Some("model.")), "other.weight");
        assert_eq!(maybe_strip_prefix("model.", Some("model.")), "");
    }

    #[test]
    fn test_is_dense_gguf_type() {
        assert!(is_dense_gguf_type(GgufQuantType::F32));
        assert!(is_dense_gguf_type(GgufQuantType::F16));
        assert!(is_dense_gguf_type(GgufQuantType::BF16));
        assert!(is_dense_gguf_type(GgufQuantType::F64));
        assert!(is_dense_gguf_type(GgufQuantType::I8));
        assert!(!is_dense_gguf_type(GgufQuantType::Q4_0));
        assert!(!is_dense_gguf_type(GgufQuantType::Q8_0));
        assert!(!is_dense_gguf_type(GgufQuantType::Q4K));
        assert!(!is_dense_gguf_type(GgufQuantType::Q6K));
    }

    #[test]
    fn test_st_dtype_to_torch() {
        assert_eq!(st_dtype_to_torch("F32"), "float32");
        assert_eq!(st_dtype_to_torch("BF16"), "bfloat16");
        assert_eq!(st_dtype_to_torch("F16"), "float16");
        assert_eq!(st_dtype_to_torch("I8"), "int8");
        assert_eq!(st_dtype_to_torch("UNKNOWN"), "float32");
    }

    #[test]
    fn test_pytorch_dtype_to_torch() {
        assert_eq!(pytorch_dtype_to_torch("float32"), "float32");
        assert_eq!(pytorch_dtype_to_torch("bfloat16"), "bfloat16");
        assert_eq!(pytorch_dtype_to_torch("int8"), "int8");
        assert_eq!(pytorch_dtype_to_torch("UNKNOWN"), "float32");
    }
}
