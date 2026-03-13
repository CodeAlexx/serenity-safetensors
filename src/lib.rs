pub mod diffusers;
pub mod format_detect;
pub mod gguf;
pub mod gguf_dequant;
pub mod probe;
pub mod pytorch;
pub mod quantized_tensor;
pub mod unified;

use memmap2::MmapOptions;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyTuple};
use safetensors::tensor::{Dtype as StDtype, View};
use safetensors::SafeTensors;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedShardSelection {
    shard_name: String,
    shard_path: String,
    tensor_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ShardedTensorLayoutEntry {
    name: String,
    shard_name: String,
    shard_path: String,
    dtype: String,
    shape: Vec<usize>,
    nbytes: usize,
    data_offsets: (usize, usize),
    absolute_offsets: (usize, usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QuantizedBlockRecord {
    id: String,
    file: String,
    offset: usize,
    nbytes: usize,
    tensor_name: Option<String>,
    tensors: Vec<String>,
    payload_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QuantizedBlockContainerEntry {
    id: String,
    tensor_name: String,
    nbytes: usize,
    absolute_offsets: (usize, usize),
    payload_sha256: String,
}

const SOURCE_MANIFEST_FORMAT: &str = "serenity_source_manifest";
const SOURCE_MANIFEST_SCHEMA_VERSION: u64 = 1;
const QUANTIZED_BLOCK_MAP_FORMAT: &str = "serenity_quantized_block_map";
const QUANTIZED_BLOCK_MAP_SCHEMA_VERSION: u64 = 1;
const QUANTIZED_BLOCK_CONTAINER_FORMAT: &str = "serenity_quantized_block_container";
const QUANTIZED_BLOCK_CONTAINER_SCHEMA_VERSION: u64 = 1;
const QUANTIZED_BLOCK_CONTAINER_FORMAT_KEY: &str = "serenity_quantized_container_format";
const QUANTIZED_BLOCK_CONTAINER_SCHEMA_KEY: &str = "serenity_quantized_container_schema_version";
const QUANTIZED_BLOCK_TENSOR_PREFIX: &str = "__quantized_block__.";

fn resolve_shard_path(index_path: &str, shard_name: &str) -> String {
    let shard_path = Path::new(shard_name);
    if shard_path.is_absolute() {
        return shard_path.to_string_lossy().into_owned();
    }
    let parent = Path::new(index_path)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    parent.join(shard_path).to_string_lossy().into_owned()
}

fn resolve_sharded_selections<F>(
    index_path: &str,
    summary: &ShardIndexSummary,
    mut include: F,
) -> Vec<ResolvedShardSelection>
where
    F: FnMut(&str) -> bool,
{
    let mut shard_names: Vec<String> = summary.shards.keys().cloned().collect();
    shard_names.sort();

    let mut resolved = Vec::new();
    for shard_name in shard_names {
        let mut tensor_names: Vec<String> = summary
            .shards
            .get(&shard_name)
            .into_iter()
            .flat_map(|names| names.iter())
            .filter(|name| include(name))
            .cloned()
            .collect();
        if tensor_names.is_empty() {
            continue;
        }
        tensor_names.sort();
        resolved.push(ResolvedShardSelection {
            shard_path: resolve_shard_path(index_path, &shard_name),
            shard_name,
            tensor_names,
        });
    }

    resolved
}

fn collect_sharded_tensor_layout(
    index_path: &str,
) -> Result<
    (
        ShardIndexSummary,
        HashMap<String, String>,
        HashMap<String, HashMap<String, String>>,
        Vec<ShardedTensorLayoutEntry>,
    ),
    SsError,
> {
    let parsed = read_shard_index(index_path)?;
    let summary = summarize_shard_index(&parsed);
    let selections = resolve_sharded_selections(index_path, &summary, |_| true);

    let mut shard_paths = HashMap::new();
    let mut shard_metadata = HashMap::new();
    let mut tensors = Vec::new();

    for selection in selections {
        let (metadata, entries) = collect_tensor_layout(&selection.shard_path)?;
        let expected: HashSet<&str> = selection.tensor_names.iter().map(String::as_str).collect();
        let mut found = 0usize;

        shard_paths.insert(selection.shard_name.clone(), selection.shard_path.clone());
        shard_metadata.insert(selection.shard_name.clone(), metadata);

        for entry in entries {
            if expected.contains(entry.name.as_str()) {
                found += 1;
                tensors.push(ShardedTensorLayoutEntry {
                    name: entry.name,
                    shard_name: selection.shard_name.clone(),
                    shard_path: selection.shard_path.clone(),
                    dtype: entry.dtype,
                    shape: entry.shape,
                    nbytes: entry.nbytes,
                    data_offsets: entry.data_offsets,
                    absolute_offsets: entry.absolute_offsets,
                });
            }
        }

        if found != expected.len() {
            return Err(SsError::Other(format!(
                "Shard {} is missing one or more indexed tensors",
                selection.shard_name
            )));
        }
    }

    Ok((summary, shard_paths, shard_metadata, tensors))
}

fn build_quantized_block_tensor_name(block_id: &str) -> String {
    format!("{QUANTIZED_BLOCK_TENSOR_PREFIX}{block_id}")
}

fn parse_quantized_block_tensor_name(name: &str) -> Option<&str> {
    name.strip_prefix(QUANTIZED_BLOCK_TENSOR_PREFIX)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push_str(format!("{byte:02x}").as_str());
    }
    encoded
}

fn block_map_tensor_names_from_py(
    py: Python<'_>,
    block_tensors: Option<&Bound<'_, PyDict>>,
) -> Result<HashMap<String, Vec<String>>, SsError> {
    let mut result = HashMap::new();
    let Some(block_tensors) = block_tensors else {
        return Ok(result);
    };
    for (key, value) in block_tensors.iter() {
        let block_id = key
            .extract::<String>()
            .map_err(|e| SsError::Other(format!("Invalid block_tensors key: {e}")))?;
        let tensor_names: Vec<String> = if let Ok(names) = value.extract::<Vec<String>>() {
            names
        } else if let Ok(py_list) = value.downcast::<PyList>() {
            py_list
                .iter()
                .map(|item| {
                    item.extract::<String>().map_err(|e| {
                        SsError::Other(format!(
                            "block_tensors[{block_id:?}] must contain only strings: {e}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            return Err(SsError::Other(format!(
                "block_tensors[{block_id:?}] must be a list of strings"
            )));
        };
        result.insert(block_id, tensor_names);
    }
    let _ = py;
    Ok(result)
}

fn prepare_python_quantized_block_views(
    payloads: &Bound<'_, PyDict>,
) -> Result<Vec<(String, PythonTensorView)>, SsError> {
    let mut views = Vec::with_capacity(payloads.len());

    for (key, value) in payloads.iter() {
        let block_id: String = key
            .extract()
            .map_err(|e| SsError::Other(format!("Invalid block id: {e}")))?;

        let dtype_str: String = value
            .getattr("dtype")
            .and_then(|d| d.str().map(|s| s.to_string()))
            .map_err(|e| SsError::Other(format!("Cannot get dtype for block {block_id}: {e}")))?;
        let st_dtype = dtype_from_torch_str(&dtype_str)?;
        if st_dtype != StDtype::U8 {
            return Err(SsError::Other(format!(
                "Quantized block payload {block_id} must be torch.uint8, got {dtype_str}"
            )));
        }

        let cpu_tensor = value
            .call_method0("detach")
            .and_then(|t| t.call_method0("cpu"))
            .and_then(|t| t.call_method0("contiguous"))
            .and_then(|t| t.call_method1("reshape", (PyTuple::new_bound(value.py(), [(-1i64)]),)))
            .map_err(|e| {
                SsError::Other(format!("Cannot prepare quantized block {block_id}: {e}"))
            })?;

        let nbytes = cpu_tensor
            .call_method0("numel")
            .and_then(|v| v.extract::<usize>())
            .map_err(|e| SsError::Other(format!("Cannot get numel for block {block_id}: {e}")))?;

        let storage = cpu_tensor.call_method0("untyped_storage").map_err(|e| {
            SsError::Other(format!("untyped_storage failed for block {block_id}: {e}"))
        })?;
        let data_ptr = storage
            .call_method0("data_ptr")
            .map_err(|e| SsError::Other(format!("data_ptr failed for block {block_id}: {e}")))?
            .extract::<usize>()
            .map_err(|e| SsError::Other(format!("ptr extract failed for block {block_id}: {e}")))?;

        if data_ptr == 0 && nbytes > 0 {
            return Err(SsError::Other(format!(
                "Null data_ptr for block {block_id}"
            )));
        }

        views.push((
            build_quantized_block_tensor_name(&block_id),
            PythonTensorView {
                dtype: StDtype::U8,
                shape: vec![nbytes],
                data_ptr,
                nbytes,
                tensor: cpu_tensor.unbind(),
            },
        ));
    }

    Ok(views)
}

fn build_quantized_block_container_metadata(
    metadata: Option<HashMap<String, String>>,
) -> HashMap<String, String> {
    let mut metadata = metadata.unwrap_or_default();
    metadata.insert(
        QUANTIZED_BLOCK_CONTAINER_FORMAT_KEY.to_string(),
        QUANTIZED_BLOCK_CONTAINER_FORMAT.to_string(),
    );
    metadata.insert(
        QUANTIZED_BLOCK_CONTAINER_SCHEMA_KEY.to_string(),
        QUANTIZED_BLOCK_CONTAINER_SCHEMA_VERSION.to_string(),
    );
    metadata
}

fn collect_quantized_block_container_entries(
    path: &str,
) -> Result<Option<Vec<QuantizedBlockContainerEntry>>, SsError> {
    let (metadata, entries) = collect_tensor_layout(path)?;
    let Some(container_format) = metadata.get(QUANTIZED_BLOCK_CONTAINER_FORMAT_KEY) else {
        return Ok(None);
    };
    if container_format != QUANTIZED_BLOCK_CONTAINER_FORMAT {
        return Err(SsError::Other(format!(
            "Unsupported quantized container format {container_format} in {path}"
        )));
    }
    let schema_version = metadata
        .get(QUANTIZED_BLOCK_CONTAINER_SCHEMA_KEY)
        .ok_or_else(|| {
            SsError::Other(format!(
                "Missing {QUANTIZED_BLOCK_CONTAINER_SCHEMA_KEY} metadata in {path}"
            ))
        })?
        .parse::<u64>()
        .map_err(|_| {
            SsError::Other(format!(
                "{QUANTIZED_BLOCK_CONTAINER_SCHEMA_KEY} must be an integer string in {path}"
            ))
        })?;
    if schema_version != QUANTIZED_BLOCK_CONTAINER_SCHEMA_VERSION {
        return Err(SsError::Other(format!(
            "Unsupported quantized container schema version {schema_version} in {path}"
        )));
    }

    let file = fs::File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let mut blocks = Vec::with_capacity(entries.len());
    for entry in entries {
        if entry.dtype != "U8" {
            return Err(SsError::Other(format!(
                "Quantized block container tensors must use U8 payloads: {} has dtype {}",
                entry.name, entry.dtype
            )));
        }
        if entry.shape.len() != 1 {
            return Err(SsError::Other(format!(
                "Quantized block container tensors must be 1D byte payloads: {} has shape {:?}",
                entry.name, entry.shape
            )));
        }
        let block_id = parse_quantized_block_tensor_name(&entry.name).ok_or_else(|| {
            SsError::Other(format!(
                "Quantized block container tensor name {} does not use the {} prefix",
                entry.name, QUANTIZED_BLOCK_TENSOR_PREFIX
            ))
        })?;
        let payload = &mmap[entry.absolute_offsets.0..entry.absolute_offsets.1];
        blocks.push(QuantizedBlockContainerEntry {
            id: block_id.to_string(),
            tensor_name: entry.name,
            nbytes: entry.nbytes,
            absolute_offsets: entry.absolute_offsets,
            payload_sha256: sha256_hex(payload),
        });
    }
    Ok(Some(blocks))
}

fn parse_quantized_block_records(
    block_map: &serde_json::Value,
) -> Result<Vec<QuantizedBlockRecord>, SsError> {
    let root = ensure_object(block_map, "block_map")?;
    let blocks = root
        .get("blocks")
        .and_then(|value| value.as_array())
        .ok_or_else(|| SsError::Other("block_map.blocks must be an array".into()))?;

    let mut seen_ids = HashSet::new();
    let mut records = Vec::with_capacity(blocks.len());
    for block in blocks {
        let block_obj = ensure_object(block, "block_map.blocks[]")?;
        let id = expect_string(block_obj, "id", "block_map.blocks[]")?;
        if !seen_ids.insert(id.clone()) {
            return Err(SsError::Other(format!(
                "block_map.blocks contains duplicate id {id}"
            )));
        }
        let file = expect_string(block_obj, "file", "block_map.blocks[]")?;
        let offset = block_obj
            .get("offset")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| SsError::Other("block_map.blocks[].offset must be an integer".into()))?
            as usize;
        let nbytes = block_obj
            .get("nbytes")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| SsError::Other("block_map.blocks[].nbytes must be an integer".into()))?
            as usize;
        let tensor_name = optional_string(block_obj, "tensor_name", "block_map.blocks[]")?;
        let payload_sha256 = optional_string(block_obj, "payload_sha256", "block_map.blocks[]")?;
        let tensors = match block_obj.get("tensors") {
            Some(_) => ensure_string_array(block_obj, "tensors", "block_map.blocks[]")?,
            None => Vec::new(),
        };
        records.push(QuantizedBlockRecord {
            id,
            file,
            offset,
            nbytes,
            tensor_name,
            tensors,
            payload_sha256,
        });
    }

    Ok(records)
}

fn build_quantized_block_map_for_container(
    container_path: &str,
    file_field: &str,
    block_tensors: &HashMap<String, Vec<String>>,
) -> Result<serde_json::Value, SsError> {
    let container_entries =
        collect_quantized_block_container_entries(container_path)?.ok_or_else(|| {
            SsError::Other(format!(
                "{container_path} is not a quantized block container"
            ))
        })?;

    let mut blocks = Vec::with_capacity(container_entries.len());
    for entry in container_entries {
        let tensors = block_tensors.get(&entry.id).cloned().unwrap_or_default();
        blocks.push(serde_json::json!({
            "id": entry.id,
            "file": file_field,
            "offset": entry.absolute_offsets.0,
            "nbytes": entry.nbytes,
            "tensor_name": entry.tensor_name,
            "tensors": tensors,
            "payload_sha256": entry.payload_sha256,
        }));
    }
    blocks.sort_by(|left, right| {
        left.get("id")
            .and_then(|value| value.as_str())
            .cmp(&right.get("id").and_then(|value| value.as_str()))
    });

    Ok(serde_json::json!({
        "metadata": {
            "container_format": QUANTIZED_BLOCK_CONTAINER_FORMAT,
            "container_schema_version": QUANTIZED_BLOCK_CONTAINER_SCHEMA_VERSION.to_string(),
            "payload_dtype": "U8",
        },
        "blocks": blocks,
    }))
}

fn read_quantized_block_map_reference_value(
    path: &str,
    resolve_paths: bool,
) -> Result<serde_json::Value, SsError> {
    if let Ok(manifest) = read_manifest_value(path, resolve_paths) {
        let root = ensure_object(&manifest, "manifest")?;
        let source = root
            .get("source")
            .ok_or_else(|| SsError::Other("manifest.source is required".into()))?;
        let source_obj = ensure_object(source, "manifest.source")?;
        let source_kind = expect_string(source_obj, "kind", "manifest.source")?;
        if source_kind == "quantized_blocks" {
            let artifacts = root
                .get("artifacts")
                .ok_or_else(|| SsError::Other("manifest.artifacts is required".into()))?;
            let artifacts_obj = ensure_object(artifacts, "manifest.artifacts")?;
            let block_map_path = expect_string(artifacts_obj, "block_map", "manifest.artifacts")?;
            return read_quantized_block_map_value(&block_map_path, resolve_paths);
        }
    }

    read_quantized_block_map_value(path, resolve_paths)
}

#[cfg_attr(not(test), allow(dead_code))]
fn load_quantized_block_payloads_value(
    reference_path: &str,
    requested_ids: Option<&HashSet<String>>,
) -> Result<HashMap<String, Vec<u8>>, SsError> {
    let block_map = read_quantized_block_map_reference_value(reference_path, true)?;
    let records = parse_quantized_block_records(&block_map)?;

    let mut grouped: HashMap<String, Vec<QuantizedBlockRecord>> = HashMap::new();
    for record in records {
        if requested_ids.is_some_and(|ids| !ids.contains(record.id.as_str())) {
            continue;
        }
        grouped.entry(record.file.clone()).or_default().push(record);
    }

    let mut payloads = HashMap::new();
    let mut file_paths: Vec<String> = grouped.keys().cloned().collect();
    file_paths.sort();
    for file_path in file_paths {
        let file = fs::File::open(&file_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let file_len = mmap.len();
        let mut file_records = grouped.remove(&file_path).unwrap_or_default();
        file_records.sort_by(|left, right| left.id.cmp(&right.id));
        for record in file_records {
            let end = record.offset.saturating_add(record.nbytes);
            if end > file_len {
                return Err(SsError::Other(format!(
                    "block {} exceeds file bounds in {}: offset={} nbytes={} file_size={}",
                    record.id, file_path, record.offset, record.nbytes, file_len
                )));
            }
            payloads.insert(record.id, mmap[record.offset..end].to_vec());
        }
    }

    Ok(payloads)
}

fn py_to_json_value(py: Python<'_>, obj: &Bound<'_, PyAny>) -> Result<serde_json::Value, SsError> {
    let json = py
        .import_bound("json")
        .map_err(|e| SsError::Other(e.to_string()))?;
    let dumped: String = json
        .call_method1("dumps", (obj,))
        .and_then(|v| v.extract())
        .map_err(|e| SsError::Other(format!("json.dumps failed: {e}")))?;
    let parsed = serde_json::from_str(&dumped)?;
    Ok(parsed)
}

fn json_value_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    let json = py.import_bound("json")?;
    let dumped =
        serde_json::to_string(value).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(json.call_method1("loads", (dumped,))?.into())
}

fn ensure_object<'a>(
    value: &'a serde_json::Value,
    context: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, SsError> {
    value
        .as_object()
        .ok_or_else(|| SsError::Other(format!("{context} must be a JSON object")))
}

fn ensure_object_mut<'a>(
    value: &'a mut serde_json::Value,
    context: &str,
) -> Result<&'a mut serde_json::Map<String, serde_json::Value>, SsError> {
    value
        .as_object_mut()
        .ok_or_else(|| SsError::Other(format!("{context} must be a JSON object")))
}

fn expect_string(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    context: &str,
) -> Result<String, SsError> {
    object
        .get(key)
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .ok_or_else(|| SsError::Other(format!("{context}.{key} must be a string")))
}

fn optional_string(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    context: &str,
) -> Result<Option<String>, SsError> {
    match object.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(|v| Some(v.to_string()))
            .ok_or_else(|| SsError::Other(format!("{context}.{key} must be a string"))),
    }
}

fn ensure_string_array(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    context: &str,
) -> Result<Vec<String>, SsError> {
    let values = object
        .get(key)
        .and_then(|value| value.as_array())
        .ok_or_else(|| SsError::Other(format!("{context}.{key} must be an array of strings")))?;
    values
        .iter()
        .map(|value| {
            value.as_str().map(str::to_string).ok_or_else(|| {
                SsError::Other(format!("{context}.{key} must be an array of strings"))
            })
        })
        .collect()
}

fn validate_tensor_policy(
    source: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<(), SsError> {
    if !source.contains_key("tensor_policy") {
        source.insert(
            "tensor_policy".to_string(),
            serde_json::json!({"mode": "all"}),
        );
    }
    let tensor_policy = source
        .get_mut("tensor_policy")
        .ok_or_else(|| SsError::Other("source.tensor_policy is missing".into()))?;
    let tensor_policy_obj = ensure_object_mut(tensor_policy, "source.tensor_policy")?;
    let mode = expect_string(tensor_policy_obj, "mode", "source.tensor_policy")?;
    match mode.as_str() {
        "all" => {}
        "prefixes" => {
            let prefixes =
                ensure_string_array(tensor_policy_obj, "prefixes", "source.tensor_policy")?;
            if prefixes.is_empty() {
                return Err(SsError::Other(
                    "source.tensor_policy.prefixes must not be empty".into(),
                ));
            }
        }
        "names" => {
            let names = ensure_string_array(tensor_policy_obj, "names", "source.tensor_policy")?;
            if names.is_empty() {
                return Err(SsError::Other(
                    "source.tensor_policy.names must not be empty".into(),
                ));
            }
        }
        other => {
            return Err(SsError::Other(format!(
                "source.tensor_policy.mode must be one of all/prefixes/names, got {other}"
            )))
        }
    }
    Ok(())
}

fn resolve_manifest_path(base_dir: &Path, value: &str) -> String {
    let path = Path::new(value);
    if path.is_absolute() {
        value.to_string()
    } else {
        base_dir.join(path).to_string_lossy().into_owned()
    }
}

fn resolve_manifest_path_field(
    object: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    base_dir: &Path,
) -> Result<(), SsError> {
    if let Some(value) = object.get_mut(key) {
        match value {
            serde_json::Value::String(path) => {
                let resolved = resolve_manifest_path(base_dir, path);
                *path = resolved;
            }
            serde_json::Value::Array(paths) => {
                for item in paths {
                    let path = item.as_str().ok_or_else(|| {
                        SsError::Other(format!("{key} must be a string or list of strings"))
                    })?;
                    *item = serde_json::Value::String(resolve_manifest_path(base_dir, path));
                }
            }
            _ => {
                return Err(SsError::Other(format!(
                    "{key} must be a string or list of strings"
                )))
            }
        }
    }
    Ok(())
}

fn normalize_manifest_value(
    mut manifest: serde_json::Value,
    manifest_path: Option<&str>,
    resolve_paths: bool,
) -> Result<serde_json::Value, SsError> {
    let base_dir = manifest_path
        .and_then(|path| Path::new(path).parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));

    let root = ensure_object_mut(&mut manifest, "manifest")?;
    match root.get("schema_version") {
        None => {
            root.insert(
                "schema_version".to_string(),
                serde_json::Value::Number(SOURCE_MANIFEST_SCHEMA_VERSION.into()),
            );
        }
        Some(value) => {
            let schema_version = value.as_u64().ok_or_else(|| {
                SsError::Other("manifest.schema_version must be an integer".into())
            })?;
            if schema_version != SOURCE_MANIFEST_SCHEMA_VERSION {
                return Err(SsError::Other(format!(
                    "Unsupported manifest schema_version {schema_version}"
                )));
            }
        }
    }
    match root.get("format") {
        None => {
            root.insert(
                "format".to_string(),
                serde_json::Value::String(SOURCE_MANIFEST_FORMAT.to_string()),
            );
        }
        Some(value) => {
            let format = value
                .as_str()
                .ok_or_else(|| SsError::Other("manifest.format must be a string".into()))?;
            if format != SOURCE_MANIFEST_FORMAT {
                return Err(SsError::Other(format!(
                    "Unsupported manifest.format {format}"
                )));
            }
        }
    }

    let model = root
        .get_mut("model")
        .ok_or_else(|| SsError::Other("manifest.model is required".into()))?;
    let model_obj = ensure_object_mut(model, "manifest.model")?;
    expect_string(model_obj, "family", "manifest.model")?;
    expect_string(model_obj, "version", "manifest.model")?;
    optional_string(model_obj, "variant", "manifest.model")?;

    let source = root
        .get_mut("source")
        .ok_or_else(|| SsError::Other("manifest.source is required".into()))?;
    let source_obj = ensure_object_mut(source, "manifest.source")?;
    let source_kind = expect_string(source_obj, "kind", "manifest.source")?;
    match source_kind.as_str() {
        "single_file" | "sharded_index" | "materialized_subset" | "quantized_blocks" => {}
        other => {
            return Err(SsError::Other(format!(
                "manifest.source.kind must be one of single_file/sharded_index/materialized_subset/quantized_blocks, got {other}"
            )))
        }
    }
    optional_string(source_obj, "path", "manifest.source")?;
    optional_string(source_obj, "original", "manifest.source")?;
    optional_string(source_obj, "signature", "manifest.source")?;
    optional_string(source_obj, "dtype", "manifest.source")?;
    validate_tensor_policy(source_obj)?;

    if resolve_paths {
        resolve_manifest_path_field(source_obj, "path", &base_dir)?;
    }

    if let Some(artifacts) = root.get_mut("artifacts") {
        let artifacts_obj = ensure_object_mut(artifacts, "manifest.artifacts")?;
        if resolve_paths {
            for key in [
                "path",
                "index",
                "block_map",
                "weights",
                "data_files",
                "files",
            ] {
                resolve_manifest_path_field(artifacts_obj, key, &base_dir)?;
            }
        }
    }

    if let Some(quantization) = root.get_mut("quantization") {
        let quant_obj = ensure_object_mut(quantization, "manifest.quantization")?;
        expect_string(quant_obj, "mode", "manifest.quantization")?;
        if let Some(value) = quant_obj.get("block_count") {
            value.as_u64().ok_or_else(|| {
                SsError::Other("manifest.quantization.block_count must be an integer".into())
            })?;
        }
        if let Some(value) = quant_obj.get("group_size") {
            value.as_u64().ok_or_else(|| {
                SsError::Other("manifest.quantization.group_size must be an integer".into())
            })?;
        }
        if !quant_obj.contains_key("frozen") {
            quant_obj.insert("frozen".to_string(), serde_json::Value::Bool(true));
        } else if quant_obj
            .get("frozen")
            .and_then(|value| value.as_bool())
            .is_none()
        {
            return Err(SsError::Other(
                "manifest.quantization.frozen must be a boolean".into(),
            ));
        }
    }

    if let Some(compatibility) = root.get_mut("compatibility") {
        let compatibility_obj = ensure_object_mut(compatibility, "manifest.compatibility")?;
        for key in [
            "minimum_serenity_version",
            "stagehand_layout",
            "required_source_signature",
            "required_quant_mode",
        ] {
            optional_string(compatibility_obj, key, "manifest.compatibility")?;
        }
    }

    if let Some(metadata) = root.get("metadata") {
        ensure_object(metadata, "manifest.metadata")?;
    }

    Ok(manifest)
}

fn build_source_manifest_value(
    model_family: &str,
    model_version: &str,
    source_kind: &str,
    source_path: Option<&str>,
    original_source: Option<&str>,
    source_signature: Option<&str>,
    dtype: Option<&str>,
    variant: Option<&str>,
    tensor_prefixes: Option<Vec<String>>,
    tensor_names: Option<Vec<String>>,
    minimum_serenity_version: Option<&str>,
    stagehand_layout: Option<&str>,
    extra_metadata: Option<serde_json::Value>,
) -> Result<serde_json::Value, SsError> {
    if tensor_prefixes.as_ref().is_some_and(|v| !v.is_empty())
        && tensor_names.as_ref().is_some_and(|v| !v.is_empty())
    {
        return Err(SsError::Other(
            "Provide either tensor_prefixes or tensor_names, not both".into(),
        ));
    }

    let tensor_policy = if let Some(names) = tensor_names.filter(|v| !v.is_empty()) {
        serde_json::json!({ "mode": "names", "names": names })
    } else if let Some(prefixes) = tensor_prefixes.filter(|v| !v.is_empty()) {
        serde_json::json!({ "mode": "prefixes", "prefixes": prefixes })
    } else {
        serde_json::json!({ "mode": "all" })
    };

    let mut root = serde_json::json!({
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "format": SOURCE_MANIFEST_FORMAT,
        "model": {
            "family": model_family,
            "version": model_version,
        },
        "source": {
            "kind": source_kind,
            "tensor_policy": tensor_policy,
        },
    });

    {
        let root_obj = ensure_object_mut(&mut root, "manifest")?;
        let model_obj = root_obj
            .get_mut("model")
            .and_then(|value| value.as_object_mut())
            .expect("model");
        if let Some(variant) = variant {
            model_obj.insert(
                "variant".to_string(),
                serde_json::Value::String(variant.to_string()),
            );
        }

        let source_obj = root_obj
            .get_mut("source")
            .and_then(|value| value.as_object_mut())
            .expect("source");
        if let Some(path) = source_path {
            source_obj.insert(
                "path".to_string(),
                serde_json::Value::String(path.to_string()),
            );
        }
        if let Some(original) = original_source {
            source_obj.insert(
                "original".to_string(),
                serde_json::Value::String(original.to_string()),
            );
        }
        if let Some(signature) = source_signature {
            source_obj.insert(
                "signature".to_string(),
                serde_json::Value::String(signature.to_string()),
            );
        }
        if let Some(dtype) = dtype {
            source_obj.insert(
                "dtype".to_string(),
                serde_json::Value::String(dtype.to_string()),
            );
        }

        if minimum_serenity_version.is_some() || stagehand_layout.is_some() {
            let mut compatibility = serde_json::Map::new();
            if let Some(minimum_serenity_version) = minimum_serenity_version {
                compatibility.insert(
                    "minimum_serenity_version".to_string(),
                    serde_json::Value::String(minimum_serenity_version.to_string()),
                );
            }
            if let Some(stagehand_layout) = stagehand_layout {
                compatibility.insert(
                    "stagehand_layout".to_string(),
                    serde_json::Value::String(stagehand_layout.to_string()),
                );
            }
            root_obj.insert(
                "compatibility".to_string(),
                serde_json::Value::Object(compatibility),
            );
        }

        if let Some(extra_metadata) = extra_metadata {
            ensure_object(&extra_metadata, "manifest.metadata")?;
            root_obj.insert("metadata".to_string(), extra_metadata);
        }
    }

    normalize_manifest_value(root, None, false)
}

fn build_quantized_source_manifest_value(
    model_family: &str,
    model_version: &str,
    source_path: Option<&str>,
    original_source: &str,
    source_signature: &str,
    block_map_path: &str,
    data_files: Vec<String>,
    quant_mode: &str,
    dtype: Option<&str>,
    variant: Option<&str>,
    tensor_prefixes: Option<Vec<String>>,
    block_count: Option<u64>,
    group_size: Option<u64>,
    minimum_serenity_version: Option<&str>,
    stagehand_layout: Option<&str>,
    extra_metadata: Option<serde_json::Value>,
) -> Result<serde_json::Value, SsError> {
    let mut manifest = build_source_manifest_value(
        model_family,
        model_version,
        "quantized_blocks",
        source_path,
        Some(original_source),
        Some(source_signature),
        dtype,
        variant,
        tensor_prefixes,
        None,
        minimum_serenity_version,
        stagehand_layout,
        extra_metadata,
    )?;

    let root = ensure_object_mut(&mut manifest, "manifest")?;
    root.insert(
        "artifacts".to_string(),
        serde_json::json!({
            "block_map": block_map_path,
            "data_files": data_files,
        }),
    );
    let mut quantization = serde_json::Map::new();
    quantization.insert(
        "mode".to_string(),
        serde_json::Value::String(quant_mode.to_string()),
    );
    quantization.insert("frozen".to_string(), serde_json::Value::Bool(true));
    if let Some(block_count) = block_count {
        quantization.insert(
            "block_count".to_string(),
            serde_json::Value::Number(block_count.into()),
        );
    }
    if let Some(group_size) = group_size {
        quantization.insert(
            "group_size".to_string(),
            serde_json::Value::Number(group_size.into()),
        );
    }
    root.insert(
        "quantization".to_string(),
        serde_json::Value::Object(quantization),
    );
    if let Some(compatibility) = root.get_mut("compatibility") {
        let compatibility_obj = ensure_object_mut(compatibility, "manifest.compatibility")?;
        compatibility_obj.insert(
            "required_source_signature".to_string(),
            serde_json::Value::String(source_signature.to_string()),
        );
        compatibility_obj.insert(
            "required_quant_mode".to_string(),
            serde_json::Value::String(quant_mode.to_string()),
        );
    } else {
        root.insert(
            "compatibility".to_string(),
            serde_json::json!({
                "required_source_signature": source_signature,
                "required_quant_mode": quant_mode,
            }),
        );
    }

    normalize_manifest_value(manifest, None, false)
}

fn read_manifest_value(path: &str, resolve_paths: bool) -> Result<serde_json::Value, SsError> {
    let raw = fs::read_to_string(path)?;
    let manifest = serde_json::from_str(&raw)?;
    normalize_manifest_value(manifest, Some(path), resolve_paths)
}

fn write_manifest_value(path: &str, manifest: serde_json::Value) -> Result<(), SsError> {
    let manifest = normalize_manifest_value(manifest, Some(path), false)?;
    let serialized = serde_json::to_vec_pretty(&manifest)?;
    fs::write(path, serialized)?;
    Ok(())
}

fn evaluate_manifest_compatibility(
    manifest: &serde_json::Value,
    model_family: &str,
    model_version: &str,
    source_signature: Option<&str>,
    quant_mode: Option<&str>,
    stagehand_layout: Option<&str>,
) -> serde_json::Value {
    let mut reasons = Vec::new();
    let root = manifest.as_object().expect("normalized manifest root");
    let model = root
        .get("model")
        .and_then(|value| value.as_object())
        .expect("normalized model");
    let source = root
        .get("source")
        .and_then(|value| value.as_object())
        .expect("normalized source");

    let manifest_family = model.get("family").and_then(|v| v.as_str()).unwrap_or("");
    let manifest_version = model.get("version").and_then(|v| v.as_str()).unwrap_or("");
    if manifest_family != model_family {
        reasons.push(format!(
            "model family mismatch: manifest={manifest_family}, requested={model_family}"
        ));
    }
    if manifest_version != model_version {
        reasons.push(format!(
            "model version mismatch: manifest={manifest_version}, requested={model_version}"
        ));
    }

    if let Some(required_signature) = root
        .get("compatibility")
        .and_then(|value| value.as_object())
        .and_then(|compatibility| compatibility.get("required_source_signature"))
        .and_then(|value| value.as_str())
        .or_else(|| source.get("signature").and_then(|value| value.as_str()))
    {
        if let Some(source_signature) = source_signature {
            if required_signature != source_signature {
                reasons.push(format!(
                    "source signature mismatch: manifest={required_signature}, requested={source_signature}"
                ));
            }
        }
    }

    if let Some(required_quant_mode) = root
        .get("compatibility")
        .and_then(|value| value.as_object())
        .and_then(|compatibility| compatibility.get("required_quant_mode"))
        .and_then(|value| value.as_str())
        .or_else(|| {
            root.get("quantization")
                .and_then(|value| value.as_object())
                .and_then(|quantization| quantization.get("mode"))
                .and_then(|value| value.as_str())
        })
    {
        if let Some(quant_mode) = quant_mode {
            if required_quant_mode != quant_mode {
                reasons.push(format!(
                    "quant mode mismatch: manifest={required_quant_mode}, requested={quant_mode}"
                ));
            }
        }
    }

    if let Some(required_stagehand_layout) = root
        .get("compatibility")
        .and_then(|value| value.as_object())
        .and_then(|compatibility| compatibility.get("stagehand_layout"))
        .and_then(|value| value.as_str())
    {
        if let Some(stagehand_layout) = stagehand_layout {
            if required_stagehand_layout != stagehand_layout {
                reasons.push(format!(
                    "stagehand layout mismatch: manifest={required_stagehand_layout}, requested={stagehand_layout}"
                ));
            }
        }
    }

    serde_json::json!({
        "ok": reasons.is_empty(),
        "reasons": reasons,
    })
}

fn normalize_quantized_block_map_value(
    mut block_map: serde_json::Value,
    block_map_path: Option<&str>,
    resolve_paths: bool,
) -> Result<serde_json::Value, SsError> {
    let base_dir = block_map_path
        .and_then(|path| Path::new(path).parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    let root = ensure_object_mut(&mut block_map, "block_map")?;

    match root.get("schema_version") {
        None => {
            root.insert(
                "schema_version".to_string(),
                serde_json::Value::Number(QUANTIZED_BLOCK_MAP_SCHEMA_VERSION.into()),
            );
        }
        Some(value) => {
            let schema_version = value.as_u64().ok_or_else(|| {
                SsError::Other("block_map.schema_version must be an integer".into())
            })?;
            if schema_version != QUANTIZED_BLOCK_MAP_SCHEMA_VERSION {
                return Err(SsError::Other(format!(
                    "Unsupported block_map.schema_version {schema_version}"
                )));
            }
        }
    }

    match root.get("format") {
        None => {
            root.insert(
                "format".to_string(),
                serde_json::Value::String(QUANTIZED_BLOCK_MAP_FORMAT.to_string()),
            );
        }
        Some(value) => {
            let format = value
                .as_str()
                .ok_or_else(|| SsError::Other("block_map.format must be a string".into()))?;
            if format != QUANTIZED_BLOCK_MAP_FORMAT {
                return Err(SsError::Other(format!(
                    "Unsupported block_map.format {format}"
                )));
            }
        }
    }

    if let Some(metadata) = root.get("metadata") {
        ensure_object(metadata, "block_map.metadata")?;
    }

    let blocks = root
        .get_mut("blocks")
        .ok_or_else(|| SsError::Other("block_map.blocks is required".into()))?;
    let blocks_array = blocks
        .as_array_mut()
        .ok_or_else(|| SsError::Other("block_map.blocks must be an array".into()))?;
    if blocks_array.is_empty() {
        return Err(SsError::Other("block_map.blocks must not be empty".into()));
    }

    for (index, block) in blocks_array.iter_mut().enumerate() {
        let block_obj = ensure_object_mut(block, "block_map.blocks[]")?;
        if !block_obj.contains_key("id") {
            block_obj.insert(
                "id".to_string(),
                serde_json::Value::String(index.to_string()),
            );
        } else {
            optional_string(block_obj, "id", "block_map.blocks[]")?;
        }
        optional_string(block_obj, "file", "block_map.blocks[]")?
            .ok_or_else(|| SsError::Other("block_map.blocks[].file is required".into()))?;
        if block_obj.get("offset").and_then(|v| v.as_u64()).is_none() {
            return Err(SsError::Other(
                "block_map.blocks[].offset must be an integer".into(),
            ));
        }
        if block_obj.get("nbytes").and_then(|v| v.as_u64()).is_none() {
            return Err(SsError::Other(
                "block_map.blocks[].nbytes must be an integer".into(),
            ));
        }
        optional_string(block_obj, "tensor_name", "block_map.blocks[]")?;
        optional_string(block_obj, "payload_sha256", "block_map.blocks[]")?;
        if let Some(tensors) = block_obj.get("tensors") {
            let array = tensors.as_array().ok_or_else(|| {
                SsError::Other("block_map.blocks[].tensors must be an array of strings".into())
            })?;
            for value in array {
                value.as_str().ok_or_else(|| {
                    SsError::Other("block_map.blocks[].tensors must be an array of strings".into())
                })?;
            }
        }
        if resolve_paths {
            resolve_manifest_path_field(block_obj, "file", &base_dir)?;
        }
    }

    Ok(block_map)
}

fn write_quantized_block_map_value(
    path: &str,
    block_map: serde_json::Value,
) -> Result<(), SsError> {
    let block_map = normalize_quantized_block_map_value(block_map, Some(path), false)?;
    let serialized = serde_json::to_vec_pretty(&block_map)?;
    fs::write(path, serialized)?;
    Ok(())
}

fn read_quantized_block_map_value(
    path: &str,
    resolve_paths: bool,
) -> Result<serde_json::Value, SsError> {
    let raw = fs::read_to_string(path)?;
    let block_map = serde_json::from_str(&raw)?;
    normalize_quantized_block_map_value(block_map, Some(path), resolve_paths)
}

fn validate_quantized_block_records(
    records: &[QuantizedBlockRecord],
) -> Result<Vec<String>, SsError> {
    let mut reasons = Vec::new();
    let mut grouped: HashMap<String, Vec<&QuantizedBlockRecord>> = HashMap::new();
    for record in records {
        grouped.entry(record.file.clone()).or_default().push(record);
    }

    for (file, file_records) in grouped {
        if !Path::new(&file).exists() {
            reasons.push(format!("missing data file referenced by block_map: {file}"));
            continue;
        }

        let is_safetensors = Path::new(&file)
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"));

        if is_safetensors {
            match collect_quantized_block_container_entries(&file) {
                Ok(Some(container_entries)) => {
                    let entry_map: HashMap<String, QuantizedBlockContainerEntry> =
                        container_entries
                            .into_iter()
                            .map(|entry| (entry.tensor_name.clone(), entry))
                            .collect();
                    for record in file_records {
                        let tensor_name = record
                            .tensor_name
                            .clone()
                            .unwrap_or_else(|| build_quantized_block_tensor_name(&record.id));
                        let Some(entry) = entry_map.get(&tensor_name) else {
                            reasons.push(format!(
                                "block {} references missing container tensor {} in {}",
                                record.id, tensor_name, file
                            ));
                            continue;
                        };
                        if record.offset != entry.absolute_offsets.0 {
                            reasons.push(format!(
                                "block {} offset mismatch: block_map={} container={}",
                                record.id, record.offset, entry.absolute_offsets.0
                            ));
                        }
                        if record.nbytes != entry.nbytes {
                            reasons.push(format!(
                                "block {} nbytes mismatch: block_map={} container={}",
                                record.id, record.nbytes, entry.nbytes
                            ));
                        }
                        if let Some(expected_hash) = &record.payload_sha256 {
                            if expected_hash != &entry.payload_sha256 {
                                reasons.push(format!(
                                    "block {} payload hash mismatch: block_map={} container={}",
                                    record.id, expected_hash, entry.payload_sha256
                                ));
                            }
                        }
                    }
                }
                Ok(None) => {
                    reasons.push(format!(
                        "safetensors file {} is missing {} metadata",
                        file, QUANTIZED_BLOCK_CONTAINER_FORMAT_KEY
                    ));
                }
                Err(err) => {
                    reasons.push(format!("invalid quantized block container {file}: {err}"));
                }
            }
        } else {
            let file_len = fs::metadata(&file)?.len() as usize;
            let file_handle = fs::File::open(&file)?;
            let mmap = unsafe { MmapOptions::new().map(&file_handle)? };
            for record in file_records {
                let end = record.offset.saturating_add(record.nbytes);
                if end > file_len {
                    reasons.push(format!(
                        "block {} exceeds file bounds: offset={} nbytes={} file_size={}",
                        record.id, record.offset, record.nbytes, file_len
                    ));
                    continue;
                }
                if let Some(expected_hash) = &record.payload_sha256 {
                    let payload = &mmap[record.offset..end];
                    let actual_hash = sha256_hex(payload);
                    if expected_hash != &actual_hash {
                        reasons.push(format!(
                            "block {} payload hash mismatch: block_map={} file={}",
                            record.id, expected_hash, actual_hash
                        ));
                    }
                }
            }
        }
    }

    Ok(reasons)
}

fn verify_quantized_manifest_artifacts_value(path: &str) -> Result<serde_json::Value, SsError> {
    let manifest = read_manifest_value(path, true)?;
    let root = ensure_object(&manifest, "manifest")?;
    let source = root
        .get("source")
        .ok_or_else(|| SsError::Other("manifest.source is required".into()))?;
    let source_obj = ensure_object(source, "manifest.source")?;
    let source_kind = expect_string(source_obj, "kind", "manifest.source")?;
    if source_kind != "quantized_blocks" {
        return Err(SsError::Other(
            "verify_quantized_manifest_artifacts requires a quantized_blocks manifest".into(),
        ));
    }

    let artifacts = root
        .get("artifacts")
        .ok_or_else(|| SsError::Other("manifest.artifacts is required".into()))?;
    let artifacts_obj = ensure_object(artifacts, "manifest.artifacts")?;
    let block_map_path = expect_string(artifacts_obj, "block_map", "manifest.artifacts")?;
    let data_files = ensure_string_array(artifacts_obj, "data_files", "manifest.artifacts")?;

    let mut reasons = Vec::new();
    if !Path::new(&block_map_path).exists() {
        reasons.push(format!("missing block_map file: {block_map_path}"));
    }
    for data_file in &data_files {
        if !Path::new(data_file).exists() {
            reasons.push(format!("missing data file: {data_file}"));
        }
    }

    let mut block_count = None;
    let mut declared_files = data_files.clone();
    declared_files.sort();
    declared_files.dedup();
    let mut validated_files = Vec::new();

    if reasons.is_empty() {
        let block_map = read_quantized_block_map_value(&block_map_path, true)?;
        let records = parse_quantized_block_records(&block_map)?;
        block_count = Some(records.len() as u64);

        let mut referenced_files = Vec::new();
        for record in &records {
            referenced_files.push(record.file.clone());
        }
        referenced_files.sort();
        referenced_files.dedup();
        validated_files = referenced_files.clone();

        for file in &referenced_files {
            if !declared_files.contains(file) {
                reasons.push(format!("block_map references undeclared data file: {file}"));
            }
        }

        reasons.extend(validate_quantized_block_records(&records)?);

        if let Some(quantization) = root.get("quantization") {
            let quant_obj = ensure_object(quantization, "manifest.quantization")?;
            if let Some(expected_count) = quant_obj
                .get("block_count")
                .and_then(|value| value.as_u64())
            {
                if Some(expected_count) != block_count {
                    reasons.push(format!(
                        "block_count mismatch: manifest={expected_count}, block_map={}",
                        block_count.unwrap_or(0)
                    ));
                }
            }
        }
    }

    Ok(serde_json::json!({
        "ok": reasons.is_empty(),
        "block_map": block_map_path,
        "declared_data_files": data_files,
        "validated_data_files": validated_files,
        "block_count": block_count,
        "reasons": reasons,
    }))
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
        header.insert("__metadata__".to_string(), serde_json::to_value(metadata)?);
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

        let copy_segment =
            |segment: &[u8], buffered: &mut usize, exact_len: &mut usize| -> Result<(), SsError> {
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

fn materialize_single_file_selection<F>(
    input_path: &str,
    output_path: &str,
    direct: bool,
    mut include: F,
) -> Result<usize, SsError>
where
    F: FnMut(&str) -> bool,
{
    let (metadata, _) = collect_tensor_layout(input_path)?;
    let file = fs::File::open(input_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    let mut selected = Vec::new();
    for name in tensors.names() {
        if include(name) {
            selected.push((name.clone(), tensors.tensor(name)?));
        }
    }

    let selected_count = selected.len();
    let metadata = if metadata.is_empty() {
        None
    } else {
        Some(metadata)
    };
    let (header_len, header_bytes, selected, total_tensor_bytes) =
        prepare_write_plan(selected, &metadata)?;
    if direct {
        write_direct_streaming(
            Path::new(output_path),
            header_len,
            &header_bytes,
            &selected,
            total_tensor_bytes,
        )?;
    } else {
        write_standard_streaming(Path::new(output_path), header_len, &header_bytes, &selected)?;
    }

    Ok(selected_count)
}

fn materialize_sharded_selection<F>(
    index_path: &str,
    output_path: &str,
    direct: bool,
    mut include: F,
) -> Result<usize, SsError>
where
    F: FnMut(&str) -> bool,
{
    let (summary, _, shard_metadata, _) = collect_sharded_tensor_layout(index_path)?;
    let selections = resolve_sharded_selections(index_path, &summary, |name| include(name));

    let mut metadata = HashMap::new();
    for selection in &selections {
        if let Some(shard_meta) = shard_metadata.get(&selection.shard_name) {
            if metadata.is_empty() && !shard_meta.is_empty() {
                metadata = shard_meta.clone();
            }
        }
    }
    for (k, v) in &summary.metadata_strings {
        metadata.entry(k.clone()).or_insert(v.clone());
    }
    for (k, v) in &summary.metadata_numbers {
        metadata.entry(k.clone()).or_insert(v.to_string());
    }

    let mut mmaps = Vec::with_capacity(selections.len());
    for selection in &selections {
        let file = fs::File::open(&selection.shard_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        mmaps.push(mmap);
    }

    let mut selected = Vec::new();
    for (selection, mmap) in selections.iter().zip(mmaps.iter()) {
        let tensors = SafeTensors::deserialize(mmap)?;
        let requested: HashSet<&str> = selection.tensor_names.iter().map(String::as_str).collect();
        for name in tensors.names() {
            if requested.contains(name.as_str()) {
                selected.push((name.clone(), tensors.tensor(name)?));
            }
        }
    }

    let selected_count = selected.len();
    let metadata = if metadata.is_empty() {
        None
    } else {
        Some(metadata)
    };
    let (header_len, header_bytes, selected, total_tensor_bytes) =
        prepare_write_plan(selected, &metadata)?;
    if direct {
        write_direct_streaming(
            Path::new(output_path),
            header_len,
            &header_bytes,
            &selected,
            total_tensor_bytes,
        )?;
    } else {
        write_standard_streaming(Path::new(output_path), header_len, &header_bytes, &selected)?;
    }

    Ok(selected_count)
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
    let header_len = u64::from_le_bytes(
        header_len_bytes
            .try_into()
            .map_err(|_| PyRuntimeError::new_err("Failed to read header length"))?,
    ) as usize;

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
                    (
                        vals.get(0).copied().unwrap_or(0),
                        vals.get(1).copied().unwrap_or(0),
                    )
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

fn open_readonly_mmap(py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
    let mmap_mod = py.import_bound("mmap")?;
    let builtins = py.import_bound("builtins")?;

    let file = builtins.call_method1("open", (path, "rb"))?;
    let fileno = file.call_method0("fileno")?;
    let mmap_kwargs = PyDict::new_bound(py);
    mmap_kwargs.set_item("access", mmap_mod.getattr("ACCESS_READ")?)?;
    let py_mmap = mmap_mod.call_method("mmap", (fileno, 0i64), Some(&mmap_kwargs))?;
    file.call_method0("close")?;
    Ok(py_mmap.unbind())
}

fn load_selected_tensors_from_mmap(
    py: Python<'_>,
    py_mmap: &Bound<'_, PyAny>,
    requested_names: &HashSet<&str>,
    device: &str,
    result: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let builtins = py.import_bound("builtins")?;
    let torch = py.import_bound("torch")?;
    let (tensors, data_start) = parse_tensor_layout(py, py_mmap)?;
    let memview = builtins.call_method1("memoryview", (py_mmap,))?;

    for (name, dtype, shape, start, end) in &tensors {
        if requested_names.contains(name.as_str()) {
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
    }

    Ok(())
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

/// Write a new safetensors file containing only the named tensors from a source file.
#[pyfunction]
#[pyo3(signature = (path, output_path, names, direct=false))]
fn materialize_selective(
    path: &str,
    output_path: &str,
    names: Vec<String>,
    direct: bool,
) -> PyResult<usize> {
    let requested: HashSet<&str> = names.iter().map(String::as_str).collect();
    materialize_single_file_selection(path, output_path, direct, |name| requested.contains(name))
        .map_err(PyErr::from)
}

/// Write a new safetensors file containing only tensors matching a prefix.
#[pyfunction]
#[pyo3(signature = (path, output_path, prefix, direct=false))]
fn materialize_by_prefix(
    path: &str,
    output_path: &str,
    prefix: &str,
    direct: bool,
) -> PyResult<usize> {
    materialize_single_file_selection(path, output_path, direct, |name| name.starts_with(prefix))
        .map_err(PyErr::from)
}

/// Write a new safetensors file containing only the named tensors from a sharded index.
#[pyfunction]
#[pyo3(signature = (index_path, output_path, names, direct=false))]
fn materialize_sharded_selective(
    index_path: &str,
    output_path: &str,
    names: Vec<String>,
    direct: bool,
) -> PyResult<usize> {
    let requested: HashSet<&str> = names.iter().map(String::as_str).collect();
    materialize_sharded_selection(index_path, output_path, direct, |name| {
        requested.contains(name)
    })
    .map_err(PyErr::from)
}

/// Write a new safetensors file containing only prefix-matched tensors from a sharded index.
#[pyfunction]
#[pyo3(signature = (index_path, output_path, prefix, direct=false))]
fn materialize_sharded_by_prefix(
    index_path: &str,
    output_path: &str,
    prefix: &str,
    direct: bool,
) -> PyResult<usize> {
    materialize_sharded_selection(index_path, output_path, direct, |name| {
        name.starts_with(prefix)
    })
    .map_err(PyErr::from)
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
    }

    let ret = PyTuple::new_bound(py, &[result.as_any(), py_mmap.as_any()]);
    Ok(ret.into())
}

/// Load all tensors referenced by a sharded safetensors index.
#[pyfunction]
#[pyo3(signature = (index_path, device="cpu"))]
fn _load_sharded_raw(py: Python<'_>, index_path: &str, device: &str) -> PyResult<PyObject> {
    let parsed = read_shard_index(index_path)?;
    let summary = summarize_shard_index(&parsed);
    let selections = resolve_sharded_selections(index_path, &summary, |_| true);

    let result = PyDict::new_bound(py);
    let handles = PyList::empty_bound(py);
    for selection in selections {
        let py_mmap = open_readonly_mmap(py, &selection.shard_path)?;
        let bound = py_mmap.bind(py);
        let requested: HashSet<&str> = selection.tensor_names.iter().map(String::as_str).collect();
        load_selected_tensors_from_mmap(py, bound, &requested, device, &result)?;
        handles.append(py_mmap)?;
    }

    let ret = PyTuple::new_bound(py, &[result.as_any(), handles.as_any()]);
    Ok(ret.into())
}

/// Load selected tensors by name from a sharded safetensors index.
#[pyfunction]
#[pyo3(signature = (index_path, names, device="cpu"))]
fn _load_sharded_selective_raw(
    py: Python<'_>,
    index_path: &str,
    names: Vec<String>,
    device: &str,
) -> PyResult<PyObject> {
    let parsed = read_shard_index(index_path)?;
    let summary = summarize_shard_index(&parsed);
    let requested: HashSet<&str> = names.iter().map(String::as_str).collect();
    let selections =
        resolve_sharded_selections(index_path, &summary, |name| requested.contains(name));

    let result = PyDict::new_bound(py);
    let handles = PyList::empty_bound(py);
    for selection in selections {
        let py_mmap = open_readonly_mmap(py, &selection.shard_path)?;
        let bound = py_mmap.bind(py);
        let selection_names: HashSet<&str> =
            selection.tensor_names.iter().map(String::as_str).collect();
        load_selected_tensors_from_mmap(py, bound, &selection_names, device, &result)?;
        handles.append(py_mmap)?;
    }

    let ret = PyTuple::new_bound(py, &[result.as_any(), handles.as_any()]);
    Ok(ret.into())
}

/// Load tensors matching a prefix from a sharded safetensors index.
#[pyfunction]
#[pyo3(signature = (index_path, prefix, device="cpu"))]
fn _load_sharded_by_prefix_raw(
    py: Python<'_>,
    index_path: &str,
    prefix: &str,
    device: &str,
) -> PyResult<PyObject> {
    let parsed = read_shard_index(index_path)?;
    let summary = summarize_shard_index(&parsed);
    let selections =
        resolve_sharded_selections(index_path, &summary, |name| name.starts_with(prefix));

    let result = PyDict::new_bound(py);
    let handles = PyList::empty_bound(py);
    for selection in selections {
        let py_mmap = open_readonly_mmap(py, &selection.shard_path)?;
        let bound = py_mmap.bind(py);
        let selection_names: HashSet<&str> =
            selection.tensor_names.iter().map(String::as_str).collect();
        load_selected_tensors_from_mmap(py, bound, &selection_names, device, &result)?;
        handles.append(py_mmap)?;
    }

    let ret = PyTuple::new_bound(py, &[result.as_any(), handles.as_any()]);
    Ok(ret.into())
}

/// Write a deterministic safetensors container for opaque quantized block payloads.
///
/// Each payload is stored as a single 1D U8 tensor named after its block id using
/// the Serenity quantized-block prefix. The returned value is a validated block map
/// that can be written with `write_quantized_block_map`.
#[pyfunction]
#[pyo3(signature = (payloads, path, block_tensors=None, metadata=None, direct=false))]
fn write_quantized_block_container(
    py: Python<'_>,
    payloads: &Bound<'_, PyDict>,
    path: &str,
    block_tensors: Option<&Bound<'_, PyDict>>,
    metadata: Option<&Bound<'_, PyDict>>,
    direct: bool,
) -> PyResult<PyObject> {
    let views = prepare_python_quantized_block_views(payloads)?;
    if views.is_empty() {
        return Err(PyRuntimeError::new_err(
            "write_quantized_block_container requires at least one payload",
        ));
    }

    let block_tensors = block_map_tensor_names_from_py(py, block_tensors)?;
    let metadata = Some(build_quantized_block_container_metadata(
        metadata_from_py_dict(metadata)?,
    ));
    let (header_len, header_bytes, views, total_tensor_bytes) =
        prepare_write_plan(views, &metadata)?;

    if direct {
        write_direct_streaming(
            Path::new(path),
            header_len,
            &header_bytes,
            &views,
            total_tensor_bytes,
        )?;
    } else {
        write_standard_streaming(Path::new(path), header_len, &header_bytes, &views)?;
    }

    let block_map = build_quantized_block_map_for_container(path, path, &block_tensors)?;
    json_value_to_py(py, &block_map)
}

/// Load opaque quantized block payloads from a block map or quantized-source manifest.
///
/// Returns `(dict, mmap_handles)` where the dict maps block id -> `torch.uint8`
/// tensor view over the persisted payload bytes.
#[pyfunction]
#[pyo3(signature = (reference_path, block_ids=None, device="cpu"))]
fn _load_quantized_blocks_raw(
    py: Python<'_>,
    reference_path: &str,
    block_ids: Option<Vec<String>>,
    device: &str,
) -> PyResult<PyObject> {
    let block_map = read_quantized_block_map_reference_value(reference_path, true)?;
    let records = parse_quantized_block_records(&block_map)?;
    let requested: Option<HashSet<&str>> = block_ids
        .as_ref()
        .map(|ids| ids.iter().map(String::as_str).collect());

    let mut grouped: HashMap<String, Vec<QuantizedBlockRecord>> = HashMap::new();
    for record in records {
        if requested
            .as_ref()
            .is_some_and(|ids| !ids.contains(record.id.as_str()))
        {
            continue;
        }
        grouped.entry(record.file.clone()).or_default().push(record);
    }

    let builtins = py.import_bound("builtins")?;
    let torch = py.import_bound("torch")?;
    let result = PyDict::new_bound(py);
    let handles = PyList::empty_bound(py);

    let mut file_paths: Vec<String> = grouped.keys().cloned().collect();
    file_paths.sort();
    for file_path in file_paths {
        let mut file_records = grouped.remove(&file_path).unwrap_or_default();
        file_records.sort_by(|left, right| left.id.cmp(&right.id));
        let py_mmap = open_readonly_mmap(py, &file_path)?;
        let bound = py_mmap.bind(py);
        let memview = builtins.call_method1("memoryview", (bound,))?;
        let file_len = fs::metadata(&file_path)?.len() as usize;

        for record in file_records {
            let end = record.offset.saturating_add(record.nbytes);
            if end > file_len {
                return Err(PyRuntimeError::new_err(format!(
                    "block {} exceeds file bounds in {}: offset={} nbytes={} file_size={}",
                    record.id, file_path, record.offset, record.nbytes, file_len
                )));
            }
            let tensor = memview_slice_to_tensor(
                py,
                &torch,
                &memview,
                "U8",
                &[record.nbytes],
                record.offset,
                end,
                device,
            )?;
            result.set_item(record.id, tensor)?;
        }

        handles.append(py_mmap)?;
    }

    let ret = PyTuple::new_bound(py, &[result.as_any(), handles.as_any()]);
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
    let (_, entries) =
        collect_tensor_layout(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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

/// List tensor names in a sharded safetensors index without opening shard data.
#[pyfunction]
fn sharded_tensor_names(index_path: &str) -> PyResult<Vec<String>> {
    let parsed = read_shard_index(index_path)?;
    let summary = summarize_shard_index(&parsed);
    let mut names: Vec<String> = summary.weight_map.keys().cloned().collect();
    names.sort();
    Ok(names)
}

/// Return tensor layout across all shards with resolved shard paths and offsets.
#[pyfunction]
fn sharded_tensor_layout(py: Python<'_>, index_path: &str) -> PyResult<PyObject> {
    let (summary, shard_paths, shard_metadata, entries) =
        collect_sharded_tensor_layout(index_path)?;
    let result = PyDict::new_bound(py);
    result.set_item("index_path", index_path)?;

    let metadata_dict = PyDict::new_bound(py);
    for (k, v) in &summary.metadata_strings {
        metadata_dict.set_item(k, v)?;
    }
    for (k, v) in &summary.metadata_numbers {
        metadata_dict.set_item(k, v)?;
    }
    result.set_item("metadata", metadata_dict)?;

    let shards_dict = PyDict::new_bound(py);
    let mut shard_names: Vec<String> = shard_paths.keys().cloned().collect();
    shard_names.sort();
    for shard_name in shard_names {
        let item = PyDict::new_bound(py);
        if let Some(path) = shard_paths.get(&shard_name) {
            item.set_item("path", path)?;
        }
        let shard_meta = PyDict::new_bound(py);
        if let Some(metadata) = shard_metadata.get(&shard_name) {
            for (k, v) in metadata {
                shard_meta.set_item(k, v)?;
            }
        }
        item.set_item("metadata", shard_meta)?;
        shards_dict.set_item(&shard_name, item)?;
    }
    result.set_item("shards", shards_dict)?;

    let tensors_dict = PyDict::new_bound(py);
    for entry in &entries {
        let item = PyDict::new_bound(py);
        let shape: Vec<u64> = entry.shape.iter().map(|&n| n as u64).collect();
        item.set_item("shard", &entry.shard_name)?;
        item.set_item("path", &entry.shard_path)?;
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
        tensors_dict.set_item(&entry.name, item)?;
    }
    result.set_item("tensors", tensors_dict)?;

    Ok(result.into())
}

/// Build a canonical Serenity source manifest.
#[pyfunction]
#[pyo3(signature = (
    model_family,
    model_version,
    source_kind,
    source_path=None,
    original_source=None,
    source_signature=None,
    dtype=None,
    variant=None,
    tensor_prefixes=None,
    tensor_names=None,
    minimum_serenity_version=None,
    stagehand_layout=None,
    extra_metadata=None
))]
fn source_manifest(
    py: Python<'_>,
    model_family: &str,
    model_version: &str,
    source_kind: &str,
    source_path: Option<&str>,
    original_source: Option<&str>,
    source_signature: Option<&str>,
    dtype: Option<&str>,
    variant: Option<&str>,
    tensor_prefixes: Option<Vec<String>>,
    tensor_names: Option<Vec<String>>,
    minimum_serenity_version: Option<&str>,
    stagehand_layout: Option<&str>,
    extra_metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let extra_metadata = extra_metadata
        .map(|value| py_to_json_value(py, value))
        .transpose()?;
    let manifest = build_source_manifest_value(
        model_family,
        model_version,
        source_kind,
        source_path,
        original_source,
        source_signature,
        dtype,
        variant,
        tensor_prefixes,
        tensor_names,
        minimum_serenity_version,
        stagehand_layout,
        extra_metadata,
    )?;
    json_value_to_py(py, &manifest)
}

/// Build a canonical Serenity manifest for persisted quantized sources.
#[pyfunction]
#[pyo3(signature = (
    model_family,
    model_version,
    original_source,
    source_signature,
    block_map_path,
    data_files,
    source_path=None,
    quant_mode="eriquant",
    dtype=None,
    variant=None,
    tensor_prefixes=None,
    block_count=None,
    group_size=None,
    minimum_serenity_version=None,
    stagehand_layout=None,
    extra_metadata=None
))]
fn quantized_source_manifest(
    py: Python<'_>,
    model_family: &str,
    model_version: &str,
    original_source: &str,
    source_signature: &str,
    block_map_path: &str,
    data_files: Vec<String>,
    source_path: Option<&str>,
    quant_mode: &str,
    dtype: Option<&str>,
    variant: Option<&str>,
    tensor_prefixes: Option<Vec<String>>,
    block_count: Option<u64>,
    group_size: Option<u64>,
    minimum_serenity_version: Option<&str>,
    stagehand_layout: Option<&str>,
    extra_metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let extra_metadata = extra_metadata
        .map(|value| py_to_json_value(py, value))
        .transpose()?;
    let manifest = build_quantized_source_manifest_value(
        model_family,
        model_version,
        source_path,
        original_source,
        source_signature,
        block_map_path,
        data_files,
        quant_mode,
        dtype,
        variant,
        tensor_prefixes,
        block_count,
        group_size,
        minimum_serenity_version,
        stagehand_layout,
        extra_metadata,
    )?;
    json_value_to_py(py, &manifest)
}

/// Read and validate a Serenity source manifest.
#[pyfunction]
#[pyo3(signature = (path, resolve_paths=true))]
fn read_manifest(py: Python<'_>, path: &str, resolve_paths: bool) -> PyResult<PyObject> {
    let manifest = read_manifest_value(path, resolve_paths)?;
    json_value_to_py(py, &manifest)
}

/// Validate and write a Serenity source manifest.
#[pyfunction]
fn write_manifest(py: Python<'_>, path: &str, manifest: &Bound<'_, PyAny>) -> PyResult<()> {
    let manifest = py_to_json_value(py, manifest)?;
    write_manifest_value(path, manifest)?;
    Ok(())
}

/// Check whether a manifest matches the requested runtime/source constraints.
#[pyfunction]
#[pyo3(signature = (
    path,
    model_family,
    model_version,
    source_signature=None,
    quant_mode=None,
    stagehand_layout=None
))]
fn check_manifest_compatibility(
    py: Python<'_>,
    path: &str,
    model_family: &str,
    model_version: &str,
    source_signature: Option<&str>,
    quant_mode: Option<&str>,
    stagehand_layout: Option<&str>,
) -> PyResult<PyObject> {
    let manifest = read_manifest_value(path, false)?;
    let result = evaluate_manifest_compatibility(
        &manifest,
        model_family,
        model_version,
        source_signature,
        quant_mode,
        stagehand_layout,
    );
    json_value_to_py(py, &result)
}

/// Read and validate a Serenity quantized block-map file.
#[pyfunction]
#[pyo3(signature = (path, resolve_paths=true))]
fn read_quantized_block_map(py: Python<'_>, path: &str, resolve_paths: bool) -> PyResult<PyObject> {
    let block_map = read_quantized_block_map_value(path, resolve_paths)?;
    json_value_to_py(py, &block_map)
}

/// Validate and write a Serenity quantized block-map file.
#[pyfunction]
fn write_quantized_block_map(
    py: Python<'_>,
    path: &str,
    block_map: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let block_map = py_to_json_value(py, block_map)?;
    write_quantized_block_map_value(path, block_map)?;
    Ok(())
}

/// Verify that a quantized-source manifest points to a consistent block-map/data-file set.
#[pyfunction]
fn verify_quantized_manifest_artifacts(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let result = verify_quantized_manifest_artifacts_value(path)?;
    json_value_to_py(py, &result)
}

// ── Format detection & probing ───────────────────────────────────────────────

#[pyfunction]
#[pyo3(name = "detect_format")]
fn py_detect_format(_py: Python<'_>, path: &str) -> PyResult<String> {
    let fmt = format_detect::detect_format(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(fmt.to_string())
}

#[pyfunction]
#[pyo3(name = "probe_model")]
fn py_probe_model(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let info = probe::probe_model(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(e))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("format", info.format.to_string())?;
    dict.set_item("path", info.path.to_string_lossy().to_string())?;
    dict.set_item("tensor_count", info.tensor_count)?;
    dict.set_item("tensor_names", info.tensor_names)?;
    dict.set_item("param_count", info.param_count)?;
    dict.set_item("total_file_bytes", info.total_file_bytes)?;

    // tensor_shapes: dict of name → list of ints
    let shapes_dict = PyDict::new_bound(py);
    for (name, shape) in &info.tensor_shapes {
        shapes_dict.set_item(name, shape)?;
    }
    dict.set_item("tensor_shapes", shapes_dict)?;

    // tensor_dtypes: dict of name → str
    let dtypes_dict = PyDict::new_bound(py);
    for (name, dtype) in &info.tensor_dtypes {
        dtypes_dict.set_item(name, dtype)?;
    }
    dict.set_item("tensor_dtypes", dtypes_dict)?;

    // metadata: dict of str → str
    let meta_dict = PyDict::new_bound(py);
    for (k, v) in &info.metadata {
        meta_dict.set_item(k, v)?;
    }
    dict.set_item("metadata", meta_dict)?;

    // quant_types: None or dict
    match &info.quant_types {
        Some(qt) => {
            let qt_dict = PyDict::new_bound(py);
            for (k, v) in qt {
                qt_dict.set_item(k, v)?;
            }
            dict.set_item("quant_types", qt_dict)?;
        }
        None => {
            dict.set_item("quant_types", py.None())?;
        }
    }

    Ok(dict.into())
}

// ── Diffusers directory probing ──────────────────────────────────────────────

/// Probe a diffusers model directory and return component layout information.
///
/// Returns a dict with:
///   - root: str (directory path)
///   - pipeline_class: str | None
///   - components: list of dicts with name, class_name, library_name, weight_source, has_config
///   - tensor_count: int (total across all components)
///   - param_count: int
///   - total_file_bytes: int
///   - tensor_names: list of str (prefixed with component name)
#[pyfunction]
fn probe_diffusers(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let dir = std::path::Path::new(path);
    let layout = diffusers::DiffusersLayout::open(dir)
        .map_err(|e| PyValueError::new_err(e))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("root", layout.root.to_string_lossy().to_string())?;

    // Pipeline class name
    let pipeline_class = layout
        .model_index
        .get("_class_name")
        .and_then(|v| v.as_str())
        .map(String::from);
    match pipeline_class {
        Some(cls) => dict.set_item("pipeline_class", cls)?,
        None => dict.set_item("pipeline_class", py.None())?,
    }

    // Components list
    let comp_list = PyList::empty_bound(py);
    for comp in &layout.components {
        let cd = PyDict::new_bound(py);
        cd.set_item("name", &comp.name)?;
        cd.set_item(
            "class_name",
            comp.class_name.as_deref().map(|s| s.to_string()),
        )?;
        cd.set_item(
            "library_name",
            comp.library_name.as_deref().map(|s| s.to_string()),
        )?;
        let ws_str = match &comp.weight_source {
            diffusers::WeightSource::SingleSafetensors(p) => {
                format!("single_safetensors:{}", p.display())
            }
            diffusers::WeightSource::ShardedSafetensors {
                shard_paths, ..
            } => format!("sharded_safetensors:{}_shards", shard_paths.len()),
            diffusers::WeightSource::SinglePytorch(p) => {
                format!("single_pytorch:{}", p.display())
            }
            diffusers::WeightSource::ShardedPytorch {
                shard_paths, ..
            } => format!("sharded_pytorch:{}_shards", shard_paths.len()),
            diffusers::WeightSource::None => "none".to_string(),
        };
        cd.set_item("weight_source", ws_str)?;
        cd.set_item("has_config", comp.config.is_some())?;
        comp_list.append(cd)?;
    }
    dict.set_item("components", comp_list)?;

    // Aggregated info
    let info = layout
        .to_model_info()
        .map_err(|e| PyRuntimeError::new_err(e))?;
    dict.set_item("tensor_count", info.tensor_count)?;
    dict.set_item("param_count", info.param_count)?;
    dict.set_item("total_file_bytes", info.total_file_bytes)?;
    dict.set_item("tensor_names", info.tensor_names)?;

    Ok(dict.into())
}

// ── PyTorch checkpoint loading ───────────────────────────────────────────────

#[pyfunction]
fn load_pickle_index(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let index = pytorch::PickleIndex::open(std::path::Path::new(path))
        .map_err(|e| PyRuntimeError::new_err(e))?;

    let dict = PyDict::new_bound(py);

    let names: Vec<&str> = index.tensors.iter().map(|t| t.name.as_str()).collect();
    dict.set_item("tensor_names", names)?;

    let shapes_dict = PyDict::new_bound(py);
    for t in &index.tensors {
        shapes_dict.set_item(&t.name, &t.shape)?;
    }
    dict.set_item("tensor_shapes", shapes_dict)?;

    let dtypes_dict = PyDict::new_bound(py);
    for t in &index.tensors {
        dtypes_dict.set_item(&t.name, &t.dtype)?;
    }
    dict.set_item("tensor_dtypes", dtypes_dict)?;

    let storage_dict = PyDict::new_bound(py);
    for (k, v) in &index.storage_files {
        storage_dict.set_item(k, v)?;
    }
    dict.set_item("storage_map", storage_dict)?;

    dict.set_item("tensor_count", index.tensors.len())?;

    let param_count: u64 = index.tensors.iter().map(|t| t.numel as u64).sum();
    dict.set_item("param_count", param_count)?;

    Ok(dict.into())
}

#[pyfunction]
fn load_pickle_tensor(py: Python<'_>, path: &str, name: &str) -> PyResult<PyObject> {
    let index = pytorch::PickleIndex::open(std::path::Path::new(path))
        .map_err(|e| PyRuntimeError::new_err(e))?;

    let info = index
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| {
            PyValueError::new_err(format!("tensor '{}' not found in checkpoint", name))
        })?;

    let raw_bytes = index
        .read_tensor_bytes(info)
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Convert dtype string to torch dtype string for Python side
    let torch_dtype = match info.dtype.as_str() {
        "float32" => "torch.float32",
        "float16" => "torch.float16",
        "bfloat16" => "torch.bfloat16",
        "float64" => "torch.float64",
        "int64" => "torch.int64",
        "int32" => "torch.int32",
        "int16" => "torch.int16",
        "int8" => "torch.int8",
        "uint8" => "torch.uint8",
        _ => "torch.float32",
    };

    // Use torch.frombuffer to create tensor from raw bytes, then reshape
    let torch = py.import_bound("torch")?;

    // Get actual dtype
    let actual_dtype = py.eval_bound(torch_dtype, None, None)?;

    let py_bytes = pyo3::types::PyBytes::new_bound(py, &raw_bytes);

    let tensor = torch.call_method(
        "frombuffer",
        (py_bytes,),
        Some(&{
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("dtype", actual_dtype)?;
            kwargs
        }),
    )?;

    // Clone to own the data (frombuffer returns a view of the bytes)
    let tensor = tensor.call_method0("clone")?;

    // Reshape
    let shape: Vec<i64> = info.shape.iter().map(|&s| s as i64).collect();
    let py_shape = PyTuple::new_bound(py, &shape);
    let tensor = tensor.call_method1("reshape", (py_shape,))?;

    Ok(tensor.into())
}

// ── GGUF loading ────────────────────────────────────────────────────────────

/// Open a GGUF file and return its index as a Python dict.
///
/// Returns: dict with keys "version", "alignment", "tensor_count",
/// "tensors" (list of dicts), "metadata" (dict).
#[pyfunction]
fn load_gguf_index(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let idx = gguf::GgufIndex::open(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(e))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("version", idx.version)?;
    dict.set_item("alignment", idx.alignment)?;
    dict.set_item("tensor_count", idx.tensors.len())?;
    dict.set_item("data_offset", idx.data_offset)?;

    // Tensors list
    let tensors_list = PyList::new_bound(py, Vec::<PyObject>::new());
    for t in &idx.tensors {
        let td = PyDict::new_bound(py);
        td.set_item("name", &t.name)?;
        td.set_item("shape", &t.shape)?;
        td.set_item("quant_type", t.quant_type.name())?;
        td.set_item("offset", t.offset)?;
        td.set_item("byte_size", t.byte_size)?;
        td.set_item("param_count", t.param_count)?;
        tensors_list.append(td)?;
    }
    dict.set_item("tensors", tensors_list)?;

    // Metadata dict
    let meta_dict = PyDict::new_bound(py);
    for (k, v) in &idx.metadata {
        meta_dict.set_item(k, v.to_string_lossy())?;
    }
    dict.set_item("metadata", meta_dict)?;

    Ok(dict.into())
}

/// Dequantize raw bytes of a specific GGUF quant type to a BF16 torch.Tensor.
///
/// Arguments:
///   data: bytes — raw quantized data
///   quant_type: str — e.g. "Q4_0", "Q8_0", "Q6_K", "F16", etc.
///   shape: list[int] — desired output tensor shape
#[pyfunction]
fn dequant_tensor(py: Python<'_>, data: &[u8], quant_type: &str, shape: Vec<usize>) -> PyResult<PyObject> {
    let qt = quant_type_from_name(quant_type)
        .map_err(|e| PyValueError::new_err(e))?;

    let n_weights: usize = if shape.is_empty() { 0 } else { shape.iter().product() };

    let bf16_bytes = py
        .allow_threads(move || {
            let bf16_vec = gguf_dequant::dequant_to_bf16(data, qt, n_weights)?;
            let raw: &[u8] = bytemuck::cast_slice(&bf16_vec);
            Ok::<Vec<u8>, String>(raw.to_vec())
        })
        .map_err(|e| PyValueError::new_err(e))?;

    let torch = py.import_bound("torch")?;
    let bf16_dtype = torch.getattr("bfloat16")?;
    let py_bytes = pyo3::types::PyBytes::new_bound(py, &bf16_bytes);
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", bf16_dtype)?;
    let tensor = torch.call_method("frombuffer", (py_bytes,), Some(&kwargs))?;
    let tensor = tensor.call_method0("clone")?;
    let shape_tuple = PyTuple::new_bound(py, &shape);
    let tensor = tensor.call_method1("reshape", (shape_tuple,))?;
    Ok(tensor.into())
}

fn quant_type_from_name(name: &str) -> Result<gguf::GgufQuantType, String> {
    match name {
        "F32" => Ok(gguf::GgufQuantType::F32),
        "F16" => Ok(gguf::GgufQuantType::F16),
        "BF16" => Ok(gguf::GgufQuantType::BF16),
        "F64" => Ok(gguf::GgufQuantType::F64),
        "I8" => Ok(gguf::GgufQuantType::I8),
        "I16" => Ok(gguf::GgufQuantType::I16),
        "I32" => Ok(gguf::GgufQuantType::I32),
        "I64" => Ok(gguf::GgufQuantType::I64),
        "Q4_0" => Ok(gguf::GgufQuantType::Q4_0),
        "Q4_1" => Ok(gguf::GgufQuantType::Q4_1),
        "Q5_0" => Ok(gguf::GgufQuantType::Q5_0),
        "Q5_1" => Ok(gguf::GgufQuantType::Q5_1),
        "Q8_0" => Ok(gguf::GgufQuantType::Q8_0),
        "Q8_1" => Ok(gguf::GgufQuantType::Q8_1),
        "Q2_K" => Ok(gguf::GgufQuantType::Q2K),
        "Q3_K" => Ok(gguf::GgufQuantType::Q3K),
        "Q4_K" => Ok(gguf::GgufQuantType::Q4K),
        "Q5_K" => Ok(gguf::GgufQuantType::Q5K),
        "Q6_K" => Ok(gguf::GgufQuantType::Q6K),
        "Q8_K" => Ok(gguf::GgufQuantType::Q8K),
        "IQ2_XXS" => Ok(gguf::GgufQuantType::IQ2XXS),
        "IQ2_XS" => Ok(gguf::GgufQuantType::IQ2XS),
        "IQ2_S" => Ok(gguf::GgufQuantType::IQ2S),
        "IQ3_XXS" => Ok(gguf::GgufQuantType::IQ3XXS),
        "IQ3_S" => Ok(gguf::GgufQuantType::IQ3S),
        "IQ4_NL" => Ok(gguf::GgufQuantType::IQ4NL),
        "IQ4_XS" => Ok(gguf::GgufQuantType::IQ4XS),
        "IQ1_S" => Ok(gguf::GgufQuantType::IQ1S),
        "IQ1_M" => Ok(gguf::GgufQuantType::IQ1M),
        other => Err(format!("unknown quant type name: {other}")),
    }
}

// ── Module definition ───────────────────────────────────────────────────────

#[pymodule]
fn serenity_safetensors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save_file_direct, m)?)?;
    m.add_function(wrap_pyfunction!(save_file, m)?)?;
    m.add_function(wrap_pyfunction!(materialize_selective, m)?)?;
    m.add_function(wrap_pyfunction!(materialize_by_prefix, m)?)?;
    m.add_function(wrap_pyfunction!(materialize_sharded_selective, m)?)?;
    m.add_function(wrap_pyfunction!(materialize_sharded_by_prefix, m)?)?;
    m.add_function(wrap_pyfunction!(_load_file_raw, m)?)?;
    m.add_function(wrap_pyfunction!(_load_selective_raw, m)?)?;
    m.add_function(wrap_pyfunction!(_load_by_prefix_raw, m)?)?;
    m.add_function(wrap_pyfunction!(_load_sharded_raw, m)?)?;
    m.add_function(wrap_pyfunction!(_load_sharded_selective_raw, m)?)?;
    m.add_function(wrap_pyfunction!(_load_sharded_by_prefix_raw, m)?)?;
    m.add_function(wrap_pyfunction!(write_quantized_block_container, m)?)?;
    m.add_function(wrap_pyfunction!(_load_quantized_blocks_raw, m)?)?;
    m.add_function(wrap_pyfunction!(file_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_layout, m)?)?;
    m.add_function(wrap_pyfunction!(training_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_names, m)?)?;
    m.add_function(wrap_pyfunction!(shard_index, m)?)?;
    m.add_function(wrap_pyfunction!(sharded_tensor_names, m)?)?;
    m.add_function(wrap_pyfunction!(sharded_tensor_layout, m)?)?;
    m.add_function(wrap_pyfunction!(source_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(quantized_source_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(read_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(write_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(check_manifest_compatibility, m)?)?;
    m.add_function(wrap_pyfunction!(read_quantized_block_map, m)?)?;
    m.add_function(wrap_pyfunction!(write_quantized_block_map, m)?)?;
    m.add_function(wrap_pyfunction!(verify_quantized_manifest_artifacts, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_format, m)?)?;
    m.add_function(wrap_pyfunction!(py_probe_model, m)?)?;
    m.add_function(wrap_pyfunction!(probe_diffusers, m)?)?;
    m.add_function(wrap_pyfunction!(load_pickle_index, m)?)?;
    m.add_function(wrap_pyfunction!(load_pickle_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(load_gguf_index, m)?)?;
    m.add_function(wrap_pyfunction!(dequant_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(unified::load_model, m)?)?;
    m.add_class::<quantized_tensor::QuantizedTensor>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
            .join(format!(
                "serenity_safetensors_{name}_{pid}_{nanos}.{suffix}"
            ))
            .to_string_lossy()
            .into_owned()
    }

    fn write_test_safetensors(path: &str) {
        let metadata = Some(HashMap::from([
            ("family".to_string(), "ltx2".to_string()),
            ("step".to_string(), "3".to_string()),
        ]));
        let views = vec![
            (
                "weight".to_string(),
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
                "bias".to_string(),
                ByteView {
                    dtype: StDtype::I8,
                    shape: vec![3],
                    data: [5i8, 6i8, 7i8]
                        .into_iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                },
            ),
        ];
        write_byte_view_safetensors(path, metadata, views);
    }

    fn write_byte_view_safetensors(
        path: &str,
        metadata: Option<HashMap<String, String>>,
        views: Vec<(String, ByteView)>,
    ) {
        let serialized = serialize(views, &metadata).expect("serialize");
        fs::write(path, serialized).expect("write test safetensors");
    }

    fn write_sharded_fixture() -> (String, String) {
        let dir = temp_path("sharded", "dir");
        fs::create_dir_all(&dir).expect("create temp dir");

        let shard_one = Path::new(&dir).join("model-00001-of-00002.safetensors");
        let shard_two = Path::new(&dir).join("model-00002-of-00002.safetensors");
        let index_path = Path::new(&dir).join("model.safetensors.index.json");

        write_byte_view_safetensors(
            shard_one.to_string_lossy().as_ref(),
            Some(HashMap::from([("family".to_string(), "ltx2".to_string())])),
            vec![(
                "a.weight".to_string(),
                ByteView {
                    dtype: StDtype::F32,
                    shape: vec![2],
                    data: [11.0f32, 12.0f32]
                        .into_iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                },
            )],
        );

        write_byte_view_safetensors(
            shard_two.to_string_lossy().as_ref(),
            Some(HashMap::from([("family".to_string(), "ltx2".to_string())])),
            vec![
                (
                    "b.weight".to_string(),
                    ByteView {
                        dtype: StDtype::BF16,
                        shape: vec![2],
                        data: [1u16, 2u16]
                            .into_iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    },
                ),
                (
                    "b.bias".to_string(),
                    ByteView {
                        dtype: StDtype::I16,
                        shape: vec![2],
                        data: [3i16, 4i16]
                            .into_iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    },
                ),
            ],
        );

        let payload = serde_json::json!({
            "metadata": {"total_size": 20, "format": "pt"},
            "weight_map": {
                "a.weight": "model-00001-of-00002.safetensors",
                "b.weight": "model-00002-of-00002.safetensors",
                "b.bias": "model-00002-of-00002.safetensors"
            }
        });
        fs::write(
            &index_path,
            serde_json::to_vec(&payload).expect("json bytes"),
        )
        .expect("write index");

        (dir, index_path.to_string_lossy().into_owned())
    }

    fn write_quantized_block_container_fixture(
        path: &str,
        file_field: &str,
    ) -> (serde_json::Value, HashMap<String, Vec<u8>>) {
        let payloads = HashMap::from([
            ("transformer.layers.0".to_string(), vec![1u8, 3u8, 5u8, 7u8]),
            ("transformer.layers.1".to_string(), vec![2u8, 4u8, 6u8]),
        ]);
        let views = payloads
            .iter()
            .map(|(block_id, data)| {
                (
                    build_quantized_block_tensor_name(block_id),
                    ByteView {
                        dtype: StDtype::U8,
                        shape: vec![data.len()],
                        data: data.clone(),
                    },
                )
            })
            .collect::<Vec<_>>();
        let metadata = Some(build_quantized_block_container_metadata(Some(
            HashMap::from([("quant_mode".to_string(), "eriquant".to_string())]),
        )));
        let (header_len, header_bytes, views, _) =
            prepare_write_plan(views, &metadata).expect("prepare quantized container");
        write_standard_streaming(Path::new(path), header_len, &header_bytes, &views)
            .expect("write quantized container");

        let block_tensors = HashMap::from([
            (
                "transformer.layers.0".to_string(),
                vec!["linear.weight".to_string()],
            ),
            (
                "transformer.layers.1".to_string(),
                vec!["proj.weight".to_string(), "proj.bias".to_string()],
            ),
        ]);

        (
            build_quantized_block_map_for_container(path, file_field, &block_tensors)
                .expect("block map"),
            payloads,
        )
    }

    #[test]
    fn dtype_aliases_cover_fp8_and_bfloat16() {
        assert_eq!(
            dtype_from_torch_str("torch.bfloat16").expect("bf16"),
            StDtype::BF16
        );
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

        let weight = entries
            .iter()
            .find(|entry| entry.name == "weight")
            .expect("weight entry");
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
            summary.weight_map.get("a.weight").map(String::as_str),
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
    fn sharded_tensor_layout_resolves_relative_shards() {
        let (dir, index_path) = write_sharded_fixture();
        let expected_shard_one = Path::new(&dir)
            .join("model-00001-of-00002.safetensors")
            .to_string_lossy()
            .into_owned();

        let (summary, shard_paths, shard_metadata, entries) =
            collect_sharded_tensor_layout(&index_path).expect("sharded layout");
        assert_eq!(
            summary.metadata_numbers.get("total_size").copied(),
            Some(20)
        );
        assert_eq!(entries.len(), 3);
        assert_eq!(
            shard_paths
                .get("model-00001-of-00002.safetensors")
                .map(String::as_str),
            Some(expected_shard_one.as_str())
        );
        assert_eq!(
            shard_metadata
                .get("model-00002-of-00002.safetensors")
                .and_then(|md| md.get("family"))
                .map(String::as_str),
            Some("ltx2")
        );
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            vec!["a.weight", "b.weight", "b.bias"]
        );
        assert!(entries.iter().all(|entry| entry.absolute_offsets.0 >= 8));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn resolve_sharded_selections_filters_prefix() {
        let (_, index_path) = write_sharded_fixture();
        let parsed = read_shard_index(&index_path).expect("read shard index");
        let summary = summarize_shard_index(&parsed);

        let selections =
            resolve_sharded_selections(&index_path, &summary, |name| name.starts_with("b."));
        assert_eq!(selections.len(), 1);
        assert_eq!(selections[0].shard_name, "model-00002-of-00002.safetensors");
        assert_eq!(
            selections[0].tensor_names,
            vec!["b.bias".to_string(), "b.weight".to_string()]
        );

        let parent = Path::new(&index_path).parent().expect("index parent");
        fs::remove_dir_all(parent).ok();
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
        assert_eq!(
            parsed.tensor("linear.weight").expect("weight").shape(),
            &[2, 2]
        );

        let (layout_metadata, entries) = collect_tensor_layout(&path).expect("layout");
        assert_eq!(
            layout_metadata.get("family").map(String::as_str),
            Some("ltx2")
        );
        assert_eq!(
            layout_metadata.get("quant").map(String::as_str),
            Some("eriquant")
        );
        assert_eq!(entries.len(), 2);
        assert!(entries
            .iter()
            .all(|entry| entry.absolute_offsets.1 > entry.absolute_offsets.0));

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
        assert_eq!(
            parsed.tensor("tensor").expect("tensor").dtype(),
            StDtype::BF16
        );
        let (layout_metadata, _) = collect_tensor_layout(&path).expect("layout");
        assert_eq!(layout_metadata.get("step").map(String::as_str), Some("42"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn materialize_selective_writes_subset_file() {
        let source = temp_path("materialize_source", "safetensors");
        let output = temp_path("materialize_subset", "safetensors");
        write_test_safetensors(&source);

        let written =
            materialize_single_file_selection(&source, &output, false, |name| name == "bias")
                .expect("materialize selective");
        assert_eq!(written, 1);

        let names = tensor_names(&output).expect("subset names");
        assert_eq!(names, vec!["bias".to_string()]);
        let (metadata, entries) = collect_tensor_layout(&output).expect("subset layout");
        assert_eq!(metadata.get("family").map(String::as_str), Some("ltx2"));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "bias");

        fs::remove_file(source).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn materialize_sharded_by_prefix_writes_subset_file() {
        let (dir, index_path) = write_sharded_fixture();
        let output = temp_path("materialize_sharded_subset", "safetensors");

        let written = materialize_sharded_selection(&index_path, &output, false, |name| {
            name.starts_with("b.")
        })
        .expect("materialize sharded prefix");
        assert_eq!(written, 2);

        let names = tensor_names(&output).expect("subset names");
        assert_eq!(names, vec!["b.weight".to_string(), "b.bias".to_string()]);
        let (metadata, entries) = collect_tensor_layout(&output).expect("subset layout");
        assert_eq!(metadata.get("family").map(String::as_str), Some("ltx2"));
        assert_eq!(metadata.get("total_size").map(String::as_str), Some("20"));
        assert_eq!(entries.len(), 2);

        fs::remove_dir_all(dir).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn source_manifest_roundtrip_resolves_relative_paths() {
        let dir = temp_path("manifest_dir", "dir");
        fs::create_dir_all(&dir).expect("create manifest dir");
        let manifest_path = Path::new(&dir).join("source_manifest.json");

        let manifest = build_source_manifest_value(
            "ltx2_19b",
            "2.3",
            "materialized_subset",
            Some("transformer_only.safetensors"),
            Some("hf://Lightricks/LTX-2.3-distilled"),
            Some("sha256:abc123"),
            Some("bfloat16"),
            Some("distilled"),
            Some(vec!["transformer.".to_string()]),
            None,
            Some("0.4.0"),
            Some("transformer_blocks_v1"),
            Some(serde_json::json!({"note": "test"})),
        )
        .expect("build manifest");
        write_manifest_value(manifest_path.to_string_lossy().as_ref(), manifest).expect("write");

        let raw =
            read_manifest_value(manifest_path.to_string_lossy().as_ref(), false).expect("read raw");
        let resolved = read_manifest_value(manifest_path.to_string_lossy().as_ref(), true)
            .expect("read resolved");

        let raw_path = raw
            .get("source")
            .and_then(|v| v.get("path"))
            .and_then(|v| v.as_str())
            .expect("raw source path");
        assert_eq!(raw_path, "transformer_only.safetensors");

        let resolved_path = resolved
            .get("source")
            .and_then(|v| v.get("path"))
            .and_then(|v| v.as_str())
            .expect("resolved source path");
        assert_eq!(
            resolved_path,
            Path::new(&dir)
                .join("transformer_only.safetensors")
                .to_string_lossy()
                .as_ref()
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn quantized_manifest_compatibility_checks_signature_and_layout() {
        let manifest = build_quantized_source_manifest_value(
            "ltx2_19b",
            "2.3",
            Some("eriquant_cache"),
            "hf://Lightricks/LTX-2.3-distilled",
            "sha256:source123",
            "blocks.json",
            vec!["block_000.bin".to_string(), "block_001.bin".to_string()],
            "eriquant",
            Some("bfloat16"),
            Some("distilled"),
            Some(vec!["transformer.".to_string()]),
            Some(4128),
            Some(64),
            Some("0.4.0"),
            Some("transformer_blocks_v1"),
            Some(serde_json::json!({"frozen_base": true})),
        )
        .expect("build quant manifest");

        let ok = evaluate_manifest_compatibility(
            &manifest,
            "ltx2_19b",
            "2.3",
            Some("sha256:source123"),
            Some("eriquant"),
            Some("transformer_blocks_v1"),
        );
        assert_eq!(ok.get("ok").and_then(|v| v.as_bool()), Some(true));

        let bad = evaluate_manifest_compatibility(
            &manifest,
            "ltx2_19b",
            "2.3",
            Some("sha256:wrong"),
            Some("squareq"),
            Some("other_layout"),
        );
        assert_eq!(bad.get("ok").and_then(|v| v.as_bool()), Some(false));
        let reasons = bad
            .get("reasons")
            .and_then(|v| v.as_array())
            .expect("reasons");
        assert!(reasons.iter().any(|v| v
            .as_str()
            .unwrap_or("")
            .contains("source signature mismatch")));
        assert!(reasons
            .iter()
            .any(|v| v.as_str().unwrap_or("").contains("quant mode mismatch")));
        assert!(reasons.iter().any(|v| v
            .as_str()
            .unwrap_or("")
            .contains("stagehand layout mismatch")));
    }

    #[test]
    fn quantized_block_map_roundtrip_resolves_relative_paths() {
        let dir = temp_path("block_map_dir", "dir");
        fs::create_dir_all(&dir).expect("create block_map dir");
        let block_map_path = Path::new(&dir).join("blocks.json");

        let block_map = serde_json::json!({
            "metadata": {"quant_mode": "eriquant"},
            "blocks": [
                {"id": "0", "file": "block_000.bin", "offset": 0, "nbytes": 128, "tensors": ["a.weight"]},
                {"id": "1", "file": "block_001.bin", "offset": 0, "nbytes": 256, "tensors": ["b.weight"]}
            ]
        });
        write_quantized_block_map_value(block_map_path.to_string_lossy().as_ref(), block_map)
            .expect("write block map");

        let raw = read_quantized_block_map_value(block_map_path.to_string_lossy().as_ref(), false)
            .expect("read block map");
        let resolved =
            read_quantized_block_map_value(block_map_path.to_string_lossy().as_ref(), true)
                .expect("read resolved block map");
        let raw_file = raw
            .get("blocks")
            .and_then(|v| v.as_array())
            .and_then(|blocks| blocks.first())
            .and_then(|block| block.get("file"))
            .and_then(|v| v.as_str())
            .expect("raw file");
        assert_eq!(raw_file, "block_000.bin");

        let resolved_file = resolved
            .get("blocks")
            .and_then(|v| v.as_array())
            .and_then(|blocks| blocks.first())
            .and_then(|block| block.get("file"))
            .and_then(|v| v.as_str())
            .expect("resolved file");
        assert_eq!(
            resolved_file,
            Path::new(&dir)
                .join("block_000.bin")
                .to_string_lossy()
                .as_ref()
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn verify_quantized_manifest_artifacts_checks_block_map_and_files() {
        let dir = temp_path("quant_verify", "dir");
        fs::create_dir_all(&dir).expect("create verify dir");
        let block_map_path = Path::new(&dir).join("blocks.json");
        let data_file = Path::new(&dir).join("block_000.bin");
        fs::write(&data_file, [0u8; 16]).expect("write data file");

        write_quantized_block_map_value(
            block_map_path.to_string_lossy().as_ref(),
            serde_json::json!({
                "blocks": [
                    {"file": "block_000.bin", "offset": 0, "nbytes": 16, "tensors": ["a.weight"]}
                ]
            }),
        )
        .expect("write block map");

        let manifest_path = Path::new(&dir).join("quantized.source.json");
        let manifest = build_quantized_source_manifest_value(
            "ltx2_19b",
            "2.3",
            Some("eriquant_cache"),
            "hf://Lightricks/LTX-2.3-distilled",
            "sha256:source123",
            "blocks.json",
            vec!["block_000.bin".to_string()],
            "eriquant",
            Some("bfloat16"),
            None,
            Some(vec!["transformer.".to_string()]),
            Some(1),
            Some(64),
            None,
            Some("transformer_blocks_v1"),
            None,
        )
        .expect("build manifest");
        write_manifest_value(manifest_path.to_string_lossy().as_ref(), manifest)
            .expect("write manifest");

        let ok =
            verify_quantized_manifest_artifacts_value(manifest_path.to_string_lossy().as_ref())
                .expect("verify ok");
        assert_eq!(ok.get("ok").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(ok.get("block_count").and_then(|v| v.as_u64()), Some(1));

        write_quantized_block_map_value(
            block_map_path.to_string_lossy().as_ref(),
            serde_json::json!({
                "blocks": [
                    {"file": "block_001.bin", "offset": 0, "nbytes": 16, "tensors": ["a.weight"]}
                ]
            }),
        )
        .expect("rewrite block map");
        let bad =
            verify_quantized_manifest_artifacts_value(manifest_path.to_string_lossy().as_ref())
                .expect("verify bad");
        assert_eq!(bad.get("ok").and_then(|v| v.as_bool()), Some(false));
        let reasons = bad
            .get("reasons")
            .and_then(|v| v.as_array())
            .expect("reasons");
        assert!(reasons
            .iter()
            .any(|v| { v.as_str().unwrap_or("").contains("undeclared data file") }));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn quantized_block_container_roundtrip_builds_offsets_and_hashes() {
        let path = temp_path("quant_container", "safetensors");
        let (block_map, payloads) =
            write_quantized_block_container_fixture(&path, "quant_blocks.safetensors");

        let container_entries = collect_quantized_block_container_entries(&path)
            .expect("collect container entries")
            .expect("container entries");
        assert_eq!(container_entries.len(), 2);

        let records = parse_quantized_block_records(&block_map).expect("parse block map");
        assert_eq!(records.len(), 2);
        for record in &records {
            let container_entry = container_entries
                .iter()
                .find(|entry| entry.id == record.id)
                .expect("matching container entry");
            let payload = payloads.get(&record.id).expect("payload");
            assert_eq!(record.file, "quant_blocks.safetensors");
            assert_eq!(record.offset, container_entry.absolute_offsets.0);
            assert_eq!(record.nbytes, payload.len());
            assert_eq!(
                record.payload_sha256.as_deref(),
                Some(sha256_hex(payload).as_str())
            );
            assert_eq!(
                record.tensor_name.as_deref(),
                Some(build_quantized_block_tensor_name(&record.id).as_str())
            );
        }

        fs::remove_file(path).ok();
    }

    #[test]
    fn load_quantized_block_payloads_resolves_manifest_reference() {
        let dir = temp_path("quant_payloads", "dir");
        fs::create_dir_all(&dir).expect("create payload dir");
        let container_path = Path::new(&dir).join("blocks.safetensors");
        let block_map_path = Path::new(&dir).join("blocks.json");
        let manifest_path = Path::new(&dir).join("quantized.source.json");

        let (block_map, payloads) = write_quantized_block_container_fixture(
            container_path.to_string_lossy().as_ref(),
            "blocks.safetensors",
        );
        write_quantized_block_map_value(block_map_path.to_string_lossy().as_ref(), block_map)
            .expect("write block map");
        let manifest = build_quantized_source_manifest_value(
            "ltx2_19b",
            "2.3",
            Some("eriquant_cache"),
            "hf://Lightricks/LTX-2.3-distilled",
            "sha256:source123",
            "blocks.json",
            vec!["blocks.safetensors".to_string()],
            "eriquant",
            Some("bfloat16"),
            None,
            Some(vec!["transformer.".to_string()]),
            Some(2),
            Some(64),
            None,
            Some("transformer_blocks_v1"),
            None,
        )
        .expect("build manifest");
        write_manifest_value(manifest_path.to_string_lossy().as_ref(), manifest)
            .expect("write manifest");

        let requested = HashSet::from(["transformer.layers.1".to_string()]);
        let loaded = load_quantized_block_payloads_value(
            manifest_path.to_string_lossy().as_ref(),
            Some(&requested),
        )
        .expect("load payloads");
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded.get("transformer.layers.1"),
            payloads.get("transformer.layers.1")
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn verify_quantized_manifest_artifacts_checks_container_offsets_and_hashes() {
        let dir = temp_path("quant_verify_container", "dir");
        fs::create_dir_all(&dir).expect("create verify dir");
        let container_path = Path::new(&dir).join("blocks.safetensors");
        let block_map_path = Path::new(&dir).join("blocks.json");
        let manifest_path = Path::new(&dir).join("quantized.source.json");

        let (block_map, _payloads) = write_quantized_block_container_fixture(
            container_path.to_string_lossy().as_ref(),
            "blocks.safetensors",
        );
        write_quantized_block_map_value(
            block_map_path.to_string_lossy().as_ref(),
            block_map.clone(),
        )
        .expect("write block map");

        let manifest = build_quantized_source_manifest_value(
            "ltx2_19b",
            "2.3",
            Some("eriquant_cache"),
            "hf://Lightricks/LTX-2.3-distilled",
            "sha256:source123",
            "blocks.json",
            vec!["blocks.safetensors".to_string()],
            "eriquant",
            Some("bfloat16"),
            None,
            Some(vec!["transformer.".to_string()]),
            Some(2),
            Some(64),
            None,
            Some("transformer_blocks_v1"),
            None,
        )
        .expect("build manifest");
        write_manifest_value(manifest_path.to_string_lossy().as_ref(), manifest)
            .expect("write manifest");

        let ok =
            verify_quantized_manifest_artifacts_value(manifest_path.to_string_lossy().as_ref())
                .expect("verify container ok");
        assert_eq!(ok.get("ok").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(ok.get("block_count").and_then(|v| v.as_u64()), Some(2));

        let mut tampered = block_map;
        let blocks = tampered
            .get_mut("blocks")
            .and_then(|value| value.as_array_mut())
            .expect("blocks");
        blocks[0]["offset"] = serde_json::Value::Number(0u64.into());
        blocks[1]["payload_sha256"] = serde_json::Value::String("deadbeef".to_string());
        write_quantized_block_map_value(block_map_path.to_string_lossy().as_ref(), tampered)
            .expect("rewrite tampered block map");

        let bad =
            verify_quantized_manifest_artifacts_value(manifest_path.to_string_lossy().as_ref())
                .expect("verify container bad");
        assert_eq!(bad.get("ok").and_then(|v| v.as_bool()), Some(false));
        let reasons = bad
            .get("reasons")
            .and_then(|v| v.as_array())
            .expect("reasons");
        assert!(reasons
            .iter()
            .any(|v| { v.as_str().unwrap_or("").contains("offset mismatch") }));
        assert!(reasons
            .iter()
            .any(|v| { v.as_str().unwrap_or("").contains("payload hash mismatch") }));

        fs::remove_dir_all(dir).ok();
    }
}
