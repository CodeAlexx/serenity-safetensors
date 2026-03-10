use memmap2::MmapOptions;
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyTuple};
use safetensors::tensor::{Dtype as StDtype, View};
use safetensors::SafeTensors;
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

const SOURCE_MANIFEST_FORMAT: &str = "serenity_source_manifest";
const SOURCE_MANIFEST_SCHEMA_VERSION: u64 = 1;
const QUANTIZED_BLOCK_MAP_FORMAT: &str = "serenity_quantized_block_map";
const QUANTIZED_BLOCK_MAP_SCHEMA_VERSION: u64 = 1;

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
    let dumped = serde_json::to_string(value)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(json.call_method1("loads", (dumped,))?.into())
}

fn ensure_object<'a>(
    value: &'a serde_json::Value,
    context: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, SsError> {
    value.as_object()
        .ok_or_else(|| SsError::Other(format!("{context} must be a JSON object")))
}

fn ensure_object_mut<'a>(
    value: &'a mut serde_json::Value,
    context: &str,
) -> Result<&'a mut serde_json::Map<String, serde_json::Value>, SsError> {
    value.as_object_mut()
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
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| SsError::Other(format!("{context}.{key} must be an array of strings")))
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
            let prefixes = ensure_string_array(tensor_policy_obj, "prefixes", "source.tensor_policy")?;
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
            let format = value.as_str().ok_or_else(|| {
                SsError::Other("manifest.format must be a string".into())
            })?;
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
            for key in ["path", "index", "block_map", "weights", "data_files", "files"] {
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
        } else if quant_obj.get("frozen").and_then(|value| value.as_bool()).is_none() {
            return Err(SsError::Other(
                "manifest.quantization.frozen must be a boolean".into(),
            ));
        }
    }

    if let Some(compatibility) = root.get_mut("compatibility") {
        let compatibility_obj =
            ensure_object_mut(compatibility, "manifest.compatibility")?;
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
            source_obj.insert("path".to_string(), serde_json::Value::String(path.to_string()));
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
        let compatibility_obj =
            ensure_object_mut(compatibility, "manifest.compatibility")?;
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
            let format = value.as_str().ok_or_else(|| {
                SsError::Other("block_map.format must be a string".into())
            })?;
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
    let blocks_array = blocks.as_array_mut().ok_or_else(|| {
        SsError::Other("block_map.blocks must be an array".into())
    })?;
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

fn write_quantized_block_map_value(path: &str, block_map: serde_json::Value) -> Result<(), SsError> {
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

    if reasons.is_empty() {
        let block_map = read_quantized_block_map_value(&block_map_path, true)?;
        let block_map_obj = ensure_object(&block_map, "block_map")?;
        let blocks = block_map_obj
            .get("blocks")
            .and_then(|value| value.as_array())
            .ok_or_else(|| SsError::Other("block_map.blocks must be an array".into()))?;
        block_count = Some(blocks.len() as u64);

        let mut referenced_files = Vec::new();
        for block in blocks {
            let block_obj = ensure_object(block, "block_map.blocks[]")?;
            let file = expect_string(block_obj, "file", "block_map.blocks[]")?;
            referenced_files.push(file);
        }
        referenced_files.sort();
        referenced_files.dedup();

        for file in &referenced_files {
            if !declared_files.contains(file) {
                reasons.push(format!(
                    "block_map references undeclared data file: {file}"
                ));
            }
        }

        if let Some(quantization) = root.get("quantization") {
            let quant_obj = ensure_object(quantization, "manifest.quantization")?;
            if let Some(expected_count) = quant_obj.get("block_count").and_then(|value| value.as_u64()) {
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
    let metadata = if metadata.is_empty() { None } else { Some(metadata) };
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
    let metadata = if metadata.is_empty() { None } else { Some(metadata) };
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
    materialize_sharded_selection(index_path, output_path, direct, |name| requested.contains(name))
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
    materialize_sharded_selection(index_path, output_path, direct, |name| name.starts_with(prefix))
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
    let (summary, shard_paths, shard_metadata, entries) = collect_sharded_tensor_layout(index_path)?;
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
fn write_quantized_block_map(py: Python<'_>, path: &str, block_map: &Bound<'_, PyAny>) -> PyResult<()> {
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
            .join(format!("serenity_safetensors_{name}_{pid}_{nanos}.{suffix}"))
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
        fs::write(&index_path, serde_json::to_vec(&payload).expect("json bytes")).expect("write index");

        (
            dir,
            index_path.to_string_lossy().into_owned(),
        )
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
            entries.iter().map(|entry| entry.name.as_str()).collect::<Vec<_>>(),
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

    #[test]
    fn materialize_selective_writes_subset_file() {
        let source = temp_path("materialize_source", "safetensors");
        let output = temp_path("materialize_subset", "safetensors");
        write_test_safetensors(&source);

        let written = materialize_single_file_selection(&source, &output, false, |name| name == "bias")
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

        let written =
            materialize_sharded_selection(&index_path, &output, false, |name| name.starts_with("b."))
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

        let raw = read_manifest_value(manifest_path.to_string_lossy().as_ref(), false).expect("read raw");
        let resolved =
            read_manifest_value(manifest_path.to_string_lossy().as_ref(), true).expect("read resolved");

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
        assert!(reasons.iter().any(|v| v.as_str().unwrap_or("").contains("source signature mismatch")));
        assert!(reasons.iter().any(|v| v.as_str().unwrap_or("").contains("quant mode mismatch")));
        assert!(reasons.iter().any(|v| v.as_str().unwrap_or("").contains("stagehand layout mismatch")));
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
        write_manifest_value(manifest_path.to_string_lossy().as_ref(), manifest).expect("write manifest");

        let ok = verify_quantized_manifest_artifacts_value(manifest_path.to_string_lossy().as_ref())
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
        let bad = verify_quantized_manifest_artifacts_value(manifest_path.to_string_lossy().as_ref())
            .expect("verify bad");
        assert_eq!(bad.get("ok").and_then(|v| v.as_bool()), Some(false));
        let reasons = bad
            .get("reasons")
            .and_then(|v| v.as_array())
            .expect("reasons");
        assert!(reasons.iter().any(|v| {
            v.as_str()
                .unwrap_or("")
                .contains("undeclared data file")
        }));

        fs::remove_dir_all(dir).ok();
    }
}
