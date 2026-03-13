//! Diffusers directory layout discovery and probing.
//!
//! Diffusers models are stored as directories with a `model_index.json` root
//! file and component subdirectories (transformer, text_encoder, vae, etc.)
//! each containing weight files and optional config.json.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::format_detect::ModelFormat;
use crate::probe::ModelInfo;

/// How a component's weights are stored on disk.
#[derive(Debug, Clone)]
pub enum WeightSource {
    /// Single safetensors file.
    SingleSafetensors(PathBuf),
    /// Sharded safetensors with an index JSON and shard files.
    ShardedSafetensors {
        index_path: PathBuf,
        shard_paths: Vec<PathBuf>,
    },
    /// Single PyTorch .bin file.
    SinglePytorch(PathBuf),
    /// Sharded PyTorch .bin files with index JSON.
    ShardedPytorch {
        index_path: PathBuf,
        shard_paths: Vec<PathBuf>,
    },
    /// No weights (scheduler, tokenizer, feature_extractor, etc.).
    None,
}

/// A single component within a diffusers model directory.
#[derive(Debug, Clone)]
pub struct DiffusersComponent {
    /// Component name (e.g. "transformer", "text_encoder", "vae").
    pub name: String,
    /// Python class name from model_index.json (e.g. "LTXVideoTransformer3DModel").
    pub class_name: Option<String>,
    /// Library name from model_index.json (e.g. "diffusers", "transformers").
    pub library_name: Option<String>,
    /// How the weights are stored on disk.
    pub weight_source: WeightSource,
    /// Parsed config.json if present.
    pub config: Option<serde_json::Value>,
}

/// Parsed diffusers directory layout.
#[derive(Debug, Clone)]
pub struct DiffusersLayout {
    /// Root directory path.
    pub root: PathBuf,
    /// Parsed model_index.json.
    pub model_index: serde_json::Value,
    /// Discovered components.
    pub components: Vec<DiffusersComponent>,
}

impl DiffusersLayout {
    /// Open and parse a diffusers model directory.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let index_path = dir.join("model_index.json");
        if !index_path.is_file() {
            return Err(format!(
                "not a diffusers directory: {} (missing model_index.json)",
                dir.display()
            ));
        }

        let index_raw = fs::read_to_string(&index_path)
            .map_err(|e| format!("cannot read {}: {e}", index_path.display()))?;
        let model_index: serde_json::Value = serde_json::from_str(&index_raw)
            .map_err(|e| format!("invalid JSON in {}: {e}", index_path.display()))?;

        let obj = model_index
            .as_object()
            .ok_or_else(|| "model_index.json is not a JSON object".to_string())?;

        let mut components = Vec::new();

        for (key, value) in obj {
            // Skip metadata keys (start with underscore).
            if key.starts_with('_') {
                continue;
            }

            let (library_name, class_name) = parse_component_entry(value);

            let comp_dir = dir.join(key);
            let (weight_source, config) = if comp_dir.is_dir() {
                let ws = discover_weight_source(&comp_dir);
                let cfg = read_config_json(&comp_dir);
                (ws, cfg)
            } else {
                (WeightSource::None, None)
            };

            components.push(DiffusersComponent {
                name: key.clone(),
                class_name,
                library_name,
                weight_source,
                config,
            });
        }

        // Sort components by name for deterministic ordering.
        components.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(DiffusersLayout {
            root: dir.to_path_buf(),
            model_index,
            components,
        })
    }

    /// Get tensor names for a specific component, prefixed with "{component}/".
    pub fn component_tensor_names(&self, component: &str) -> Result<Vec<String>, String> {
        let comp = self
            .components
            .iter()
            .find(|c| c.name == component)
            .ok_or_else(|| format!("component not found: {component}"))?;

        let raw_names = weight_source_tensor_names(&comp.weight_source)?;
        let prefix = format!("{}/", component);
        Ok(raw_names.into_iter().map(|n| format!("{prefix}{n}")).collect())
    }

    /// Get all tensor names across all components, prefixed with component names.
    pub fn all_tensor_names(&self) -> Result<Vec<String>, String> {
        let mut names = Vec::new();
        for comp in &self.components {
            if let Ok(comp_names) = self.component_tensor_names(&comp.name) {
                names.extend(comp_names);
            }
        }
        names.sort();
        Ok(names)
    }

    /// Build a ModelInfo by aggregating across all components.
    pub fn to_model_info(&self) -> Result<ModelInfo, String> {
        let mut all_names = Vec::new();
        let mut all_shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let mut all_dtypes: HashMap<String, String> = HashMap::new();
        let mut param_count: u64 = 0;
        let mut total_bytes: u64 = 0;

        for comp in &self.components {
            let prefix = format!("{}/", comp.name);
            let entries = weight_source_tensor_entries(&comp.weight_source)?;
            for entry in entries {
                let prefixed = format!("{prefix}{}", entry.name);
                let params: u64 = if entry.shape.is_empty() {
                    0
                } else {
                    entry.shape.iter().map(|&d| d as u64).product()
                };
                param_count += params;
                all_shapes.insert(prefixed.clone(), entry.shape);
                all_dtypes.insert(prefixed.clone(), entry.dtype);
                all_names.push(prefixed);
            }
            total_bytes += weight_source_file_bytes(&comp.weight_source);
        }

        all_names.sort();

        // Build metadata from model_index.
        let mut metadata = HashMap::new();
        if let Some(cls) = self
            .model_index
            .get("_class_name")
            .and_then(|v| v.as_str())
        {
            metadata.insert("pipeline_class".to_string(), cls.to_string());
        }

        // List components with weights.
        let comp_list: Vec<&str> = self
            .components
            .iter()
            .filter(|c| !matches!(c.weight_source, WeightSource::None))
            .map(|c| c.name.as_str())
            .collect();
        metadata.insert("components".to_string(), comp_list.join(", "));

        // Include all component class names.
        for comp in &self.components {
            if let Some(ref cls) = comp.class_name {
                metadata.insert(format!("{}.class_name", comp.name), cls.clone());
            }
            if let Some(ref lib) = comp.library_name {
                metadata.insert(format!("{}.library_name", comp.name), lib.clone());
            }
        }

        Ok(ModelInfo {
            format: ModelFormat::Diffusers,
            path: self.root.clone(),
            tensor_count: all_names.len(),
            tensor_names: all_names,
            tensor_shapes: all_shapes,
            tensor_dtypes: all_dtypes,
            metadata,
            quant_types: None,
            total_file_bytes: total_bytes,
            param_count,
        })
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Parse a model_index.json component entry.
/// Values are typically `["library", "ClassName"]` or null.
fn parse_component_entry(value: &serde_json::Value) -> (Option<String>, Option<String>) {
    if let Some(arr) = value.as_array() {
        let lib = arr.first().and_then(|v| v.as_str()).map(String::from);
        let cls = arr.get(1).and_then(|v| v.as_str()).map(String::from);
        (lib, cls)
    } else {
        (None, None)
    }
}

/// Read config.json from a component directory if present.
fn read_config_json(dir: &Path) -> Option<serde_json::Value> {
    let cfg_path = dir.join("config.json");
    if !cfg_path.is_file() {
        return None;
    }
    let raw = fs::read_to_string(&cfg_path).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Discover weight files in a component subdirectory.
fn discover_weight_source(comp_dir: &Path) -> WeightSource {
    // Sharded safetensors (check index files first — they imply sharding).
    for index_name in &[
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
    ] {
        let idx = comp_dir.join(index_name);
        if idx.is_file() {
            if let Ok(shards) = parse_shard_index_files(&idx) {
                return WeightSource::ShardedSafetensors {
                    index_path: idx,
                    shard_paths: shards,
                };
            }
        }
    }

    // Single safetensors.
    for name in &["model.safetensors", "diffusion_pytorch_model.safetensors"] {
        let f = comp_dir.join(name);
        if f.is_file() {
            return WeightSource::SingleSafetensors(f);
        }
    }

    // Sharded pytorch.
    for index_name in &[
        "pytorch_model.bin.index.json",
        "diffusion_pytorch_model.bin.index.json",
    ] {
        let idx = comp_dir.join(index_name);
        if idx.is_file() {
            if let Ok(shards) = parse_shard_index_files(&idx) {
                return WeightSource::ShardedPytorch {
                    index_path: idx,
                    shard_paths: shards,
                };
            }
        }
    }

    // Single pytorch.
    for name in &[
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin",
        "model.bin",
    ] {
        let f = comp_dir.join(name);
        if f.is_file() {
            return WeightSource::SinglePytorch(f);
        }
    }

    WeightSource::None
}

/// Parse a shard index JSON and return resolved shard file paths.
fn parse_shard_index_files(index_path: &Path) -> Result<Vec<PathBuf>, String> {
    let raw = fs::read_to_string(index_path)
        .map_err(|e| format!("cannot read {}: {e}", index_path.display()))?;
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("invalid JSON: {e}"))?;

    let weight_map = parsed
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "missing weight_map in shard index".to_string())?;

    let parent = index_path
        .parent()
        .unwrap_or_else(|| Path::new("."));

    let mut shard_names: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    shard_names.sort();
    shard_names.dedup();

    let shard_paths: Vec<PathBuf> = shard_names
        .into_iter()
        .map(|name| parent.join(name))
        .collect();

    Ok(shard_paths)
}

/// Minimal tensor entry for aggregation.
struct TensorEntry {
    name: String,
    dtype: String,
    shape: Vec<usize>,
}

/// Get tensor names from a weight source (safetensors only; pytorch returns empty).
fn weight_source_tensor_names(source: &WeightSource) -> Result<Vec<String>, String> {
    Ok(weight_source_tensor_entries(source)?
        .into_iter()
        .map(|e| e.name)
        .collect())
}

/// Get tensor entries from a weight source.
fn weight_source_tensor_entries(source: &WeightSource) -> Result<Vec<TensorEntry>, String> {
    match source {
        WeightSource::SingleSafetensors(path) => read_safetensors_header_entries(path),
        WeightSource::ShardedSafetensors {
            index_path,
            shard_paths,
        } => {
            // Use the index JSON weight_map for tensor names, then read dtypes/shapes
            // from shard headers. For efficiency, we parse shard headers.
            read_sharded_safetensors_entries(index_path, shard_paths)
        }
        WeightSource::SinglePytorch(_)
        | WeightSource::ShardedPytorch { .. }
        | WeightSource::None => Ok(Vec::new()),
    }
}

/// Total file bytes for a weight source.
fn weight_source_file_bytes(source: &WeightSource) -> u64 {
    match source {
        WeightSource::SingleSafetensors(p) | WeightSource::SinglePytorch(p) => {
            fs::metadata(p).map(|m| m.len()).unwrap_or(0)
        }
        WeightSource::ShardedSafetensors { shard_paths, .. }
        | WeightSource::ShardedPytorch { shard_paths, .. } => shard_paths
            .iter()
            .map(|p| fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .sum(),
        WeightSource::None => 0,
    }
}

/// Read tensor entries from a single safetensors file header.
fn read_safetensors_header_entries(path: &Path) -> Result<Vec<TensorEntry>, String> {
    let mut file =
        fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;

    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|e| format!("cannot read header length: {e}"))?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    if header_len == 0 || header_len > 100_000_000 {
        return Err(format!("unreasonable header length: {header_len}"));
    }

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)
        .map_err(|e| format!("cannot read header: {e}"))?;

    let header: serde_json::Value =
        serde_json::from_slice(&header_buf).map_err(|e| format!("invalid JSON header: {e}"))?;

    let obj = header
        .as_object()
        .ok_or_else(|| "header is not a JSON object".to_string())?;

    let mut entries = Vec::new();
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
        entries.push(TensorEntry { name: name.clone(), dtype, shape });
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

/// Read tensor entries from sharded safetensors by parsing each shard header.
fn read_sharded_safetensors_entries(
    _index_path: &Path,
    shard_paths: &[PathBuf],
) -> Result<Vec<TensorEntry>, String> {
    let mut all_entries = Vec::new();
    for shard in shard_paths {
        if shard.is_file() {
            let entries = read_safetensors_header_entries(shard)?;
            all_entries.extend(entries);
        }
    }
    all_entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(all_entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype as StDtype, View};
    use std::borrow::Cow;

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

    fn make_temp_dir(name: &str) -> PathBuf {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("diffusers_test_{name}_{ts}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_tiny_safetensors(path: &Path, tensors: &[(&str, Vec<usize>)]) {
        let views: Vec<(String, ByteView)> = tensors
            .iter()
            .map(|(name, shape)| {
                let numel: usize = shape.iter().product();
                (
                    name.to_string(),
                    ByteView {
                        dtype: StDtype::BF16,
                        shape: shape.clone(),
                        data: vec![0u8; numel * 2], // BF16 = 2 bytes
                    },
                )
            })
            .collect();
        let data =
            safetensors::serialize(views.iter().map(|(n, v)| (n.as_str(), v.clone())).collect::<Vec<_>>(), &None)
                .unwrap();
        fs::write(path, data).unwrap();
    }

    fn write_shard_index(index_path: &Path, weight_map: &[(&str, &str)]) {
        let mut wm = serde_json::Map::new();
        for (tensor, shard) in weight_map {
            wm.insert(tensor.to_string(), serde_json::Value::String(shard.to_string()));
        }
        let index = serde_json::json!({
            "metadata": { "total_size": 1000 },
            "weight_map": wm,
        });
        fs::write(index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();
    }

    fn build_basic_diffusers_dir() -> PathBuf {
        let dir = make_temp_dir("basic");

        // model_index.json
        let index = serde_json::json!({
            "_class_name": "LTXPipeline",
            "_diffusers_version": "0.32.0",
            "transformer": ["diffusers", "LTXVideoTransformer3DModel"],
            "text_encoder": ["transformers", "GemmaForCausalLM"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "vae": ["diffusers", "AutoencoderKLLTXVideo"],
        });
        fs::write(dir.join("model_index.json"), serde_json::to_string_pretty(&index).unwrap()).unwrap();

        // transformer/ with single safetensors
        let tf_dir = dir.join("transformer");
        fs::create_dir_all(&tf_dir).unwrap();
        write_tiny_safetensors(
            &tf_dir.join("diffusion_pytorch_model.safetensors"),
            &[("blocks.0.weight", vec![4, 4]), ("blocks.0.bias", vec![4])],
        );
        fs::write(
            tf_dir.join("config.json"),
            r#"{"num_layers": 28}"#,
        ).unwrap();

        // text_encoder/ with model.safetensors
        let te_dir = dir.join("text_encoder");
        fs::create_dir_all(&te_dir).unwrap();
        write_tiny_safetensors(
            &te_dir.join("model.safetensors"),
            &[("embed_tokens.weight", vec![256, 32])],
        );

        // vae/ with diffusion_pytorch_model.safetensors
        let vae_dir = dir.join("vae");
        fs::create_dir_all(&vae_dir).unwrap();
        write_tiny_safetensors(
            &vae_dir.join("diffusion_pytorch_model.safetensors"),
            &[("encoder.weight", vec![3, 3])],
        );

        // scheduler/ with only config
        let sched_dir = dir.join("scheduler");
        fs::create_dir_all(&sched_dir).unwrap();
        fs::write(sched_dir.join("scheduler_config.json"), r#"{"type": "euler"}"#).unwrap();

        dir
    }

    #[test]
    fn open_basic_diffusers_layout() {
        let dir = build_basic_diffusers_dir();
        let layout = DiffusersLayout::open(&dir).unwrap();

        assert_eq!(layout.components.len(), 4);

        // Components sorted: scheduler, text_encoder, transformer, vae
        let names: Vec<&str> = layout.components.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["scheduler", "text_encoder", "transformer", "vae"]);

        // Transformer has weights
        let tf = layout.components.iter().find(|c| c.name == "transformer").unwrap();
        assert!(matches!(tf.weight_source, WeightSource::SingleSafetensors(_)));
        assert_eq!(tf.class_name.as_deref(), Some("LTXVideoTransformer3DModel"));
        assert_eq!(tf.library_name.as_deref(), Some("diffusers"));
        assert!(tf.config.is_some());

        // Scheduler has no weights
        let sched = layout.components.iter().find(|c| c.name == "scheduler").unwrap();
        assert!(matches!(sched.weight_source, WeightSource::None));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn component_tensor_names_prefixed() {
        let dir = build_basic_diffusers_dir();
        let layout = DiffusersLayout::open(&dir).unwrap();

        let names = layout.component_tensor_names("transformer").unwrap();
        assert!(names.contains(&"transformer/blocks.0.weight".to_string()));
        assert!(names.contains(&"transformer/blocks.0.bias".to_string()));
        assert_eq!(names.len(), 2);

        let te_names = layout.component_tensor_names("text_encoder").unwrap();
        assert_eq!(te_names, vec!["text_encoder/embed_tokens.weight"]);

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn all_tensor_names_aggregated() {
        let dir = build_basic_diffusers_dir();
        let layout = DiffusersLayout::open(&dir).unwrap();

        let names = layout.all_tensor_names().unwrap();
        // transformer (2) + text_encoder (1) + vae (1) = 4
        assert_eq!(names.len(), 4);
        assert!(names.contains(&"transformer/blocks.0.weight".to_string()));
        assert!(names.contains(&"text_encoder/embed_tokens.weight".to_string()));
        assert!(names.contains(&"vae/encoder.weight".to_string()));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn to_model_info_aggregates() {
        let dir = build_basic_diffusers_dir();
        let layout = DiffusersLayout::open(&dir).unwrap();

        let info = layout.to_model_info().unwrap();
        assert_eq!(info.format, ModelFormat::Diffusers);
        assert_eq!(info.tensor_count, 4);
        // param_count: blocks.0.weight=16, blocks.0.bias=4, embed_tokens.weight=8192, encoder.weight=9
        assert_eq!(info.param_count, 16 + 4 + 8192 + 9);
        assert!(info.total_file_bytes > 0);
        assert_eq!(
            info.metadata.get("pipeline_class").map(String::as_str),
            Some("LTXPipeline")
        );
        assert!(info.metadata.get("components").unwrap().contains("transformer"));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn sharded_component_discovery() {
        let dir = make_temp_dir("sharded");

        let index = serde_json::json!({
            "_class_name": "TestPipeline",
            "transformer": ["diffusers", "TestModel"],
        });
        fs::write(dir.join("model_index.json"), serde_json::to_string_pretty(&index).unwrap()).unwrap();

        let tf_dir = dir.join("transformer");
        fs::create_dir_all(&tf_dir).unwrap();

        // Create two shard files
        write_tiny_safetensors(
            &tf_dir.join("model-00001-of-00002.safetensors"),
            &[("layer.0.weight", vec![4, 4])],
        );
        write_tiny_safetensors(
            &tf_dir.join("model-00002-of-00002.safetensors"),
            &[("layer.1.weight", vec![4, 4])],
        );

        // Create shard index
        write_shard_index(
            &tf_dir.join("model.safetensors.index.json"),
            &[
                ("layer.0.weight", "model-00001-of-00002.safetensors"),
                ("layer.1.weight", "model-00002-of-00002.safetensors"),
            ],
        );

        let layout = DiffusersLayout::open(&dir).unwrap();
        let tf = layout.components.iter().find(|c| c.name == "transformer").unwrap();
        match &tf.weight_source {
            WeightSource::ShardedSafetensors { shard_paths, .. } => {
                assert_eq!(shard_paths.len(), 2);
            }
            other => panic!("expected ShardedSafetensors, got {:?}", other),
        }

        // Tensor names should come from the shard files
        let names = layout.component_tensor_names("transformer").unwrap();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"transformer/layer.0.weight".to_string()));
        assert!(names.contains(&"transformer/layer.1.weight".to_string()));

        let info = layout.to_model_info().unwrap();
        assert_eq!(info.tensor_count, 2);
        assert_eq!(info.param_count, 32); // 2 * 16

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn missing_model_index_errors() {
        let dir = make_temp_dir("no_index");
        let result = DiffusersLayout::open(&dir);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("missing model_index.json"));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn component_with_no_weights_returns_none() {
        let dir = make_temp_dir("no_weights");
        let index = serde_json::json!({
            "_class_name": "TestPipeline",
            "scheduler": ["diffusers", "Scheduler"],
        });
        fs::write(dir.join("model_index.json"), serde_json::to_string_pretty(&index).unwrap()).unwrap();
        let sched_dir = dir.join("scheduler");
        fs::create_dir_all(&sched_dir).unwrap();
        fs::write(sched_dir.join("scheduler_config.json"), "{}").unwrap();

        let layout = DiffusersLayout::open(&dir).unwrap();
        let sched = layout.components.iter().find(|c| c.name == "scheduler").unwrap();
        assert!(matches!(sched.weight_source, WeightSource::None));

        // No tensor names for scheduler
        let names = layout.component_tensor_names("scheduler").unwrap();
        assert!(names.is_empty());

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn nonexistent_component_errors() {
        let dir = build_basic_diffusers_dir();
        let layout = DiffusersLayout::open(&dir).unwrap();
        let result = layout.component_tensor_names("nonexistent");
        assert!(result.is_err());
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn component_without_subdirectory_gets_no_weights() {
        let dir = make_temp_dir("no_subdir");
        let index = serde_json::json!({
            "_class_name": "TestPipeline",
            "ghost_module": ["diffusers", "GhostModel"],
        });
        fs::write(dir.join("model_index.json"), serde_json::to_string_pretty(&index).unwrap()).unwrap();
        // Don't create ghost_module/ directory

        let layout = DiffusersLayout::open(&dir).unwrap();
        let ghost = layout.components.iter().find(|c| c.name == "ghost_module").unwrap();
        assert!(matches!(ghost.weight_source, WeightSource::None));

        fs::remove_dir_all(dir).ok();
    }
}
