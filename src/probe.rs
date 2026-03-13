//! Model probing — extract tensor inventory and metadata without loading data.

use crate::format_detect::{detect_format, ModelFormat};
use crate::pytorch::PickleIndex;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Summary of a model file's contents.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub format: ModelFormat,
    pub path: PathBuf,
    pub tensor_count: usize,
    pub tensor_names: Vec<String>,
    pub tensor_shapes: HashMap<String, Vec<usize>>,
    pub tensor_dtypes: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
    /// Per-tensor quantization type (GGUF only).
    pub quant_types: Option<HashMap<String, String>>,
    pub total_file_bytes: u64,
    pub param_count: u64,
}

/// Probe a model file and return its tensor inventory.
///
/// Currently implemented for safetensors. Other formats return a descriptive
/// error indicating they are not yet supported.
pub fn probe_model(path: &Path) -> Result<ModelInfo, String> {
    let format = detect_format(path)?;
    match format {
        ModelFormat::Safetensors => probe_safetensors(path, format),
        ModelFormat::PyTorchZip => probe_pytorch_zip(path, format),
        ModelFormat::Gguf => probe_gguf(path),
        ModelFormat::Diffusers => {
            let layout = crate::diffusers::DiffusersLayout::open(path)?;
            layout.to_model_info()
        }
        other => Err(format!("not yet implemented: {other}")),
    }
}

/// Parse the safetensors JSON header and extract tensor metadata.
///
/// Reimplements the header read locally so we don't depend on `lib.rs` private
/// helpers — keeps the boundary clean and avoids modifying existing code.
fn probe_safetensors(path: &Path, format: ModelFormat) -> Result<ModelInfo, String> {
    let file_bytes = fs::metadata(path)
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
        .len();

    let mut file =
        fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;

    // Read 8-byte LE header length.
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|e| format!("cannot read header length: {e}"))?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    if header_len == 0 || header_len > 100_000_000 {
        return Err(format!("unreasonable header length: {header_len}"));
    }

    // Read and parse JSON header.
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)
        .map_err(|e| format!("cannot read header: {e}"))?;

    let header: serde_json::Value =
        serde_json::from_slice(&header_buf).map_err(|e| format!("invalid JSON header: {e}"))?;

    let obj = header
        .as_object()
        .ok_or_else(|| "header is not a JSON object".to_string())?;

    // Collect metadata.
    let mut metadata = HashMap::new();
    if let Some(md) = obj.get("__metadata__") {
        if let Some(md_obj) = md.as_object() {
            for (k, v) in md_obj {
                if let Some(s) = v.as_str() {
                    metadata.insert(k.clone(), s.to_string());
                }
            }
        }
    }

    // Collect tensor entries.
    let mut tensor_names = Vec::new();
    let mut tensor_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut tensor_dtypes: HashMap<String, String> = HashMap::new();
    let mut param_count: u64 = 0;

    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }

        tensor_names.push(name.clone());

        let dtype = info
            .get("dtype")
            .and_then(|d| d.as_str())
            .unwrap_or("F32")
            .to_string();
        tensor_dtypes.insert(name.clone(), dtype);

        let shape: Vec<usize> = info
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();

        let params: u64 = if shape.is_empty() {
            0
        } else {
            shape.iter().map(|&d| d as u64).product()
        };
        param_count += params;

        tensor_shapes.insert(name.clone(), shape);
    }

    tensor_names.sort();

    Ok(ModelInfo {
        format,
        path: path.to_path_buf(),
        tensor_count: tensor_names.len(),
        tensor_names,
        tensor_shapes,
        tensor_dtypes,
        metadata,
        quant_types: None,
        total_file_bytes: file_bytes,
        param_count,
    })
}

/// Probe a GGUF file — reads tensor index and metadata from the header.
fn probe_gguf(path: &Path) -> Result<ModelInfo, String> {
    let idx = crate::gguf::GgufIndex::open(path)?;
    Ok(idx.to_model_info(path))
}

/// Probe a PyTorch ZIP checkpoint (.pt / .pth / .bin).
fn probe_pytorch_zip(path: &Path, format: ModelFormat) -> Result<ModelInfo, String> {
    let file_bytes = fs::metadata(path)
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
        .len();

    let index = PickleIndex::open(path)?;

    let mut tensor_names = Vec::new();
    let mut tensor_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut tensor_dtypes: HashMap<String, String> = HashMap::new();
    let mut param_count: u64 = 0;

    for t in &index.tensors {
        tensor_names.push(t.name.clone());
        tensor_shapes.insert(t.name.clone(), t.shape.clone());
        tensor_dtypes.insert(t.name.clone(), t.dtype.clone());
        param_count += t.numel as u64;
    }

    tensor_names.sort();

    Ok(ModelInfo {
        format,
        path: path.to_path_buf(),
        tensor_count: tensor_names.len(),
        tensor_names,
        tensor_shapes,
        tensor_dtypes,
        metadata: HashMap::new(),
        quant_types: None,
        total_file_bytes: file_bytes,
        param_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype as StDtype, View};
    use safetensors::serialize;

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
        fn data(&self) -> std::borrow::Cow<'_, [u8]> {
            std::borrow::Cow::Borrowed(&self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("probe_{name}_{ts}.safetensors"))
    }

    fn write_test_safetensors(path: &Path) {
        let weight = ByteView {
            dtype: StDtype::F32,
            shape: vec![3, 4],
            data: vec![0u8; 3 * 4 * 4], // 12 floats
        };
        let bias = ByteView {
            dtype: StDtype::F32,
            shape: vec![4],
            data: vec![0u8; 4 * 4], // 4 floats
        };

        let mut metadata = HashMap::new();
        metadata.insert("family".to_string(), "test".to_string());

        let tensors: Vec<(&str, ByteView)> = vec![("weight", weight), ("bias", bias)];
        let data = serialize(tensors, &Some(metadata)).unwrap();
        fs::write(path, &data).unwrap();
    }

    #[test]
    fn probe_safetensors_basic() {
        let path = temp_path("basic");
        write_test_safetensors(&path);

        let info = probe_model(&path).unwrap();
        assert_eq!(info.format, ModelFormat::Safetensors);
        assert_eq!(info.tensor_count, 2);
        assert!(info.tensor_names.contains(&"weight".to_string()));
        assert!(info.tensor_names.contains(&"bias".to_string()));
        assert_eq!(info.tensor_shapes["weight"], vec![3, 4]);
        assert_eq!(info.tensor_shapes["bias"], vec![4]);
        assert_eq!(info.tensor_dtypes["weight"], "F32");
        assert_eq!(info.tensor_dtypes["bias"], "F32");
        assert_eq!(info.param_count, 3 * 4 + 4); // 16
        assert!(info.total_file_bytes > 0);
        assert_eq!(info.metadata.get("family").map(String::as_str), Some("test"));
        assert!(info.quant_types.is_none());

        fs::remove_file(path).ok();
    }

    #[test]
    fn probe_gguf_returns_not_implemented() {
        let path = temp_path("gguf").with_extension("gguf");
        let mut f = fs::File::create(&path).unwrap();
        use std::io::Write;
        // GGUF magic
        f.write_all(&[0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00]).unwrap();
        drop(f);

        let err = probe_model(&path).unwrap_err();
        // Tiny GGUF file should fail to parse (too small for valid header)
        assert!(err.contains("too small") || err.contains("not yet implemented"), "got: {err}");
        fs::remove_file(path).ok();
    }

    #[test]
    fn probe_nonexistent_errors() {
        let result = probe_model(Path::new("/tmp/nonexistent_probe_test.safetensors"));
        assert!(result.is_err());
    }
}
