//! Model format detection from file magic bytes or directory structure.

use std::fmt;
use std::fs;
use std::io::Read;
use std::path::Path;

/// Recognized model container formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    /// safetensors — header-length prefixed JSON + raw tensor data.
    Safetensors,
    /// GGUF — llama.cpp quantized format (magic 0x47475546).
    Gguf,
    /// PyTorch ZIP archive (.pt / .pth / .bin).
    PyTorchZip,
    /// Legacy pickle checkpoint (.ckpt, pre-zip era).
    PyTorchLegacy,
    /// Diffusers directory layout with model_index.json.
    Diffusers,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ModelFormat::Safetensors => "safetensors",
            ModelFormat::Gguf => "gguf",
            ModelFormat::PyTorchZip => "pytorch_zip",
            ModelFormat::PyTorchLegacy => "pytorch_legacy",
            ModelFormat::Diffusers => "diffusers",
        };
        f.write_str(s)
    }
}

/// Detect the model format from a file or directory path.
///
/// For directories: checks for `model_index.json` (Diffusers layout).
/// For files: uses magic bytes and heuristics to identify the format.
pub fn detect_format(path: &Path) -> Result<ModelFormat, String> {
    if path.is_dir() {
        return detect_directory_format(path);
    }
    if !path.is_file() {
        return Err(format!("path does not exist: {}", path.display()));
    }
    detect_file_format(path)
}

fn detect_directory_format(path: &Path) -> Result<ModelFormat, String> {
    // Diffusers layout has model_index.json at root
    let index = path.join("model_index.json");
    if index.is_file() {
        return Ok(ModelFormat::Diffusers);
    }
    // Check for loose safetensors files
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "safetensors" {
                    return Ok(ModelFormat::Safetensors);
                }
            }
        }
    }
    Err(format!(
        "cannot determine model format for directory: {}",
        path.display()
    ))
}

fn detect_file_format(path: &Path) -> Result<ModelFormat, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("cannot open {}: {}", path.display(), e))?;

    let mut magic = [0u8; 8];
    let bytes_read = file
        .read(&mut magic)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;

    if bytes_read < 8 {
        return Err(format!(
            "file too small ({bytes_read} bytes): {}",
            path.display()
        ));
    }

    // GGUF magic: bytes "GGUF" at offset 0
    if magic[0..4] == [0x47, 0x47, 0x55, 0x46] {
        return Ok(ModelFormat::Gguf);
    }

    // ZIP magic: PK\x03\x04
    if magic[0..4] == [0x50, 0x4B, 0x03, 0x04] {
        return Ok(ModelFormat::PyTorchZip);
    }

    // Safetensors heuristic: first 8 bytes = u64 LE header length,
    // then next byte should be '{' (start of JSON header).
    let header_len = u64::from_le_bytes(magic);
    if header_len > 0 && header_len < 100_000_000 {
        // Read one more byte to check for '{'
        let mut peek = [0u8; 1];
        if file.read(&mut peek).unwrap_or(0) == 1 && peek[0] == b'{' {
            return Ok(ModelFormat::Safetensors);
        }
    }

    // Fallback: legacy pickle checkpoint
    Ok(ModelFormat::PyTorchLegacy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_path(name: &str, ext: &str) -> std::path::PathBuf {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("fmt_detect_{name}_{ts}.{ext}"))
    }

    #[test]
    fn detect_safetensors_file() {
        // Build a minimal safetensors file: 8-byte header len + JSON header
        let header = b"{}";
        let header_len = header.len() as u64;
        let path = temp_path("st", "safetensors");
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header).unwrap();
        drop(f);

        let fmt = detect_format(&path).unwrap();
        assert_eq!(fmt, ModelFormat::Safetensors);
        assert_eq!(fmt.to_string(), "safetensors");
        fs::remove_file(path).ok();
    }

    #[test]
    fn detect_gguf_file() {
        let path = temp_path("gguf", "gguf");
        let mut f = fs::File::create(&path).unwrap();
        // GGUF magic + 4 padding bytes
        f.write_all(&[0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00]).unwrap();
        drop(f);

        let fmt = detect_format(&path).unwrap();
        assert_eq!(fmt, ModelFormat::Gguf);
        assert_eq!(fmt.to_string(), "gguf");
        fs::remove_file(path).ok();
    }

    #[test]
    fn detect_zip_file() {
        let path = temp_path("zip", "pt");
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(&[0x50, 0x4B, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00]).unwrap();
        drop(f);

        let fmt = detect_format(&path).unwrap();
        assert_eq!(fmt, ModelFormat::PyTorchZip);
        assert_eq!(fmt.to_string(), "pytorch_zip");
        fs::remove_file(path).ok();
    }

    #[test]
    fn detect_legacy_pickle() {
        let path = temp_path("legacy", "ckpt");
        let mut f = fs::File::create(&path).unwrap();
        // Random bytes that don't match any known magic
        f.write_all(&[0x80, 0x02, 0x63, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]).unwrap();
        drop(f);

        let fmt = detect_format(&path).unwrap();
        assert_eq!(fmt, ModelFormat::PyTorchLegacy);
        assert_eq!(fmt.to_string(), "pytorch_legacy");
        fs::remove_file(path).ok();
    }

    #[test]
    fn detect_diffusers_directory() {
        let dir = std::env::temp_dir().join(format!("fmt_detect_diffusers_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()));
        fs::create_dir_all(&dir).unwrap();
        let index = dir.join("model_index.json");
        fs::write(&index, b"{}").unwrap();

        let fmt = detect_format(&dir).unwrap();
        assert_eq!(fmt, ModelFormat::Diffusers);
        assert_eq!(fmt.to_string(), "diffusers");
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn detect_directory_with_safetensors() {
        let dir = std::env::temp_dir().join(format!("fmt_detect_dir_st_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()));
        fs::create_dir_all(&dir).unwrap();
        let st = dir.join("model.safetensors");
        let header = b"{}";
        let header_len = header.len() as u64;
        let mut f = fs::File::create(&st).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header).unwrap();
        drop(f);

        let fmt = detect_format(&dir).unwrap();
        assert_eq!(fmt, ModelFormat::Safetensors);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn display_all_formats() {
        assert_eq!(ModelFormat::Safetensors.to_string(), "safetensors");
        assert_eq!(ModelFormat::Gguf.to_string(), "gguf");
        assert_eq!(ModelFormat::PyTorchZip.to_string(), "pytorch_zip");
        assert_eq!(ModelFormat::PyTorchLegacy.to_string(), "pytorch_legacy");
        assert_eq!(ModelFormat::Diffusers.to_string(), "diffusers");
    }

    #[test]
    fn nonexistent_path_errors() {
        let result = detect_format(Path::new("/tmp/does_not_exist_12345.bin"));
        assert!(result.is_err());
    }
}
