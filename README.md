# serenity-safetensors

High-performance safetensors I/O for Serenity. Rust core, PyO3 bindings.

## What's different from `safetensors.torch`

| Issue with `safetensors.torch` | serenity-safetensors |
|---|---|
| `save_file` goes through page cache | `save_file_direct` uses O_DIRECT, 4MB chunked writes |
| `load_file` reads entire file into heap | mmap-based loading via `memmap2` |
| No partial loading | `load_selective` (by name), `load_by_prefix` (by prefix) |
| Getting metadata loads entire file | `file_metadata` reads only the 8-byte header length + header |
| No FP8 support | F8_E4M3 (`float8_e4m3fn`) and F8_E5M2 (`float8_e5m2`) |
| Serialization copies through numpy | Fast path via `data_ptr()` + `ctypes.string_at()`, numpy fallback |

## Install

```bash
cd serenity-safetensors
pip install maturin
maturin develop --release
```

## Usage

### Training checkpoint (the main event)

```python
from serenity_safetensors import save_file_direct, training_metadata

meta = training_metadata(step=max_steps, lr=learning_rate, loss=loss_value)
save_file_direct(model.state_dict(), str(checkpoint_path), metadata=meta)
```

### Drop-in replacement

```python
# Before:
from safetensors.torch import load_file, save_file

# After:
from serenity_safetensors.torch import load_file, save_file
```

### Partial loading

```python
from serenity_safetensors.torch import load_selective, load_by_prefix

# Specific tensors
lora = load_selective("model.safetensors", [
    "lora_unet.down_blocks.0.weight",
    "lora_unet.down_blocks.0.bias",
], device="cuda")

# All tensors with prefix
te = load_by_prefix("model.safetensors", "text_encoder.", device="cuda")
```

### Inspect without loading data

```python
from serenity_safetensors.torch import file_metadata

info = file_metadata("model.safetensors")
print(info["metadata"])  # {step, lr, loss, ...}
for name, ti in info["tensors"].items():
    print(f"{name}: {ti['dtype']} {ti['shape']} ({ti['nbytes']} bytes)")
```

## API

| Function | Description |
|---|---|
| `save_file_direct(state_dict, path, metadata=None)` | O_DIRECT save, 4MB chunked writes |
| `save_file(state_dict, path, metadata=None)` | Normal save (drop-in) |
| `load_file(path, device="cpu")` | mmap load all tensors (drop-in) |
| `load_selective(path, names, device="cpu")` | Load named tensors only |
| `load_by_prefix(path, prefix, device="cpu")` | Load prefix-matched tensors |
| `file_metadata(path)` | Header-only read: metadata + tensor info |
| `training_metadata(step, lr, loss, epoch, extra)` | Build metadata dict |

## O_DIRECT details

- 4MB aligned chunks (not one giant blocking write)
- 4096-byte buffer alignment
- Falls back to normal write on unsupported FS (tmpfs, NFS)
- Truncates to exact size after final padded chunk
- Linux only; normal write on other platforms
