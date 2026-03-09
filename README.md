# serenity-safetensors

High-performance safetensors I/O for Serenity. Rust core, PyO3 bindings.

## Benchmarks

Tested on Gemma 3 12B shard (500 tensors, ~5GB):

| Operation | serenity-safetensors | safetensors.torch | Speedup |
|---|---|---|---|
| `load_file` (lazy mmap) | **2.1ms** | 8.5ms | **4x faster** |
| `file_metadata` (header-only) | **0.7ms** | N/A | — |

Loading returns lazy mmap-backed views via `memoryview` — zero-copy, no data read until tensor is accessed. OS pages in data on demand.

Checkpoint saves now stream tensors directly to the destination file instead of first materializing one giant serialized payload in Rust memory. That removes the worst write-time RAM spike and makes the package a better fit for Stagehand-style training workflows.

### Why O_DIRECT saves matter for training

Standard `safetensors.torch.save_file` writes through the page cache. A 4GB checkpoint evicts ~4GB of cached dataset pages. For video training, this means:

- Next 10-50 batches read dataset from disk instead of cache
- Throughput dip after every checkpoint save
- Latent caches get evicted, have to be re-read

`save_file_direct` bypasses the page cache entirely. Dataset and latent caches stay hot. No throughput dips.

## What's different from `safetensors.torch`

| Issue with `safetensors.torch` | serenity-safetensors |
|---|---|
| `save_file` goes through page cache | `save_file_direct` uses O_DIRECT, 4MB chunked writes |
| `load_file` returns in ~8ms (lazy) | `load_file` returns in ~2ms (lazy, memoryview zero-copy) |
| No partial loading | `load_selective` (by name), `load_by_prefix` (by prefix) |
| Getting metadata loads entire file | `file_metadata` reads only the 8-byte header + JSON |
| No tensor name listing | `tensor_names` returns keys without loading data |
| No Stagehand-friendly layout API | `tensor_layout` returns dtype, shape, logical offsets, and absolute byte offsets |
| No shard index parser | `shard_index` reads diffusers safetensors index files and groups tensors by shard |
| No FP8 support | F8_E4M3 (`float8_e4m3fn`) and F8_E5M2 (`float8_e5m2`) |
| Save path builds one big output blob | `save_file` / `save_file_direct` stream header + tensor payloads incrementally |

## Install

```bash
cd serenity-safetensors
pip install maturin
maturin develop --release
```

Requires Rust toolchain (`rustup.rs`).

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
from serenity_safetensors.torch import file_metadata, tensor_layout, tensor_names, shard_index

# Just the key names (fastest)
names = tensor_names("model.safetensors")

# Full metadata + tensor shapes/sizes
info = file_metadata("model.safetensors")
print(info["metadata"])  # {step, lr, loss, ...}
for name, ti in info["tensors"].items():
    print(f"{name}: {ti['dtype']} {ti['shape']} ({ti['nbytes']} bytes)")

# Stagehand-friendly offsets
layout = tensor_layout("model.safetensors")
print(layout["tensors"]["transformer.blocks.0.attn.qkv.weight"]["absolute_offsets"])

# Diffusers shard index inspection
index = shard_index("model.safetensors.index.json")
print(index["shards"]["model-00001-of-00004.safetensors"])
```

## API

| Function | Description |
|---|---|
| `save_file_direct(state_dict, path, metadata=None)` | O_DIRECT save, 4MB chunked writes — no page cache pollution |
| `save_file(state_dict, path, metadata=None)` | Normal save (drop-in for `safetensors.torch.save_file`) |
| `load_file(path, device="cpu")` | Lazy mmap load, memoryview zero-copy (drop-in, 4x faster) |
| `load_selective(path, names, device="cpu")` | Load only named tensors |
| `load_by_prefix(path, prefix, device="cpu")` | Load prefix-matched tensors |
| `file_metadata(path)` | Header-only read: metadata + tensor info (0.7ms) |
| `tensor_layout(path)` | Header-only layout read with absolute byte offsets for file-backed consumers |
| `tensor_names(path)` | List tensor names without loading data |
| `shard_index(path)` | Parse a diffusers safetensors shard index and group tensor names per shard |
| `training_metadata(step, lr, loss, epoch, extra)` | Build metadata dict for checkpoint saves |

## Verification

Current Rust verification covers:

- dtype aliases for `bf16` and FP8
- header-only layout extraction with absolute offsets
- deterministic tensor-name ordering from on-disk offsets
- diffusers shard-index grouping
- normal streaming safetensors writes
- direct/O_DIRECT streaming safetensors writes

Run it with:

```bash
cargo test
```

## Roadmap

The current package is now good enough to act as a real Serenity source layer for:

- file-backed Stagehand readers
- transformer-only source materialization
- lower-RAM checkpoint saves

The next major steps are:

- sharded data-source loading helpers beyond index inspection
- subset/materialization writers for transformer-only or prefix-only sources
- persisted quantized source containers for `EriQuant + Stagehand`
- richer training metadata/manifests for model family, quant mode, compatibility, and source signatures

Detailed implementation phases are tracked in [ROADMAP.md](ROADMAP.md).

## How the loading works

```
mmap(file, ACCESS_READ)     # OS maps file pages, no read yet
  |
memoryview(mmap)            # zero-copy view of the mapped region
  |
memoryview[start:end]       # zero-copy slice — just pointer + length
  |
torch.frombuffer(slice)     # tensor backed by mmap page — no copy
  |
tensor.reshape(shape)       # view, no copy
```

Data is paged in by the OS when you actually access the tensor (e.g., during `load_state_dict`). The `SafeTensorsDict` returned by `load_file` keeps the mmap alive via Python refcount.

## O_DIRECT details

- 4MB aligned chunks (not one giant blocking write)
- 4096-byte buffer alignment
- Falls back to normal write on unsupported FS (tmpfs, NFS)
- Truncates to exact size after final padded chunk
- Linux only; normal write on other platforms
