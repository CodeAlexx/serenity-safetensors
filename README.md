# serenity-safetensors

Universal model loading for Serenity. Rust core, PyO3 bindings.

Loads safetensors, GGUF (quantized), PyTorch checkpoints (.pt/.pth/.bin), and diffusers directories through a single `load_model(path)` call. GGUF quantized tensors stay compact in memory and dequantize to BF16 on demand during Stagehand H2D transfer.

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

## Universal model loading

```python
from serenity_safetensors import load_model, probe_model, detect_format

# Any format — safetensors, GGUF, PyTorch .pt/.pth, diffusers directory
data = load_model("model.gguf")
data = load_model("model.safetensors")
data = load_model("checkpoint.pt")
data = load_model("stable-diffusion-xl/")  # diffusers directory

# Dict-like access to tensors
tensor = data["layer.0.weight"]     # torch.Tensor (BF16) or QuantizedTensor

# QuantizedTensor — GGUF quant types stay compact, dequant on demand
if hasattr(tensor, "dequant"):
    print(tensor.quant_type_name)   # "Q4_K"
    print(tensor.compression_ratio) # 4.0x
    bf16_tensor = tensor.dequant()  # → BF16 torch.Tensor

# Header-only probe — no tensor data loaded (~1ms)
info = probe_model("flux1-dev-Q4_K.gguf")
print(info["format"])        # "gguf"
print(info["tensor_count"])  # 291
print(info["param_count"])   # 8030000000

# Format detection from magic bytes
fmt = detect_format("model.safetensors")  # "safetensors"
```

### Supported formats

| Format | Extensions | Features |
|--------|-----------|----------|
| **Safetensors** | `.safetensors` | mmap, lazy loading, zero-copy |
| **GGUF** | `.gguf` | Q4_0–Q8_K dequant, lazy QuantizedTensor, mmap |
| **PyTorch** | `.pt`, `.pth`, `.bin` | Safe pickle scanner (no code exec), ZIP extraction |
| **Diffusers** | directory with `model_index.json` | Component discovery, sharded files, auto-aggregation |

### GGUF quantization types

Full dequant support for: F16, F32, BF16, F64, I8, I16, I32, I64, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K. IQ types (IQ2_XXS, IQ3_S, etc.) are preserved as QuantizedTensor but dequant is not yet implemented.

## What's different from `safetensors.torch`

| Issue with `safetensors.torch` | serenity-safetensors |
|---|---|
| `save_file` goes through page cache | `save_file_direct` uses O_DIRECT, 4MB chunked writes |
| `load_file` returns in ~8ms (lazy) | `load_file` returns in ~2ms (lazy, memoryview zero-copy) |
| No partial loading | `load_selective` (by name), `load_by_prefix` (by prefix) |
| No sharded loading helpers | `load_sharded`, `load_sharded_selective`, `load_sharded_by_prefix` |
| No subset/source writer | `materialize_selective`, `materialize_by_prefix`, and sharded variants write reduced safetensors files |
| Getting metadata loads entire file | `file_metadata` reads only the 8-byte header + JSON |
| No tensor name listing | `tensor_names` returns keys without loading data |
| No Stagehand-friendly layout API | `tensor_layout` returns dtype, shape, logical offsets, and absolute byte offsets |
| No shard index parser | `shard_index` reads diffusers safetensors index files and groups tensors by shard |
| No sharded layout inspection | `sharded_tensor_layout` resolves shard paths and per-tensor offsets across the full index |
| No reusable source manifest layer | `source_manifest`, `quantized_source_manifest`, `read_manifest`, `write_manifest`, and `check_manifest_compatibility` |
| No quantized artifact verifier | `read_quantized_block_map`, `write_quantized_block_map`, and `verify_quantized_manifest_artifacts` |
| No persisted quantized payload container | `write_quantized_block_container` and `load_quantized_blocks` store/reload opaque block payloads with deterministic offsets |
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
from serenity_safetensors.torch import (
    load_selective,
    load_by_prefix,
    load_sharded,
    load_sharded_selective,
    load_sharded_by_prefix,
)

# Specific tensors
lora = load_selective("model.safetensors", [
    "lora_unet.down_blocks.0.weight",
    "lora_unet.down_blocks.0.bias",
], device="cuda")

# All tensors with prefix
te = load_by_prefix("model.safetensors", "text_encoder.", device="cuda")

# All tensors from a diffusers-style sharded index
model = load_sharded("model.safetensors.index.json", device="cpu")

# Only a few tensors from a sharded index
subset = load_sharded_selective("model.safetensors.index.json", [
    "transformer.blocks.0.attn.qkv.weight",
    "transformer.blocks.0.attn.qkv.bias",
])

# Prefix match across shards
prefix_subset = load_sharded_by_prefix(
    "model.safetensors.index.json",
    "transformer.blocks.0.",
)
```

### Source materialization

```python
from serenity_safetensors.torch import (
    materialize_selective,
    materialize_by_prefix,
    materialize_sharded_selective,
    materialize_sharded_by_prefix,
)

# Write a smaller file from a monolithic checkpoint
materialize_selective(
    "model.safetensors",
    "subset.safetensors",
    ["transformer.blocks.0.attn.qkv.weight"],
)

# Write everything under a prefix
materialize_by_prefix(
    "model.safetensors",
    "transformer_only.safetensors",
    "transformer.",
)

# Do the same starting from a sharded diffusers index
materialize_sharded_by_prefix(
    "model.safetensors.index.json",
    "transformer_only.safetensors",
    "transformer.",
)
```

### Source manifests

```python
import torch

from serenity_safetensors.torch import (
    source_manifest,
    quantized_source_manifest,
    write_manifest,
    read_manifest,
    check_manifest_compatibility,
    write_quantized_block_container,
    write_quantized_block_map,
    load_quantized_blocks,
    read_quantized_block_map,
    verify_quantized_manifest_artifacts,
)

manifest = source_manifest(
    model_family="ltx2_19b",
    model_version="2.3",
    source_kind="materialized_subset",
    source_path="transformer_only.safetensors",
    original_source="hf://Lightricks/LTX-2.3-distilled",
    source_signature="sha256:base123",
    dtype="bfloat16",
    tensor_prefixes=["transformer."],
    stagehand_layout="transformer_blocks_v1",
)
write_manifest("transformer_only.source.json", manifest)

quant_manifest = quantized_source_manifest(
    model_family="ltx2_19b",
    model_version="2.3",
    original_source="hf://Lightricks/LTX-2.3-distilled",
    source_signature="sha256:base123",
    block_map_path="blocks.json",
    data_files=["block_000.bin", "block_001.bin"],
    source_path="eriquant_cache",
    quant_mode="eriquant",
    tensor_prefixes=["transformer."],
    block_count=4128,
    group_size=64,
    stagehand_layout="transformer_blocks_v1",
)
write_manifest("eriquant.source.json", quant_manifest)

loaded = read_manifest("eriquant.source.json")
compat = check_manifest_compatibility(
    "eriquant.source.json",
    model_family="ltx2_19b",
    model_version="2.3",
    source_signature="sha256:base123",
    quant_mode="eriquant",
    stagehand_layout="transformer_blocks_v1",
)
assert compat["ok"]

write_quantized_block_map("blocks.json", {
    "blocks": [
        {"file": "block_000.bin", "offset": 0, "nbytes": 4096, "tensors": ["transformer.blocks.0.attn.qkv.weight"]},
    ],
})
block_map = read_quantized_block_map("blocks.json")
verify = verify_quantized_manifest_artifacts("eriquant.source.json")
assert verify["ok"]

payloads = {
    "transformer.layers.0": torch.tensor([1, 3, 5, 7], dtype=torch.uint8),
    "transformer.layers.1": torch.tensor([2, 4, 6], dtype=torch.uint8),
}
block_map = write_quantized_block_container(
    payloads,
    "blocks.safetensors",
    block_tensors={
        "transformer.layers.0": ["linear.weight"],
        "transformer.layers.1": ["proj.weight", "proj.bias"],
    },
)
write_quantized_block_map("blocks.json", block_map)
payload_views = load_quantized_blocks("eriquant.source.json")
assert payload_views["transformer.layers.0"].dtype == torch.uint8
```

### Inspect without loading data

```python
from serenity_safetensors.torch import (
    file_metadata,
    tensor_layout,
    tensor_names,
    shard_index,
    sharded_tensor_names,
    sharded_tensor_layout,
)

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

# Cross-shard tensor inspection
names = sharded_tensor_names("model.safetensors.index.json")
layout = sharded_tensor_layout("model.safetensors.index.json")
print(layout["tensors"]["transformer.blocks.0.attn.qkv.weight"]["path"])
```

## API

| Function | Description |
|---|---|
| `save_file_direct(state_dict, path, metadata=None)` | O_DIRECT save, 4MB chunked writes — no page cache pollution |
| `save_file(state_dict, path, metadata=None)` | Normal save (drop-in for `safetensors.torch.save_file`) |
| `materialize_selective(path, output_path, names, direct=False)` | Write a subset file from explicit tensor names |
| `materialize_by_prefix(path, output_path, prefix, direct=False)` | Write a subset file from a prefix |
| `materialize_sharded_selective(index_path, output_path, names, direct=False)` | Write a subset file from a sharded index by tensor name |
| `materialize_sharded_by_prefix(index_path, output_path, prefix, direct=False)` | Write a subset file from a sharded index by prefix |
| `source_manifest(...)` | Build a canonical Serenity source manifest |
| `quantized_source_manifest(...)` | Build a canonical Serenity quantized-source manifest |
| `read_manifest(path, resolve_paths=True)` | Read and validate a source manifest |
| `write_manifest(path, manifest)` | Validate and write a source manifest |
| `check_manifest_compatibility(path, ...)` | Check model/source/quant/Stagehand compatibility against a manifest |
| `read_quantized_block_map(path, resolve_paths=True)` | Read and validate a quantized block-map file |
| `write_quantized_block_map(path, block_map)` | Validate and write a quantized block-map file |
| `verify_quantized_manifest_artifacts(path)` | Verify that a quantized-source manifest matches its block map and data files |
| `write_quantized_block_container(payloads, path, block_tensors=None, metadata=None, direct=False)` | Write a deterministic opaque block-payload container and return its block map |
| `load_quantized_blocks(reference_path, block_ids=None, device="cpu")` | Load opaque block payload bytes from a block map or quantized-source manifest |
| `load_file(path, device="cpu")` | Lazy mmap load, memoryview zero-copy (drop-in, 4x faster) |
| `load_selective(path, names, device="cpu")` | Load only named tensors |
| `load_by_prefix(path, prefix, device="cpu")` | Load prefix-matched tensors |
| `load_sharded(index_path, device="cpu")` | Load all tensors referenced by a sharded safetensors index |
| `load_sharded_selective(index_path, names, device="cpu")` | Load only named tensors across shards |
| `load_sharded_by_prefix(index_path, prefix, device="cpu")` | Load prefix-matched tensors across shards |
| `file_metadata(path)` | Header-only read: metadata + tensor info (0.7ms) |
| `tensor_layout(path)` | Header-only layout read with absolute byte offsets for file-backed consumers |
| `tensor_names(path)` | List tensor names without loading data |
| `shard_index(path)` | Parse a diffusers safetensors shard index and group tensor names per shard |
| `sharded_tensor_names(index_path)` | List all tensor names referenced by a sharded index |
| `sharded_tensor_layout(index_path)` | Resolve shard paths and tensor byte offsets across a sharded index |
| `training_metadata(step, lr, loss, epoch, extra)` | Build metadata dict for checkpoint saves |
| **Universal loading** | |
| `load_model(path, strip_prefix=None)` | Load any format — returns `ModelData` with tensors + info |
| `probe_model(path)` | Header-only probe — format, tensor count, shapes, dtypes (~1ms) |
| `detect_format(path)` | Magic-byte format detection → `"safetensors"`, `"gguf"`, `"pytorch_zip"`, `"diffusers"` |
| `load_gguf_index(path)` | Open GGUF file, return tensor index for selective access |
| `dequant_tensor(data, quant_type, shape)` | Dequantize raw GGUF bytes to BF16 torch.Tensor |
| `load_pickle_index(path)` | Parse PyTorch checkpoint pickle, return tensor metadata |
| `load_pickle_tensor(path, name)` | Extract a single tensor from a PyTorch checkpoint |
| `probe_diffusers(path)` | Probe a diffusers directory — components, tensor counts, shapes |
| `QuantizedTensor` | GGUF quantized tensor wrapper with `.dequant()`, `.shape`, `.quant_type_name`, `.compression_ratio` |
| `ModelData` | Dict-like result from `load_model()` with `.tensors`, `.info`, `.format` |

## Verification

Current Rust verification covers:

- dtype aliases for `bf16` and FP8
- header-only layout extraction with absolute offsets
- deterministic tensor-name ordering from on-disk offsets
- diffusers shard-index grouping
- sharded layout resolution across relative shard paths
- prefix-based shard selection resolution
- single-file subset materialization
- sharded subset materialization
- source-manifest read/write/normalize helpers
- quantized-source manifest generation and compatibility checks
- quantized block-map read/write and artifact verification
- quantized block-container write/reload and offset/hash validation
- normal streaming safetensors writes
- direct/O_DIRECT streaming safetensors writes
- magic-byte format detection (safetensors, GGUF, PyTorch ZIP, diffusers directory)
- header-only probe for all formats
- GGUF v2/v3 parser with mmap-backed tensor index
- 19 GGUF dequant kernels (Q4_0–Q8_K, F16/F32/BF16, I8–I64)
- safe pickle scanner for PyTorch checkpoints (no arbitrary code execution)
- diffusers directory layout discovery with sharded component support
- unified load_model dispatcher across all formats

Run it with:

```bash
cargo test
```

## Roadmap

The current package is now good enough to act as a real Serenity source layer for:

- file-backed Stagehand readers
- transformer-only source materialization
- lower-RAM checkpoint saves
- persisted quantized block payload containers

All roadmap phases are complete. The package is fully adopted in the Serenity trainer and serves as the universal model loading layer for SerenityFlow.

Current uses:
- O_DIRECT checkpoint saves (saver.py, trainer.py)
- source manifest resolution (Stagehand source resolver)
- persisted EriQuant frozen base reuse (eriquant_stagehand.py)
- universal model loading for SerenityFlow (safetensors, GGUF, PyTorch, diffusers)

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
