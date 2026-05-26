# serenity-safetensors — Mojo

Pure-Mojo safetensors **read path**. A faithful port of this crate's Rust reader
(`src/mmap.rs`, `src/lib.rs`), mirroring its logic op-for-op. No Python in the
runtime path, no external dependencies beyond the Mojo stdlib — the JSON header
is hand-rolled.

Built and verified against **Mojo 1.0.0b1 / MAX 26.3**, **Linux x86-64 only**
(the libc constants in `ffi.mojo` are hardcoded for the x86-64 ABI).

## Scope

This is the **reader** — open a file (or a sharded diffusers index), inspect
tensor metadata, and get a lazily-mmap'd view of a tensor's bytes. It is the
right tool when you want zero-copy, low-RAM model loading from Mojo.

| Capability | Mojo (`mojo/`) | Rust (`src/`) / Python (`python/`) |
|---|:---:|:---:|
| Read single-file safetensors (lazy mmap) | ✅ | ✅ |
| Read sharded diffusers index (`*.index.json`) | ✅ | ✅ |
| Header-only metadata / tensor layout | ✅ | ✅ |
| Origin-bound (lifetime-safe) tensor views | ✅ | — |
| Write / `save_file` / O_DIRECT | ❌ | ✅ |
| GGUF dequant, PyTorch pickle, manifests, quant | ❌ | ✅ |

If you need writing, GGUF, PyTorch checkpoints, or manifests, use the Rust crate
or the Python bindings — see the top-level [README](../README.md).

## How loading works

```
open(path)                      # O_RDONLY
  → pread first 8 bytes          # header_len (little-endian u64)
  → pread header_len bytes        # the JSON header (≤100 MB, the only eager I/O)
  → mmap the DATA segment         # MAP_PRIVATE | MAP_NORESERVE — uncommitted
  → build name → TensorRef index
```

The data segment is **never read into RAM**. It is mapped uncommitted; the OS
page cache pages tensor bytes in on first access. The only eager read is the
small bounded header.

## Public API

`from serenity_safetensors.safetensors import SafeTensors`

| Method | Description |
|---|---|
| `SafeTensors.open(path)` | Open a single-file `.safetensors`; reads the header, mmaps the data segment. |
| `.names()` | All tensor names (excludes `__metadata__`). |
| `.tensor_info(name)` | `TensorRef` — `dtype`, `shape`, `offset`, `size`. |
| `.tensor_bytes(name)` | `Span[UInt8, origin_of(self)]` — lifetime-bound view of the raw bytes. |
| `.count()` / `.data_size()` | Tensor count / data-segment byte length. |

`from serenity_safetensors.sharded import ShardedSafeTensors`

| Method | Description |
|---|---|
| `ShardedSafeTensors.open(dir)` | Open a diffusers dir: parses `*.safetensors.index.json` `weight_map`, opens each unique shard, builds a unified name → shard map. Falls back to the single `*.safetensors` in `dir`. |
| `.num_shards()` / `.num_tensors()` | Counts. |
| `.names()` / `.tensor_info(name)` / `.tensor_bytes(name)` | As above, resolved across shards. |
| `.tensor_view(name)` | A `TensorView` (metadata + origin-bound bytes) over the owning shard. |

`from serenity_safetensors.dtype import STDtype` — the safetensors dtype enum
(`BOOL, U8, I8, F8_E5M2, F8_E4M3, I16, U16, F16, BF16, I32, U32, F32, F64,
I64, U64`) with `.byte_size()`, `.name()`, `STDtype.from_name(s)`,
`.to_mojo_dtype()`.

`from serenity_safetensors.tensor_view import TensorView, from_parts` — typed
metadata bundled with an origin-bound byte `Span` (see lifetime note below).

## Usage

```mojo
from sys import argv
from serenity_safetensors.safetensors import SafeTensors

def main() raises:
    var path = String(argv()[1])           # e.g. model/vae/diffusion_pytorch_model.safetensors
    var st = SafeTensors.open(path)
    print("tensors:", st.count(), "| data bytes:", st.data_size())

    var name = String("decoder.conv_in.weight")
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)        # Span[UInt8, origin_of(st)] — zero-copy
    print(name, "dtype=", info.dtype.name(), "size=", info.size, "first_byte=", Int(bytes[0]))
```

Run it (the `-I mojo` is **required** so the `serenity_safetensors` package
resolves):

```bash
mojo run -I mojo your_program.mojo path/to/model.safetensors
```

Two runnable smokes are included — `smoke_safetensors.mojo` (single file) and
`smoke_sharded.mojo` (diffusers index). They contain hardcoded local model
paths from the original port; edit the path to point at a model you have:

```bash
mojo run -I mojo mojo/serenity_safetensors/smoke_safetensors.mojo
```

## The lifetime contract (the headline Mojo feature)

`tensor_bytes` / `tensor_view` return a `Span` / `TensorView` whose **origin is
bound to the source** (`origin_of(self)`). The compiler keeps the source — and
therefore its mmap'd region — alive for exactly as long as any view over it is
in use. No bare pointers, no `unsafe_origin_cast`.

This **compile-rejects** the two common footguns (both verified): returning a
view that escapes past its source, and using a view after explicitly destroying
the source (`src^.__del__()`).

**Known limitation (Mojo 1.0.0b1 origin tracking):** it does *not* catch
*reassigning* the source binding while a view is live —

```mojo
var v = st.tensor_bytes(name)
st = SafeTensors.open(other)   # compiles, but v is now a use-after-free
```

Not reassigning the source while a view is live is the caller's contract.

## Layout

```
mojo/serenity_safetensors/
  __init__.mojo      package
  ffi.mojo           libc externs (mmap/munmap/madvise/sysconf/open/close/pread)
  mmap.mojo          MmapRegion — uncommitted MAP_NORESERVE mmap of the data segment
  json_header.mojo   hand-rolled parser for the flat safetensors header schema
  dtype.mojo         STDtype enum (mirrors the Rust crate's dtype set)
  safetensors.mojo   SafeTensors — single-file reader
  tensor_view.mojo   TensorView — typed metadata + origin-bound byte span
  sharded.mojo       ShardedSafeTensors — diffusers index / multi-shard loader
  smoke_*.mojo       runnable smokes
```
