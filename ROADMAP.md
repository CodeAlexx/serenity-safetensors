# serenity-safetensors Roadmap

## Goal

Make `serenity-safetensors` the canonical Serenity source layer for:

- file-backed Stagehand loading
- low-RAM checkpoint saves
- transformer-only materialization
- persisted quantized frozen bases for `EriQuant + Stagehand`

The package should stay focused on file/source concerns, not training math. It should answer:

- where tensor bytes live
- how to inspect them cheaply
- how to materialize subsets predictably
- how to persist compatible training sources and manifests

## Current State

Implemented now:

- lazy mmap file loading
- selective and prefix-based loading
- header-only metadata reads
- header-only tensor layout reads with absolute offsets
- diffusers shard-index parsing with grouped shard membership
- streaming `save_file`
- streaming `save_file_direct` with `O_DIRECT` fallback
- Rust unit coverage for layout, shard index, and streaming writes

Current gap:

- no sharded tensor reader yet, only shard-index inspection
- no subset/materialization writer
- no persisted quantized container for `EriQuant`
- metadata schema is still checkpoint-oriented, not source-manifest-oriented

## Phase 1: Stable Inspection Surface

Status: completed

Deliverables:

- `tensor_layout(path)` for Stagehand-style consumers
- `shard_index(path)` for diffusers shard manifests
- deterministic ordering based on on-disk offsets
- Rust tests for layout/index correctness

Why it matters:

- Serenity should not have to rebuild byte offsets or shard grouping in multiple repos.
- File-backed systems need a small, trustworthy API for cheap inspection.

## Phase 2: Streaming Writers

Status: completed

Deliverables:

- `save_file` streams header and tensors directly to the output file
- `save_file_direct` streams header and tensors through the aligned direct-write path
- no giant serialized safetensors blob in Rust memory before save

Why it matters:

- Large checkpoint saves should not create avoidable RAM spikes.
- Long training runs should keep cache pressure low and avoid page-cache pollution.

## Phase 3: Sharded Source Access

Status: completed

Deliverables:

- open a diffusers-style shard index and resolve tensor name -> shard path
- selective loads across shards
- prefix loads across shards
- header-only shard summaries without materializing all shard metadata separately

Tests:

- multi-shard fixture with repeated lookups
- selective loads spanning more than one shard
- prefix loads returning tensors from several shards
- correct metadata and absolute offsets per resolved shard

## Phase 4: Subset Materialization

Status: completed

Deliverables:

- write a new safetensors file from:
  - a prefix
  - an explicit tensor list
  - a source manifest
- preserve metadata and emit deterministic layouts
- avoid loading unrelated tensors

Why it matters:

- Serenity currently needs transformer-only sources and other reduced views.
- This should be a first-class package feature rather than adapter-local logic.

Tests:

- materialize a prefix-only source from a synthetic checkpoint
- validate output via `SafeTensors::deserialize`
- validate offsets and metadata with `tensor_layout`

## Phase 5: Source Manifests

Status: completed

Deliverables:

- a stable manifest schema for persisted training sources:
  - model family
  - model version
  - original checkpoint identity
  - dtype
  - quant mode
  - tensor prefix policy
  - compatibility/version markers
- manifest read/write helpers

Implemented:

- canonical `source_manifest` and `quantized_source_manifest` builders
- `read_manifest` / `write_manifest`
- manifest path resolution for relative artifact/source paths
- manifest compatibility checks for model/version/signature/quant mode/Stagehand layout

Why it matters:

- Stagehand and training adapters need to know what a source actually is before loading it.
- This keeps compatibility logic out of model-specific code.

## Phase 6: Persisted Quantized Sources

Status: in progress

Deliverables:

- persisted `EriQuant` frozen block sources in Serenity-owned format
- block map metadata compatible with Stagehand scheduling
- source signatures so quantized caches can be reused safely across runs

Implemented so far:

- quantized-source manifest schema
- block-map and data-file metadata in manifests
- compatibility/signature checks for quantized manifests
- quantized block-map read/write helpers
- artifact verification against block-map/data-file declarations

Still missing:

- actual Serenity-owned persisted quant-block container format
- deterministic block-map validation against on-disk payloads
- reload helpers for quantized block sources

Why it matters:

- quantization only becomes operationally convenient when the frozen base can be reused, not rebuilt every run
- this is the right boundary between `EriQuant`, Stagehand, and Serenity training

Tests:

- quantized manifest roundtrip
- block map integrity
- deterministic reload of persisted quantized sources
- compatibility rejection when source signature mismatches

## Phase 7: Serenity Integration

Status: planned

Adoption order:

1. use streaming saves for training checkpoints
2. use `tensor_layout` for file-backed Stagehand inspection
3. move transformer-only materialization onto subset writers
4. adopt sharded readers where large checkpoints currently have custom logic
5. add quantized-source reuse to the standard Serenity memory path

## Verification Policy

Every phase should include:

- pure Rust unit tests for file-format behavior
- fixture-based tests for offsets and metadata
- at least one real Serenity smoke after adoption when the change affects training/runtime paths

The package should remain verifiable without requiring a full Python training stack just to test file-layout logic.
