"""
serenity_safetensors.torch — drop-in replacement for safetensors.torch

Replace:
    from safetensors.torch import load_file, save_file
With:
    from serenity_safetensors.torch import load_file, save_file

Additional:
    save_file_direct  — O_DIRECT checkpoint save (4MB chunked, no page cache)
    source_manifest — build a canonical Serenity source manifest
    quantized_source_manifest — build a canonical Serenity quantized-source manifest
    read_manifest / write_manifest — persist and reload source manifests
    check_manifest_compatibility — verify manifest/runtime compatibility
    read_quantized_block_map / write_quantized_block_map — persist and reload quantized block maps
    verify_quantized_manifest_artifacts — verify quantized payload consistency
    materialize_selective — write a subset file by explicit tensor names
    materialize_by_prefix — write a subset file by prefix
    materialize_sharded_selective — subset materialization from a sharded index
    materialize_sharded_by_prefix — prefix subset materialization from a sharded index
    load_selective    — load only named tensors
    load_by_prefix    — load tensors matching a prefix
    load_sharded      — load all tensors from a sharded safetensors index
    load_sharded_selective — load named tensors from a sharded safetensors index
    load_sharded_by_prefix — load prefix-matched tensors from a sharded safetensors index
    file_metadata     — header-only inspect (no tensor data read)
    training_metadata — build metadata dict for checkpoints
    tensor_names      — list tensor names without loading data
"""

from . import (
    save_file_direct,
    save_file,
    source_manifest,
    quantized_source_manifest,
    read_manifest,
    write_manifest,
    check_manifest_compatibility,
    read_quantized_block_map,
    write_quantized_block_map,
    verify_quantized_manifest_artifacts,
    materialize_selective,
    materialize_by_prefix,
    materialize_sharded_selective,
    materialize_sharded_by_prefix,
    load_file,
    load_selective,
    load_by_prefix,
    load_sharded,
    load_sharded_selective,
    load_sharded_by_prefix,
    file_metadata,
    tensor_layout,
    training_metadata,
    tensor_names,
    shard_index,
    sharded_tensor_names,
    sharded_tensor_layout,
)

__all__ = [
    "save_file_direct",
    "save_file",
    "source_manifest",
    "quantized_source_manifest",
    "read_manifest",
    "write_manifest",
    "check_manifest_compatibility",
    "read_quantized_block_map",
    "write_quantized_block_map",
    "verify_quantized_manifest_artifacts",
    "materialize_selective",
    "materialize_by_prefix",
    "materialize_sharded_selective",
    "materialize_sharded_by_prefix",
    "load_file",
    "load_selective",
    "load_by_prefix",
    "load_sharded",
    "load_sharded_selective",
    "load_sharded_by_prefix",
    "file_metadata",
    "tensor_layout",
    "training_metadata",
    "tensor_names",
    "shard_index",
    "sharded_tensor_names",
    "sharded_tensor_layout",
]
