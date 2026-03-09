"""
serenity_safetensors.torch — drop-in replacement for safetensors.torch

Replace:
    from safetensors.torch import load_file, save_file
With:
    from serenity_safetensors.torch import load_file, save_file

Additional:
    save_file_direct  — O_DIRECT checkpoint save (4MB chunked, no page cache)
    load_selective    — load only named tensors
    load_by_prefix    — load tensors matching a prefix
    file_metadata     — header-only inspect (no tensor data read)
    training_metadata — build metadata dict for checkpoints
    tensor_names      — list tensor names without loading data
"""

from . import (
    save_file_direct,
    save_file,
    load_file,
    load_selective,
    load_by_prefix,
    file_metadata,
    tensor_layout,
    training_metadata,
    tensor_names,
    shard_index,
)

__all__ = [
    "save_file_direct",
    "save_file",
    "load_file",
    "load_selective",
    "load_by_prefix",
    "file_metadata",
    "tensor_layout",
    "training_metadata",
    "tensor_names",
    "shard_index",
]
