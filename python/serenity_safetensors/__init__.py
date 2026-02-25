"""
serenity-safetensors: High-performance safetensors I/O with O_DIRECT and mmap support.

Drop-in replacement for safetensors.torch with training-optimized checkpoint saves.

Key improvements over safetensors.torch:
  - save_file_direct: O_DIRECT checkpoint save, no page cache pollution, 4MB chunked writes
  - load_file: mmap-based loading, no full-file heap copy
  - load_selective / load_by_prefix: partial loading without materializing all tensors
  - file_metadata: header-only read, no tensor data touched
  - FP8 dtype support (float8_e4m3fn, float8_e5m2)
  - Fast tensor serialization via data_ptr + ctypes (falls back to numpy)
"""

from .serenity_safetensors import (
    save_file_direct,
    save_file,
    load_file,
    load_selective,
    load_by_prefix,
    file_metadata,
    training_metadata,
)

from . import torch

__all__ = [
    "save_file_direct",
    "save_file",
    "load_file",
    "load_selective",
    "load_by_prefix",
    "file_metadata",
    "training_metadata",
    "torch",
]
