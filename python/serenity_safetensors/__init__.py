"""
serenity-safetensors: High-performance safetensors I/O with O_DIRECT and mmap support.

Drop-in replacement for safetensors.torch with training-optimized checkpoint saves.

Key features:
  - save_file_direct: O_DIRECT save, no page cache pollution, 4MB chunked writes
  - load_file: lazy mmap views, no data copy until tensor accessed
  - load_selective / load_by_prefix: partial loading without materializing all tensors
  - file_metadata: header-only read, no tensor data touched
  - FP8 dtype support (float8_e4m3fn, float8_e5m2)
  - Fast serialization via data_ptr+ctypes (numpy fallback)
"""

from .serenity_safetensors import (
    save_file_direct,
    save_file,
    _load_file_raw,
    _load_selective_raw,
    _load_by_prefix_raw,
    file_metadata,
    tensor_layout,
    training_metadata,
    tensor_names,
    shard_index,
)


class SafeTensorsDict(dict):
    """Dict subclass that keeps the mmap handle alive.

    Tensors are lazy views into the memory-mapped file. The OS pages in data
    on access — no upfront copy. This dict holds a reference to the mmap so
    it stays alive as long as any tensor might reference it.

    Behaves exactly like a regular dict. The _mmap attribute is internal.
    """

    __slots__ = ("_mmap",)

    def __init__(self, tensor_dict, mmap_handle):
        super().__init__(tensor_dict)
        self._mmap = mmap_handle

    def close(self):
        """Explicitly close the mmap. Tensors become invalid after this."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None


def load_file(path, device="cpu"):
    """Load all tensors from a safetensors file.

    Returns a dict-like object (SafeTensorsDict) mapping tensor names to
    torch.Tensor. Tensors are lazy mmap views — data is paged in by the OS
    on first access. No upfront copy.

    Drop-in replacement for safetensors.torch.load_file.
    """
    tensor_dict, mmap_handle = _load_file_raw(path, device)
    return SafeTensorsDict(tensor_dict, mmap_handle)


def load_selective(path, names, device="cpu"):
    """Load only the named tensors from a safetensors file.

    Returns a SafeTensorsDict with only the requested tensors.
    """
    tensor_dict, mmap_handle = _load_selective_raw(path, names, device)
    return SafeTensorsDict(tensor_dict, mmap_handle)


def load_by_prefix(path, prefix, device="cpu"):
    """Load tensors whose names start with the given prefix.

    Returns a SafeTensorsDict with only prefix-matched tensors.
    """
    tensor_dict, mmap_handle = _load_by_prefix_raw(path, prefix, device)
    return SafeTensorsDict(tensor_dict, mmap_handle)


from . import torch

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
    "SafeTensorsDict",
    "torch",
]
