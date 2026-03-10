"""
serenity-safetensors: High-performance safetensors I/O with O_DIRECT and mmap support.

Drop-in replacement for safetensors.torch with training-optimized checkpoint saves.

Key features:
  - save_file_direct: O_DIRECT save, no page cache pollution, 4MB chunked writes
  - streaming save path: no giant serialized output blob before write
  - load_file: lazy mmap views, no data copy until tensor accessed
  - load_selective / load_by_prefix: partial loading without materializing all tensors
  - load_sharded / load_sharded_selective / load_sharded_by_prefix: multi-shard loading
  - materialize_selective / materialize_by_prefix: subset writers for reduced source files
  - materialize_sharded_selective / materialize_sharded_by_prefix: subset writers from sharded sources
  - source_manifest / quantized_source_manifest: canonical Serenity source-manifest builders
  - read_manifest / write_manifest / check_manifest_compatibility: manifest persistence and checks
  - read_quantized_block_map / write_quantized_block_map / verify_quantized_manifest_artifacts: quantized payload validation
  - write_quantized_block_container / load_quantized_blocks: persisted opaque block payload containers
  - file_metadata: header-only read, no tensor data touched
  - tensor_layout / sharded_tensor_layout: Stagehand-friendly byte offsets
  - FP8 dtype support (float8_e4m3fn, float8_e5m2)
"""

from .serenity_safetensors import (
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
    write_quantized_block_container,
    materialize_selective,
    materialize_by_prefix,
    materialize_sharded_selective,
    materialize_sharded_by_prefix,
    _load_file_raw,
    _load_selective_raw,
    _load_by_prefix_raw,
    _load_sharded_raw,
    _load_sharded_selective_raw,
    _load_sharded_by_prefix_raw,
    _load_quantized_blocks_raw,
    file_metadata,
    tensor_layout,
    training_metadata,
    tensor_names,
    shard_index,
    sharded_tensor_names,
    sharded_tensor_layout,
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
        """Explicitly close backing mmaps. Tensors become invalid after this."""
        if self._mmap is None:
            return
        if isinstance(self._mmap, (list, tuple)):
            for handle in self._mmap:
                handle.close()
        else:
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


def load_sharded(index_path, device="cpu"):
    """Load all tensors referenced by a sharded safetensors index."""
    tensor_dict, mmap_handles = _load_sharded_raw(index_path, device)
    return SafeTensorsDict(tensor_dict, mmap_handles)


def load_sharded_selective(index_path, names, device="cpu"):
    """Load only the named tensors referenced by a sharded safetensors index."""
    tensor_dict, mmap_handles = _load_sharded_selective_raw(index_path, names, device)
    return SafeTensorsDict(tensor_dict, mmap_handles)


def load_sharded_by_prefix(index_path, prefix, device="cpu"):
    """Load prefix-matched tensors from a sharded safetensors index."""
    tensor_dict, mmap_handles = _load_sharded_by_prefix_raw(index_path, prefix, device)
    return SafeTensorsDict(tensor_dict, mmap_handles)


def load_quantized_blocks(reference_path, block_ids=None, device="cpu"):
    """Load opaque quantized block payloads from a block-map or source manifest."""
    tensor_dict, mmap_handles = _load_quantized_blocks_raw(reference_path, block_ids, device)
    return SafeTensorsDict(tensor_dict, mmap_handles)


from . import torch

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
    "write_quantized_block_container",
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
    "load_quantized_blocks",
    "file_metadata",
    "tensor_layout",
    "training_metadata",
    "tensor_names",
    "shard_index",
    "sharded_tensor_names",
    "sharded_tensor_layout",
    "SafeTensorsDict",
    "torch",
]
