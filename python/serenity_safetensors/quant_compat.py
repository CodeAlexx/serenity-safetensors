"""Quantized checkpoint format detection and normalization.

Detects three known quantized checkpoint formats from state dict keys and
safetensors file metadata, then normalizes them to a single canonical form
that downstream consumers (EriQuant, inference engines) can handle uniformly.

Canonical output form per quantized layer:
  {prefix}{layer}.weight         — original storage tensor (unchanged)
  {prefix}{layer}.weight_scale   — float32 scale (renamed from scale_weight if needed)
  {prefix}{layer}.weight_scale_2 — float32 tensor scale (NVFP4 only, unchanged)
  {prefix}{layer}.input_scale    — float32 input scale (only if value != 1.0)
  {prefix}{layer}.quant_meta     — JSON string: {"format": "...", "full_precision": bool}

All non-quantized keys pass through unchanged. Input dict is never mutated.
"""

from __future__ import annotations

import json

__all__ = [
    "detect_quant_format",
    "normalize_quant_checkpoint",
    "is_quantized_checkpoint",
]


def detect_quant_format(
    state_dict: dict,
    prefix: str = "",
    *,
    metadata: dict | None = None,
) -> str | None:
    """Detect which quantized checkpoint format is present.

    Returns 'scaled_fp8_v1', 'comfy_quant_v2', 'metadata_v3', or None.
    """
    # Format A: LTX 2.3 scaled_fp8 marker
    marker_key = f"{prefix}scaled_fp8"
    if marker_key in state_dict:
        return "scaled_fp8_v1"

    # Format B: ComfyUI .comfy_quant keys
    comfy_suffix = ".comfy_quant"
    for key in state_dict:
        if key.startswith(prefix) and key.endswith(comfy_suffix):
            return "comfy_quant_v2"

    # Format C: safetensors file metadata
    if metadata and "_quantization_metadata" in metadata:
        return "metadata_v3"

    return None


def is_quantized_checkpoint(
    state_dict: dict,
    prefix: str = "",
    *,
    metadata: dict | None = None,
) -> bool:
    """Quick check — returns True if any supported quant format detected."""
    return detect_quant_format(state_dict, prefix, metadata=metadata) is not None


def normalize_quant_checkpoint(
    state_dict: dict,
    prefix: str = "",
    metadata: dict | None = None,
) -> dict:
    """Normalize any supported quantized checkpoint to canonical form.

    Returns a new dict — does not mutate input.
    """
    fmt = detect_quant_format(state_dict, prefix, metadata=metadata)

    if fmt == "scaled_fp8_v1":
        return _normalize_scaled_fp8(state_dict, prefix)
    elif fmt == "comfy_quant_v2":
        return _normalize_comfy_quant(state_dict, prefix)
    elif fmt == "metadata_v3":
        return _normalize_metadata(state_dict, prefix, metadata)
    else:
        # No quant format — passthrough copy
        return dict(state_dict)


# ---------------------------------------------------------------------------
# Format A — LTX 2.3 scaled_fp8
# ---------------------------------------------------------------------------

def _normalize_scaled_fp8(state_dict: dict, prefix: str) -> dict:
    marker_key = f"{prefix}scaled_fp8"
    marker = state_dict[marker_key]
    full_precision = marker.nelement() == 2

    # Collect layers that have .scale_weight (these are the quantized layers)
    scale_weight_suffix = ".scale_weight"
    quant_layers: set[str] = set()
    for key in state_dict:
        if key.startswith(prefix) and key.endswith(scale_weight_suffix):
            layer = key[: -len(scale_weight_suffix)]
            quant_layers.add(layer)

    meta_json = json.dumps(
        {"format": "float8_e4m3fn", "full_precision": full_precision}
    )

    out: dict = {}
    for key, tensor in state_dict.items():
        # Drop the marker key
        if key == marker_key:
            continue

        # Rename .scale_weight → .weight_scale and inject metadata
        matched_layer = None
        for layer in quant_layers:
            if key.startswith(layer):
                matched_layer = layer
                break

        if matched_layer is not None:
            suffix = key[len(matched_layer):]

            if suffix == ".scale_weight":
                out[f"{matched_layer}.weight_scale"] = tensor
                continue

            if suffix == ".scale_input":
                # Drop if value == 1.0
                if tensor.nelement() == 1 and float(tensor) == 1.0:
                    continue
                out[f"{matched_layer}.input_scale"] = tensor
                continue

        # Everything else passes through
        out[key] = tensor

    # Inject .quant_meta for each quantized layer
    for layer in quant_layers:
        out[f"{layer}.quant_meta"] = meta_json

    return out


# ---------------------------------------------------------------------------
# Format B — ComfyUI new format
# ---------------------------------------------------------------------------

def _normalize_comfy_quant(state_dict: dict, prefix: str) -> dict:
    comfy_suffix = ".comfy_quant"

    # First pass: parse all .comfy_quant entries
    layer_meta: dict[str, str] = {}
    for key in state_dict:
        if key.startswith(prefix) and key.endswith(comfy_suffix):
            layer = key[: -len(comfy_suffix)]
            raw = state_dict[key]
            # uint8 tensor → decode as UTF-8 JSON
            json_str = bytes(raw.tolist()).decode("utf-8")
            parsed = json.loads(json_str)

            full_precision = parsed.get("full_precision_matrix_mult", False)
            meta_json = json.dumps(
                {"format": parsed["format"], "full_precision": bool(full_precision)}
            )
            layer_meta[layer] = meta_json

    # Second pass: copy everything except .comfy_quant keys
    out: dict = {}
    for key, tensor in state_dict.items():
        if key.endswith(comfy_suffix):
            continue
        out[key] = tensor

    # Inject .quant_meta
    for layer, meta_json in layer_meta.items():
        out[f"{layer}.quant_meta"] = meta_json

    return out


# ---------------------------------------------------------------------------
# Format C — safetensors file metadata
# ---------------------------------------------------------------------------

def _normalize_metadata(
    state_dict: dict, prefix: str, metadata: dict | None
) -> dict:
    out: dict = dict(state_dict)

    if not metadata or "_quantization_metadata" not in metadata:
        return out

    qmeta = json.loads(metadata["_quantization_metadata"])
    layers = qmeta.get("layers", {})

    for layer_name, layer_info in layers.items():
        fmt = layer_info.get("format", "unknown")
        full_precision = layer_info.get("full_precision_matrix_mult", False)
        meta_json = json.dumps(
            {"format": fmt, "full_precision": bool(full_precision)}
        )
        out[f"{layer_name}.quant_meta"] = meta_json

    return out
