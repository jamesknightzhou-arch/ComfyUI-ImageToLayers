"""Places the separated layers back on one canvas. Pure Pillow/numpy, no ComfyUI or torch.

Detailed Layers describes each layer by a box on the canvas, in the pixels of the background plate (the layer
without a box). The background plate is the canvas.
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

Box = Tuple[int, int, int, int]  # x, y, width, height


def layer_box(layer: Dict[str, Any]) -> Optional[Box]:
    """The layer's box, or None for the background plate (it covers the canvas)."""
    bbox = layer.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        x, y, w, h = (float(bbox[k]) for k in ("x", "y", "width", "height"))
        if w > 0 and h > 0:
            return int(round(x)), int(round(y)), int(round(w)), int(round(h))
    except (KeyError, TypeError, ValueError):
        pass
    absolute = bbox.get("absolute")
    if isinstance(absolute, (list, tuple)) and len(absolute) == 4:
        try:
            x1, y1, x2, y2 = (float(v) for v in absolute)
            if x2 > x1 and y2 > y1:
                return int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))
        except (TypeError, ValueError):
            pass
    return None


def sort_bottom_to_top(layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """By the provider's z-index (0 is the background). Without one, the server lists the top layer first."""
    def z(item: Tuple[int, Dict[str, Any]]) -> float:
        index, layer = item
        bbox = layer.get("bbox")
        try:
            return float(bbox["zIndex"]) if isinstance(bbox, dict) and "zIndex" in bbox else float(len(layers) - 1 - index)
        except (TypeError, ValueError):
            return float(len(layers) - 1 - index)

    return [layer for _, layer in sorted(enumerate(layers), key=lambda item: z(item))]


def fit_within(box: Box, size: Tuple[int, int]) -> Box:
    """The largest rectangle with the image's proportions that fits inside the box, centred in it."""
    bx, by, bw, bh = box
    iw, ih = size
    if iw <= 0 or ih <= 0:
        return box
    if iw / ih > bw / bh:
        w, h = bw, max(1, round(bw * ih / iw))
    else:
        w, h = max(1, round(bh * iw / ih)), bh
    return bx + (bw - w) // 2, by + (bh - h) // 2, w, h


def compose_layers(layers: List[Dict[str, Any]], blobs: List[bytes]) -> Tuple[List[np.ndarray], List[str]]:
    """RGBA float arrays (0..1), all of the canvas size, bottom to top, plus their names.

    `layers` and `blobs` are index-aligned and already sorted bottom to top.
    """
    images = [Image.open(io.BytesIO(blob)).convert("RGBA") for blob in blobs]
    boxes = [layer_box(layer) for layer in layers]

    plate = next((i for i, box in enumerate(boxes) if box is None), None)
    if plate is not None:
        canvas = images[plate].size
    else:
        right = max((x + w for x, _, w, _ in (b for b in boxes if b)), default=0)
        bottom = max((y + h for _, y, _, h in (b for b in boxes if b)), default=0)
        if right <= 0 or bottom <= 0:
            raise ValueError("These layers have no usable size.")
        canvas = (right, bottom)

    arrays: List[np.ndarray] = []
    names: List[str] = []
    for index, (layer, image, box) in enumerate(zip(layers, images, boxes)):
        full = Image.new("RGBA", canvas, (0, 0, 0, 0))
        if box is None:
            full = image if image.size == canvas else image.resize(canvas, Image.LANCZOS)
        else:
            x, y, w, h = fit_within(box, image.size)
            placed = image if image.size == (w, h) else image.resize((w, h), Image.LANCZOS)
            full.alpha_composite(placed, (max(0, x), max(0, y)), (max(0, -x), max(0, -y), placed.width, placed.height))
        arrays.append(np.asarray(full).astype(np.float32) / 255.0)
        name = layer.get("name")
        names.append(name.strip() if isinstance(name, str) and name.strip() else f"Layer {index + 1}")
    return arrays, names
