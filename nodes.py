"""ComfyUI nodes for Image to Layers (https://www.imagetolayers.com)."""
from __future__ import annotations

import io
import json
from typing import Any, List

import numpy as np
import torch
from PIL import Image

from .client import ApiError, Client, SignInRequired
from .layout import compose_layers, sort_bottom_to_top

CATEGORY = "ImageToLayers"
_client = Client()


def _to_png_bytes(image: torch.Tensor) -> bytes:
    """First image of a ComfyUI IMAGE batch (B, H, W, C floats 0..1) as PNG bytes."""
    array = np.clip(255.0 * image[0].cpu().numpy(), 0, 255).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(array).save(out, format="PNG")
    return out.getvalue()


def _ensure_signed_in() -> None:
    if _client.has_session():
        return
    pending = _client.start_sign_in()
    raise RuntimeError(
        "Sign in to Image to Layers to continue:\n"
        f"1. Open {pending['url']}\n"
        f"2. Check that the code is {pending['userCode']} and approve.\n"
        "3. Queue the workflow again (this node picks the approval up by itself).\n"
        "The sign-in stays on this computer; no password or key is stored."
    )


class ImageToLayersDetailed:
    """Detailed Layers: separates one image into editable layers, the same mode as the website and plugins."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "parts_to_separate": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional. Name the parts you want as their own layers, e.g. 'bottle, leaves, podium'. Leave empty to detect everything."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("layers", "masks", "layer_names", "job_id")
    OUTPUT_IS_LIST = (False, False, False, False)
    FUNCTION = "separate"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Separates an image into editable layers with Image to Layers. Returns the layers bottom to top as an image batch "
        "(all the size of the background layer), their alpha as a mask batch, and their names. Uses credits from your account."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # a paid remote job: never serve a cached result silently

    def separate(self, image: torch.Tensor, parts_to_separate: str = ""):
        _ensure_signed_in()
        try:
            file_key = _client.upload_image(_to_png_bytes(image))
            job_id = _client.start_job(file_key, (parts_to_separate or "").strip() or None)
            layers = _client.wait_for_layers(job_id)
            if not layers:
                raise ApiError("No layers were found in this image.")
            ordered = sort_bottom_to_top(layers)
            blobs = [_client.download_layer(job_id, layer["id"]) for layer in ordered]
        except SignInRequired:
            _ensure_signed_in()
            raise
        arrays, names = compose_layers(ordered, blobs)
        stack = np.stack(arrays)  # (N, H, W, 4)
        rgb = torch.from_numpy(stack[..., :3].copy())
        alpha = torch.from_numpy(stack[..., 3].copy())
        return (rgb, alpha, json.dumps(names, ensure_ascii=False), job_id)


class ImageToLayersPick:
    """Takes one layer out of the batch and returns it as an image with its alpha, ready for Preview/Save nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers": ("IMAGE",),
                "masks": ("MASK",),
                "index": ("INT", {"default": 0, "min": 0, "max": 255, "tooltip": "0 is the background; higher numbers are layers closer to the top."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("layer", "mask")
    FUNCTION = "pick"
    CATEGORY = CATEGORY

    def pick(self, layers: torch.Tensor, masks: torch.Tensor, index: int):
        i = max(0, min(int(index), layers.shape[0] - 1))
        return (layers[i : i + 1], masks[i : i + 1])


class ImageToLayersAccount:
    """Shows who is signed in and how many credits are left; also signs out."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sign_out": ("BOOLEAN", {"default": False, "label_on": "Sign out", "label_off": "Keep signed in"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "status"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def status(self, sign_out: bool):
        if sign_out:
            _client.sign_out()
            text = "Signed out."
        else:
            _ensure_signed_in()
            remaining = _client.credits()
            who = _client.email() or "your account"
            text = f"Signed in as {who}." + (f" Credits left: {remaining}." if remaining is not None else "")
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "ImageToLayersDetailed": ImageToLayersDetailed,
    "ImageToLayersPick": ImageToLayersPick,
    "ImageToLayersAccount": ImageToLayersAccount,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToLayersDetailed": "Image to Layers (Detailed)",
    "ImageToLayersPick": "Image to Layers: Pick Layer",
    "ImageToLayersAccount": "Image to Layers: Account",
}
