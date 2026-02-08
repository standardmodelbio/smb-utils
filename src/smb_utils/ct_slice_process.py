#! /usr/bin/env python3
#
# Copyright © 2026 Standard Model Biomedicine, Inc. <zach@standardmodel.bio>
#
# Distributed under terms of the Apache License 2.0 license.
"""
CT Slice Processing for MedGemma-style Models

This module provides utilities for processing CT imaging data as a series
of 2D RGB slices with multi-window Hounsfield unit windowing, following
the MedGemma 1.5 CT processing pipeline.

Unlike the 3D volume-based approach in imaging_process.py (for VJEPA2-style
models), this module represents CTs as ordered collections of 2D windowed
slices suitable for vision-language models like MedGemma.

The RGB channels encode three simultaneous CT window representations:
  - Red:   Wide window       (-1024, 1024 HU)
  - Green: Mediastinum/Chest (-135, 215 HU)
  - Blue:  Brain             (0, 80 HU)

Example usage:
    from smb_utils.ct_slice_process import (
        process_ct_to_slices,
        build_ct_slice_messages,
        fetch_ct_slices,
        process_ct_slice_info,
    )

    # Process a CT from R2 into windowed PIL Images
    images = process_ct_to_slices("r2://bucket/series.tar")

    # Build MedGemma-style messages
    messages = build_ct_slice_messages(
        images,
        instruction="Analyze this CT scan.",
        query="Are there any abnormalities?",
    )
"""

import base64
import io
import os
import tempfile

import numpy as np
import PIL.Image

from .imaging_process import download_medical_image


# ============================================================
# Constants
# ============================================================

MAX_SLICES = 85

# CT windowing definitions (min_hu, max_hu)
# These match the MedGemma 1.5 training windows.
CT_WINDOW_WIDE = (-1024, 1024)
CT_WINDOW_MEDIASTINUM = (-135, 215)
CT_WINDOW_BRAIN = (0, 80)

DEFAULT_CT_WINDOWS: list[tuple[float, float]] = [
    CT_WINDOW_WIDE,
    CT_WINDOW_MEDIASTINUM,
    CT_WINDOW_BRAIN,
]


# ============================================================
# Core windowing functions
# ============================================================


def normalize_window(
    ct_slice: np.ndarray,
    min_hu: float,
    max_hu: float,
) -> np.ndarray:
    """Window and normalize CT Hounsfield values to 0-255.

    Args:
        ct_slice: 2D array of Hounsfield unit values.
        min_hu: Lower bound of the window.
        max_hu: Upper bound of the window.

    Returns:
        2D array with values scaled to 0-255 (float32).
    """
    ct_slice = np.clip(ct_slice, min_hu, max_hu).astype(np.float32)
    ct_slice -= min_hu
    ct_slice /= max_hu - min_hu
    ct_slice *= 255.0
    return ct_slice


def apply_ct_windowing(
    ct_slice: np.ndarray,
    windows: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Apply multi-window CT windowing to create an RGB representation.

    Each window is mapped to one channel of the output image.  By default
    the three MedGemma windows are used (wide, mediastinum, brain).

    Args:
        ct_slice: 2D array of Hounsfield unit values.
        windows: List of (min_hu, max_hu) tuples, one per channel.

    Returns:
        3D uint8 array with shape (H, W, 3) suitable for PIL Image creation.
    """
    if windows is None:
        windows = DEFAULT_CT_WINDOWS

    channels = [normalize_window(ct_slice, w[0], w[1]) for w in windows]
    stacked = np.stack(channels, axis=-1)
    return np.round(stacked, 0).astype(np.uint8)


# ============================================================
# DICOM / NIfTI loading
# ============================================================


def _require_pydicom():
    try:
        import pydicom

        return pydicom
    except ImportError as e:
        raise ImportError(
            "pydicom is required for DICOM loading. "
            "Install with: pip install pydicom"
        ) from e


def _require_nibabel():
    try:
        import nibabel as nib

        return nib
    except ImportError as e:
        raise ImportError(
            "nibabel is required for NIfTI loading. "
            "Install with: pip install nibabel"
        ) from e


def load_dicom_series(series_dir: str) -> list[np.ndarray]:
    """Load and sort DICOM slices from a directory, converting to HU.

    Reads all DICOM files in *series_dir*, sorts them by InstanceNumber,
    applies the rescale slope/intercept to convert pixel values to
    Hounsfield units.

    Args:
        series_dir: Path to directory containing DICOM files.

    Returns:
        List of 2D numpy arrays (one per slice) in Hounsfield units,
        ordered by InstanceNumber.

    Raises:
        ValueError: If no valid DICOM files are found.
    """
    pydicom = _require_pydicom()

    dicom_files: list = []
    for fname in os.listdir(series_dir):
        filepath = os.path.join(series_dir, fname)
        if not os.path.isfile(filepath):
            continue
        try:
            dcm = pydicom.dcmread(filepath)
            # Only keep files that have pixel data
            if hasattr(dcm, "pixel_array"):
                dicom_files.append(dcm)
        except Exception:
            continue

    if not dicom_files:
        raise ValueError(f"No valid DICOM files found in {series_dir}")

    # Sort by InstanceNumber (fallback to 0 if missing)
    dicom_files.sort(
        key=lambda d: (
            int(d.InstanceNumber) if hasattr(d, "InstanceNumber") else 0
        )
    )

    slices: list[np.ndarray] = []
    for dcm in dicom_files:
        hu_data = pydicom.pixels.apply_rescale(dcm.pixel_array, dcm)
        slices.append(hu_data)

    return slices


def load_nifti_slices(nifti_path: str) -> list[np.ndarray]:
    """Load axial slices from a NIfTI file as Hounsfield-unit arrays.

    Args:
        nifti_path: Path to a ``.nii`` or ``.nii.gz`` file.

    Returns:
        List of 2D numpy arrays (one per axial slice).
    """
    nib = _require_nibabel()

    img = nib.load(nifti_path)
    data = img.get_fdata()  # (H, W, D) or (H, W, D, ...)

    # Take first volume if 4D
    if data.ndim == 4:
        data = data[..., 0]

    # Extract axial slices along the last axis
    slices = [data[:, :, i] for i in range(data.shape[2])]
    return slices


# ============================================================
# Slice sampling
# ============================================================


def sample_slices(
    slices: list,
    max_slices: int = MAX_SLICES,
) -> tuple[list, list[int]]:
    """Uniformly sample slices from a CT volume.

    If the number of slices is already <= *max_slices*, all slices are
    returned unchanged.

    Args:
        slices: Ordered list of slices.
        max_slices: Maximum number of slices to keep.

    Returns:
        Tuple of (sampled_slices, original_indices) where
        *original_indices* are the 1-based positions in the full volume.
    """
    total = len(slices)
    if total <= max_slices:
        return slices, list(range(1, total + 1))

    indices = [
        int(round(i / max_slices * (total - 1)))
        for i in range(1, max_slices + 1)
    ]
    return [slices[i] for i in indices], [i + 1 for i in indices]


# ============================================================
# Encoding helpers
# ============================================================


def slice_to_pil(
    ct_slice_hu: np.ndarray,
    windows: list[tuple[float, float]] | None = None,
) -> PIL.Image.Image:
    """Convert a single HU slice to a windowed RGB PIL Image.

    Args:
        ct_slice_hu: 2D Hounsfield unit array.
        windows: Optional custom window definitions.

    Returns:
        RGB PIL Image.
    """
    windowed = apply_ct_windowing(ct_slice_hu, windows=windows)
    return PIL.Image.fromarray(windowed)


def encode_image_to_data_uri(
    image: PIL.Image.Image,
    fmt: str = "jpeg",
) -> str:
    """Encode a PIL Image as a ``data:`` URI with base64 payload.

    Args:
        image: PIL Image to encode.
        fmt: Image format (``jpeg`` or ``png``).

    Returns:
        Base64-encoded data URI string.
    """
    with io.BytesIO() as buf:
        image.save(buf, format=fmt)
        buf.seek(0)
        encoded = base64.b64encode(buf.getbuffer()).decode("utf-8")
    return f"data:image/{fmt};base64,{encoded}"


# ============================================================
# Main processing pipeline
# ============================================================


def process_ct_to_slices(
    image_path: str,
    max_slices: int = MAX_SLICES,
    windows: list[tuple[float, float]] | None = None,
) -> tuple[list[PIL.Image.Image], list[int], int]:
    """Process a CT series into windowed RGB PIL Images.

    Supports DICOM directories, NIfTI files, and R2/S3 paths (including
    ``.tar`` archives).  For remote paths the series is downloaded to a
    temporary directory which is cleaned up after loading.

    Args:
        image_path: Local path (directory or NIfTI) or ``r2://``/``s3://``
            remote path.
        max_slices: Maximum number of slices to sample.
        windows: Optional custom CT window definitions.

    Returns:
        Tuple of ``(images, slice_indices, total_slices)`` where
        *images* are RGB PIL Images, *slice_indices* are the 1-based
        positions in the original volume, and *total_slices* is the
        full slice count before sampling.
    """
    temp_dir_ctx = None
    local_path = image_path

    # Download from R2/S3 if needed
    if not os.path.exists(image_path):
        temp_dir_ctx = tempfile.TemporaryDirectory()
        local_path = download_medical_image(
            image_path, save_dir=temp_dir_ctx.name
        )

    try:
        # Load slices based on path type
        if os.path.isdir(local_path):
            hu_slices = load_dicom_series(local_path)
        elif local_path.endswith((".nii", ".nii.gz")):
            hu_slices = load_nifti_slices(local_path)
        else:
            raise ValueError(
                f"Unsupported image format: {local_path}. "
                "Expected a DICOM directory or NIfTI file."
            )

        total_slices = len(hu_slices)

        # Sample slices uniformly (preserving original indices)
        hu_slices, slice_indices = sample_slices(
            hu_slices, max_slices=max_slices
        )

        # Window and convert to PIL Images
        images = [slice_to_pil(s, windows=windows) for s in hu_slices]

    finally:
        if temp_dir_ctx:
            temp_dir_ctx.cleanup()

    return images, slice_indices, total_slices


def fetch_ct_slices(
    ele: dict,
) -> tuple[list[PIL.Image.Image], list[int], int]:
    """Convenience wrapper to process CT slices from a dict element.

    Looks for the image path in ``ele["image"]`` and optional overrides
    for ``max_slices`` and ``windows``.

    Args:
        ele: Dictionary with at least an ``"image"`` key pointing to a
            DICOM directory, NIfTI file, or ``r2://``/``s3://`` path.

    Returns:
        Tuple of ``(images, slice_indices, total_slices)`` where
        *images* are RGB PIL Images, *slice_indices* are the 1-based
        positions in the original volume, and *total_slices* is the
        full slice count before sampling.

    Raises:
        ValueError: If no valid image path is found.
    """
    image_path = ele.get("image")
    if not isinstance(image_path, str):
        raise ValueError(
            "fetch_ct_slices expects an 'image' key with a string path"
        )

    max_slices = int(ele.get("max_slices", MAX_SLICES))
    windows = ele.get("windows")

    return process_ct_to_slices(
        image_path=image_path,
        max_slices=max_slices,
        windows=windows,
    )


# ============================================================
# Conversation processing
# ============================================================


def _is_ct_image_path(ele: dict) -> bool:
    """Return ``True`` if *ele* is a CT image element with an unprocessed path.

    Already-encoded data URIs (``data:…``) are skipped so that the
    pipeline is idempotent.
    """
    path_val = ele.get("image")
    if not isinstance(path_val, str):
        return False
    if path_val.startswith("data:"):
        return False
    modality = ele.get("modality", "CT").upper()
    return modality == "CT"


def _expand_ct_element(
    ele: dict,
    encode_format: str = "jpeg",
) -> list[dict]:
    """Expand a single CT image element into encoded slice elements.

    Returns a list of alternating ``{"type": "image", …}`` and
    ``{"type": "text", …}`` dicts — one pair per sampled slice.
    """
    images, indices, total = fetch_ct_slices(ele)
    expanded: list[dict] = []
    for pos, image in enumerate(images):
        data_uri = encode_image_to_data_uri(image, fmt=encode_format)
        expanded.append({"type": "image", "image": data_uri})
        idx = indices[pos]
        expanded.append({"type": "text", "text": f"SLICE {idx}"})
    return expanded


def extract_ct_slice_info(
    conversations: list[dict] | list[list[dict]],
) -> list[dict]:
    """Extract CT slice imaging entries from a conversation structure.

    Looks for elements containing an ``"image"`` key whose ``"modality"``
    is ``"CT"`` (or unset, defaulting to CT).

    Args:
        conversations: List of message dicts, or list of conversations
            (each a list of message dicts).

    Returns:
        List of element dicts that should be processed as CT slices.
    """
    imaging_infos: list[dict] = []
    if not conversations:
        return imaging_infos

    if isinstance(conversations[0], dict):
        conversations = [conversations]

    for conversation in conversations:
        for message in conversation:
            if isinstance(message.get("content"), list):
                for ele in message["content"]:
                    if _is_ct_image_path(ele):
                        imaging_infos.append(ele)

    return imaging_infos


def process_ct_slice_info(
    conversations: list[dict] | list[list[dict]],
    encode_format: str = "jpeg",
) -> list[dict] | list[list[dict]]:
    """Process CT image paths in a conversation into encoded slices.

    Walks through the conversation structure and replaces every CT
    image element (identified by an ``"image"`` key with a file /
    remote path) with the corresponding windowed, base64-encoded slice
    images interleaved with ``SLICE N/TOTAL`` labels.  All other
    elements and messages are kept unchanged.

    The return value has the same shape as the input: a single
    conversation (``list[dict]``) or a batch (``list[list[dict]]``).

    Args:
        conversations: A single conversation (list of message dicts) or
            a batch of conversations (list of lists).
        encode_format: Image encoding format (``jpeg`` or ``png``).

    Returns:
        The conversation(s) with CT image paths replaced by encoded
        slice images.
    """
    if not conversations:
        return conversations

    single = isinstance(conversations[0], dict)
    if single:
        conversations = [conversations]

    result: list[list[dict]] = []
    for conversation in conversations:
        new_conversation: list[dict] = []
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                # Text-only or non-list content — keep as-is
                new_conversation.append(message)
                continue

            new_content: list[dict] = []
            for ele in content:
                if _is_ct_image_path(ele):
                    new_content.extend(_expand_ct_element(ele, encode_format))
                else:
                    new_content.append(ele)

            new_conversation.append({**message, "content": new_content})
        result.append(new_conversation)

    return result[0] if single else result
