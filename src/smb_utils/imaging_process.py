#! /usr/bin/env python3
#
# Copyright © 2026 Standard Model Biomedicine, Inc. <zach@standardmodel.bio>
#
# Distributed under terms of the Apache License 2.0 license.
import os
import tarfile
import tempfile

import boto3
import torch
from botocore.config import Config
from dotenv import load_dotenv


load_dotenv()

# CONFIGURATION
MAX_WORKERS = 50  # Increased since we are IO-bound
TIMEOUT = 5  # Seconds to wait for a HEAD request

# ==========================
# Medical imaging utilities
# ==========================

# CT Defines
CT_DEFAULT_SPATIAL_SIZE = (416, 416, 192)
CT_DEFAULT_PIXDIM = (1.0, 1.0, 2.0)
CT_DEFAULT_A_MIN = -1000
CT_DEFAULT_A_MAX = 1000
CT_DEFAULT_B_MIN = 0.0
CT_DEFAULT_B_MAX = 1.0

# MRI Defines (Percentile scaling default)
MRI_DEFAULT_SPATIAL_SIZE = (416, 416, 192)
MRI_DEFAULT_PIXDIM = (1.0, 1.0, 2.0)
MRI_DEFAULT_A_MIN = 0.0
MRI_DEFAULT_A_MAX = 0.0  # Dynamic (placeholder)

# PET Defines
# SUV typically 0-10 or 0-20, but can be higher.
PET_DEFAULT_SPATIAL_SIZE = (384, 384, 192)
PET_DEFAULT_PIXDIM = (2.0, 2.0, 3.0)  # Lower res usually
PET_DEFAULT_A_MIN = 0.0
PET_DEFAULT_A_MAX = 15.0  # Reasonable SUV max default
PET_DEFAULT_B_MIN = 0.0
PET_DEFAULT_B_MAX = 1.0

# X-Ray Defines (2D)
XRAY_DEFAULT_SPATIAL_SIZE = (768, 768, -1)
XRAY_DEFAULT_PIXDIM = (0.5, 0.5, -1.0)  # High res in plane
XRAY_DEFAULT_A_MIN = 0.0
XRAY_DEFAULT_A_MAX = (
    255.0  # Assuming 8-bit usually, but handled dynamically preferred
)
XRAY_DEFAULT_B_MIN = 0.0
XRAY_DEFAULT_B_MAX = 1.0

# Ultrasound Defines
US_DEFAULT_SPATIAL_SIZE = (768, 768, -1)  # Often 2D, sometimes 3D
US_DEFAULT_PIXDIM = (0.5, 0.5, -1.0)
US_DEFAULT_A_MIN = 0.0
US_DEFAULT_A_MAX = 255.0
US_DEFAULT_B_MIN = 0.0
US_DEFAULT_B_MAX = 1.0

DEPTH_PATCH_SIZE = 16
PATCH_SIZE = 16


def _require_monai():
    try:
        import monai.transforms

        return monai.transforms
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MONAI is required for medical imaging preprocessing. Install with: pip install monai nibabel"
        ) from e


def _get_base_transforms(
    spatial_size: tuple[int, int, int],
    patch_size: int,
    intensity_transform,
    pixdim: tuple[float, float, float] | None = None,
    use_3d_orientation: bool = True,
    resize_mode: str = "crop",  # "crop" or "resize"
    reader: str = "ITKReader",
):
    """Refactored base transform builder."""
    t = _require_monai()

    class PermuteImage(t.MapTransform):
        def __init__(self, keys=["image"], allow_missing_keys=False):
            super().__init__(keys, allow_missing_keys)

        def __call__(self, data):
            # ensure the image only has one channel
            if data["image"].shape[0] != 1:
                data["image"] = data["image"][0:1]
            data["image"] = data["image"].permute(0, 3, 1, 2)  # (C, D, H, W)
            return data

    transforms_list = [
        t.LoadImaged(keys=["image"], reader=reader, ensure_channel_first=True),
    ]

    # if use_3d_orientation:
    #     transforms_list.append(t.Orientationd(keys=["image"], axcodes="RAS"))

    if pixdim:
        transforms_list.append(
            t.Spacingd(
                keys=["image"],
                pixdim=pixdim,
                mode=("bilinear"),
                min_pixdim=pixdim,
            )
        )

    # Add specific intensity transform
    if intensity_transform:
        transforms_list.append(intensity_transform)

    # Spatial sizing
    if resize_mode == "crop":
        transforms_list.append(
            t.CenterSpatialCropd(keys=["image"], roi_size=list(spatial_size))
        )
    elif resize_mode == "resize":
        transforms_list.append(
            t.ResizeWithPadOrCropd(
                keys=["image"], spatial_size=list(spatial_size)
            )
        )

    transforms_list.append(t.DivisiblePadd(keys=["image"], k=patch_size * 2))
    transforms_list.append(t.ToTensord(keys=["image"], track_meta=False))
    transforms_list.append(PermuteImage())

    return t.Compose(transforms_list)


def get_ct_transforms(
    spatial_size: tuple[int, int, int] = CT_DEFAULT_SPATIAL_SIZE,
    pixdim: tuple[float, float, float] = CT_DEFAULT_PIXDIM,
    patch_size: int = PATCH_SIZE,
    a_min: float = CT_DEFAULT_A_MIN,
    a_max: float = CT_DEFAULT_A_MAX,
    b_min: float = CT_DEFAULT_B_MIN,
    b_max: float = CT_DEFAULT_B_MAX,
    reader: str = "ITKReader",
):
    """Create a MONAI Compose for CT NIfTI preprocessing.

    Returns a transform that produces a tensor with shape (C, D, H, W), where C=1.
    """
    t = _require_monai()
    return _get_base_transforms(
        spatial_size=spatial_size,
        patch_size=patch_size,
        intensity_transform=t.ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            clip=True,
        ),
        pixdim=pixdim,
        use_3d_orientation=True,
        resize_mode="crop",
        reader=reader,
    )


def get_mri_transforms(
    spatial_size: tuple[int, int, int] = MRI_DEFAULT_SPATIAL_SIZE,
    pixdim: tuple[float, float, float] = MRI_DEFAULT_PIXDIM,
    patch_size: int = PATCH_SIZE,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    b_min: float = 0.0,
    b_max: float = 1.0,
    reader: str = "ITKReader",
):
    """Create a MONAI Compose for MRI preprocessing."""
    t = _require_monai()
    return _get_base_transforms(
        spatial_size=spatial_size,
        patch_size=patch_size,
        intensity_transform=t.ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=lower_percentile,
            upper=upper_percentile,
            b_min=b_min,
            b_max=b_max,
            clip=True,
            relative=False,
        ),
        pixdim=pixdim,
        use_3d_orientation=True,
        resize_mode="crop",
        reader=reader,
    )


def get_pet_transforms(
    spatial_size: tuple[int, int, int] = PET_DEFAULT_SPATIAL_SIZE,
    pixdim: tuple[float, float, float] = PET_DEFAULT_PIXDIM,
    patch_size: int = PATCH_SIZE,
    a_min: float = PET_DEFAULT_A_MIN,
    a_max: float = PET_DEFAULT_A_MAX,
    b_min: float = PET_DEFAULT_B_MIN,
    b_max: float = PET_DEFAULT_B_MAX,
    reader: str = "ITKReader",
):
    """Create a MONAI Compose for PET preprocessing (SUV scaling)."""
    t = _require_monai()
    return _get_base_transforms(
        spatial_size=spatial_size,
        patch_size=patch_size,
        intensity_transform=t.ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            clip=True,
        ),
        pixdim=pixdim,
        use_3d_orientation=True,
        resize_mode="crop",
        reader=reader,
    )


def get_xray_transforms(
    spatial_size: tuple[
        int, int
    ] = XRAY_DEFAULT_SPATIAL_SIZE,  # Changed to 2-tuple for XRAY
    pixdim: tuple[
        float, float
    ] = XRAY_DEFAULT_PIXDIM,  # Changed to 2-tuple for XRAY
    patch_size: int = PATCH_SIZE,
    a_min: float = XRAY_DEFAULT_A_MIN,
    a_max: float = XRAY_DEFAULT_A_MAX,
    b_min: float = XRAY_DEFAULT_B_MIN,
    b_max: float = XRAY_DEFAULT_B_MAX,
    reader: str = "ITKReader",
):
    """Create a MONAI Compose for X-ray preprocessing (2D)."""
    t = _require_monai()
    return _get_base_transforms(
        spatial_size=spatial_size,
        patch_size=patch_size,
        intensity_transform=t.ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            clip=True,
        ),
        pixdim=pixdim,
        # X-rays are often 2D so we skip 3D orientation/resampling to avoid errors
        # unless we are sure metadata exists. Since it's often jpeg/png converted or 2D DICOM,
        # safer to skip.
        use_3d_orientation=False,
        resize_mode="resize",  # Resize for Xrays typically
        reader=reader,
    )


def get_ultrasound_transforms(
    spatial_size: tuple[
        int, int
    ] = US_DEFAULT_SPATIAL_SIZE,  # Changed to 2-tuple for US
    pixdim: tuple[
        float, float
    ] = US_DEFAULT_PIXDIM,  # Changed to 2-tuple for US
    patch_size: int = PATCH_SIZE,
    a_min: float = US_DEFAULT_A_MIN,
    a_max: float = US_DEFAULT_A_MAX,
    b_min: float = US_DEFAULT_B_MIN,
    b_max: float = US_DEFAULT_B_MAX,
    reader: str = "ITKReader",
):
    """Create a MONAI Compose for Ultrasound preprocessing."""
    # Reuse X-ray logic
    return get_xray_transforms(
        spatial_size=spatial_size,
        pixdim=pixdim,
        patch_size=patch_size,
        a_min=a_min,
        a_max=a_max,
        b_min=b_min,
        b_max=b_max,
        reader=reader,
    )


def get_s3_client():
    # Fix: Increase connection pool size to handle concurrent threads
    config = Config(
        max_pool_connections=MAX_WORKERS + 10,
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=5,
        read_timeout=5,
    )

    return boto3.client(
        service_name="s3",
        endpoint_url=os.getenv("ENDPOINT_URL"),
        aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        region_name="auto",
        config=config,
    )


def download_medical_image(image_path: str, save_dir: str = "./tmp/") -> str:
    """Download a medical image file from a s3 or r2 object to local temp directory."""
    # get the bucket and object name from the nifti path
    # Parse S3/R2 URL to extract bucket and object key
    if image_path.startswith("s3://") or image_path.startswith("r2://"):
        # Remove s3:// prefix and split into bucket and object key
        path_without_prefix = image_path[5:]  # Remove "s3://"
        bucket, object_name = path_without_prefix.split("/", 1)
    else:
        # Assume it's a local file if not s3/r2 or raises separate error?
        # For now, keeping existing logic that raises if it claims to be s3 path but isn't?
        # Actually original just checked startswith and elsed into raise.
        raise ValueError(f"Invalid S3/R2 path format: {image_path}")

    # create s3 client
    s3 = get_s3_client()

    # create save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    local_path = os.path.join(save_dir, os.path.basename(object_name))
    s3.download_file(bucket, object_name, local_path)

    if not local_path.endswith((".tar", ".tar.gz", ".tgz")):
        return local_path

    # extract the file
    extract_dir = local_path
    for ext in [".tar.gz", ".tgz", ".tar"]:
        if local_path.endswith(ext):
            extract_dir = local_path[: -len(ext)]
            break

    with tarfile.open(local_path) as f:
        f.extractall(extract_dir, filter="data")

    # check if there's 'instances' subfolder, if yes, add this to path
    if os.path.isdir(os.path.join(extract_dir, "instances")):
        extract_dir = os.path.join(extract_dir, "instances")

    return extract_dir


# Modality configurations: (transform_fn, spatial_size, pixdim, a_min, a_max, b_min, b_max, uses_intensity_range)
_MODALITY_CONFIGS = {
    "CT": (
        get_ct_transforms,
        CT_DEFAULT_SPATIAL_SIZE,
        CT_DEFAULT_PIXDIM,
        CT_DEFAULT_A_MIN,
        CT_DEFAULT_A_MAX,
        CT_DEFAULT_B_MIN,
        CT_DEFAULT_B_MAX,
        True,
    ),
    "MRI": (
        get_mri_transforms,
        MRI_DEFAULT_SPATIAL_SIZE,
        MRI_DEFAULT_PIXDIM,
        None,
        None,
        None,
        None,
        False,
    ),
    "PET": (
        get_pet_transforms,
        PET_DEFAULT_SPATIAL_SIZE,
        PET_DEFAULT_PIXDIM,
        PET_DEFAULT_A_MIN,
        PET_DEFAULT_A_MAX,
        PET_DEFAULT_B_MIN,
        PET_DEFAULT_B_MAX,
        True,
    ),
    "XRAY": (
        get_xray_transforms,
        XRAY_DEFAULT_SPATIAL_SIZE,
        XRAY_DEFAULT_PIXDIM,
        XRAY_DEFAULT_A_MIN,
        XRAY_DEFAULT_A_MAX,
        XRAY_DEFAULT_B_MIN,
        XRAY_DEFAULT_B_MAX,
        True,
    ),
    "ULTRASOUND": (
        get_ultrasound_transforms,
        US_DEFAULT_SPATIAL_SIZE,
        US_DEFAULT_PIXDIM,
        US_DEFAULT_A_MIN,
        US_DEFAULT_A_MAX,
        US_DEFAULT_B_MIN,
        US_DEFAULT_B_MAX,
        True,
    ),
    "US": (
        get_ultrasound_transforms,
        US_DEFAULT_SPATIAL_SIZE,
        US_DEFAULT_PIXDIM,
        US_DEFAULT_A_MIN,
        US_DEFAULT_A_MAX,
        US_DEFAULT_B_MIN,
        US_DEFAULT_B_MAX,
        True,
    ),
}


def preprocess_image(
    image_path: str,
    modality: str = "CT",
    spatial_size: tuple[int, int, int] | None = None,
    pixdim: tuple[float, float, float] | None = None,
    a_min: float | None = None,
    a_max: float | None = None,
    b_min: float | None = None,
    b_max: float | None = None,
    depth_patch_size: int = 16,
    patch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and preprocess a medical image file to VJEPA2-ready shape [1, C, D, H, W].

    Supported modalities: CT, MRI, PET, XRAY, ULTRASOUND.
    """
    modality = modality.upper()
    if modality not in _MODALITY_CONFIGS:
        raise ValueError(f"Unsupported modality: {modality}")

    reader = (
        "ITKReader"
        if os.path.isdir(image_path)
        else "NibabelReader"
        if image_path.endswith(".nii.gz")
        else "ITKReader"
    )

    (
        transform_fn,
        default_spatial,
        default_pixdim,
        default_a_min,
        default_a_max,
        default_b_min,
        default_b_max,
        uses_intensity,
    ) = _MODALITY_CONFIGS[modality]

    spatial_size = spatial_size or default_spatial
    pixdim = pixdim or default_pixdim

    if uses_intensity:
        a_min = a_min if a_min is not None else default_a_min
        a_max = a_max if a_max is not None else default_a_max
        b_min = b_min if b_min is not None else default_b_min
        b_max = b_max if b_max is not None else default_b_max
        transforms = transform_fn(
            spatial_size=spatial_size,
            pixdim=pixdim,
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            reader=reader,
        )
    else:
        # MRI uses percentile-based scaling, not intensity range
        transforms = transform_fn(
            spatial_size=spatial_size, pixdim=pixdim, reader=reader
        )

    data_dict = {"image": image_path}
    transformed = transforms(data_dict)
    volume_dc_hw = transformed["image"]  # (C, D, H, W)
    if volume_dc_hw.dim() != 4:
        raise ValueError(
            f"Expected 4D tensor (C, D, H, W), got shape {tuple(volume_dc_hw.shape)}"
        )

    # get grid_thw
    grid_thw = torch.tensor(
        [
            volume_dc_hw.shape[1] // depth_patch_size,
            volume_dc_hw.shape[2] // patch_size,
            volume_dc_hw.shape[3] // patch_size,
        ]
    )

    # 1. Chain unfold calls for a cleaner look
    patches = (
        volume_dc_hw.unfold(1, depth_patch_size, depth_patch_size)
        .unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size)
    )

    # 2. Permute to group grid dimensions and patch dimensions separately
    # Initial shape: (C, nD, nH, nW, d_p, p, p)
    # Target shape:  (nD, nH, nW, C, d_p, p, p)
    patches = patches.permute(1, 2, 3, 0, 4, 5, 6)

    # 3. Explicitly create a contiguous tensor, then flatten
    # This is the key optimization step.
    # The first three dimensions (nD, nH, nW) are flattened into `total_patches`.
    # The last four dimensions (C, d_p, p, p) are flattened into the feature dimension.
    patches = patches.contiguous().view(
        -1, volume_dc_hw.shape[0] * depth_patch_size * patch_size * patch_size
    )
    return patches, grid_thw
    # return volume_dc_hw, grid_thw


def fetch_medical_volume(ele: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper to preprocess medical volumes.

    Supported keys:
      - "nifti_path" or "image": path to image file
      - "modality": CT, MRI, PET, XRAY, ULTRASOUND (default: CT)
      - Optional overrides: spatial_size, pixdim, a_min, a_max, b_min, b_max
    """
    image_path = ele.get("nifti_path") or ele.get("image")
    if not isinstance(image_path, str):
        raise ValueError(
            "fetch_medical_volume expects 'nifti_path' or 'image' string path"
        )

    modality = ele.get("modality", "CT")

    # helper to parse tuple or None
    def get_tuple(key, default):
        val = ele.get(key)
        return tuple(val) if val else default

    spatial_size = get_tuple("spatial_size", None)
    pixdim = get_tuple("pixdim", None)

    a_min = ele.get("a_min")
    a_max = ele.get("a_max")
    b_min = ele.get("b_min")
    b_max = ele.get("b_max")

    # Cast to float only if they are not None
    a_min = float(a_min) if a_min is not None else None
    a_max = float(a_max) if a_max is not None else None
    b_min = float(b_min) if b_min is not None else None
    b_max = float(b_max) if b_max is not None else None

    depth_patch_size = int(ele.get("depth_patch_size", DEPTH_PATCH_SIZE))
    patch_size = int(ele.get("patch_size", PATCH_SIZE))

    # download file if it is not a local path
    temp_dir_context = None
    if not os.path.exists(image_path):
        temp_dir_context = tempfile.TemporaryDirectory()
        image_path = download_medical_image(image_path, temp_dir_context.name)

    try:
        volume_dc_hw, grid_thw = preprocess_image(
            image_path=image_path,
            modality=modality,
            spatial_size=spatial_size,
            pixdim=pixdim,
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            depth_patch_size=depth_patch_size,
            patch_size=patch_size,
        )
    finally:
        if temp_dir_context:
            temp_dir_context.cleanup()

    return volume_dc_hw, grid_thw


def extract_imaging_info(
    conversations: list[dict] | list[list[dict]],
) -> list[dict]:
    """Extract medical imaging entries from a conversation-like structure.

    Accepts either a list of dicts, or a list of list of dicts (messages with content).
    Looks for elements containing 'image' or 'nifti_path' pointing to a .nii/.nii.gz.
    """
    imaging_infos: list[dict] = []
    if not conversations:
        return imaging_infos
    if isinstance(conversations[0], dict):
        conversations = [conversations]  # type: ignore[assignment]
    for conversation in conversations:  # type: ignore[assignment]
        for message in conversation:
            if isinstance(message.get("content"), list):
                for ele in message["content"]:
                    path_val = ele.get("image") or ele.get("nifti_path")
                    if isinstance(path_val, str):
                        imaging_infos.append(ele)
    return imaging_infos


def process_imaging_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Process medical imaging info into preprocessed tensors.

    Returns a list of tensors with shape [1, C, D, H, W] or None if no entries.
    """
    imaging_infos = extract_imaging_info(conversations)
    volume_inputs: list[torch.Tensor] = []
    grid_thws: list[torch.Tensor] = []
    for info in imaging_infos:
        volume, grid_thw = fetch_medical_volume(info)
        volume_inputs.append(volume)
        grid_thws.append(grid_thw)
    if len(volume_inputs) == 0:
        return None
    return torch.cat(volume_inputs, dim=0), torch.stack(grid_thws)
