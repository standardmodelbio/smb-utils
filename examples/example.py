import torch
from transformers import AutoModel

from smb_utils import process_mm_info


if __name__ == "__main__":
    # Prepare message spec for your volume(s). Each "image" can be a path to NIfTI/DICOM.
    messages = [
        {
            "content": [
                {
                    "type": "image",
                    "image": "dummy.nii.gz",
                },  # Volume size is [1, 64, 160, 160]
                {"type": "image", "image": "dummy.nii.gz"},
            ]
        }
    ]

    # Convert to patch tokens and grid descriptor expected by SMB‑Vision
    # Default patch size is 16 for all dimensions
    images, grid_thw = process_mm_info(
        messages
    )  # images size is [800(400*2), 4096]

    # Optional - Dummy images and grid_thw
    images, grid_thw = (
        torch.randn(800, 4096),
        torch.tensor([[4, 10, 10], [4, 10, 10]]),
    )

    # Load backbone from HF Hub (uses this repo's modeling with trust_remote_code)
    model = AutoModel.from_pretrained(
        "standardmodelbio/smb-vision-v0",
        trust_remote_code=True,  # must be True
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # optional
    )
    model.to("cuda")

    # Encode features
    encoded_patches, deepstack_features = model.forward_features(
        images.to("cuda"), grid_thw=grid_thw.to("cuda")
    )
    print(encoded_patches.shape)
    # (800, 1152)
