# smb-utils

SMB Utils contains a set of helper functions for processing and integrating arbitrary biomedical modality with SMB's Foundational Model. It supports CT imaging, ECG (coming soon), and EHR data processing.

## Install

```bash
uv pip install "git+https://github.com/standardmodelbio/smb-utils"
```

## Usage

### EHR Data Processing

Process structured EHR data in MEDS format for model inference. The formatter groups events by code, shows values chronologically, and supports flexible category handling.

#### Basic Usage - Date Grouping

```python
import pandas as pd
from smb_utils import process_ehr_info

# Load MEDS format data
df = pd.read_parquet("patient_data.parquet")

# Format for a single patient with date grouping
text = process_ehr_info(
    df,
    subject_id="patient_123",
    code_column="code",
    category_column="table",  # OMOP-style category column
)

print(text)
# Output:
# Birth: 1980-05-15
#
# [2024-01-01]
# Hypertension
# Diabetes Type 2
# Glucose (mg/dL): 95.00, 102.00, 98.00
```

#### Time Bins

Organize events into time bins going backwards from an anchor date (most recent first):

```python
text = process_ehr_info(
    df,
    subject_id="patient_123",
    end_time=pd.Timestamp("2024-01-01"),  # anchor point
    time_bins=[(90, 30), (30, 7), (7, 0)],  # days back: (start, end)
)

# Output shows bins with absolute dates (most recent first):
# 2023-12-25 - 2024-01-01
# Glucose (mg/dL): 98.00
#
# 2023-12-04 - 2023-12-25
# Blood Pressure (mmHg): 138.00
#
# 2023-10-02 - 2023-12-04
# Annual Wellness Visit
```

#### Flexible Category Handling

```python
# No category column - all events formatted together
text = process_ehr_info(df, category_column=None)

# Custom category column name
text = process_ehr_info(df, category_column="event_type")

# Custom category mapping
custom_mapping = {
    "vitals": ["bp", "hr", "temp"],
    "labs": ["glucose", "hba1c", "creatinine"],
    "diagnoses": ["condition", "diagnosis"],
}
text = process_ehr_info(
    df,
    category_column="type",
    category_mapping=custom_mapping,
)
```

#### With Imaging Data

```python
# Include imaging paths for multimodal models
# <image> tokens are placed inline at the timepoint where imaging occurs
result = process_ehr_info(
    df,
    subject_id="patient_123",
    include_imaging=True
)

# Returns dict with text and image paths
print(result["text"])
# Output shows <image> at the event with imaging:
# [2024-01-20]
# <image>CT Scan Chest
# ...

print(result["images"]) # ['/data/scans/ct_001.nii.gz']
```

#### Parameters

- `df`: DataFrame in MEDS format
- `subject_id`: Patient ID to filter (optional)
- `code_column`: Column containing event codes (default: "code")
- `category_column`: Column for categorizing events (default: "table"). Set to `None` if no category column exists
- `category_mapping`: Custom mapping from group names to category values (optional)
- `start_time`: Absolute lower bound for events (optional)
- `end_time`: Upper bound or anchor point for time bins (optional)
- `include_demographics`: Include demographics section (default: True)
- `time_bins`: List of (days_back_start, days_back_end) tuples (optional)
- `include_imaging`: Return dict with text and imaging paths (default: False)

See [examples/ehr_example.py](examples/ehr_example.py) for more detailed examples.


### SMB-Vision Series

```python
from smb_utils import process_mm_info
from transformers import AutoModel


# Prepare message spec for your volume(s). Each "image" can be a path to NIfTI/DICOM.
messages = [
    {
        "content": [
            {"type": "image", "image": "dummy.nii.gz"}, # Volume size is [1, 64, 160, 160]
            {"type": "image", "image": "dummy.nii.gz"},
        ]
    }
]

# Convert to patch tokens and grid descriptor expected by SMB‑Vision
# Default patch size is 16 for all dimensions
images, grid_thw = process_mm_info(messages) # images size is [800(400*2), 4096]

# Optional - Dummy images and grid_thw
images, grid_thw = torch.randn(800, 4096), torch.tensor([[4, 10, 10], [4, 10, 10]])

# Load backbone from HF Hub (uses this repo's modeling with trust_remote_code)
model = AutoModel.from_pretrained(
    "standardmodelbio/smb-vision-v0",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.to("cuda")

# Encode features
encoded_patches, deepstack_features = model.forward_features(
    images.to("cuda"), grid_thw=grid_thw.to("cuda")
)
print(encoded_patches.shape)
# (800, 1152)
```
