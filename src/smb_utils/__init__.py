#! /usr/bin/env python3
#
# Copyright © 2026 Standard Model Biomedicine, Inc. <zach@standardmodel.bio>
#
# Distributed under terms of the Apache License 2.0 license.

# Core EHR functionality — always available.
from .ehr_process import process_ehr_info

# Imaging functionality — requires optional `imaging` extras:
#   pip install smb-utils[imaging]
try:
    from .ct_slice_process import (
        extract_ct_slice_info,
        fetch_ct_slices,
        process_ct_slice_info,
        process_ct_to_slices,
    )
    from .imaging_process import (
        extract_imaging_info,
        fetch_medical_volume,
        process_imaging_info,
    )
    from .imaging_text_pairs import create_imaging_text_pairs

    # Alias for backward compatibility
    process_mm_info = process_imaging_info
except ImportError:
    pass

__all__ = [
    "process_ehr_info",
    # Imaging (available when installed with [imaging] extra)
    "fetch_medical_volume",
    "extract_imaging_info",
    "process_imaging_info",
    "process_mm_info",
    "create_imaging_text_pairs",
    "process_ct_to_slices",
    "fetch_ct_slices",
    "extract_ct_slice_info",
    "process_ct_slice_info",
]
