"""Example: Processing EHR data for model inference.

This example demonstrates how to use the EHR processing utilities to format
patient event timelines into structured text for language models.
"""

import pandas as pd

from smb_utils import process_ehr_info


def create_sample_data() -> pd.DataFrame:
    """Create sample MEDS-format EHR data for demonstration."""
    data = {
        "subject_id": ["patient_001"] * 12,
        "time": pd.to_datetime(
            [
                "1985-03-15",  # birth
                "2024-01-05",  # condition
                "2024-01-05",  # condition
                "2024-01-10",  # lab
                "2024-01-10",  # lab
                "2024-01-10",  # lab
                "2024-01-15",  # vital
                "2024-01-15",  # vital
                "2024-01-20",  # medication
                "2024-01-20",  # procedure
                "2024-01-25",  # lab
                "2024-01-25",  # lab
            ]
        ),
        "table": [
            "person",
            "condition",
            "condition",
            "lab",
            "lab",
            "lab",
            "measurement",
            "measurement",
            "drug_exposure",
            "procedure",
            "lab",
            "lab",
        ],
        "code": [
            "Birth",
            "Essential Hypertension",
            "Type 2 Diabetes Mellitus",
            "Glucose",
            "HbA1c",
            "Creatinine",
            "Blood Pressure Systolic",
            "Blood Pressure Diastolic",
            "Metformin 500mg",
            "Annual Wellness Visit",
            "Glucose",
            "HbA1c",
        ],
        "numeric_value": [
            None,
            None,
            None,
            126.0,
            7.2,
            1.1,
            138.0,
            88.0,
            None,
            None,
            118.0,
            6.9,
        ],
        "text_value": [None] * 12,
        "unit": [
            None,
            None,
            None,
            "mg/dL",
            "%",
            "mg/dL",
            "mmHg",
            "mmHg",
            None,
            None,
            "mg/dL",
            "%",
        ],
    }
    return pd.DataFrame(data)


def example_basic_usage():
    """Basic usage with date grouping."""
    print("=" * 60)
    print("Example 1: Basic Usage with Date Grouping")
    print("=" * 60)

    df = create_sample_data()

    text = process_ehr_info(
        df,
        subject_id="patient_001",
        code_column="code",
        category_column="table",
    )

    print(text)
    print()


def example_time_bins():
    """Organize events into time bins."""
    print("=" * 60)
    print("Example 2: Time Bins (Most Recent First)")
    print("=" * 60)

    df = create_sample_data()

    text = process_ehr_info(
        df,
        subject_id="patient_001",
        code_column="code",
        category_column="table",
        end_time=pd.Timestamp("2024-01-31"),
        time_bins=[
            (30, 15),
            (15, 7),
            (7, 0),
        ],  # 30-15 days, 15-7 days, 7-0 days
    )

    print(text)
    print()


def example_no_category():
    """Process data without category grouping."""
    print("=" * 60)
    print("Example 3: No Category Column")
    print("=" * 60)

    df = create_sample_data()

    # Remove the table column to simulate data without categories
    df_no_cat = df.drop(columns=["table"])

    text = process_ehr_info(
        df_no_cat,
        subject_id="patient_001",
        code_column="code",
        category_column=None,  # No category grouping
    )

    print(text)
    print()


def example_custom_category():
    """Use custom category mapping."""
    print("=" * 60)
    print("Example 4: Custom Category Mapping")
    print("=" * 60)

    df = create_sample_data()

    # Define custom mapping
    custom_mapping = {
        "vitals": ["measurement"],
        "labs": ["lab"],
        "diagnoses": ["condition"],
        "treatments": ["drug_exposure", "procedure"],
        "demographics": ["person"],
    }

    text = process_ehr_info(
        df,
        subject_id="patient_001",
        code_column="code",
        category_column="table",
        category_mapping=custom_mapping,
    )

    print(text)
    print()


def example_with_imaging():
    """Include imaging paths for multimodal models.

    Image tokens are placed inline at the timepoint where imaging occurs,
    not at the header.
    """
    print("=" * 60)
    print("Example 5: With Imaging Data (Inline Tokens)")
    print("=" * 60)

    df = create_sample_data()

    # Add imaging path column - CT scan done during procedure visit
    df["img_path"] = None
    df.loc[df["code"] == "Annual Wellness Visit", "img_path"] = (
        "/data/scans/ct_001.nii.gz"
    )

    result = process_ehr_info(
        df,
        subject_id="patient_001",
        code_column="code",
        category_column="table",
        include_imaging=True,
    )

    print("Text output (notice <image> at the procedure timepoint):")
    print(result["text"])
    print()
    print("Image paths:", result["images"])
    print()


if __name__ == "__main__":
    example_basic_usage()
    example_time_bins()
    example_no_category()
    example_custom_category()
    example_with_imaging()
