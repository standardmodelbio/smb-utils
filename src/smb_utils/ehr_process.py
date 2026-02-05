#! /usr/bin/env python3
#
# Copyright © 2026 Standard Model Biomedicine, Inc. <zach@standardmodel.bio>
#
# Distributed under terms of the Apache License 2.0 license.
"""MEDS EHR data formatter for inference.

This module formats patient event timelines from MEDS format into structured
text suitable for model inference. It provides utilities for extracting and
formatting demographics, conditions, measurements, observations, procedures,
drugs, and imaging paths.

Supports flexible category columns or no category column at all.
"""

# pyright: reportMissingTypeStubs=false

from typing import Any

import pandas as pd


# Default mapping from category values to semantic groups (used for OMOP-style data)
DEFAULT_CATEGORY_MAPPING: dict[str, list[str]] = {
    "demographics": ["person"],
    "conditions": ["condition", "diagnosis", "condition_occurrence"],
    "observations": [
        "observation",
        "ed_out",
        "icu_discharge",
        "ed_registration",
        "hospital_admission",
        "icu_admission",
        "hospital_discharge",
    ],
    "measurements": [
        "measurement",
        "lab",
        "blood pressure sitting",
        "bmi",
        "bmi (kg/m2)",
        "blood pressure standing (1 min)",
        "egfr",
        "blood pressure",
        "height",
        "height (inches)",
        "weight",
        "blood pressure standing",
        "blood pressure lying",
        "subject_fluid_output",
        "blood pressure standing (3 mins)",
        "weight (lbs)",
    ],
    "procedures": ["procedure", "device_exposure", "procedure_occurrence"],
    "drugs": ["drug_exposure", "medication", "infusion_start", "infusion_end"],
    "death": ["death", "meds_death"],
    "notes": ["note"],
}


def _build_reverse_mapping(category_mapping: dict[str, list[str]]) -> dict[str, str]:
    """Build reverse mapping from category values to group names."""
    reverse = {}
    for group, values in category_mapping.items():
        for val in values:
            reverse[val] = group
    return reverse


def _format_event_value(row: pd.Series) -> str:
    """Format a single event's value from numeric or text fields."""
    if pd.notna(row.get("numeric_value")):
        return f"{row['numeric_value']:.2f}"
    elif pd.notna(row.get("text_value")):
        return str(row["text_value"])
    return ""


IMAGE_PATH_COLUMNS = ("img_path", "ct_path", "file_path")


def _count_images_in_group(group: pd.DataFrame) -> int:
    """Count the number of imaging paths in a group of events."""
    count = 0
    for col in IMAGE_PATH_COLUMNS:
        if col in group.columns:
            count += group[col].notna().sum()
    return count


def _format_code_events(
    events_df: pd.DataFrame,
    code_column: str,
    max_values_per_code: int = 5,
    include_image_tokens: bool = False,
    image_token: str = "<image>",
) -> list[str]:
    """
    Format events grouped by code into lines of text.

    Args:
        events_df: DataFrame containing events, pre-sorted by time.
        code_column: Column name containing the code/event type.
        max_values_per_code: Maximum number of values to show per code.
        include_image_tokens: Whether to include image tokens inline.
        image_token: Token string for images (default: "<image>").

    Returns:
        List of formatted lines.
    """
    if events_df.empty:
        return []

    aggregated_lines: list[str] = []

    for code, group in events_df.groupby(code_column, sort=False):
        # Count images associated with this code
        num_images = _count_images_in_group(group) if include_image_tokens else 0

        # Special handling for birth
        if "birth" in str(code).lower():
            birth_date = group["time"].iloc[0].strftime("%Y-%m-%d")
            line = f"Birth: {birth_date}"
            if num_images > 0:
                line = (image_token * num_images) + line
            aggregated_lines.append(line)
            continue

        # Collect all values in chronological order
        values_with_units: list[tuple[str, str]] = []

        for _, row in group.iterrows():
            value_str = _format_event_value(row)
            if value_str:
                unit = row.get("unit")
                unit_str = str(unit) if pd.notna(unit) else ""
                values_with_units.append((value_str, unit_str))

        if values_with_units:
            recent_values = values_with_units[-max_values_per_code:]
            last_unit = recent_values[-1][1] if recent_values else ""
            unit_suffix = f" ({last_unit})" if last_unit else ""
            value_strings = [v[0] for v in recent_values]
            line = f"{code}{unit_suffix}: {', '.join(value_strings)}"
        else:
            line = str(code)

        if num_images > 0:
            line = (image_token * num_images) + line
        aggregated_lines.append(line)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_lines: list[str] = []
    for line in aggregated_lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    return unique_lines


def format_events_chronological(
    events_df: Any,
    code_column: str,
    category_column: str | None = None,
    category_mapping: dict[str, list[str]] | None = None,
    max_values_per_code: int = 5,
    include_image_tokens: bool = False,
    image_token: str = "<image>",
) -> str:
    """
    Format events into a structured string, optionally grouped by category.

    Args:
        events_df: DataFrame containing patient events, pre-sorted by time.
        code_column: Column name containing the code/event type information.
        category_column: Optional column name for categorizing events.
            If None, all events are formatted together without grouping.
        category_mapping: Optional mapping from group names to category values.
            If None and category_column is provided, events are grouped by
            unique values in the category column directly.
        max_values_per_code: Maximum number of values to show per code.
        include_image_tokens: Whether to include image tokens inline.
        image_token: Token string for images (default: "<image>").

    Returns:
        Formatted string with events grouped by category (if provided).
    """
    if events_df is None or events_df.empty:
        return ""

    # No category column - format all events together
    if category_column is None or category_column not in events_df.columns:
        lines = _format_code_events(events_df, code_column, max_values_per_code, include_image_tokens, image_token)
        return "\n".join(lines)

    # With category column but no mapping - group by unique category values
    if category_mapping is None:
        formatted_parts: list[str] = []
        for category_val in events_df[category_column].unique():
            cat_df = events_df[events_df[category_column] == category_val]
            lines = _format_code_events(cat_df, code_column, max_values_per_code, include_image_tokens, image_token)
            if lines:
                formatted_parts.append("\n".join(lines))
        return "\n".join(formatted_parts)

    # With category mapping - group by mapped categories
    output_parts: dict[str, list[str]] = {}

    for group_name, category_values in category_mapping.items():
        group_df = events_df[events_df[category_column].isin(category_values)]
        if group_df.empty:
            continue

        lines = _format_code_events(group_df, code_column, max_values_per_code, include_image_tokens, image_token)
        if lines:
            output_parts[group_name] = lines

    # Format output by category order
    formatted_parts = []
    for group_name in category_mapping.keys():
        if group_name in output_parts and output_parts[group_name]:
            formatted_parts.append("\n".join(output_parts[group_name]))

    return "\n".join(formatted_parts)


def format_events_by_date(
    events_df: Any,
    code_column: str,
    category_column: str | None = None,
    category_mapping: dict[str, list[str]] | None = None,
    include_image_tokens: bool = False,
    image_token: str = "<image>",
) -> str:
    """
    Format events grouped by date with date headers.

    Args:
        events_df: DataFrame containing patient events, pre-sorted by time.
        code_column: Column name containing the code/event type information.
        category_column: Optional column name for categorizing events.
        category_mapping: Optional mapping from group names to category values.
        include_image_tokens: Whether to include image tokens inline.
        image_token: Token string for images (default: "<image>").

    Returns:
        Formatted string with date headers and event sections.
    """
    if events_df is None or events_df.empty:
        return ""

    events_df = events_df.copy()
    events_df["date"] = events_df["time"].dt.date

    output_lines: list[str] = []

    for date, date_group in events_df.groupby("date", sort=True):
        date_str = date.strftime("%Y-%m-%d")
        output_lines.append(f"\n[{date_str}]")

        date_content = format_events_chronological(
            date_group,
            code_column,
            category_column,
            category_mapping,
            include_image_tokens=include_image_tokens,
            image_token=image_token,
        )
        if date_content:
            output_lines.append(date_content)

    return "\n".join(output_lines)


def extract_imaging_paths(events_df: Any) -> list[str]:
    """
    Extract imaging file paths from an events dataframe.

    Looks for 'img_path', 'ct_path', or 'file_path' columns.

    Args:
        events_df: DataFrame containing patient events.

    Returns:
        List of unique imaging file paths in chronological order.
    """
    if events_df is None or events_df.empty:
        return []

    paths: list[str] = []

    for col in ("img_path", "ct_path", "file_path"):
        if col in events_df.columns:
            col_vals = events_df[col].dropna().astype(str).tolist()
            paths.extend(col_vals)

    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            ordered.append(p)

    return ordered


def format_patient_history(
    patient_df: Any,
    code_column: str = "code",
    category_column: str | None = "table",
    category_mapping: dict[str, list[str]] | None = None,
    demographics_values: list[str] | None = None,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    include_demographics: bool = True,
    time_bins: list[tuple[int, int]] | None = None,
    include_image_tokens: bool = False,
    image_token: str = "<image>",
) -> str:
    """
    Format patient history as a structured text string for inference.

    Args:
        patient_df: DataFrame of events for a single patient (MEDS format).
        code_column: Column name containing the code/event type information.
        category_column: Column name for categorizing events. Set to None if
            the data has no category column.
        category_mapping: Optional mapping from group names to category values.
            If None and category_column exists, uses DEFAULT_CATEGORY_MAPPING.
            Pass an empty dict {} to disable grouping.
        demographics_values: Category values that represent demographics/person data.
            Used to separate demographics from timeline. Default: ["person"].
        start_time: Optional start time to filter events (lower bound, inclusive).
        end_time: Optional end time / anchor time for filtering and time bins.
        include_demographics: Whether to include demographics section.
        time_bins: Optional list of (days_back_start, days_back_end) tuples for binned history.
        include_image_tokens: Whether to include image tokens inline at their timepoints.
        image_token: Token string for images (default: "<image>").

    Returns:
        Formatted text string with event sections.
    """
    if patient_df.empty:
        return ""

    # Determine if category column exists
    has_category = category_column is not None and category_column in patient_df.columns

    # Set defaults
    if demographics_values is None:
        demographics_values = ["person"]

    if category_mapping is None and has_category:
        category_mapping = DEFAULT_CATEGORY_MAPPING

    # Sort by time (and category if available)
    sort_cols = ["time", category_column] if has_category else ["time"]
    patient_df = patient_df.sort_values(sort_cols).reset_index(drop=True)

    # Apply time filtering
    filtered_df = patient_df.copy()

    if time_bins and end_time:
        filtered_df = filtered_df[filtered_df["time"] <= end_time]
        if start_time is not None:
            filtered_df = filtered_df[filtered_df["time"] >= start_time]
    else:
        if start_time is not None:
            filtered_df = filtered_df[filtered_df["time"] >= start_time]
        if end_time is not None:
            filtered_df = filtered_df[filtered_df["time"] <= end_time]

    if filtered_df.empty:
        return ""

    output_parts: list[str] = []

    # Add demographics first if requested and category column exists
    if include_demographics and has_category:
        person_rows = filtered_df[filtered_df[category_column].isin(demographics_values)]
        if not person_rows.empty:
            demographics_str = format_events_chronological(
                person_rows,
                code_column,
                category_column,
                category_mapping,
                include_image_tokens=include_image_tokens,
                image_token=image_token,
            )
            if demographics_str:
                output_parts.append(demographics_str)
        # Remove demographics from main timeline
        filtered_df = filtered_df[~filtered_df[category_column].isin(demographics_values)]

    if filtered_df.empty:
        return "\n\n".join(output_parts)

    # Format events
    if time_bins and end_time:
        for days_back_start, days_back_end in time_bins:
            if days_back_start < days_back_end:
                raise ValueError(
                    f"Invalid time bin: ({days_back_start}, {days_back_end}). First value must be >= second value"
                )

        for days_back_start, days_back_end in reversed(time_bins):
            bin_start_time = end_time - pd.Timedelta(days=days_back_start)
            bin_end_time = end_time - pd.Timedelta(days=days_back_end)

            bin_df = filtered_df[(filtered_df["time"] > bin_start_time) & (filtered_df["time"] <= bin_end_time)]

            if not bin_df.empty:
                bin_label = f"{bin_start_time.strftime('%Y-%m-%d')} - {bin_end_time.strftime('%Y-%m-%d')}"
                bin_content = format_events_chronological(
                    bin_df,
                    code_column,
                    category_column,
                    category_mapping,
                    include_image_tokens=include_image_tokens,
                    image_token=image_token,
                )
                if bin_content:
                    output_parts.append(f"{bin_label}\n{bin_content}")
    else:
        content = format_events_by_date(
            filtered_df,
            code_column,
            category_column,
            category_mapping,
            include_image_tokens=include_image_tokens,
            image_token=image_token,
        )
        if content:
            output_parts.append(content)

    return "\n\n".join(output_parts)


def process_ehr_info(
    df: Any,
    subject_id: Any | None = None,
    code_column: str = "code",
    category_column: str | None = "table",
    category_mapping: dict[str, list[str]] | None = None,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    include_demographics: bool = True,
    time_bins: list[tuple[int, int]] | None = None,
    include_imaging: bool = False,
) -> str | dict[str, Any]:
    """
    High-level API to format MEDS data for inference.

    Args:
        df: DataFrame in MEDS format (can contain multiple subjects).
        subject_id: Optional subject ID to filter to a specific patient.
        code_column: Column name containing the code/event type information.
        category_column: Column name for categorizing events (e.g., "table", "event_type").
            Set to None if the data has no category column.
        category_mapping: Optional mapping from group names to category values.
            If None and category_column exists, uses DEFAULT_CATEGORY_MAPPING.
        start_time: Optional start time (absolute lower bound for events).
        end_time: Optional end time / anchor time for time bins.
        include_demographics: Whether to include demographics section.
        time_bins: Optional list of (days_back_start, days_back_end) tuples.
        include_imaging: If True, returns dict with 'text' and 'images' keys.

    Returns:
        Formatted text string ready for model inference, or dict if include_imaging=True.

    Example:
        >>> import pandas as pd
        >>> df = pd.read_parquet("patient_data.parquet")
        >>>
        >>> # With OMOP-style category column
        >>> text = process_ehr_info(df, subject_id="patient_123", category_column="table")
        >>>
        >>> # With custom category column
        >>> text = process_ehr_info(df, category_column="event_type")
        >>>
        >>> # No category column - all events formatted together
        >>> text = process_ehr_info(df, category_column=None)
        >>>
        >>> # Custom category mapping
        >>> mapping = {"vitals": ["bp", "hr", "temp"], "labs": ["cbc", "cmp"]}
        >>> text = process_ehr_info(df, category_column="type", category_mapping=mapping)
    """
    if df.empty:
        return "" if not include_imaging else {"text": "", "images": []}

    # Ensure the dataframe has required columns
    if "subject_id" not in df.columns and subject_id is None:
        df = df.copy()
        df["subject_id"] = "default"

    # Filter to specific subject if requested
    if subject_id is not None:
        df = df[df["subject_id"] == subject_id]
        if df.empty:
            return "" if not include_imaging else {"text": "", "images": []}

    # If processing multiple subjects, format each separately
    if len(df["subject_id"].unique()) > 1:
        results = []
        for sid in df["subject_id"].unique():
            patient_df = df[df["subject_id"] == sid]
            text = format_patient_history(
                patient_df=patient_df,
                code_column=code_column,
                category_column=category_column,
                category_mapping=category_mapping,
                start_time=start_time,
                end_time=end_time,
                include_demographics=include_demographics,
                time_bins=time_bins,
                include_image_tokens=include_imaging,
            )
            if text:
                results.append(f"Subject: {sid}\n{text}")
        formatted_text = "\n\n" + ("=" * 80 + "\n\n").join(results)
    else:
        formatted_text = format_patient_history(
            patient_df=df,
            code_column=code_column,
            category_column=category_column,
            category_mapping=category_mapping,
            start_time=start_time,
            end_time=end_time,
            include_demographics=include_demographics,
            time_bins=time_bins,
            include_image_tokens=include_imaging,
        )

    if include_imaging:
        imaging_paths = extract_imaging_paths(df)
        return {"text": formatted_text, "images": imaging_paths}

    return formatted_text
