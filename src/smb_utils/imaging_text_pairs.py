#! /usr/bin/env python3
#
# Copyright © 2026 Standard Model Biomedicine, Inc. <zach@standardmodel.bio>
#
# Distributed under terms of the Apache License 2.0 license.
"""
Imaging-Text Pair Data Creation for Model Training

This module provides utilities for creating imaging-text pair datasets in OpenAI
conversation format, suitable for training vision-language models on medical imaging data.

Example usage:
    from smb_utils.imaging_text_pairs import (
        ImagingTextPairCreator,
        create_imaging_text_pairs,
        convert_impressions_to_conversations,
    )

    # From a CSV file with impressions
    conversations = convert_impressions_to_conversations(
        csv_path="impressions.csv",
        image_dir="/path/to/images",
        impression_id_col="impression_id",
        impression_text_col="impressions",
    )

    # Or use the creator class for more control
    creator = ImagingTextPairCreator(
        system_prompt="You are a radiologist analyzing medical images.",
        image_extensions=[".nii.gz", ".dcm", ".png"],
    )
    conversations = creator.create_from_dataframe(df, image_dir="/path/to/images")
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Union

import pandas as pd


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: Union[str, list[dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible dictionary format."""
        return {"role": self.role, "content": self.content}


@dataclass
class ImageContent:
    """Represents image content in a multimodal message."""

    type: Literal["image_url", "image"] = "image_url"
    image_url: Optional[dict[str, str]] = None
    image_path: Optional[str] = None

    def to_dict(self, use_base64: bool = False) -> dict[str, Any]:
        """Convert to OpenAI-compatible dictionary format."""
        if self.image_url:
            return {"type": "image_url", "image_url": self.image_url}
        elif self.image_path:
            if use_base64:
                # Return base64-encoded image
                import base64

                with open(self.image_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                ext = Path(self.image_path).suffix.lower()
                media_type = _get_media_type(ext)
                return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}}
            else:
                return {"type": "image", "image": self.image_path}
        return {"type": "image", "image": ""}


@dataclass
class TextContent:
    """Represents text content in a multimodal message."""

    text: str
    type: Literal["text"] = "text"

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible dictionary format."""
        return {"type": "text", "text": self.text}


@dataclass
class Conversation:
    """Represents a complete conversation with optional metadata."""

    messages: list[ConversationMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_metadata: bool = False) -> dict[str, Any]:
        """Convert to OpenAI-compatible dictionary format."""
        result = {"messages": [msg.to_dict() for msg in self.messages]}
        if include_metadata and self.metadata:
            result["metadata"] = self.metadata
        return result

    def to_jsonl_line(self, include_metadata: bool = False) -> str:
        """Convert to a JSONL line."""
        return json.dumps(self.to_dict(include_metadata), ensure_ascii=False)


def _get_media_type(extension: str) -> str:
    """Get MIME type from file extension."""
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".nii": "application/octet-stream",
        ".nii.gz": "application/gzip",
        ".dcm": "application/dicom",
    }
    return media_types.get(extension.lower(), "application/octet-stream")


def _clean_impression_text(text: str) -> str:
    """Clean and normalize impression text."""
    if pd.isna(text):
        return ""

    text = str(text).strip()

    # Remove common prefixes
    prefixes_to_remove = [
        r"^IMPRESSION:\s*",
        r"^FINDINGS:\s*",
        r"^END OF IMPRESSION:\s*",
    ]
    for prefix in prefixes_to_remove:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)

    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


class ImagingTextPairCreator:
    """
    Creates imaging-text pair datasets for model training.

    This class provides flexible options for creating training data in
    OpenAI conversation format from medical imaging impression data.

    Args:
        system_prompt: Optional system prompt to include in conversations.
        user_prompt_template: Template for user prompts. Use {image_placeholder}
            where images should be inserted.
        image_extensions: List of valid image file extensions.
        image_placeholder: Placeholder text for images in content.
        include_image_paths: Whether to include image paths in output.
        clean_impressions: Whether to clean impression text.

    Example:
        >>> creator = ImagingTextPairCreator(
        ...     system_prompt="You are an expert radiologist.",
        ...     user_prompt_template="Analyze this CT scan and provide your impression."
        ... )
        >>> conversations = creator.create_from_dataframe(df, image_dir="./images")
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert radiologist providing accurate and detailed impressions of medical imaging studies."
    )

    DEFAULT_USER_PROMPT = "Please analyze this medical image and provide your clinical impression."

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        image_extensions: Optional[list[str]] = None,
        image_placeholder: str = "<image>",
        include_image_paths: bool = True,
        clean_impressions: bool = True,
    ):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template or self.DEFAULT_USER_PROMPT
        self.image_extensions = image_extensions or [".nii.gz", ".nii", ".dcm", ".png", ".jpg", ".jpeg"]
        self.image_placeholder = image_placeholder
        self.include_image_paths = include_image_paths
        self.clean_impressions = clean_impressions

    def find_images_for_id(
        self,
        impression_id: str,
        image_dir: Union[str, Path],
        image_pattern: Optional[str] = None,
        find_all: bool = False,
    ) -> Union[Optional[Path], list[Path]]:
        """
        Find image file(s) corresponding to an impression ID.

        Args:
            impression_id: The impression/study ID to find images for.
            image_dir: Directory containing images.
            image_pattern: Optional pattern for matching. Use {id} placeholder.
            find_all: If True, return all matching images. Otherwise return first match.

        Returns:
            Path to found image, list of paths (if find_all=True), or None.
        """
        image_dir = Path(image_dir)

        if not image_dir.exists():
            return [] if find_all else None

        found_images = []

        # Try direct match with various extensions
        for ext in self.image_extensions:
            if image_pattern:
                filename = image_pattern.format(id=impression_id)
            else:
                filename = f"{impression_id}{ext}"

            path = image_dir / filename
            if path.exists():
                if not find_all:
                    return path
                found_images.append(path)

        # Try glob pattern matching
        for ext in self.image_extensions:
            pattern = f"*{impression_id}*{ext}"
            matches = list(image_dir.glob(pattern))
            if matches:
                if not find_all:
                    return matches[0]
                found_images.extend(matches)

        if find_all:
            return list(set(found_images))  # Remove duplicates
        return None

    def find_image_for_id(
        self,
        impression_id: str,
        image_dir: Union[str, Path],
        image_pattern: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Find image file corresponding to an impression ID.

        Args:
            impression_id: The impression/study ID to find images for.
            image_dir: Directory containing images.
            image_pattern: Optional pattern for matching. Use {id} placeholder.

        Returns:
            Path to found image or None.
        """
        return self.find_images_for_id(impression_id, image_dir, image_pattern, find_all=False)

    def create_conversation(
        self,
        impression_text: str,
        image_path: Optional[Union[str, Path, list[Union[str, Path]]]] = None,
        impression_id: Optional[str] = None,
        additional_metadata: Optional[dict[str, Any]] = None,
        use_multimodal_format: bool = True,
    ) -> Conversation:
        """
        Create a single conversation from impression text and optional image(s).

        Args:
            impression_text: The radiology impression text.
            image_path: Optional path(s) to the associated image(s). Can be a single
                path or a list of paths for multiple images.
            impression_id: Optional ID for the impression.
            additional_metadata: Additional metadata to include.
            use_multimodal_format: Use multimodal content format for user message.

        Returns:
            A Conversation object.
        """
        messages = []

        # Normalize image_path to list
        image_paths = []
        if image_path:
            if isinstance(image_path, (list, tuple)):
                image_paths = [str(p) for p in image_path]
            else:
                image_paths = [str(image_path)]

        # Create user message content
        if use_multimodal_format and image_paths:
            # Multimodal format: [image(s), text] in user content
            content = []

            # Add image content(s) first
            for img_path in image_paths:
                content.append({"type": "image", "image": img_path})

            # Add text content
            content.append({"type": "text", "text": self.user_prompt_template})

            messages.append(ConversationMessage(role="user", content=content))
        else:
            # Text-only format
            user_content = self.user_prompt_template
            if image_paths:
                user_content = f"{self.image_placeholder}\n{user_content}"
            messages.append(ConversationMessage(role="user", content=user_content))

        # Clean and add assistant response
        assistant_content = impression_text
        if self.clean_impressions:
            assistant_content = _clean_impression_text(impression_text)

        messages.append(ConversationMessage(role="assistant", content=assistant_content))

        # Build metadata
        metadata = {}
        if impression_id:
            metadata["impression_id"] = impression_id
        if image_paths and self.include_image_paths:
            metadata["image_paths"] = image_paths
        if additional_metadata:
            metadata.update(additional_metadata)

        return Conversation(messages=messages, metadata=metadata)

    def create_from_dataframe(
        self,
        df: pd.DataFrame,
        image_dir: Optional[Union[str, Path]] = None,
        impression_id_col: str = "impression_id",
        impression_text_col: str = "impressions",
        image_path_col: Optional[str] = None,
        image_url_col: Optional[str] = None,
        image_pattern: Optional[str] = None,
        use_multimodal_format: bool = True,
        skip_missing_images: bool = False,
        find_all_images: bool = False,
    ) -> list[Conversation]:
        """
        Create conversations from a DataFrame of impressions.

        Args:
            df: DataFrame containing impression data.
            image_dir: Directory to search for images.
            impression_id_col: Column name for impression IDs.
            impression_text_col: Column name for impression text.
            image_path_col: Optional column name for pre-specified image paths.
            image_url_col: Optional column name for image URLs.
            image_pattern: Pattern for finding images. Use {id} placeholder.
            use_multimodal_format: Use multimodal content format.
            skip_missing_images: Skip rows where no image is found.
            find_all_images: Find all matching images for each ID (for multi-image cases).

        Returns:
            List of Conversation objects.
        """
        conversations = []

        for _, row in df.iterrows():
            impression_id = row.get(impression_id_col)
            impression_text = row.get(impression_text_col, "")

            if pd.isna(impression_text) or not str(impression_text).strip():
                continue

            # Get image path(s)
            image_paths = None

            # Check for image URL column first
            if image_url_col and image_url_col in row:
                img_url = row[image_url_col]
                if pd.notna(img_url):
                    image_paths = str(img_url)
            # Then check for image path column
            elif image_path_col and image_path_col in row:
                img_path = row[image_path_col]
                if pd.notna(img_path):
                    image_paths = str(img_path)
            # Finally try to find images in directory
            elif image_dir and impression_id:
                if find_all_images:
                    found = self.find_images_for_id(str(impression_id), image_dir, image_pattern, find_all=True)
                    if found:
                        image_paths = [str(p) for p in found]
                else:
                    found = self.find_image_for_id(str(impression_id), image_dir, image_pattern)
                    if found:
                        image_paths = str(found)

            if skip_missing_images and image_paths is None:
                continue

            # Gather additional metadata from other columns
            additional_metadata = {}
            excluded_cols = {impression_id_col, impression_text_col, image_path_col, image_url_col}
            for col in df.columns:
                if col not in excluded_cols and pd.notna(row.get(col)):
                    additional_metadata[col] = row[col]

            conversation = self.create_conversation(
                impression_text=str(impression_text),
                image_path=image_paths,
                impression_id=str(impression_id) if impression_id else None,
                additional_metadata=additional_metadata if additional_metadata else None,
                use_multimodal_format=use_multimodal_format,
            )
            conversations.append(conversation)

        return conversations

    def create_from_csv(
        self,
        csv_path: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
        impression_id_col: str = "impression_id",
        impression_text_col: str = "impressions",
        **kwargs,
    ) -> list[Conversation]:
        """
        Create conversations from a CSV file.

        Args:
            csv_path: Path to CSV file.
            image_dir: Directory to search for images.
            impression_id_col: Column name for impression IDs.
            impression_text_col: Column name for impression text.
            **kwargs: Additional arguments passed to create_from_dataframe.

        Returns:
            List of Conversation objects.
        """
        df = pd.read_csv(csv_path)
        return self.create_from_dataframe(
            df,
            image_dir=image_dir,
            impression_id_col=impression_id_col,
            impression_text_col=impression_text_col,
            **kwargs,
        )


def convert_impressions_to_conversations(
    csv_path: Optional[Union[str, Path]] = None,
    df: Optional[pd.DataFrame] = None,
    image_dir: Optional[Union[str, Path]] = None,
    impression_id_col: str = "impression_id",
    impression_text_col: str = "impressions",
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    use_multimodal_format: bool = True,
    output_format: Literal["list", "jsonl", "dict"] = "list",
    include_metadata: bool = True,
) -> Union[list[Conversation], list[dict], str]:
    """
    Convert radiology impressions to OpenAI conversation format.

    This is a convenience function that wraps ImagingTextPairCreator for
    simple use cases.

    Args:
        csv_path: Path to CSV file with impressions.
        df: DataFrame with impressions (alternative to csv_path).
        image_dir: Directory containing associated images.
        impression_id_col: Column name for impression IDs.
        impression_text_col: Column name for impression text.
        system_prompt: Optional system prompt for conversations.
        user_prompt: Optional user prompt template.
        use_multimodal_format: Use multimodal content format.
        output_format: Output format ("list", "jsonl", or "dict").
        include_metadata: Include metadata in output.

    Returns:
        Conversations in requested format.

    Example:
        >>> # From CSV
        >>> conversations = convert_impressions_to_conversations(
        ...     csv_path="impressions.csv",
        ...     image_dir="./ct_scans",
        ...     system_prompt="You are a radiologist.",
        ... )

        >>> # To JSONL for training
        >>> jsonl_data = convert_impressions_to_conversations(
        ...     df=my_dataframe,
        ...     output_format="jsonl",
        ... )
    """
    if csv_path is None and df is None:
        raise ValueError("Either csv_path or df must be provided")

    creator = ImagingTextPairCreator(
        system_prompt=system_prompt,
        user_prompt_template=user_prompt,
    )

    if csv_path is not None:
        conversations = creator.create_from_csv(
            csv_path,
            image_dir=image_dir,
            impression_id_col=impression_id_col,
            impression_text_col=impression_text_col,
            use_multimodal_format=use_multimodal_format,
        )
    else:
        conversations = creator.create_from_dataframe(
            df,
            image_dir=image_dir,
            impression_id_col=impression_id_col,
            impression_text_col=impression_text_col,
            use_multimodal_format=use_multimodal_format,
        )

    if output_format == "list":
        return conversations
    elif output_format == "dict":
        return [conv.to_dict(include_metadata) for conv in conversations]
    elif output_format == "jsonl":
        lines = [conv.to_jsonl_line(include_metadata) for conv in conversations]
        return "\n".join(lines)
    else:
        raise ValueError(f"Unknown output_format: {output_format}")


def create_imaging_text_pairs(
    impressions_data: Union[str, Path, pd.DataFrame],
    image_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    impression_id_col: str = "impression_id",
    impression_text_col: str = "impressions",
    image_url_col: Optional[str] = None,
    image_path_col: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    use_multimodal_format: bool = True,
    include_metadata: bool = True,
    conversation_style: Literal["single_turn", "multi_turn"] = "single_turn",
) -> Union[list[dict], str]:
    """
    Create imaging-text pair training data from impression data.

    This is the main entry point for creating training datasets. It supports
    various input formats and outputs data in conversation format.

    Args:
        impressions_data: Path to CSV/parquet file or DataFrame with impressions.
        image_dir: Directory containing associated images.
        output_path: Optional path to save output JSONL file.
        impression_id_col: Column name for impression IDs.
        impression_text_col: Column name for impression text.
        image_url_col: Column name for image URLs.
        image_path_col: Column name for image file paths.
        system_prompt: Optional system prompt (stored in metadata, not in messages).
        user_prompt: Optional user prompt template.
        use_multimodal_format: Use multimodal content format for images.
        include_metadata: Include metadata in output.
        conversation_style: "single_turn" or "multi_turn" conversation style.

    Returns:
        List of conversation dictionaries or path to saved file.

    Example:
        >>> # Create training data from CSV with image URLs
        >>> data = create_imaging_text_pairs(
        ...     "impressions.csv",
        ...     image_url_col="image_url",
        ...     output_path="training_data.jsonl",
        ... )

        >>> # From DataFrame with local images
        >>> import pandas as pd
        >>> df = pd.read_csv("impressions.csv")
        >>> data = create_imaging_text_pairs(df, image_dir="./images")
    """
    # Load data if path provided
    if isinstance(impressions_data, (str, Path)):
        path = Path(impressions_data)
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    else:
        df = impressions_data

    # Create conversations
    creator = ImagingTextPairCreator(
        system_prompt=system_prompt,
        user_prompt_template=user_prompt,
    )

    conversations = creator.create_from_dataframe(
        df,
        image_dir=image_dir,
        impression_id_col=impression_id_col,
        impression_text_col=impression_text_col,
        image_url_col=image_url_col,
        image_path_col=image_path_col,
        use_multimodal_format=use_multimodal_format,
    )

    # Convert to dict format
    result = [conv.to_dict(include_metadata) for conv in conversations]

    # Save if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(conv.to_jsonl_line(include_metadata) + "\n")

        return str(output_path)

    return result


class ImagingTextDataset:
    """
    Iterator-based dataset for large-scale imaging-text pair processing.

    This class provides memory-efficient iteration over large datasets
    without loading everything into memory.

    Args:
        data_source: Path to CSV/parquet file or DataFrame.
        image_dir: Directory containing images.
        batch_size: Number of conversations per batch.
        **kwargs: Additional arguments for ImagingTextPairCreator.

    Example:
        >>> dataset = ImagingTextDataset(
        ...     "large_impressions.csv",
        ...     image_dir="./images",
        ...     batch_size=100,
        ... )
        >>> for batch in dataset:
        ...     # Process batch of conversations
        ...     save_to_training_file(batch)
    """

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        image_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 100,
        impression_id_col: str = "impression_id",
        impression_text_col: str = "impressions",
        **creator_kwargs,
    ):
        self.data_source = data_source
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.impression_id_col = impression_id_col
        self.impression_text_col = impression_text_col
        self.creator = ImagingTextPairCreator(**creator_kwargs)

        # Load or reference data
        if isinstance(data_source, (str, Path)):
            path = Path(data_source)
            if path.suffix == ".csv":
                self.df = pd.read_csv(path)
            elif path.suffix in [".parquet", ".pq"]:
                self.df = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            self.df = data_source

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterator[list[Conversation]]:
        for i in range(0, len(self.df), self.batch_size):
            batch_df = self.df.iloc[i : i + self.batch_size]
            conversations = self.creator.create_from_dataframe(
                batch_df,
                image_dir=self.image_dir,
                impression_id_col=self.impression_id_col,
                impression_text_col=self.impression_text_col,
            )
            yield conversations

    def to_jsonl(
        self,
        output_path: Union[str, Path],
        include_metadata: bool = True,
    ) -> str:
        """
        Stream dataset to JSONL file.

        Args:
            output_path: Path for output file.
            include_metadata: Include metadata in output.

        Returns:
            Path to saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for batch in self:
                for conv in batch:
                    f.write(conv.to_jsonl_line(include_metadata) + "\n")

        return str(output_path)


# Convenience functions for specific use cases


def create_ct_impression_pairs(
    impressions_csv: Union[str, Path],
    ct_image_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Union[list[dict], str]:
    """
    Create training pairs specifically for CT imaging and impressions.

    Args:
        impressions_csv: Path to CSV with CT impressions.
        ct_image_dir: Directory containing CT images (NIfTI/DICOM).
        output_path: Optional output path for JSONL file.
        **kwargs: Additional arguments for create_imaging_text_pairs.

    Returns:
        Training data in conversation format.
    """
    default_system = (
        "You are an expert radiologist specializing in CT imaging. "
        "Analyze the provided CT scan and provide a detailed clinical impression."
    )
    default_user = (
        "Please review this CT scan and provide your clinical impression, including any significant findings."
    )

    return create_imaging_text_pairs(
        impressions_csv,
        image_dir=ct_image_dir,
        output_path=output_path,
        system_prompt=kwargs.pop("system_prompt", default_system),
        user_prompt=kwargs.pop("user_prompt", default_user),
        **kwargs,
    )


def create_pulmonary_embolism_pairs(
    impressions_data: Union[str, Path, pd.DataFrame],
    image_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Union[list[dict], str]:
    """
    Create training pairs for pulmonary embolism detection.

    Specialized function for PE-focused imaging-text pairs with
    appropriate system prompts.

    Args:
        impressions_data: Path to file or DataFrame with PE impressions.
        image_dir: Directory containing CT pulmonary angiography images.
        output_path: Optional output path for JSONL file.
        **kwargs: Additional arguments.

    Returns:
        Training data in conversation format.
    """
    pe_system_prompt = (
        "You are an expert radiologist specializing in CT pulmonary angiography. "
        "Your task is to analyze the imaging study and provide a detailed impression, "
        "specifically evaluating for the presence or absence of pulmonary embolism, "
        "deep venous thrombosis, and any additional pulmonary or cardiovascular findings."
    )

    pe_user_prompt = (
        "Please analyze this CT pulmonary angiography study and provide your "
        "clinical impression, including evaluation for pulmonary embolism and "
        "any other significant findings."
    )

    return create_imaging_text_pairs(
        impressions_data,
        image_dir=image_dir,
        output_path=output_path,
        system_prompt=kwargs.pop("system_prompt", pe_system_prompt),
        user_prompt=kwargs.pop("user_prompt", pe_user_prompt),
        **kwargs,
    )
