"""
Input validation utilities for the learning app.
"""
import os
from typing import Optional, Tuple


def validate_file_upload(file_path: Optional[str]) -> Tuple[bool, str]:
    """
    Validate uploaded file.

    Args:
        file_path: Path to uploaded file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if file_path is None:
        return False, "No file uploaded"

    if not os.path.exists(file_path):
        return False, "File does not exist"

    # Check file size (max 10MB)
    file_size = os.path.getsize(file_path)
    max_size = 10 * 1024 * 1024  # 10MB
    if file_size > max_size:
        return False, f"File too large ({file_size / 1024 / 1024:.1f}MB). Maximum size is 10MB."

    # Check file extension
    allowed_extensions = {'.pdf', '.docx', '.txt', '.json', '.csv'}
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in allowed_extensions:
        return False, f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}"

    return True, ""


def validate_text_input(text: str, min_length: int = 1, max_length: int = 10000) -> Tuple[bool, str]:
    """
    Validate text input.

    Args:
        text: Input text
        min_length: Minimum required length
        max_length: Maximum allowed length

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Text input is empty"

    text_length = len(text.strip())

    if text_length < min_length:
        return False, f"Text too short (minimum {min_length} characters)"

    if text_length > max_length:
        return False, f"Text too long (maximum {max_length:,} characters)"

    return True, ""


def validate_top_k(top_k: int, max_documents: int = 1000) -> Tuple[bool, str]:
    """
    Validate top_k parameter.

    Args:
        top_k: Number of results to return
        max_documents: Maximum number of documents available

    Returns:
        Tuple of (is_valid, error_message)
    """
    if top_k < 1:
        return False, "top_k must be at least 1"

    if top_k > 100:
        return False, "top_k cannot exceed 100"

    if top_k > max_documents:
        return False, f"top_k ({top_k}) exceeds number of documents ({max_documents})"

    return True, ""


def validate_chunk_size(chunk_size: int) -> Tuple[bool, str]:
    """
    Validate chunk_size parameter.

    Args:
        chunk_size: Size of text chunks

    Returns:
        Tuple of (is_valid, error_message)
    """
    if chunk_size < 50:
        return False, "chunk_size too small (minimum 50 characters)"

    if chunk_size > 5000:
        return False, "chunk_size too large (maximum 5000 characters)"

    return True, ""


def validate_weight(weight: float, name: str = "weight") -> Tuple[bool, str]:
    """
    Validate weight parameter.

    Args:
        weight: Weight value
        name: Name of weight parameter for error message

    Returns:
        Tuple of (is_valid, error_message)
    """
    if weight < 0.0 or weight > 1.0:
        return False, f"{name} must be between 0.0 and 1.0"

    return True, ""
