"""
Utility functions for formatting output in the learning app.
"""
from typing import List, Dict, Any
import pandas as pd


def format_results_table(results: List[Dict[str, Any]], show_content: bool = True) -> pd.DataFrame:
    """
    Format search results as a pandas DataFrame for display.

    Args:
        results: List of result dictionaries
        show_content: Whether to include full content (vs truncated)

    Returns:
        Formatted DataFrame
    """
    if not results:
        return pd.DataFrame({"Message": ["No results found"]})

    formatted = []
    for i, result in enumerate(results, 1):
        row = {"Rank": i}

        if "id" in result:
            row["ID"] = result["id"]

        if "score" in result:
            row["Score"] = f"{result['score']:.4f}"

        if "content" in result:
            content = result["content"]
            if not show_content and len(content) > 150:
                content = content[:150] + "..."
            row["Content"] = content

        formatted.append(row)

    return pd.DataFrame(formatted)


def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    Format metrics dictionary as readable text.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Formatted string
    """
    lines = ["=" * 60, "PERFORMANCE METRICS", "=" * 60]

    for key, value in metrics.items():
        # Format key nicely
        display_key = key.replace("_", " ").title()

        # Format value based on type
        if isinstance(value, float):
            if "time" in key.lower() or "latency" in key.lower():
                display_value = f"{value:.4f}s"
            elif "rate" in key.lower() or "percent" in key.lower():
                display_value = f"{value:.2f}%"
            else:
                display_value = f"{value:.4f}"
        elif isinstance(value, int):
            display_value = f"{value:,}"
        else:
            display_value = str(value)

        lines.append(f"{display_key:.<40} {display_value:>18}")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_document_preview(text: str, max_chars: int = 500) -> str:
    """
    Format document text for preview display.

    Args:
        text: Document text
        max_chars: Maximum characters to show

    Returns:
        Formatted preview string
    """
    lines = [
        "=" * 60,
        "DOCUMENT PREVIEW",
        "=" * 60,
        ""
    ]

    if len(text) <= max_chars:
        lines.append(text)
    else:
        lines.append(text[:max_chars])
        lines.append(f"\n... (truncated, showing {max_chars} of {len(text):,} characters)")

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Total characters: {len(text):,}")
    lines.append(f"Total words: {len(text.split()):,}")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_embedding_info(embedding: List[float], model_name: str) -> str:
    """
    Format embedding vector information.

    Args:
        embedding: Embedding vector
        model_name: Name of the model used

    Returns:
        Formatted string
    """
    lines = [
        "=" * 60,
        "EMBEDDING INFORMATION",
        "=" * 60,
        f"Model: {model_name}",
        f"Dimensions: {len(embedding)}",
        "",
        "First 10 values:",
        str(embedding[:10]),
        "",
        "Last 10 values:",
        str(embedding[-10:]),
        "=" * 60
    ]

    return "\n".join(lines)


def format_comparison_table(items: List[Dict[str, Any]], title: str = "COMPARISON") -> str:
    """
    Format comparison results as a table.

    Args:
        items: List of items to compare
        title: Table title

    Returns:
        Formatted table string
    """
    if not items:
        return "No items to compare"

    df = pd.DataFrame(items)

    lines = [
        "=" * 80,
        title.center(80),
        "=" * 80,
        df.to_string(index=False),
        "=" * 80
    ]

    return "\n".join(lines)
