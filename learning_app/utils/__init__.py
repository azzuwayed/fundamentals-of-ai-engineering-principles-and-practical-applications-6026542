"""Utility modules for the AI Engineering Learning App."""

from .formatters import (
    format_results_table,
    format_metrics,
    format_document_preview,
    format_embedding_info,
    format_comparison_table
)

from .validators import (
    validate_file_upload,
    validate_text_input,
    validate_top_k,
    validate_chunk_size,
    validate_weight
)

from .plot_helpers import (
    create_plotly_config,
    format_hover_data,
    get_color_scale,
    create_annotation_box,
    create_comparison_table,
    add_metric_cards,
    create_score_gauge,
    create_legend_info,
    style_dataframe_table
)

__all__ = [
    'format_results_table',
    'format_metrics',
    'format_document_preview',
    'format_embedding_info',
    'format_comparison_table',
    'validate_file_upload',
    'validate_text_input',
    'validate_top_k',
    'validate_chunk_size',
    'validate_weight',
    'create_plotly_config',
    'format_hover_data',
    'get_color_scale',
    'create_annotation_box',
    'create_comparison_table',
    'add_metric_cards',
    'create_score_gauge',
    'create_legend_info',
    'style_dataframe_table'
]
