"""
Plot helpers module for the learning app (Phase 1 Enhancements).
Provides reusable utilities for creating consistent visualizations.
"""
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np


def create_plotly_config(
    filename: str = "visualization",
    include_mode_bar: bool = True
) -> Dict[str, any]:
    """
    Create standard Plotly configuration for all plots.

    Args:
        filename: Default filename for downloads
        include_mode_bar: Whether to show the mode bar

    Returns:
        Configuration dictionary for Plotly
    """
    return {
        'displayModeBar': include_mode_bar,
        'displaylogo': False,
        'modeBarButtonsToRemove': [
            'lasso2d',
            'select2d',
            'autoScale2d'
        ],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': filename,
            'height': 800,
            'width': 1200,
            'scale': 2
        },
        'responsive': True
    }


def format_hover_data(
    documents: List[Dict[str, any]],
    include_fields: Optional[List[str]] = None
) -> List[str]:
    """
    Format document metadata for hover tooltips.

    Args:
        documents: List of document dictionaries with metadata
        include_fields: Optional list of fields to include (None = all)

    Returns:
        List of formatted hover text strings
    """
    hover_texts = []

    for doc in documents:
        lines = []

        # Title or ID
        if 'title' in doc:
            lines.append(f"<b>{doc['title']}</b>")
        elif 'id' in doc:
            lines.append(f"<b>Document {doc['id']}</b>")

        # Additional fields
        fields_to_show = include_fields if include_fields else doc.keys()

        for field in fields_to_show:
            if field in ['title', 'id']:  # Already shown
                continue

            if field in doc:
                value = doc[field]

                # Format based on type
                if field == 'score':
                    lines.append(f"Score: {value:.4f}")
                elif field == 'similarity':
                    lines.append(f"Similarity: {value:.4f}")
                elif field == 'length':
                    lines.append(f"Length: {value:,} chars")
                elif field == 'type':
                    lines.append(f"Type: {value}")
                elif field == 'date':
                    lines.append(f"Date: {value}")
                elif isinstance(value, float):
                    lines.append(f"{field.title()}: {value:.3f}")
                elif isinstance(value, int):
                    lines.append(f"{field.title()}: {value:,}")
                else:
                    # Truncate long strings
                    str_value = str(value)
                    if len(str_value) > 50:
                        str_value = str_value[:47] + "..."
                    lines.append(f"{field.title()}: {str_value}")

        hover_text = "<br>".join(lines)
        hover_texts.append(hover_text)

    return hover_texts


def get_color_scale(
    n_items: int,
    palette: str = "Viridis",
    discrete: bool = False
) -> List[str]:
    """
    Get a color scale for visualization.

    Args:
        n_items: Number of colors needed
        palette: Plotly color palette name
        discrete: Whether to use discrete (qualitative) or continuous colors

    Returns:
        List of color codes
    """
    if discrete and n_items <= 10:
        # Use qualitative colors for small number of categories
        qualitative_palettes = {
            "Plotly": pc.qualitative.Plotly,
            "D3": pc.qualitative.D3,
            "Set1": pc.qualitative.Set1,
            "Set2": pc.qualitative.Set2,
            "Set3": pc.qualitative.Set3,
            "Pastel": pc.qualitative.Pastel,
            "Dark2": pc.qualitative.Dark2,
            "Vivid": pc.qualitative.Vivid,
            "Safe": pc.qualitative.Safe
        }

        if palette in qualitative_palettes:
            colors = qualitative_palettes[palette][:n_items]
        else:
            colors = pc.qualitative.Plotly[:n_items]
    else:
        # Use continuous color scale
        if palette in pc.named_colorscales():
            colors = pc.sample_colorscale(palette, n_items)
        else:
            colors = pc.sample_colorscale("Viridis", n_items)

    return colors


def create_annotation_box(
    text: str,
    x: float = 0.5,
    y: float = 1.15,
    xref: str = "paper",
    yref: str = "paper",
    showarrow: bool = False,
    bgcolor: str = "rgba(255, 255, 255, 0.8)",
    bordercolor: str = "#2196F3",
    borderwidth: int = 2
) -> Dict[str, any]:
    """
    Create a styled annotation box for plots.

    Args:
        text: Annotation text (supports HTML)
        x: X position
        y: Y position
        xref: X reference coordinate system
        yref: Y reference coordinate system
        showarrow: Whether to show an arrow
        bgcolor: Background color
        bordercolor: Border color
        borderwidth: Border width in pixels

    Returns:
        Annotation dictionary for Plotly layout
    """
    return {
        'text': text,
        'x': x,
        'y': y,
        'xref': xref,
        'yref': yref,
        'showarrow': showarrow,
        'bgcolor': bgcolor,
        'bordercolor': bordercolor,
        'borderwidth': borderwidth,
        'borderpad': 10,
        'font': {'size': 12},
        'align': 'left'
    }


def create_comparison_table(
    data: List[Dict[str, any]],
    columns: List[str],
    title: Optional[str] = None
) -> go.Figure:
    """
    Create a styled comparison table using Plotly.

    Args:
        data: List of row dictionaries
        columns: Column names to display
        title: Optional table title

    Returns:
        Plotly figure with table
    """
    # Extract column data
    header_values = columns
    cell_values = [[row.get(col, "") for row in data] for col in columns]

    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{col}</b>" for col in header_values],
            fill_color='#2196F3',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=cell_values,
            fill_color=['#f5f5f5', 'white'],
            align='left',
            font=dict(size=11),
            height=30
        )
    )])

    if title:
        fig.update_layout(title=title)

    fig.update_layout(
        margin=dict(l=10, r=10, t=40 if title else 10, b=10)
    )

    return fig


def add_metric_cards(
    fig: go.Figure,
    metrics: List[Dict[str, any]],
    position: str = "top"
) -> go.Figure:
    """
    Add metric cards as annotations to a figure.

    Args:
        fig: Plotly figure to add metrics to
        metrics: List of metric dictionaries with 'label' and 'value' keys
        position: Where to position cards ("top" or "bottom")

    Returns:
        Modified Plotly figure
    """
    n_metrics = len(metrics)
    spacing = 1.0 / (n_metrics + 1)

    y_pos = 1.15 if position == "top" else -0.15

    for i, metric in enumerate(metrics):
        x_pos = (i + 1) * spacing

        # Format value
        value = metric['value']
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)

        # Create annotation
        fig.add_annotation(
            text=f"<b>{metric['label']}</b><br>{formatted_value}",
            x=x_pos,
            y=y_pos,
            xref="paper",
            yref="paper",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#2196F3",
            borderwidth=2,
            borderpad=8,
            font=dict(size=11)
        )

    return fig


def create_score_gauge(
    score: float,
    title: str,
    min_val: float = 0.0,
    max_val: float = 1.0,
    threshold_low: float = 0.3,
    threshold_high: float = 0.7
) -> go.Figure:
    """
    Create a gauge chart for displaying scores.

    Args:
        score: Score value
        title: Gauge title
        min_val: Minimum value
        max_val: Maximum value
        threshold_low: Lower threshold for color change
        threshold_high: Upper threshold for color change

    Returns:
        Plotly figure with gauge
    """
    # Determine color based on thresholds
    if score < threshold_low:
        color = "#F44336"  # Red
    elif score < threshold_high:
        color = "#FF9800"  # Orange
    else:
        color = "#4CAF50"  # Green

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, threshold_low], 'color': 'rgba(244, 67, 54, 0.2)'},
                {'range': [threshold_low, threshold_high], 'color': 'rgba(255, 152, 0, 0.2)'},
                {'range': [threshold_high, max_val], 'color': 'rgba(76, 175, 80, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def create_legend_info(
    items: List[Tuple[str, str, str]],
    title: str = "Legend"
) -> str:
    """
    Create HTML legend information.

    Args:
        items: List of (label, color, description) tuples
        title: Legend title

    Returns:
        HTML string with legend
    """
    html = f'<div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0;">'
    html += f'<h4 style="margin-top: 0;">{title}</h4>'
    html += '<div style="display: grid; grid-template-columns: auto 1fr; gap: 10px; align-items: center;">'

    for label, color, description in items:
        html += f'<div style="display: flex; align-items: center;">'
        html += f'<span style="width: 20px; height: 20px; background-color: {color}; display: inline-block; margin-right: 8px; border: 1px solid #ddd;"></span>'
        html += f'<strong>{label}</strong>'
        html += f'</div>'
        html += f'<div>{description}</div>'

    html += '</div></div>'
    return html


def style_dataframe_table(
    df,
    highlight_column: Optional[str] = None,
    highlight_threshold: Optional[float] = None
) -> str:
    """
    Convert pandas DataFrame to styled HTML table.

    Args:
        df: Pandas DataFrame
        highlight_column: Column to highlight based on values
        highlight_threshold: Threshold for highlighting

    Returns:
        HTML string with styled table
    """
    html = '<table style="width: 100%; border-collapse: collapse; font-size: 14px;">'

    # Header
    html += '<thead><tr style="background-color: #2196F3; color: white;">'
    for col in df.columns:
        html += f'<th style="padding: 10px; text-align: left; border: 1px solid #ddd;">{col}</th>'
    html += '</tr></thead>'

    # Body
    html += '<tbody>'
    for idx, row in df.iterrows():
        html += '<tr style="border-bottom: 1px solid #ddd;">'
        for col in df.columns:
            value = row[col]

            # Determine cell style
            cell_style = "padding: 8px; border: 1px solid #ddd;"

            if highlight_column and col == highlight_column and highlight_threshold:
                if isinstance(value, (int, float)) and value >= highlight_threshold:
                    cell_style += " background-color: #C8E6C9; font-weight: bold;"

            # Format value
            if isinstance(value, float):
                formatted = f"{value:.4f}"
            elif isinstance(value, int):
                formatted = f"{value:,}"
            else:
                formatted = str(value)

            html += f'<td style="{cell_style}">{formatted}</td>'

        html += '</tr>'

    html += '</tbody></table>'
    return html
