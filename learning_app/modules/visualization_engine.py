"""
Visualization engine module for the learning app (Phase 1 Enhancements).
Handles embedding space visualization, similarity heatmaps, and retrieval comparisons.
"""
import time
from typing import List, Dict, Tuple, Literal, Optional, Union
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from umap import UMAP


class VisualizationEngine:
    """Handles visualization generation for embeddings and retrieval systems."""

    # Visualization configuration
    DEFAULT_CONFIG = {
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "umap_metric": "cosine",
        "tsne_perplexity": 30,
        "tsne_learning_rate": 200,
        "tsne_max_iter": 1000,
        "plot_height": 600,
        "plot_width": 800,
        "color_scale": "Viridis"
    }

    def __init__(self):
        """Initialize the visualization engine."""
        pass

    def reduce_dimensions(
        self,
        embeddings: Union[List[List[float]], np.ndarray],
        method: Literal["umap", "tsne"] = "umap",
        n_components: Literal[2, 3] = 2,
        random_state: int = 42,
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """
        Reduce high-dimensional embeddings to 2D or 3D using UMAP or t-SNE.

        Args:
            embeddings: High-dimensional embedding vectors
            method: Dimensionality reduction method ("umap" or "tsne")
            n_components: Number of dimensions to reduce to (2 or 3)
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters for the reduction algorithm

        Returns:
            Tuple of (reduced embeddings, computation time in seconds)
        """
        # Convert to numpy array if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        if embeddings.shape[0] < 2:
            raise ValueError("Need at least 2 embeddings for dimensionality reduction")

        start_time = time.time()

        if method == "umap":
            # UMAP parameters
            n_neighbors = kwargs.get("n_neighbors", self.DEFAULT_CONFIG["umap_n_neighbors"])
            min_dist = kwargs.get("min_dist", self.DEFAULT_CONFIG["umap_min_dist"])
            metric = kwargs.get("metric", self.DEFAULT_CONFIG["umap_metric"])

            # Adjust n_neighbors if we have fewer samples
            n_neighbors = min(n_neighbors, embeddings.shape[0] - 1)

            reducer = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state
            )
            reduced = reducer.fit_transform(embeddings)

        elif method == "tsne":
            # t-SNE parameters
            perplexity = kwargs.get("perplexity", self.DEFAULT_CONFIG["tsne_perplexity"])
            learning_rate = kwargs.get("learning_rate", self.DEFAULT_CONFIG["tsne_learning_rate"])
            max_iter = kwargs.get("max_iter", self.DEFAULT_CONFIG["tsne_max_iter"])

            # Adjust perplexity if we have fewer samples
            perplexity = min(perplexity, embeddings.shape[0] - 1)

            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                max_iter=max_iter,
                random_state=random_state
            )
            reduced = reducer.fit_transform(embeddings)

        else:
            raise ValueError(f"Unknown method: {method}. Choose 'umap' or 'tsne'")

        computation_time = time.time() - start_time
        return reduced, computation_time

    def plot_embedding_space(
        self,
        embeddings: Union[List[List[float]], np.ndarray],
        labels: Optional[List[str]] = None,
        method: Literal["umap", "tsne"] = "umap",
        n_components: Literal[2, 3] = 2,
        title: Optional[str] = None,
        hover_data: Optional[List[Dict]] = None,
        color_by: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[go.Figure, Dict[str, float]]:
        """
        Create interactive visualization of embedding space.

        Args:
            embeddings: High-dimensional embedding vectors
            labels: Labels for each embedding (document names, etc.)
            method: Dimensionality reduction method
            n_components: 2D or 3D visualization
            title: Plot title
            hover_data: Additional data to show on hover
            color_by: Categories for color coding
            **kwargs: Additional parameters for reduction algorithm

        Returns:
            Tuple of (Plotly figure, timing dictionary)
        """
        # Reduce dimensions
        reduced, reduction_time = self.reduce_dimensions(
            embeddings, method, n_components, **kwargs
        )

        # Prepare labels
        if labels is None:
            labels = [f"Doc {i+1}" for i in range(len(embeddings))]

        # Prepare hover text
        hover_texts = []
        for i, label in enumerate(labels):
            hover_text = f"<b>{label}</b><br>"
            if hover_data and i < len(hover_data):
                for key, value in hover_data[i].items():
                    hover_text += f"{key}: {value}<br>"
            hover_texts.append(hover_text)

        # Create plot
        if n_components == 2:
            fig = self._create_2d_plot(reduced, labels, hover_texts, color_by, title, method)
        else:
            fig = self._create_3d_plot(reduced, labels, hover_texts, color_by, title, method)

        timings = {
            "reduction_time": reduction_time,
            "method": method,
            "n_components": n_components
        }

        return fig, timings

    def _create_2d_plot(
        self,
        reduced: np.ndarray,
        labels: List[str],
        hover_texts: List[str],
        color_by: Optional[List[str]],
        title: Optional[str],
        method: str
    ) -> go.Figure:
        """Create 2D scatter plot."""
        if title is None:
            title = f"Embedding Space Visualization ({method.upper()})"

        if color_by is not None:
            # Color by categories
            fig = px.scatter(
                x=reduced[:, 0],
                y=reduced[:, 1],
                color=color_by,
                hover_name=labels,
                labels={"x": f"{method.upper()}-1", "y": f"{method.upper()}-2", "color": "Category"},
                title=title
            )
            # Update hover data
            fig.update_traces(
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_texts
            )
        else:
            # Single color
            fig = go.Figure(data=[go.Scatter(
                x=reduced[:, 0],
                y=reduced[:, 1],
                mode='markers+text',
                text=labels,
                textposition="top center",
                textfont=dict(size=8),
                hovertext=hover_texts,
                hovertemplate="%{hovertext}<extra></extra>",
                marker=dict(
                    size=12,
                    color=np.arange(len(labels)),
                    colorscale=self.DEFAULT_CONFIG["color_scale"],
                    showscale=False,
                    line=dict(width=1, color='white')
                )
            )])

            fig.update_layout(
                title=title,
                xaxis_title=f"{method.upper()}-1",
                yaxis_title=f"{method.upper()}-2"
            )

        fig.update_layout(
            height=self.DEFAULT_CONFIG["plot_height"],
            width=self.DEFAULT_CONFIG["plot_width"],
            hovermode='closest'
        )

        return fig

    def _create_3d_plot(
        self,
        reduced: np.ndarray,
        labels: List[str],
        hover_texts: List[str],
        color_by: Optional[List[str]],
        title: Optional[str],
        method: str
    ) -> go.Figure:
        """Create 3D scatter plot."""
        if title is None:
            title = f"Embedding Space Visualization - 3D ({method.upper()})"

        if color_by is not None:
            fig = px.scatter_3d(
                x=reduced[:, 0],
                y=reduced[:, 1],
                z=reduced[:, 2],
                color=color_by,
                hover_name=labels,
                labels={"x": f"{method.upper()}-1", "y": f"{method.upper()}-2", "z": f"{method.upper()}-3"},
                title=title
            )
            fig.update_traces(
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_texts
            )
        else:
            fig = go.Figure(data=[go.Scatter3d(
                x=reduced[:, 0],
                y=reduced[:, 1],
                z=reduced[:, 2],
                mode='markers+text',
                text=labels,
                textposition="top center",
                textfont=dict(size=8),
                hovertext=hover_texts,
                hovertemplate="%{hovertext}<extra></extra>",
                marker=dict(
                    size=8,
                    color=np.arange(len(labels)),
                    colorscale=self.DEFAULT_CONFIG["color_scale"],
                    showscale=False,
                    line=dict(width=1, color='white')
                )
            )])

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=f"{method.upper()}-1",
                    yaxis_title=f"{method.upper()}-2",
                    zaxis_title=f"{method.upper()}-3"
                )
            )

        fig.update_layout(
            height=self.DEFAULT_CONFIG["plot_height"],
            width=self.DEFAULT_CONFIG["plot_width"]
        )

        return fig

    def plot_similarity_heatmap(
        self,
        embeddings: Union[List[List[float]], np.ndarray],
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        metric: Literal["cosine", "euclidean", "manhattan"] = "cosine"
    ) -> Tuple[go.Figure, Dict[str, any]]:
        """
        Create interactive similarity heatmap for embeddings.

        Args:
            embeddings: High-dimensional embedding vectors
            labels: Labels for each embedding
            title: Plot title
            metric: Distance/similarity metric to use

        Returns:
            Tuple of (Plotly figure, metadata dictionary)
        """
        # Convert to numpy array
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        # Prepare labels
        if labels is None:
            labels = [f"Doc {i+1}" for i in range(len(embeddings))]

        # Compute similarity matrix
        start_time = time.time()
        if metric == "cosine":
            # Cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / norms
            similarity_matrix = np.dot(normalized, normalized.T)
        elif metric == "euclidean":
            # Euclidean distance (convert to similarity)
            from scipy.spatial.distance import cdist
            distance_matrix = cdist(embeddings, embeddings, metric='euclidean')
            # Normalize to 0-1 range and invert (smaller distance = higher similarity)
            max_dist = distance_matrix.max()
            similarity_matrix = 1 - (distance_matrix / max_dist) if max_dist > 0 else np.ones_like(distance_matrix)
        elif metric == "manhattan":
            # Manhattan distance (convert to similarity)
            from scipy.spatial.distance import cdist
            distance_matrix = cdist(embeddings, embeddings, metric='cityblock')
            max_dist = distance_matrix.max()
            similarity_matrix = 1 - (distance_matrix / max_dist) if max_dist > 0 else np.ones_like(distance_matrix)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        computation_time = time.time() - start_time

        # Create hover text
        hover_texts = []
        for i in range(len(labels)):
            row_texts = []
            for j in range(len(labels)):
                text = f"<b>{labels[i]}</b> vs <b>{labels[j]}</b><br>"
                text += f"Similarity: {similarity_matrix[i, j]:.4f}"
                row_texts.append(text)
            hover_texts.append(row_texts)

        # Create heatmap
        if title is None:
            title = f"Similarity Heatmap ({metric.capitalize()} Similarity)"

        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            hovertext=hover_texts,
            hovertemplate="%{hovertext}<extra></extra>",
            colorscale='RdYlGn',
            zmid=0.5,
            colorbar=dict(title="Similarity")
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Documents",
            yaxis_title="Documents",
            height=max(600, len(labels) * 30),
            width=max(800, len(labels) * 30),
            xaxis=dict(tickangle=-45)
        )

        metadata = {
            "computation_time": computation_time,
            "metric": metric,
            "matrix_shape": similarity_matrix.shape,
            "avg_similarity": float(np.mean(similarity_matrix[np.triu_indices(len(labels), k=1)])),
            "min_similarity": float(np.min(similarity_matrix[np.triu_indices(len(labels), k=1)])),
            "max_similarity": float(np.max(similarity_matrix[np.triu_indices(len(labels), k=1)]))
        }

        return fig, metadata

    def plot_retrieval_comparison(
        self,
        query: str,
        results: Dict[str, List[Dict[str, any]]],
        top_k: int = 5,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create side-by-side comparison of different retrieval methods.

        Args:
            query: The search query
            results: Dictionary mapping method names to result lists
                    Each result should have 'document', 'score', and optionally 'rank'
            top_k: Number of top results to display
            title: Plot title

        Returns:
            Plotly figure with comparison bars
        """
        if title is None:
            title = f"Retrieval Method Comparison<br><sub>Query: '{query[:50]}...'</sub>"

        # Prepare data for plotting
        methods = list(results.keys())
        all_documents = set()

        # Collect all unique documents
        for method_results in results.values():
            for result in method_results[:top_k]:
                all_documents.add(result['document'])

        all_documents = sorted(list(all_documents))

        # Create score matrix
        score_data = []
        for doc in all_documents:
            doc_scores = {'document': doc}
            for method in methods:
                # Find this document in the results
                score = 0
                for result in results[method][:top_k]:
                    if result['document'] == doc:
                        score = result.get('score', 0)
                        break
                doc_scores[method] = score
            score_data.append(doc_scores)

        # Create grouped bar chart
        fig = go.Figure()

        for method in methods:
            scores = [item[method] for item in score_data]
            fig.add_trace(go.Bar(
                name=method,
                x=all_documents,
                y=scores,
                text=[f"{s:.3f}" if s > 0 else "" for s in scores],
                textposition='outside'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Documents",
            yaxis_title="Relevance Score",
            barmode='group',
            height=600,
            width=max(800, len(all_documents) * 80),
            xaxis=dict(tickangle=-45),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    def plot_score_distribution(
        self,
        scores: Dict[str, List[float]],
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create distribution plot for retrieval scores.

        Args:
            scores: Dictionary mapping method names to score lists
            title: Plot title

        Returns:
            Plotly figure with distribution plots
        """
        if title is None:
            title = "Score Distribution Comparison"

        fig = go.Figure()

        for method, score_list in scores.items():
            fig.add_trace(go.Box(
                y=score_list,
                name=method,
                boxmean='sd'
            ))

        fig.update_layout(
            title=title,
            yaxis_title="Relevance Score",
            height=500,
            width=800,
            showlegend=True
        )

        return fig


def create_plotly_config() -> Dict[str, any]:
    """
    Create standard Plotly configuration for all plots.

    Returns:
        Configuration dictionary
    """
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'visualization',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    }


def format_hover_data(documents: List[Dict[str, any]]) -> List[Dict[str, str]]:
    """
    Format document metadata for hover tooltips.

    Args:
        documents: List of document dictionaries

    Returns:
        Formatted hover data
    """
    formatted = []
    for doc in documents:
        hover_dict = {}
        if 'title' in doc:
            hover_dict['Title'] = doc['title']
        if 'type' in doc:
            hover_dict['Type'] = doc['type']
        if 'length' in doc:
            hover_dict['Length'] = f"{doc['length']} chars"
        if 'score' in doc:
            hover_dict['Score'] = f"{doc['score']:.4f}"
        formatted.append(hover_dict)
    return formatted


def get_color_scale(n_items: int, palette: str = "Viridis") -> List[str]:
    """
    Get a color scale for visualization.

    Args:
        n_items: Number of colors needed
        palette: Plotly color palette name

    Returns:
        List of color codes
    """
    import plotly.colors as pc

    # Get the colorscale
    if palette in pc.named_colorscales():
        colors = pc.sample_colorscale(palette, n_items)
    else:
        # Default to qualitative colors for discrete items
        if n_items <= 10:
            colors = pc.qualitative.Plotly[:n_items]
        else:
            colors = pc.sample_colorscale("Viridis", n_items)

    return colors
