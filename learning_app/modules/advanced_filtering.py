"""
Advanced Filtering Module for Retrieval Systems.

Provides metadata filtering, similarity thresholds, diversity controls,
and Maximal Marginal Relevance (MMR) for result optimization.
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class FilterConfig:
    """Configuration for advanced filtering."""

    # Metadata filters
    document_types: Optional[List[str]] = None
    date_range: Optional[tuple] = None  # (start, end)
    categories: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    custom_metadata: Optional[Dict[str, Any]] = None

    # Similarity filtering
    min_similarity: float = 0.0
    max_similarity: float = 1.0

    # Diversity controls
    enable_diversity: bool = False
    diversity_threshold: float = 0.7  # Similarity threshold for diversity

    # MMR parameters
    enable_mmr: bool = False
    mmr_lambda: float = 0.5  # Balance relevance (1.0) vs diversity (0.0)

    # Context window
    dynamic_context: bool = False
    context_size: int = 512


class AdvancedFilter:
    """
    Advanced filtering and result optimization for retrieval systems.

    Features:
    - Metadata-based filtering
    - Similarity threshold filtering
    - Diversity-based result optimization
    - Maximal Marginal Relevance (MMR)
    - Dynamic context window management
    """

    def __init__(self):
        """Initialize advanced filter."""
        pass

    def apply_filters(
        self,
        results: List[Dict],
        config: FilterConfig
    ) -> List[Dict]:
        """
        Apply all configured filters to results.

        Args:
            results: List of retrieval results
            config: Filter configuration

        Returns:
            Filtered results
        """
        filtered = results.copy()

        # Apply metadata filters
        if config.document_types:
            filtered = self.filter_by_document_type(
                filtered,
                config.document_types
            )

        if config.date_range:
            filtered = self.filter_by_date_range(
                filtered,
                config.date_range[0],
                config.date_range[1]
            )

        if config.categories:
            filtered = self.filter_by_categories(
                filtered,
                config.categories
            )

        if config.sources:
            filtered = self.filter_by_sources(
                filtered,
                config.sources
            )

        if config.custom_metadata:
            filtered = self.filter_by_custom_metadata(
                filtered,
                config.custom_metadata
            )

        # Apply similarity thresholds
        filtered = self.filter_by_similarity(
            filtered,
            config.min_similarity,
            config.max_similarity
        )

        # Apply diversity/MMR if enabled
        if config.enable_mmr:
            filtered = self.apply_mmr(
                filtered,
                lambda_param=config.mmr_lambda,
                top_k=len(filtered)
            )
        elif config.enable_diversity:
            filtered = self.apply_diversity_filter(
                filtered,
                config.diversity_threshold
            )

        return filtered

    def filter_by_document_type(
        self,
        results: List[Dict],
        allowed_types: List[str]
    ) -> List[Dict]:
        """
        Filter by document type.

        Args:
            results: Results to filter
            allowed_types: List of allowed document types

        Returns:
            Filtered results
        """
        return [
            r for r in results
            if r.get('metadata', {}).get('type', 'unknown') in allowed_types
        ]

    def filter_by_date_range(
        self,
        results: List[Dict],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[Dict]:
        """
        Filter by date range.

        Args:
            results: Results to filter
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Filtered results
        """
        filtered = []
        for result in results:
            date_str = result.get('metadata', {}).get('date')
            if not date_str:
                continue

            try:
                doc_date = datetime.fromisoformat(date_str)
                if start_date and doc_date < start_date:
                    continue
                if end_date and doc_date > end_date:
                    continue
                filtered.append(result)
            except (ValueError, TypeError):
                # Skip if date parsing fails
                continue

        return filtered

    def filter_by_categories(
        self,
        results: List[Dict],
        allowed_categories: List[str]
    ) -> List[Dict]:
        """
        Filter by document categories.

        Args:
            results: Results to filter
            allowed_categories: List of allowed categories

        Returns:
            Filtered results
        """
        return [
            r for r in results
            if r.get('metadata', {}).get('category', 'uncategorized') in allowed_categories
        ]

    def filter_by_sources(
        self,
        results: List[Dict],
        allowed_sources: List[str]
    ) -> List[Dict]:
        """
        Filter by document sources.

        Args:
            results: Results to filter
            allowed_sources: List of allowed sources

        Returns:
            Filtered results
        """
        return [
            r for r in results
            if r.get('metadata', {}).get('source', 'unknown') in allowed_sources
        ]

    def filter_by_custom_metadata(
        self,
        results: List[Dict],
        metadata_filters: Dict[str, Any]
    ) -> List[Dict]:
        """
        Filter by custom metadata fields.

        Args:
            results: Results to filter
            metadata_filters: Dictionary of field -> value filters

        Returns:
            Filtered results
        """
        filtered = []
        for result in results:
            metadata = result.get('metadata', {})
            matches = all(
                metadata.get(key) == value
                for key, value in metadata_filters.items()
            )
            if matches:
                filtered.append(result)

        return filtered

    def filter_by_similarity(
        self,
        results: List[Dict],
        min_score: float = 0.0,
        max_score: float = 1.0
    ) -> List[Dict]:
        """
        Filter by similarity score range.

        Args:
            results: Results to filter
            min_score: Minimum similarity score
            max_score: Maximum similarity score

        Returns:
            Filtered results
        """
        filtered = []
        for result in results:
            # Try different score keys
            score = result.get('similarity', result.get('score', 0.0))
            if min_score <= score <= max_score:
                filtered.append(result)

        return filtered

    def apply_diversity_filter(
        self,
        results: List[Dict],
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Apply diversity filtering to reduce redundant results.

        Removes documents that are too similar to already selected ones.

        Args:
            results: Results to filter
            similarity_threshold: Maximum allowed similarity between results

        Returns:
            Diverse results
        """
        if not results:
            return []

        diverse_results = [results[0]]  # Always keep first result

        for candidate in results[1:]:
            # Check similarity with all selected results
            is_diverse = True
            for selected in diverse_results:
                similarity = self._compute_text_similarity(
                    candidate.get('text', ''),
                    selected.get('text', '')
                )
                if similarity > similarity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                diverse_results.append(candidate)

        return diverse_results

    def apply_mmr(
        self,
        results: List[Dict],
        query_embedding: Optional[np.ndarray] = None,
        lambda_param: float = 0.5,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Apply Maximal Marginal Relevance (MMR) for result diversification.

        MMR balances relevance to query with diversity among results.

        Formula: MMR = λ * Relevance(d, q) - (1 - λ) * max Similarity(d, d_i)
        where d_i are already selected documents.

        Args:
            results: Candidate results
            query_embedding: Query embedding (if not available, use scores)
            lambda_param: Balance parameter (1.0 = only relevance, 0.0 = only diversity)
            top_k: Number of results to return

        Returns:
            MMR-ranked results
        """
        if not results:
            return []

        # If we have embeddings, use embedding-based MMR
        if query_embedding is not None and all('embedding' in r for r in results):
            return self._mmr_with_embeddings(
                results,
                query_embedding,
                lambda_param,
                top_k
            )

        # Otherwise, use text-based MMR
        return self._mmr_text_based(results, lambda_param, top_k)

    def _mmr_with_embeddings(
        self,
        results: List[Dict],
        query_embedding: np.ndarray,
        lambda_param: float,
        top_k: int
    ) -> List[Dict]:
        """
        MMR using embedding vectors.

        Args:
            results: Results with embeddings
            query_embedding: Query embedding vector
            lambda_param: Balance parameter
            top_k: Number of results

        Returns:
            MMR-ranked results
        """
        selected = []
        candidates = results.copy()

        while len(selected) < top_k and candidates:
            mmr_scores = []

            for candidate in candidates:
                candidate_emb = np.array(candidate['embedding'])

                # Relevance: cosine similarity to query
                relevance = self._cosine_similarity(query_embedding, candidate_emb)

                # Diversity: max similarity to selected documents
                if selected:
                    max_similarity = max(
                        self._cosine_similarity(
                            candidate_emb,
                            np.array(s['embedding'])
                        )
                        for s in selected
                    )
                else:
                    max_similarity = 0.0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr_score)

            # Select candidate with highest MMR score
            best_idx = np.argmax(mmr_scores)
            selected.append(candidates[best_idx])
            candidates.pop(best_idx)

        return selected

    def _mmr_text_based(
        self,
        results: List[Dict],
        lambda_param: float,
        top_k: int
    ) -> List[Dict]:
        """
        MMR using text similarity.

        Args:
            results: Results to rank
            lambda_param: Balance parameter
            top_k: Number of results

        Returns:
            MMR-ranked results
        """
        selected = []
        candidates = results.copy()

        while len(selected) < top_k and candidates:
            mmr_scores = []

            for candidate in candidates:
                # Relevance: use existing score
                relevance = candidate.get('similarity', candidate.get('score', 0.5))

                # Diversity: max text similarity to selected
                if selected:
                    max_similarity = max(
                        self._compute_text_similarity(
                            candidate.get('text', ''),
                            s.get('text', '')
                        )
                        for s in selected
                    )
                else:
                    max_similarity = 0.0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr_score)

            # Select best
            best_idx = np.argmax(mmr_scores)
            selected.append(candidates[best_idx])
            candidates.pop(best_idx)

        return selected

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _compute_text_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute simple text similarity (Jaccard).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Simple word-level Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def format_filter_summary(self, config: FilterConfig) -> str:
        """
        Format filter configuration as HTML summary.

        Args:
            config: Filter configuration

        Returns:
            HTML summary string
        """
        html = """
        <div style="font-family: sans-serif; padding: 15px; background: #f9fafb; border-radius: 8px;">
            <h4 style="margin-top: 0;">⚙️ Active Filters</h4>
        """

        # Metadata filters
        if any([config.document_types, config.categories, config.sources]):
            html += "<div style='margin-bottom: 10px;'><strong>Metadata Filters:</strong><ul style='margin: 5px 0;'>"

            if config.document_types:
                html += f"<li>Document Types: {', '.join(config.document_types)}</li>"
            if config.categories:
                html += f"<li>Categories: {', '.join(config.categories)}</li>"
            if config.sources:
                html += f"<li>Sources: {', '.join(config.sources)}</li>"

            html += "</ul></div>"

        # Similarity filter
        if config.min_similarity > 0.0 or config.max_similarity < 1.0:
            html += f"""
            <div style='margin-bottom: 10px;'>
                <strong>Similarity Range:</strong> {config.min_similarity:.2f} - {config.max_similarity:.2f}
            </div>
            """

        # Diversity/MMR
        if config.enable_mmr:
            html += f"""
            <div style='margin-bottom: 10px;'>
                <strong>MMR Enabled:</strong> λ = {config.mmr_lambda:.2f}
                <br><small>Balance: {config.mmr_lambda:.0%} relevance, {(1-config.mmr_lambda):.0%} diversity</small>
            </div>
            """
        elif config.enable_diversity:
            html += f"""
            <div style='margin-bottom: 10px;'>
                <strong>Diversity Filter:</strong> Threshold = {config.diversity_threshold:.2f}
            </div>
            """

        html += "</div>"
        return html
