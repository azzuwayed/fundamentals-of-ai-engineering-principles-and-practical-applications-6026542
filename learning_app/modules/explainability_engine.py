"""
Explainability engine module for the learning app (Phase 1 Enhancements).
Handles token-level analysis, BM25 breakdowns, and vector similarity explanations.
"""
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import re


class ExplainabilityEngine:
    """Handles explainability features for retrieval systems."""

    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        """
        Initialize the explainability engine.

        Args:
            embedding_model: Optional pre-loaded SentenceTransformer model
        """
        self.embedding_model = embedding_model

    def explain_token_similarity(
        self,
        query: str,
        document: str,
        model: SentenceTransformer,
        top_k: int = 10
    ) -> Dict[str, any]:
        """
        Explain which tokens contribute most to similarity between query and document.

        Args:
            query: Query text
            document: Document text
            model: SentenceTransformer model
            top_k: Number of top contributing tokens to return

        Returns:
            Dictionary with token contributions and analysis
        """
        start_time = time.time()

        # Tokenize
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)

        # Get full embeddings
        query_embedding = model.encode(query, convert_to_numpy=True)
        doc_embedding = model.encode(document, convert_to_numpy=True)

        # Compute overall similarity
        overall_similarity = float(np.dot(query_embedding, doc_embedding) /
                                   (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)))

        # Analyze token-level contributions
        token_contributions = []

        # For each query token, compute its contribution
        for token in query_tokens:
            if len(token.strip()) < 2:  # Skip very short tokens
                continue

            # Create document with this token emphasized
            # Simple approximation: compute similarity when token is present/absent
            token_embedding = model.encode(token, convert_to_numpy=True)

            # Compute how similar this token is to the document
            token_doc_sim = float(np.dot(token_embedding, doc_embedding) /
                                  (np.linalg.norm(token_embedding) * np.linalg.norm(doc_embedding)))

            # Check if token appears in document
            in_document = token.lower() in document.lower()

            token_contributions.append({
                "token": token,
                "contribution": abs(token_doc_sim),
                "similarity_to_doc": token_doc_sim,
                "in_document": in_document,
                "frequency": document.lower().count(token.lower())
            })

        # Sort by contribution
        token_contributions.sort(key=lambda x: x['contribution'], reverse=True)
        top_contributions = token_contributions[:top_k]

        computation_time = time.time() - start_time

        return {
            "query": query,
            "document_preview": document[:200] + "..." if len(document) > 200 else document,
            "overall_similarity": overall_similarity,
            "token_contributions": top_contributions,
            "total_tokens_analyzed": len(query_tokens),
            "computation_time": computation_time
        }

    def explain_bm25_score(
        self,
        query: str,
        document: str,
        corpus_stats: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """
        Explain BM25 score calculation for query-document pair.

        Args:
            query: Query text
            document: Document text
            corpus_stats: Optional corpus statistics (IDF values, avg_doc_length, etc.)

        Returns:
            Dictionary with detailed BM25 breakdown
        """
        start_time = time.time()

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        # Tokenize
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)

        # Document stats
        doc_length = len(doc_tokens)

        # Default corpus stats if not provided
        if corpus_stats is None:
            avg_doc_length = doc_length
            total_docs = 1
        else:
            avg_doc_length = corpus_stats.get('avg_doc_length', doc_length)
            total_docs = corpus_stats.get('total_docs', 1)

        # Term frequency in document
        term_frequencies = {}
        for token in doc_tokens:
            token_lower = token.lower()
            term_frequencies[token_lower] = term_frequencies.get(token_lower, 0) + 1

        # Calculate BM25 components for each query term
        term_scores = []
        total_score = 0

        for token in query_tokens:
            token_lower = token.lower()
            tf = term_frequencies.get(token_lower, 0)

            # IDF calculation (simplified)
            if corpus_stats and 'idf_scores' in corpus_stats:
                idf = corpus_stats['idf_scores'].get(token_lower, 0)
            else:
                # Simple IDF approximation
                doc_freq = 1 if tf > 0 else 0
                idf = np.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

            # BM25 formula components
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))

            if denominator > 0:
                term_score = idf * (numerator / denominator)
            else:
                term_score = 0

            total_score += term_score

            term_scores.append({
                "term": token,
                "tf": tf,
                "idf": float(idf),
                "doc_freq": doc_length,
                "bm25_component": float(term_score),
                "in_document": tf > 0
            })

        # Sort by contribution
        term_scores.sort(key=lambda x: x['bm25_component'], reverse=True)

        computation_time = time.time() - start_time

        return {
            "query": query,
            "document_preview": document[:200] + "..." if len(document) > 200 else document,
            "total_bm25_score": float(total_score),
            "term_scores": term_scores,
            "parameters": {
                "k1": k1,
                "b": b,
                "doc_length": doc_length,
                "avg_doc_length": avg_doc_length
            },
            "formula_explanation": (
                f"BM25 = Σ IDF(term) × (TF(term) × (k1 + 1)) / "
                f"(TF(term) + k1 × (1 - b + b × (doc_len / avg_doc_len)))<br>"
                f"where k1={k1}, b={b}"
            ),
            "computation_time": computation_time
        }

    def explain_vector_similarity(
        self,
        query: str,
        document: str,
        model: SentenceTransformer
    ) -> Dict[str, any]:
        """
        Explain vector similarity calculation with component-level breakdown.

        Args:
            query: Query text
            document: Document text
            model: SentenceTransformer model

        Returns:
            Dictionary with detailed similarity breakdown
        """
        start_time = time.time()

        # Generate embeddings
        query_embedding = model.encode(query, convert_to_numpy=True)
        doc_embedding = model.encode(document, convert_to_numpy=True)

        # Compute cosine similarity components
        dot_product = float(np.dot(query_embedding, doc_embedding))
        query_norm = float(np.linalg.norm(query_embedding))
        doc_norm = float(np.linalg.norm(doc_embedding))

        cosine_similarity = dot_product / (query_norm * doc_norm)

        # Analyze component-wise contributions
        component_contributions = query_embedding * doc_embedding

        # Get top contributing dimensions
        top_dimensions = []
        sorted_indices = np.argsort(np.abs(component_contributions))[::-1][:10]

        for idx in sorted_indices:
            top_dimensions.append({
                "dimension": int(idx),
                "query_value": float(query_embedding[idx]),
                "doc_value": float(doc_embedding[idx]),
                "contribution": float(component_contributions[idx]),
                "percentage": float(abs(component_contributions[idx]) / abs(dot_product) * 100) if dot_product != 0 else 0
            })

        # Compute Euclidean distance as alternative metric
        euclidean_distance = float(np.linalg.norm(query_embedding - doc_embedding))

        computation_time = time.time() - start_time

        return {
            "query": query,
            "document_preview": document[:200] + "..." if len(document) > 200 else document,
            "cosine_similarity": float(cosine_similarity),
            "dot_product": dot_product,
            "query_norm": query_norm,
            "doc_norm": doc_norm,
            "euclidean_distance": euclidean_distance,
            "embedding_dimensions": len(query_embedding),
            "top_contributing_dimensions": top_dimensions,
            "formula_explanation": (
                "Cosine Similarity = (Query · Document) / (||Query|| × ||Document||)<br>"
                f"= {dot_product:.4f} / ({query_norm:.4f} × {doc_norm:.4f})<br>"
                f"= {cosine_similarity:.4f}"
            ),
            "interpretation": self._interpret_similarity(cosine_similarity),
            "computation_time": computation_time
        }

    def highlight_important_tokens(
        self,
        text: str,
        token_weights: List[Dict[str, any]],
        max_weight: Optional[float] = None
    ) -> str:
        """
        Generate HTML with highlighted tokens based on importance weights.

        Args:
            text: Original text
            token_weights: List of dicts with 'token' and 'contribution' keys
            max_weight: Maximum weight for normalization (auto-computed if None)

        Returns:
            HTML string with highlighted spans
        """
        if not token_weights:
            return text

        # Normalize weights
        if max_weight is None:
            max_weight = max([tw['contribution'] for tw in token_weights])

        if max_weight == 0:
            return text

        # Create a dict for quick lookup
        weight_dict = {tw['token'].lower(): tw['contribution'] / max_weight
                      for tw in token_weights}

        # Tokenize and highlight
        tokens = self._tokenize(text)
        highlighted_parts = []

        for token in tokens:
            token_lower = token.lower()
            if token_lower in weight_dict:
                weight = weight_dict[token_lower]
                # Color from light yellow (low) to dark red (high)
                opacity = 0.3 + (weight * 0.7)  # Range from 0.3 to 1.0
                color = self._get_heatmap_color(weight)
                highlighted = f'<span style="background-color: {color}; padding: 2px 4px; margin: 0 2px; border-radius: 3px; font-weight: {500 if weight > 0.5 else 400};" title="Contribution: {weight:.3f}">{token}</span>'
                highlighted_parts.append(highlighted)
            else:
                highlighted_parts.append(token)

        return ' '.join(highlighted_parts)

    def generate_explanation_summary(
        self,
        query: str,
        document: str,
        model: SentenceTransformer,
        include_bm25: bool = True,
        corpus_stats: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Generate comprehensive explanation combining all methods.

        Args:
            query: Query text
            document: Document text
            model: SentenceTransformer model
            include_bm25: Whether to include BM25 explanation
            corpus_stats: Optional corpus statistics for BM25

        Returns:
            Combined explanation dictionary
        """
        explanations = {}

        # Token-level analysis
        explanations['token_analysis'] = self.explain_token_similarity(
            query, document, model
        )

        # Vector similarity
        explanations['vector_similarity'] = self.explain_vector_similarity(
            query, document, model
        )

        # BM25 (if requested)
        if include_bm25:
            explanations['bm25_breakdown'] = self.explain_bm25_score(
                query, document, corpus_stats
            )

        # Generate highlighted text
        token_weights = explanations['token_analysis']['token_contributions']
        explanations['highlighted_query'] = self.highlight_important_tokens(
            query, token_weights
        )

        return explanations

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple tokenization helper.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple word tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    @staticmethod
    def _interpret_similarity(score: float) -> str:
        """
        Interpret similarity score.

        Args:
            score: Similarity score (0-1)

        Returns:
            Human-readable interpretation
        """
        if score >= 0.9:
            return "Very High - Nearly identical meaning"
        elif score >= 0.7:
            return "High - Semantically similar"
        elif score >= 0.5:
            return "Moderate - Related topics"
        elif score >= 0.3:
            return "Low - Somewhat related"
        else:
            return "Very Low - Unrelated"

    @staticmethod
    def _get_heatmap_color(weight: float) -> str:
        """
        Get color for heatmap based on weight (0-1).

        Args:
            weight: Normalized weight value

        Returns:
            RGB color string
        """
        # Gradient from light yellow (#FFF9C4) to dark red (#D32F2F)
        if weight < 0.33:
            # Light yellow to orange
            r = int(255)
            g = int(249 - (weight * 3) * 50)
            b = int(196 - (weight * 3) * 150)
        elif weight < 0.67:
            # Orange to red
            adjusted = (weight - 0.33) * 3
            r = int(255)
            g = int(199 - adjusted * 120)
            b = int(46)
        else:
            # Red to dark red
            adjusted = (weight - 0.67) * 3
            r = int(255 - adjusted * 44)
            g = int(79 - adjusted * 32)
            b = int(46 - adjusted * 30)

        return f"rgba({r}, {g}, {b}, 0.7)"


def format_bm25_table(term_scores: List[Dict[str, any]]) -> str:
    """
    Format BM25 term scores as an HTML table.

    Args:
        term_scores: List of term score dictionaries

    Returns:
        HTML table string
    """
    html = '<table style="width: 100%; border-collapse: collapse; font-size: 14px;">'
    html += '''
    <thead>
        <tr style="background-color: #f5f5f5; border-bottom: 2px solid #ddd;">
            <th style="padding: 8px; text-align: left;">Term</th>
            <th style="padding: 8px; text-align: right;">TF</th>
            <th style="padding: 8px; text-align: right;">IDF</th>
            <th style="padding: 8px; text-align: right;">BM25 Score</th>
            <th style="padding: 8px; text-align: center;">In Doc?</th>
        </tr>
    </thead>
    <tbody>
    '''

    for score in term_scores:
        in_doc_icon = "✓" if score['in_document'] else "✗"
        in_doc_color = "#4CAF50" if score['in_document'] else "#F44336"

        html += f'''
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding: 8px;"><strong>{score['term']}</strong></td>
            <td style="padding: 8px; text-align: right;">{score['tf']}</td>
            <td style="padding: 8px; text-align: right;">{score['idf']:.4f}</td>
            <td style="padding: 8px; text-align: right; font-weight: bold;">{score['bm25_component']:.4f}</td>
            <td style="padding: 8px; text-align: center; color: {in_doc_color};">{in_doc_icon}</td>
        </tr>
        '''

    html += '</tbody></table>'
    return html


def format_similarity_breakdown(explanation: Dict[str, any]) -> str:
    """
    Format vector similarity breakdown as HTML.

    Args:
        explanation: Similarity explanation dictionary

    Returns:
        HTML string with formatted breakdown
    """
    html = '<div style="font-family: monospace; background-color: #f5f5f5; padding: 15px; border-radius: 5px;">'

    # Main formula
    html += f'<div style="margin-bottom: 15px;">'
    html += f'<strong>Formula:</strong><br>'
    html += f'{explanation["formula_explanation"]}'
    html += f'</div>'

    # Components
    html += '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 15px;">'
    html += f'<div><strong>Dot Product:</strong> {explanation["dot_product"]:.4f}</div>'
    html += f'<div><strong>Cosine Similarity:</strong> {explanation["cosine_similarity"]:.4f}</div>'
    html += f'<div><strong>Query Magnitude:</strong> {explanation["query_norm"]:.4f}</div>'
    html += f'<div><strong>Document Magnitude:</strong> {explanation["doc_norm"]:.4f}</div>'
    html += '</div>'

    # Interpretation
    html += f'<div style="margin-top: 10px; padding: 10px; background-color: white; border-left: 4px solid #2196F3;">'
    html += f'<strong>Interpretation:</strong> {explanation["interpretation"]}'
    html += '</div>'

    html += '</div>'
    return html
