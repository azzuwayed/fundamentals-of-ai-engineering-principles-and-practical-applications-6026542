"""
Multi-Query Engine for Advanced Retrieval.

Provides query decomposition, parallel execution, result fusion,
and multi-query strategy management.
"""

import time
from typing import List, Dict, Tuple, Callable, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict


@dataclass
class QueryVariation:
    """A single query variation for multi-query search."""

    original_query: str
    variation: str
    variation_type: str  # decomposed, expanded, rephrased, refined
    strategy: str  # Strategy that generated this variation


@dataclass
class MultiQueryResult:
    """Results from multi-query execution."""

    original_query: str
    variations: List[QueryVariation]
    individual_results: Dict[str, List[Dict]]  # variation -> results
    fused_results: List[Dict]
    fusion_strategy: str
    metrics: Dict[str, Any]
    execution_time: float


class MultiQueryEngine:
    """
    Multi-query retrieval engine with decomposition and fusion.

    Features:
    - Query decomposition (break complex queries)
    - Query expansion (add synonyms/context)
    - Query rephrasing (alternative formulations)
    - Parallel execution
    - Result fusion with multiple strategies
    - Performance comparison
    """

    DECOMPOSITION_TEMPLATES = [
        "What is {concept}?",
        "Explain {concept}",
        "How does {concept} work?",
        "What are the key features of {concept}?",
        "What are examples of {concept}?"
    ]

    EXPANSION_PATTERNS = [
        "{query} explained",
        "{query} overview",
        "{query} fundamentals",
        "{query} in practice",
        "{query} best practices"
    ]

    def __init__(self, max_workers: int = 4):
        """
        Initialize multi-query engine.

        Args:
            max_workers: Maximum parallel query executions
        """
        self.max_workers = max_workers

    def decompose_query(
        self,
        query: str,
        num_variations: int = 3
    ) -> List[QueryVariation]:
        """
        Decompose complex query into sub-queries.

        Args:
            query: Original complex query
            num_variations: Number of decomposition variations

        Returns:
            List of QueryVariation objects
        """
        variations = []

        # Extract key concepts (simple noun phrase extraction)
        concepts = self._extract_concepts(query)

        # Generate sub-queries for each concept
        for concept in concepts[:num_variations]:
            for template in self.DECOMPOSITION_TEMPLATES[:2]:  # Limit templates
                variation = template.format(concept=concept)
                variations.append(QueryVariation(
                    original_query=query,
                    variation=variation,
                    variation_type="decomposed",
                    strategy="concept_decomposition"
                ))

        return variations[:num_variations]

    def expand_query(
        self,
        query: str,
        num_expansions: int = 3
    ) -> List[QueryVariation]:
        """
        Expand query with additional context and synonyms.

        Args:
            query: Original query
            num_expansions: Number of expansion variations

        Returns:
            List of QueryVariation objects
        """
        variations = []

        for pattern in self.EXPANSION_PATTERNS[:num_expansions]:
            expansion = pattern.format(query=query)
            variations.append(QueryVariation(
                original_query=query,
                variation=expansion,
                variation_type="expanded",
                strategy="semantic_expansion"
            ))

        return variations

    def rephrase_query(
        self,
        query: str,
        num_rephrasings: int = 3
    ) -> List[QueryVariation]:
        """
        Generate alternative phrasings of the query.

        Args:
            query: Original query
            num_rephrasings: Number of rephrasing variations

        Returns:
            List of QueryVariation objects
        """
        variations = []

        # Question reformulation patterns
        reformulations = []

        # Convert statement to question
        if not query.strip().endswith('?'):
            reformulations.append(f"What is {query}?")
            reformulations.append(f"How does {query} work?")
            reformulations.append(f"Explain {query}")

        # Convert question to statement
        if query.strip().endswith('?'):
            # Remove question words and punctuation
            statement = query.replace('?', '').strip()
            for qword in ['what is', 'how does', 'why does', 'when does', 'where is']:
                if statement.lower().startswith(qword):
                    statement = statement[len(qword):].strip()
            reformulations.append(statement)

        # Add specificity
        if len(query.split()) < 5:
            reformulations.append(f"{query} in detail")
            reformulations.append(f"{query} comprehensive")

        for rephrasing in reformulations[:num_rephrasings]:
            variations.append(QueryVariation(
                original_query=query,
                variation=rephrasing,
                variation_type="rephrased",
                strategy="query_reformulation"
            ))

        return variations

    def _extract_concepts(self, query: str) -> List[str]:
        """
        Extract key concepts from query.

        Simple implementation using capitalized words and noun-like patterns.

        Args:
            query: Query string

        Returns:
            List of extracted concepts
        """
        import re

        # Remove common question words
        cleaned = re.sub(
            r'\b(what|how|why|when|where|who|is|are|the|a|an|and|or)\b',
            '',
            query,
            flags=re.I
        )

        # Extract words (prefer capitalized as likely concepts)
        words = cleaned.split()
        concepts = []

        # Multi-word concepts (consecutive capitalized)
        i = 0
        while i < len(words):
            if words[i] and words[i][0].isupper():
                concept = words[i]
                j = i + 1
                while j < len(words) and words[j] and words[j][0].isupper():
                    concept += " " + words[j]
                    j += 1
                concepts.append(concept)
                i = j
            else:
                if words[i].strip():
                    concepts.append(words[i])
                i += 1

        return [c.strip() for c in concepts if c.strip()]

    def execute_parallel(
        self,
        variations: List[QueryVariation],
        retrieval_fn: Callable[[str], List[Dict]],
        max_results_per_query: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Execute multiple queries in parallel.

        Args:
            variations: List of query variations to execute
            retrieval_fn: Function that takes query string and returns results
            max_results_per_query: Maximum results per individual query

        Returns:
            Dictionary mapping variation text to results
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all queries
            future_to_variation = {
                executor.submit(retrieval_fn, var.variation, max_results_per_query): var
                for var in variations
            }

            # Collect results as they complete
            for future in as_completed(future_to_variation):
                variation = future_to_variation[future]
                try:
                    query_results = future.result()
                    results[variation.variation] = query_results
                except Exception as e:
                    print(f"Error executing query '{variation.variation}': {e}")
                    results[variation.variation] = []

        return results

    def fuse_results(
        self,
        individual_results: Dict[str, List[Dict]],
        strategy: str = "weighted_voting",
        top_k: int = 10
    ) -> List[Dict]:
        """
        Fuse results from multiple queries.

        Args:
            individual_results: Dictionary mapping query to results
            strategy: Fusion strategy (weighted_voting, round_robin, score_aggregation)
            top_k: Number of final results to return

        Returns:
            Fused and ranked result list
        """
        if strategy == "weighted_voting":
            return self._fuse_weighted_voting(individual_results, top_k)
        elif strategy == "round_robin":
            return self._fuse_round_robin(individual_results, top_k)
        elif strategy == "score_aggregation":
            return self._fuse_score_aggregation(individual_results, top_k)
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

    def _fuse_weighted_voting(
        self,
        individual_results: Dict[str, List[Dict]],
        top_k: int
    ) -> List[Dict]:
        """
        Fuse using weighted voting (rank-based).

        Documents appearing in multiple result sets get higher scores.

        Args:
            individual_results: Query -> results mapping
            top_k: Number of results to return

        Returns:
            Fused results
        """
        # Track document scores and data
        doc_scores = defaultdict(float)
        doc_data = {}

        for query, results in individual_results.items():
            for rank, result in enumerate(results):
                # Use content as doc ID (could also use metadata)
                doc_id = result.get('text', '')[:100]  # First 100 chars as ID

                # Weighted by inverse rank
                weight = 1.0 / (rank + 1)
                doc_scores[doc_id] += weight

                # Store full document data
                if doc_id not in doc_data:
                    doc_data[doc_id] = result

        # Sort by score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top-k with scores
        fused = []
        for doc_id, score in sorted_docs[:top_k]:
            result = doc_data[doc_id].copy()
            result['fusion_score'] = round(score, 4)
            result['fusion_rank'] = len(fused) + 1
            fused.append(result)

        return fused

    def _fuse_round_robin(
        self,
        individual_results: Dict[str, List[Dict]],
        top_k: int
    ) -> List[Dict]:
        """
        Fuse using round-robin (alternate between queries).

        Args:
            individual_results: Query -> results mapping
            top_k: Number of results to return

        Returns:
            Fused results
        """
        fused = []
        seen_docs = set()

        # Get max length
        max_len = max(len(results) for results in individual_results.values())

        # Round-robin through positions
        for i in range(max_len):
            for query, results in individual_results.items():
                if i < len(results):
                    result = results[i]
                    doc_id = result.get('text', '')[:100]

                    if doc_id not in seen_docs:
                        result_copy = result.copy()
                        result_copy['fusion_score'] = 1.0 / (len(fused) + 1)
                        result_copy['fusion_rank'] = len(fused) + 1
                        fused.append(result_copy)
                        seen_docs.add(doc_id)

                        if len(fused) >= top_k:
                            return fused

        return fused

    def _fuse_score_aggregation(
        self,
        individual_results: Dict[str, List[Dict]],
        top_k: int
    ) -> List[Dict]:
        """
        Fuse by aggregating similarity scores.

        Args:
            individual_results: Query -> results mapping
            top_k: Number of results to return

        Returns:
            Fused results
        """
        doc_scores = defaultdict(list)
        doc_data = {}

        for query, results in individual_results.items():
            for result in results:
                doc_id = result.get('text', '')[:100]

                # Get score (try different score keys)
                score = result.get('similarity', result.get('score', 0.0))
                doc_scores[doc_id].append(score)

                if doc_id not in doc_data:
                    doc_data[doc_id] = result

        # Aggregate scores (average)
        aggregated = {}
        for doc_id, scores in doc_scores.items():
            aggregated[doc_id] = sum(scores) / len(scores)

        # Sort by aggregated score
        sorted_docs = sorted(
            aggregated.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top-k
        fused = []
        for doc_id, score in sorted_docs[:top_k]:
            result = doc_data[doc_id].copy()
            result['fusion_score'] = round(score, 4)
            result['fusion_rank'] = len(fused) + 1
            fused.append(result)

        return fused

    def execute_multi_query(
        self,
        query: str,
        retrieval_fn: Callable[[str, int], List[Dict]],
        strategy: str = "hybrid",
        fusion_method: str = "weighted_voting",
        num_variations: int = 3,
        max_results_per_query: int = 5,
        top_k: int = 10
    ) -> MultiQueryResult:
        """
        Execute complete multi-query pipeline.

        Args:
            query: Original query
            retrieval_fn: Retrieval function (query, top_k) -> results
            strategy: Variation strategy (decompose, expand, rephrase, hybrid)
            fusion_method: Result fusion method
            num_variations: Number of query variations
            max_results_per_query: Results per individual query
            top_k: Final result count

        Returns:
            MultiQueryResult with complete analysis
        """
        start_time = time.time()

        # Generate variations based on strategy
        if strategy == "decompose":
            variations = self.decompose_query(query, num_variations)
        elif strategy == "expand":
            variations = self.expand_query(query, num_variations)
        elif strategy == "rephrase":
            variations = self.rephrase_query(query, num_variations)
        elif strategy == "hybrid":
            # Mix of all strategies
            variations = []
            n_each = max(1, num_variations // 3)
            variations.extend(self.decompose_query(query, n_each))
            variations.extend(self.expand_query(query, n_each))
            variations.extend(self.rephrase_query(query, n_each))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Execute queries in parallel
        individual_results = self.execute_parallel(
            variations,
            retrieval_fn,
            max_results_per_query
        )

        # Fuse results
        fused_results = self.fuse_results(
            individual_results,
            fusion_method,
            top_k
        )

        # Compute metrics
        execution_time = time.time() - start_time
        metrics = self._compute_metrics(
            individual_results,
            fused_results,
            execution_time
        )

        return MultiQueryResult(
            original_query=query,
            variations=variations,
            individual_results=individual_results,
            fused_results=fused_results,
            fusion_strategy=fusion_method,
            metrics=metrics,
            execution_time=execution_time
        )

    def _compute_metrics(
        self,
        individual_results: Dict[str, List[Dict]],
        fused_results: List[Dict],
        execution_time: float
    ) -> Dict[str, Any]:
        """
        Compute metrics for multi-query execution.

        Args:
            individual_results: Individual query results
            fused_results: Fused results
            execution_time: Total execution time

        Returns:
            Dictionary of metrics
        """
        # Count unique documents across all queries
        all_docs = set()
        for results in individual_results.values():
            for result in results:
                doc_id = result.get('text', '')[:100]
                all_docs.add(doc_id)

        # Count documents in fused results
        fused_docs = set()
        for result in fused_results:
            doc_id = result.get('text', '')[:100]
            fused_docs.add(doc_id)

        # Coverage: how many unique docs ended up in fused results
        coverage = len(fused_docs) / len(all_docs) if all_docs else 0

        return {
            'num_queries_executed': len(individual_results),
            'total_results_retrieved': sum(
                len(results) for results in individual_results.values()
            ),
            'unique_documents': len(all_docs),
            'fused_documents': len(fused_results),
            'coverage': round(coverage, 3),
            'avg_results_per_query': round(
                sum(len(r) for r in individual_results.values()) / max(len(individual_results), 1),
                2
            ),
            'execution_time_seconds': round(execution_time, 3),
            'queries_per_second': round(
                len(individual_results) / max(execution_time, 0.001),
                2
            )
        }

    def format_comparison_table(
        self,
        multi_result: MultiQueryResult
    ) -> str:
        """
        Format multi-query results as comparison HTML table.

        Args:
            multi_result: MultiQueryResult object

        Returns:
            HTML table string
        """
        html = """
        <div style="font-family: sans-serif;">
            <h3>🔍 Multi-Query Comparison</h3>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                <thead>
                    <tr style="background: var(--background-fill-secondary); text-align: left;">
                        <th style="padding: 12px; border: 1px solid var(--border-color-primary);">Query Variation</th>
                        <th style="padding: 12px; border: 1px solid var(--border-color-primary);">Type</th>
                        <th style="padding: 12px; border: 1px solid var(--border-color-primary);">Results</th>
                    </tr>
                </thead>
                <tbody>
        """

        for variation in multi_result.variations:
            result_count = len(multi_result.individual_results.get(variation.variation, []))
            type_badge_colors = {
                'decomposed': '#3b82f6',
                'expanded': '#10b981',
                'rephrased': '#8b5cf6'
            }
            badge_color = type_badge_colors.get(variation.variation_type, '#6b7280')

            html += f"""
                    <tr style="border-bottom: 1px solid var(--border-color-primary);">
                        <td style="padding: 12px;">{variation.variation}</td>
                        <td style="padding: 12px;">
                            <span style="background: {badge_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">
                                {variation.variation_type}
                            </span>
                        </td>
                        <td style="padding: 12px; font-weight: 600;">{result_count}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>

            <div style="background: var(--block-background-fill); border: 1px solid var(--border-color-primary); padding: 15px; border-radius: 6px;">
                <h4 style="margin-top: 0;">📊 Execution Metrics</h4>
                <table style="width: 100%; border-collapse: collapse;">
        """

        metrics = multi_result.metrics
        metric_rows = [
            ("Total Queries", metrics['num_queries_executed']),
            ("Unique Documents", metrics['unique_documents']),
            ("Fused Results", metrics['fused_documents']),
            ("Coverage", f"{metrics['coverage']:.1%}"),
            ("Execution Time", f"{metrics['execution_time_seconds']:.3f}s"),
            ("Queries/Second", f"{metrics['queries_per_second']:.2f}")
        ]

        for label, value in metric_rows:
            html += f"""
                    <tr style="border-bottom: 1px solid var(--border-color-primary);">
                        <td style="padding: 8px 0;">{label}</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600;">{value}</td>
                    </tr>
            """

        html += """
                </table>
            </div>
        </div>
        """

        return html
