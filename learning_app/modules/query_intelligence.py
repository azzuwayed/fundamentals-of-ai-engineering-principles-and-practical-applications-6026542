"""
Query Intelligence Module for Advanced Retrieval.

Provides query analysis, intent classification, complexity scoring,
and intelligent query expansion/suggestions.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class QueryAnalysis:
    """Structured analysis of a search query."""

    original_query: str
    tokens: List[str]
    keywords: List[str]
    intent: str
    intent_confidence: float
    complexity_score: float
    suggestions: List[str]
    expanded_terms: Dict[str, List[str]]
    metrics: Dict[str, any]


class QueryIntelligence:
    """
    Intelligent query analysis and optimization engine.

    Features:
    - Intent classification (factual, conceptual, exploratory, comparison)
    - Complexity scoring (0-1 scale)
    - Query suggestions and improvements
    - Semantic term expansion
    - Query structure analysis
    """

    # Intent patterns for classification
    INTENT_PATTERNS = {
        'factual': [
            r'\bwhat is\b', r'\bdefine\b', r'\bwho is\b', r'\bwhen\b',
            r'\bwhere\b', r'\bwhich\b', r'\bhow many\b', r'\bhow much\b'
        ],
        'conceptual': [
            r'\bwhy\b', r'\bhow does\b', r'\bexplain\b', r'\bdescribe\b',
            r'\bunderstand\b', r'\bconcept\b', r'\btheory\b', r'\bprinciple\b'
        ],
        'exploratory': [
            r'\bexplore\b', r'\bdiscover\b', r'\bfind out\b', r'\blearn about\b',
            r'\bresearch\b', r'\binvestigate\b', r'\bsurvey\b'
        ],
        'comparison': [
            r'\bcompare\b', r'\bversus\b', r'\bvs\b', r'\bdifference\b',
            r'\bsimilar\b', r'\balternative\b', r'\bbetter\b', r'\bworse\b'
        ]
    }

    # Stop words for keyword extraction
    STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how'
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize query intelligence engine.

        Args:
            model_name: SentenceTransformer model for semantic analysis
        """
        self.model = SentenceTransformer(model_name)

    def analyze_query(
        self,
        query: str,
        context_docs: Optional[List[str]] = None
    ) -> QueryAnalysis:
        """
        Perform comprehensive query analysis.

        Args:
            query: Search query to analyze
            context_docs: Optional document corpus for context-aware analysis

        Returns:
            QueryAnalysis with full analysis results
        """
        # Tokenize and extract keywords
        tokens = self._tokenize(query)
        keywords = self._extract_keywords(tokens)

        # Classify intent
        intent, intent_confidence = self._classify_intent(query)

        # Score complexity
        complexity_score = self._score_complexity(query, tokens, keywords)

        # Generate suggestions
        suggestions = self._generate_suggestions(
            query, intent, complexity_score, keywords
        )

        # Expand terms semantically
        expanded_terms = self._expand_terms(keywords, context_docs)

        # Compute metrics
        metrics = self._compute_metrics(query, tokens, keywords)

        return QueryAnalysis(
            original_query=query,
            tokens=tokens,
            keywords=keywords,
            intent=intent,
            intent_confidence=intent_confidence,
            complexity_score=complexity_score,
            suggestions=suggestions,
            expanded_terms=expanded_terms,
            metrics=metrics
        )

    def _tokenize(self, query: str) -> List[str]:
        """
        Tokenize query into words.

        Args:
            query: Query string

        Returns:
            List of tokens
        """
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', query.lower())
        return tokens

    def _extract_keywords(self, tokens: List[str]) -> List[str]:
        """
        Extract meaningful keywords from tokens.

        Args:
            tokens: List of tokens

        Returns:
            List of keywords (non-stop words)
        """
        keywords = [t for t in tokens if t not in self.STOP_WORDS and len(t) > 2]
        return keywords

    def _classify_intent(self, query: str) -> Tuple[str, float]:
        """
        Classify query intent using pattern matching.

        Args:
            query: Query string

        Returns:
            Tuple of (intent_type, confidence)
        """
        query_lower = query.lower()

        # Score each intent type
        intent_scores = {}
        for intent_type, patterns in self.INTENT_PATTERNS.items():
            score = sum(
                1 for pattern in patterns
                if re.search(pattern, query_lower)
            )
            intent_scores[intent_type] = score

        # If no patterns match, default to exploratory
        if not any(intent_scores.values()):
            return "exploratory", 0.5

        # Get intent with highest score
        max_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[max_intent]

        # Normalize confidence (cap at 1.0)
        confidence = min(max_score / 2.0, 1.0)

        return max_intent, confidence

    def _score_complexity(
        self,
        query: str,
        tokens: List[str],
        keywords: List[str]
    ) -> float:
        """
        Score query complexity (0-1 scale).

        Factors:
        - Length (longer = more complex)
        - Keyword density
        - Special characters
        - Multi-clause structure

        Args:
            query: Original query
            tokens: Tokenized query
            keywords: Extracted keywords

        Returns:
            Complexity score (0 = simple, 1 = complex)
        """
        # Factor 1: Length (normalize to 0-1, cap at 50 words)
        length_score = min(len(tokens) / 50.0, 1.0)

        # Factor 2: Keyword density (higher = more complex)
        keyword_density = len(keywords) / max(len(tokens), 1)

        # Factor 3: Special characters (quotes, operators, etc.)
        special_chars = len(re.findall(r'["\(\)\[\]&|!]', query))
        special_score = min(special_chars / 5.0, 1.0)

        # Factor 4: Multi-clause (commas, semicolons, conjunctions)
        clauses = len(re.findall(r'[,;]|\b(and|or|but)\b', query.lower()))
        clause_score = min(clauses / 3.0, 1.0)

        # Weighted average
        complexity = (
            length_score * 0.3 +
            keyword_density * 0.3 +
            special_score * 0.2 +
            clause_score * 0.2
        )

        return round(complexity, 3)

    def _generate_suggestions(
        self,
        query: str,
        intent: str,
        complexity: float,
        keywords: List[str]
    ) -> List[str]:
        """
        Generate query improvement suggestions.

        Args:
            query: Original query
            intent: Classified intent
            complexity: Complexity score
            keywords: Extracted keywords

        Returns:
            List of suggestion strings
        """
        suggestions = []

        # Check query length
        if len(query.split()) < 3:
            suggestions.append(
                "💡 Try adding more specific terms to improve relevance"
            )

        # Check keyword count
        if len(keywords) < 2:
            suggestions.append(
                "💡 Include additional keywords to narrow search scope"
            )

        # Intent-specific suggestions
        if intent == "factual":
            suggestions.append(
                "💡 Factual query detected: Results will prioritize definitions and facts"
            )
        elif intent == "conceptual":
            suggestions.append(
                "💡 Conceptual query detected: Results will prioritize explanations"
            )
        elif intent == "comparison":
            suggestions.append(
                "💡 Comparison query detected: Use 'vs' or 'versus' for best results"
            )

        # Complexity-based suggestions
        if complexity > 0.7:
            suggestions.append(
                "⚠️ Complex query detected: Consider breaking into multiple simpler queries"
            )

        # Operator suggestions
        if '"' not in query and len(keywords) > 2:
            suggestions.append(
                '💡 Use quotes "like this" for exact phrase matching'
            )

        return suggestions[:5]  # Limit to top 5

    def _expand_terms(
        self,
        keywords: List[str],
        context_docs: Optional[List[str]] = None,
        top_k: int = 3
    ) -> Dict[str, List[str]]:
        """
        Expand keywords with semantically similar terms.

        Args:
            keywords: Keywords to expand
            context_docs: Optional corpus for context-aware expansion
            top_k: Number of expansions per keyword

        Returns:
            Dictionary mapping keywords to expanded terms
        """
        # Common semantic expansions (could be enhanced with word2vec/embeddings)
        expansion_map = {
            'machine': ['artificial', 'automated', 'computational'],
            'learning': ['training', 'education', 'acquisition'],
            'algorithm': ['method', 'procedure', 'technique'],
            'data': ['information', 'dataset', 'records'],
            'model': ['framework', 'architecture', 'system'],
            'neural': ['network', 'connectionist', 'deep'],
            'network': ['architecture', 'topology', 'structure'],
            'training': ['learning', 'optimization', 'fitting'],
            'accuracy': ['precision', 'correctness', 'performance'],
            'predict': ['forecast', 'estimate', 'infer'],
            'classification': ['categorization', 'labeling', 'taxonomy'],
            'regression': ['prediction', 'estimation', 'modeling'],
            'vector': ['embedding', 'representation', 'feature'],
            'semantic': ['meaning', 'conceptual', 'contextual'],
            'similarity': ['resemblance', 'closeness', 'proximity'],
            'search': ['retrieval', 'query', 'lookup'],
            'document': ['text', 'file', 'record'],
            'query': ['question', 'search', 'request']
        }

        expanded = {}
        for keyword in keywords:
            if keyword in expansion_map:
                expanded[keyword] = expansion_map[keyword][:top_k]
            else:
                # For unknown keywords, keep original
                expanded[keyword] = []

        return expanded

    def _compute_metrics(
        self,
        query: str,
        tokens: List[str],
        keywords: List[str]
    ) -> Dict[str, any]:
        """
        Compute various query metrics.

        Args:
            query: Original query
            tokens: Tokenized query
            keywords: Extracted keywords

        Returns:
            Dictionary of metrics
        """
        return {
            'character_count': len(query),
            'word_count': len(tokens),
            'keyword_count': len(keywords),
            'avg_word_length': round(
                sum(len(t) for t in tokens) / max(len(tokens), 1), 2
            ),
            'keyword_ratio': round(
                len(keywords) / max(len(tokens), 1), 3
            ),
            'has_quotes': '"' in query,
            'has_operators': bool(re.search(r'\b(AND|OR|NOT)\b', query, re.I)),
            'question_type': self._detect_question_type(query)
        }

    def _detect_question_type(self, query: str) -> str:
        """
        Detect if query is a question and its type.

        Args:
            query: Query string

        Returns:
            Question type or 'statement'
        """
        query_lower = query.lower().strip()

        if query_lower.endswith('?'):
            # Detect question word
            if query_lower.startswith('what'):
                return 'what_question'
            elif query_lower.startswith('why'):
                return 'why_question'
            elif query_lower.startswith('how'):
                return 'how_question'
            elif query_lower.startswith('when'):
                return 'when_question'
            elif query_lower.startswith('where'):
                return 'where_question'
            elif query_lower.startswith('who'):
                return 'who_question'
            else:
                return 'general_question'

        return 'statement'

    def format_analysis_report(self, analysis: QueryAnalysis) -> str:
        """
        Format query analysis as human-readable report.

        Args:
            analysis: QueryAnalysis object

        Returns:
            Formatted HTML report
        """
        # Intent badge color
        intent_colors = {
            'factual': '#2563eb',      # blue
            'conceptual': '#7c3aed',   # purple
            'exploratory': '#059669',  # green
            'comparison': '#dc2626'    # red
        }
        intent_color = intent_colors.get(analysis.intent, '#6b7280')

        # Complexity badge color
        if analysis.complexity_score < 0.3:
            complexity_color = '#059669'  # green
            complexity_label = 'Simple'
        elif analysis.complexity_score < 0.7:
            complexity_color = '#d97706'  # orange
            complexity_label = 'Moderate'
        else:
            complexity_color = '#dc2626'  # red
            complexity_label = 'Complex'

        html = f"""
        <div style="font-family: sans-serif; padding: 20px; background: var(--block-background-fill); border: 1px solid var(--border-color-primary); border-radius: 8px;">
            <h3 style="margin-top: 0;">📊 Query Analysis Report</h3>

            <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                <span style="background: {intent_color}; color: white; padding: 4px 12px; border-radius: 4px; font-size: 14px;">
                    {analysis.intent.upper()} ({analysis.intent_confidence:.0%})
                </span>
                <span style="background: {complexity_color}; color: white; padding: 4px 12px; border-radius: 4px; font-size: 14px;">
                    {complexity_label} ({analysis.complexity_score:.2f})
                </span>
            </div>

            <div style="background: var(--background-fill-secondary); padding: 15px; border-radius: 6px; margin-bottom: 15px;">
                <h4 style="margin-top: 0;">Keywords Extracted</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    {''.join(f'<span style="background: var(--block-background-fill); padding: 4px 10px; border-radius: 4px; font-size: 14px; border: 1px solid var(--border-color-primary);">{kw}</span>' for kw in analysis.keywords)}
                </div>
            </div>

            <div style="background: var(--background-fill-secondary); padding: 15px; border-radius: 6px; margin-bottom: 15px;">
                <h4 style="margin-top: 0;">Metrics</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid var(--border-color-primary);">
                        <td style="padding: 8px 0;">Words</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600;">{analysis.metrics['word_count']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid var(--border-color-primary);">
                        <td style="padding: 8px 0;">Keywords</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600;">{analysis.metrics['keyword_count']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid var(--border-color-primary);">
                        <td style="padding: 8px 0;">Keyword Ratio</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600;">{analysis.metrics['keyword_ratio']:.1%}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0;">Question Type</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600;">{analysis.metrics['question_type'].replace('_', ' ').title()}</td>
                    </tr>
                </table>
            </div>

            <div style="background: rgba(251, 191, 36, 0.2); padding: 15px; border-radius: 6px; border-left: 4px solid rgb(251, 191, 36);">
                <h4 style="margin-top: 0;">💡 Suggestions</h4>
                {'<br>'.join(analysis.suggestions) if analysis.suggestions else 'No suggestions - query looks good!'}
            </div>
        </div>
        """

        return html
