"""
RAG Pipeline Module for Chat.

Orchestrates the complete RAG (Retrieval-Augmented Generation) process:
1. Query processing
2. Document retrieval
3. Context formatting
4. Prompt assembly
5. LLM generation
6. Response with source citations

Educational focus: Shows each step transparently to teach how RAG works.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class RAGStep:
    """Represents one step in the RAG process."""

    step_number: int
    name: str
    description: str
    content: str
    duration_ms: float = 0.0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RAGResult:
    """Complete RAG execution result."""

    query: str
    response: str
    sources: List[Dict]
    steps: List[RAGStep]
    total_duration_ms: float
    tokens_used: int
    metadata: Dict


class RAGPipeline:
    """
    Complete RAG pipeline orchestration.

    Integrates:
    - Document retrieval (from RetrievalPipeline)
    - Context management (from ContextManager)
    - LLM generation (from LLMManager)

    Educational design: Each step is captured and can be displayed
    to show learners how RAG systems work internally.
    """

    def __init__(
        self,
        retrieval_pipeline,
        llm,
        context_manager,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize RAG pipeline.

        Args:
            retrieval_pipeline: RetrievalPipeline instance for document retrieval
            llm: LLM instance for generation
            context_manager: ContextManager for token management
            system_prompt: Optional custom system prompt
        """
        self.retrieval_pipeline = retrieval_pipeline
        self.llm = llm
        self.context_manager = context_manager
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """Get default system prompt for RAG."""
        return """You are a helpful AI assistant. Answer the user's question based on the provided context documents.

Guidelines:
- Use information from the context to answer the question
- If the context doesn't contain enough information, say so honestly
- Cite specific documents when making claims
- Be concise but thorough
- If multiple sources provide conflicting information, mention this

Context documents will be provided below."""

    def execute(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 3,
        retrieval_method: str = "hybrid",
        config: Optional[Dict] = None
    ) -> RAGResult:
        """
        Execute complete RAG pipeline.

        Args:
            query: User question
            conversation_history: Previous conversation turns
            top_k: Number of documents to retrieve
            retrieval_method: "bm25", "vector", or "hybrid"
            config: Optional LLM configuration overrides

        Returns:
            RAGResult with response and process details
        """
        start_time = time.time()
        steps = []
        conversation_history = conversation_history or []

        # Step 1: Query Processing
        step1_start = time.time()
        processed_query = self._process_query(query)
        steps.append(RAGStep(
            step_number=1,
            name="Query Processing",
            description="Analyze and prepare user query",
            content=f"Original query: {query}\nProcessed: {processed_query}",
            duration_ms=(time.time() - step1_start) * 1000
        ))

        # Step 2: Document Retrieval
        step2_start = time.time()
        retrieved_docs = self._retrieve_documents(
            processed_query,
            top_k,
            retrieval_method
        )
        steps.append(RAGStep(
            step_number=2,
            name="Document Retrieval",
            description=f"Retrieved {len(retrieved_docs)} documents using {retrieval_method}",
            content=self._format_retrieved_docs(retrieved_docs),
            duration_ms=(time.time() - step2_start) * 1000,
            metadata={
                "num_docs": len(retrieved_docs),
                "method": retrieval_method
            }
        ))

        # Step 3: Context Assembly
        step3_start = time.time()
        rag_context = self._format_context(retrieved_docs)

        # Apply token budget
        (
            truncated_system,
            truncated_context,
            truncated_history,
            allocation_stats
        ) = self.context_manager.allocate_budget(
            self.system_prompt,
            rag_context,
            conversation_history,
            self.llm.count_tokens
        )

        steps.append(RAGStep(
            step_number=3,
            name="Context Assembly",
            description="Format context and manage token budget",
            content=self.context_manager.format_allocation_report(allocation_stats),
            duration_ms=(time.time() - step3_start) * 1000,
            metadata=allocation_stats
        ))

        # Step 4: Prompt Construction
        step4_start = time.time()
        full_prompt = self._build_prompt(
            truncated_system,
            truncated_context,
            truncated_history,
            query
        )

        prompt_tokens = self.llm.count_tokens(full_prompt)
        steps.append(RAGStep(
            step_number=4,
            name="Prompt Construction",
            description=f"Assembled complete prompt ({prompt_tokens:,} tokens)",
            content=self._format_prompt_preview(full_prompt),
            duration_ms=(time.time() - step4_start) * 1000,
            metadata={"tokens": prompt_tokens}
        ))

        # Step 5: LLM Generation
        step5_start = time.time()
        llm_config = self._get_llm_config(config)
        response = self.llm.generate(full_prompt, llm_config)

        steps.append(RAGStep(
            step_number=5,
            name="Response Generation",
            description=f"Generated response using {response.model_name}",
            content=self._format_llm_response(response),
            duration_ms=(time.time() - step5_start) * 1000,
            metadata={
                "model": response.model_name,
                "backend": response.backend,
                "tokens_used": response.tokens_used,
                **response.metadata
            }
        ))

        # Step 6: Source Attribution
        step6_start = time.time()
        sources = self._extract_sources(retrieved_docs)
        steps.append(RAGStep(
            step_number=6,
            name="Source Attribution",
            description=f"Linked {len(sources)} source documents",
            content=self._format_sources(sources),
            duration_ms=(time.time() - step6_start) * 1000,
            metadata={"num_sources": len(sources)}
        ))

        total_duration = (time.time() - start_time) * 1000

        return RAGResult(
            query=query,
            response=response.text,
            sources=sources,
            steps=steps,
            total_duration_ms=total_duration,
            tokens_used=response.tokens_used,
            metadata={
                "retrieval_method": retrieval_method,
                "top_k": top_k,
                "model": response.model_name,
                "backend": response.backend
            }
        )

    def _process_query(self, query: str) -> str:
        """
        Process query (placeholder for future enhancements).

        Could add:
        - Query expansion
        - Spell correction
        - Intent detection

        Args:
            query: Original query

        Returns:
            Processed query
        """
        # For now, just return as-is
        # Future: could integrate with query_intelligence module
        return query.strip()

    def _retrieve_documents(
        self,
        query: str,
        top_k: int,
        method: str
    ) -> List[Dict]:
        """
        Retrieve documents using existing retrieval pipeline.

        Args:
            query: Search query
            top_k: Number of results
            method: Retrieval method

        Returns:
            List of retrieved document dictionaries
        """
        try:
            results = self.retrieval_pipeline.retrieve(
                query=query,
                top_k=top_k,
                method=method,
                bm25_weight=0.5,
                vector_weight=0.5,
                enable_reranking=False
            )
            return results
        except Exception as e:
            # Return empty if retrieval fails
            return []

    def _format_context(self, documents: List[Dict]) -> str:
        """
        Format retrieved documents as context.

        Args:
            documents: Retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."

        context_parts = ["=== Context Documents ===\n"]

        for i, doc in enumerate(documents, 1):
            text = doc.get('text', '')
            score = doc.get('similarity', doc.get('score', 0.0))

            context_parts.append(f"\n[Document {i}] (Relevance: {score:.3f})")
            context_parts.append(text)
            context_parts.append("-" * 50)

        return "\n".join(context_parts)

    def _build_prompt(
        self,
        system_prompt: str,
        context: str,
        history: List[Dict[str, str]],
        query: str
    ) -> str:
        """
        Build complete prompt with all components.

        Args:
            system_prompt: System instructions
            context: RAG context
            history: Conversation history
            query: Current query

        Returns:
            Complete prompt string
        """
        parts = [system_prompt, "\n\n", context]

        if history:
            parts.append("\n\n=== Conversation History ===")
            for msg in history:
                role = msg["role"].capitalize()
                content = msg["content"]
                parts.append(f"\n{role}: {content}")

        parts.append(f"\n\n=== Current Question ===\n{query}")
        parts.append("\n\nAnswer:")

        return "".join(parts)

    def _get_llm_config(self, overrides: Optional[Dict]):
        """Get LLM configuration with optional overrides."""
        from .llm_manager import LLMConfig

        config = LLMConfig(
            max_tokens=150,
            temperature=0.7,
            top_p=0.9
        )

        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return config

    def _extract_sources(self, documents: List[Dict]) -> List[Dict]:
        """
        Extract source information from documents.

        Args:
            documents: Retrieved documents

        Returns:
            List of source dictionaries
        """
        sources = []
        for i, doc in enumerate(documents, 1):
            sources.append({
                "id": i,
                "text": doc.get('text', '')[:200] + "...",  # Preview
                "score": doc.get('similarity', doc.get('score', 0.0)),
                "metadata": doc.get('metadata', {})
            })
        return sources

    def _format_retrieved_docs(self, docs: List[Dict]) -> str:
        """Format retrieved documents for display."""
        if not docs:
            return "No documents retrieved"

        lines = [f"Retrieved {len(docs)} documents:\n"]
        for i, doc in enumerate(docs, 1):
            score = doc.get('similarity', doc.get('score', 0.0))
            text_preview = doc.get('text', '')[:100]
            lines.append(f"  {i}. Score: {score:.3f} | {text_preview}...")

        return "\n".join(lines)

    def _format_prompt_preview(self, prompt: str, max_chars: int = 500) -> str:
        """Format prompt preview (truncated)."""
        if len(prompt) <= max_chars:
            return prompt

        return prompt[:max_chars] + f"\n\n[... {len(prompt) - max_chars} more characters ...]"

    def _format_llm_response(self, response) -> str:
        """Format LLM response details."""
        lines = [
            f"Model: {response.model_name} ({response.backend})",
            f"Tokens used: {response.tokens_used:,}",
            ""
        ]

        # Add token breakdown if available (OpenAI provides this)
        if "prompt_tokens" in response.metadata:
            lines.append(f"Prompt tokens: {response.metadata['prompt_tokens']:,}")
        if "completion_tokens" in response.metadata:
            lines.append(f"Completion tokens: {response.metadata['completion_tokens']:,}")

        lines.append(f"\nResponse:\n{response.text}")

        return "\n".join(lines)

    def _format_sources(self, sources: List[Dict]) -> str:
        """Format sources for display."""
        if not sources:
            return "No sources"

        lines = [f"Linked {len(sources)} sources:\n"]
        for src in sources:
            lines.append(f"  [{src['id']}] Score: {src['score']:.3f}")
            lines.append(f"      {src['text']}")

        return "\n".join(lines)

    def format_rag_process(self, result: RAGResult) -> str:
        """
        Format complete RAG process as HTML for educational display.

        Args:
            result: RAGResult from execute()

        Returns:
            HTML string showing all steps
        """
        html = """
        <div style="font-family: sans-serif;">
            <h3>🔄 RAG Process Breakdown</h3>
        """

        # Add each step
        for step in result.steps:
            html += f"""
            <div style="margin: 15px 0; padding: 15px; background: var(--block-background-fill); border-left: 4px solid rgba(59, 130, 246, 0.6); border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <strong>Step {step.step_number}: {step.name}</strong>
                        <div style="opacity: 0.7; font-size: 0.9em;">{step.description}</div>
                    </div>
                    <div style="opacity: 0.7; font-size: 0.85em;">
                        {step.duration_ms:.1f}ms
                    </div>
                </div>
                <div style="background: var(--background-fill-secondary); padding: 10px; border-radius: 4px; font-size: 0.9em; white-space: pre-wrap; max-height: 300px; overflow-y: auto;">
                    {step.content}
                </div>
            </div>
            """

        # Add summary
        html += f"""
            <div style="margin-top: 20px; padding: 15px; background: rgba(59, 130, 246, 0.1); border-radius: 8px;">
                <strong>⏱️ Total Duration:</strong> {result.total_duration_ms:.1f}ms<br>
                <strong>🎯 Tokens Used:</strong> {result.tokens_used:,}<br>
                <strong>📚 Sources:</strong> {len(result.sources)}<br>
                <strong>🤖 Model:</strong> {result.metadata.get('model', 'Unknown')} ({result.metadata.get('backend', 'unknown')})
            </div>
        </div>
        """

        return html
