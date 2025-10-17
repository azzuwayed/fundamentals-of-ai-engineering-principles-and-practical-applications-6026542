# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LinkedIn Learning course "Fundamentals of AI Engineering: Principles and Practical Applications" - Jupyter notebooks demonstrating embeddings, vector search, RAG systems, and hybrid retrieval.

## Environment

- Python 3.12, virtual environment at `.venv/`
- Package manager: `uv` (fast installs)
- Install: `uv pip install -r requirements.txt`
- CPU-only, no GPU required

## Repository Structure

- `chapter_2/` - Local LLM inference (DistilGPT2)
- `chapter_3/` - Text extraction (PDF, DOCX, web, JSON, CSV, DB)
- `chapter_4/` - Embeddings & semantic search (Python exercises with TODOs - incomplete by design)
- `chapter_5/` - ChromaDB vector database operations
- `chapter_6/` - Hybrid retrieval (BM25 vs vector search)
- `doc_samples/` - Sample documents for testing
- `learning_app/` - Interactive Gradio app for hands-on learning (NEW)

## Key Technologies

**LlamaIndex:** Document readers (PDF/DOCX/Web/CSV/JSON), BM25Retriever, VectorStoreIndex, SentenceSplitter chunking
**Vector/Embeddings:** ChromaDB, SentenceTransformers (`all-MiniLM-L6-v2`)
**LLMs:** Transformers library, PyTorch CPU-only
**UI Framework:** Gradio 5.49.1

## Documentation Resources

**Gradio Documentation:**
- **Always use Context7** for up-to-date Gradio docs: `/websites/gradio_app`
- Context7 provides curated, version-specific documentation and code examples
- Official website: https://context7.com/websites/gradio_app
- Use Context7 for migration guides, breaking changes, and API references

## Common Patterns

**Document Processing:**
Reader → Clean text → Extract metadata → Create Document → Chunk → Embed → Store

**ChromaDB:**
- Metadata filtering: `where={"$and": [{"category": {"$eq": "value"}}]}`
- Content filtering: `where_document={"$contains": "text"}`

**Notebook Conventions:**
- First cells install deps with `uv pip install`
- Set `UV_LINK_MODE=copy` to avoid hardlink warnings
- `device = "cpu"` for codespaces
- Common vars: `save_directory`, `similarity_top_k` (3-5), `chunk_size` (200-2000)
- Helper functions: `display_results()`, `test_bm25_retrieval()`, `clean_text()`, `extract_metadata()`

## Notebook Polishing Standards

When polishing notebooks, apply these patterns consistently:

**Structure:**
- Add title + Learning Objectives at start
- Use `## Section Name` for major sections
- Add Summary section at end with Key Takeaways

**Code Cells:**
- Add section headers as markdown before code
- Print confirmations: "✓ Action completed successfully!"
- Use `print("=" * 80)` for visual separators
- Format output with proper indentation (2 spaces)
- Show character counts as `{len(text):,}` with commas

**Formatting:**
- Consistent use of arrows: `→` for mappings
- Check marks: `✓` for success, `✗` for failure
- Use `repr()` for token/string display
- Right-align numbers in tables
- Use f-strings for all formatting

**Caching Pattern:**
```python
if os.path.exists(save_directory) and os.listdir(save_directory):
    print(f"✓ Model already exists in {save_directory}")
    print("  Loading from local directory...")
    # load from local
else:
    # download and save
```

## Notes

- Educational notebooks for software engineers learning AI engineering
- `chapter_4/*.py` exercises are intentionally incomplete (learner exercises)

## Polished Notebooks Status

**✓ ALL CHAPTERS COMPLETE (17 notebooks total)**

All notebooks have been polished with:
- Learning Objectives sections
- Consistent formatting (✓, →, separators)
- Comprehensive summaries with Key Takeaways
- Production best practices
- Performance benchmarks where applicable

**Completed chapters:**
- **Chapter 2:** 02_02, 02_03, 02_04 (3 notebooks)
- **Chapter 3:** 03_02, 03_03, 03_04, 03_05 (4 notebooks)
- **Chapter 4:** 04_02, 04_03, 04_04 (3 notebooks)
- **Chapter 5:** 05_02, 05_03, 05_04, 05_05 (4 notebooks)
- **Chapter 6:** 06_02, 06_03, 06_04, 06_05 (4 notebooks)

## Interactive Learning App

**Location:** `learning_app/`

A comprehensive Gradio-based web application that provides hands-on experimentation with all course concepts.

### Features

1. **Document Processing Tab** (Chapter 3)
   - Upload or select pre-loaded documents (PDF, DOCX, TXT, JSON, CSV)
   - Clean and preprocess text
   - Preview chunking with adjustable parameters

2. **Embeddings Playground** (Chapter 4)
   - Compare embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM)
   - Compute semantic similarity between texts
   - View embedding dimensions and generation times

3. **Vector Search Lab** (Chapter 5)
   - Load documents into ChromaDB
   - Perform semantic search with tunable parameters
   - Benchmark query performance

4. **Hybrid Retrieval Studio** (Chapter 6)
   - Compare BM25 (lexical) vs Vector (semantic) vs Hybrid retrieval
   - Adjust BM25/Vector weights dynamically
   - Enable cross-encoder reranking
   - Side-by-side method comparison

5. **📊 Visualization Lab** (Phase 1 Enhancement)
   - Visualize embedding spaces in 2D/3D using UMAP or t-SNE
   - Create interactive similarity heatmaps
   - Understand document clustering and relationships
   - Explore dimensionality reduction techniques

6. **🔍 Explainability Studio** (Phase 1 Enhancement)
   - Token-level similarity analysis with highlighting
   - BM25 scoring breakdown (TF, IDF, final scores)
   - Vector similarity component analysis
   - Understand retrieval decision-making process

7. **🎯 Advanced Retrieval** (Phase 2 Enhancement)
   - **Query Intelligence**: Analyze query intent, complexity, and get optimization suggestions
   - **Multi-Query Search**: Decompose complex queries, execute parallel searches, fuse results
   - **Context & Filtering**: Apply metadata filters, similarity thresholds, diversity controls, and MMR
   - Production-quality retrieval optimization techniques

8. **💬 RAG Chat** (Phase 3 Enhancement)
   - **Multiple LLM Backends**: Local (DistilGPT2) and OpenAI API (GPT-3.5, GPT-4)
   - **Conversation History**: Multi-turn conversations with automatic pruning
   - **Token Budget Management**: Smart allocation of limited context window
   - **Educational Process Viewer**: See complete 6-step RAG process transparently
   - **Source Attribution**: Citations and relevance scores for every response
   - End-to-end conversational RAG system with production patterns

### Running the App

**Quick Start (Recommended):**
```bash
# Automated setup and launch with comprehensive verification
cd learning_app
./run.sh

# Open browser to http://localhost:7860
```

The `run.sh` script automatically:
- ✓ Checks Python 3.12+ installation
- ✓ Verifies/installs `uv` package manager
- ✓ Creates/activates virtual environment
- ✓ Installs/updates all dependencies
- ✓ Validates Gradio 5.49.1 installation
- ✓ Tests all module imports
- ✓ Checks port availability (auto-switches if needed)
- ✓ Launches the application

**Advanced Options:**
```bash
./run.sh --help                    # Show all options
./run.sh --force-reinstall         # Force reinstall dependencies
./run.sh --port 8080               # Run on custom port
```

**Manual Setup (Alternative):**
```bash
# Install dependencies
uv pip install -r requirements.txt

# Launch the app
cd learning_app
python app.py
```

**Note:** Uses Gradio 5.49.1 with server-side rendering for improved performance and modern UI.

### Architecture

```
learning_app/
├── app.py                       # Main Gradio application (8 tabs)
├── run.sh                       # Automated launch script with verification
├── modules/                     # Core functionality
│   ├── document_processor.py    # Document extraction & chunking
│   ├── embeddings_engine.py     # Embedding generation & similarity
│   ├── vector_store.py          # ChromaDB operations
│   ├── retrieval_pipeline.py    # BM25, vector, hybrid retrieval
│   ├── visualization_engine.py  # Phase 1: UMAP/t-SNE, heatmaps
│   ├── explainability_engine.py # Phase 1: Token/BM25/vector analysis
│   ├── query_intelligence.py    # Phase 2: Query analysis & optimization
│   ├── multi_query_engine.py    # Phase 2: Multi-query & result fusion
│   ├── advanced_filtering.py    # Phase 2: Metadata filters & MMR
│   ├── llm_manager.py           # Phase 3: LLM backend abstraction
│   ├── context_manager.py       # Phase 3: Token budget management
│   ├── rag_pipeline.py          # Phase 3: RAG orchestration
│   └── conversation_engine.py   # Phase 3: Conversation history
├── utils/                       # Helper functions
│   ├── formatters.py            # Output formatting
│   ├── validators.py            # Input validation
│   └── plot_helpers.py          # Phase 1: Plotly utilities
├── test_enhancements.py         # Phase 1: Test suite
├── test_phase2.py               # Phase 2: Test suite
├── test_phase3.py               # Phase 3: Test suite
└── data/
    ├── preloaded/               # Sample documents (from doc_samples/)
    └── uploads/                 # User-uploaded files
```

### Phase 1 Enhancements

**New Dependencies:**
- `plotly==5.24.1` - Interactive visualizations
- `umap-learn==0.5.7` - Dimensionality reduction

**New Capabilities:**
- Embedding space visualization (UMAP/t-SNE in 2D/3D)
- Document similarity heatmaps
- Token-level contribution analysis
- BM25 scoring breakdown with term-by-term details
- Vector similarity component analysis
- Interactive Plotly charts with hover information

**Testing:**
```bash
cd learning_app
python test_enhancements.py  # Run Phase 1 test suite
```

### Phase 2 Enhancements

**New Modules:**
- `query_intelligence.py` - Query analysis, intent classification, complexity scoring
- `multi_query_engine.py` - Multi-query execution with parallel processing and result fusion
- `advanced_filtering.py` - Metadata filters, similarity thresholds, diversity controls, MMR

**New Capabilities:**
- **Query Intelligence**: Intent classification (factual/conceptual/exploratory/comparison), complexity scoring, keyword extraction, term expansion, optimization suggestions
- **Multi-Query Strategies**: Query decomposition, expansion, rephrasing, hybrid approach
- **Parallel Execution**: ThreadPoolExecutor-based concurrent query execution
- **Result Fusion**: Weighted voting (rank-based), round robin, score aggregation
- **Advanced Filtering**: Document type/category/source filters, date range filtering, similarity thresholds
- **Diversity Controls**: Remove redundant results based on text similarity
- **MMR (Maximal Marginal Relevance)**: Balance relevance vs diversity with λ parameter
- **Production Patterns**: Enterprise-grade retrieval optimization techniques

**Key Algorithms:**
- **Intent Classification**: Pattern-based classification with confidence scoring
- **Query Decomposition**: Template-based sub-query generation
- **Weighted Voting Fusion**: Rank-based scoring (score = 1/(rank+1), higher is better)
- **MMR Formula**: λ * Relevance - (1-λ) * max Similarity (supports text and embedding-based)
- **Diversity Filtering**: Jaccard similarity for text-based redundancy detection

**Testing:**
```bash
cd learning_app
python test_phase2.py  # Run Phase 2 test suite
```

### Phase 3 Enhancements

**New Dependencies:**
- `openai==1.58.1` - OpenAI API for GPT models
- `tiktoken==0.8.0` - Token counting for OpenAI models
- `transformers==4.48.3` - Hugging Face transformers for local models
- `torch==2.6.0` - PyTorch for local model inference

**New Modules:**
- `llm_manager.py` - Unified LLM interface with multiple backends (LocalLLM, OpenAILLM)
- `context_manager.py` - Token budget allocation and context window management
- `rag_pipeline.py` - Complete RAG orchestration with 6-step process tracking
- `conversation_engine.py` - Multi-turn conversation history with automatic pruning

**New Capabilities:**
- **LLM Backends**: Support for local (DistilGPT2) and OpenAI API (GPT-3.5-turbo, GPT-4) models
- **Factory Pattern**: LLMManager provides unified interface across different backends
- **Token Counting**: Backend-specific token counting (tiktoken for OpenAI, AutoTokenizer for local)
- **Context Management**: Intelligent allocation of limited context window with configurable ratios
- **Budget Allocation**: System prompt (10%), RAG context (50%), conversation history (20%), generation (20%)
- **Automatic Truncation**: Smart truncation strategies when content exceeds budget
- **RAG Process Tracking**: Capture and display all 6 steps with timing and metadata
- **Conversation History**: Multi-turn conversations with automatic pruning (10 turns default)
- **Source Attribution**: Link responses to retrieved documents with similarity scores
- **Educational Transparency**: Complete RAG process visible to learners

**RAG Pipeline Architecture:**
1. **Query Processing**: Analyze and prepare user query
2. **Document Retrieval**: Fetch relevant documents using existing retrieval pipeline
3. **Context Assembly**: Format documents and apply token budget allocation
4. **Prompt Construction**: Build complete prompt with system/context/history/query
5. **Response Generation**: Generate answer using selected LLM backend
6. **Source Attribution**: Extract and format source citations

**Key Design Patterns:**
- **Abstract Base Class**: LLM interface for extensibility
- **Factory Pattern**: LLMManager.create_llm() for backend instantiation
- **Dataclasses**: LLMConfig, TokenBudget, RAGStep, RAGResult for type safety
- **Graceful Degradation**: Missing OpenAI API key handled with informative messages
- **Educational Focus**: All steps transparent with HTML formatting for Gradio display

**Testing:**
```bash
cd learning_app
python test_phase3.py  # Run Phase 3 test suite
```

**Test Coverage:**
- LLM manager: Factory creation, model info, token counting, generation
- Context manager: Budget allocation, truncation strategies, HTML reporting
- Conversation engine: Message management, history retrieval, pruning, export/import
- RAG pipeline: Complete execution with mock retrieval, 6-step validation

### Usage Notes

- **Progressive learning**: Tabs are ordered logically (1→2→3→4→5→6→7→8)
- **Document flow**: Process documents in Tab 1, then use in other tabs
- **Visualization workflow**: Tab 1 → Tab 5 (generate embeddings first)
- **Explainability workflow**: Use any query/document pair in Tab 6
- **Advanced retrieval workflow**: Tab 1 (process docs) → Tab 7 (optimize queries and filter results)
- **RAG Chat workflow**: Tab 1 (process docs) → Tab 8 (initialize RAG, ask questions)
- **OpenAI API**: Set `OPENAI_API_KEY` environment variable to use GPT-3.5/GPT-4 backends
- **Local LLM**: No API key needed for DistilGPT2 (free, CPU-only)
- **Parameter experimentation**: All key parameters exposed as interactive controls
- **Pre-loaded samples**: Quick start with included documents
- **Session persistence**: Data persists across tabs during session

For detailed usage instructions, see `learning_app/README.md`.
