# AI Engineering Interactive Learning App

A comprehensive interactive application for learning AI engineering fundamentals through hands-on experimentation.

## Overview

This Gradio-based application provides an interactive environment for exploring core AI engineering concepts covered in the LinkedIn Learning course "Fundamentals of AI Engineering: Principles and Practical Applications."

### Features

- **Document Processing**: Extract and process PDFs, DOCX, TXT, JSON, and CSV files
- **Embeddings Playground**: Compare embedding models and compute semantic similarity
- **Vector Search Lab**: Experiment with ChromaDB for semantic search
- **Hybrid Retrieval Studio**: Combine BM25 + vector search with cross-encoder reranking
- **📊 Visualization Lab** (Phase 1): Visualize embedding spaces in 2D/3D using UMAP/t-SNE
- **🔍 Explainability Studio** (Phase 1): Understand token contributions, BM25 scoring, and vector similarity
- **🎯 Advanced Retrieval** (Phase 2): Query intelligence, multi-query search, and context filtering
- **💬 RAG Chat** (Phase 3): Conversational RAG with LLM backends and educational process viewer

## Installation

### Prerequisites

- Python 3.12+
- Virtual environment (recommended)

### Setup

1. **Install dependencies:**

```bash
# From the main project directory
uv pip install -r requirements.txt
```

All dependencies including Gradio 5.49.1 (latest version with performance improvements) and sentence-transformers are included in requirements.txt.

2. **Launch the app:**

```bash
cd learning_app
python app.py
```

The app will start on `http://localhost:7860`

**For GitHub Codespaces:** The port will be automatically forwarded. Check the PORTS tab in VS Code to access the app URL.

### Ollama Setup (Optional - For High-Quality Local LLMs)

To use production-quality open-source models locally through the Ollama backend:

1. **Install Ollama:**

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download installer from https://ollama.com/download
```

2. **Start Ollama server:**

```bash
ollama serve
```

3. **Pull a model** (recommended: llama3.2:3b for speed):

```bash
# Fastest option - 2GB, great for learning
ollama pull llama3.2:3b

# Balanced option - 4.1GB, production quality
ollama pull mistral:7b

# High quality option - 4.7GB, excellent responses
ollama pull llama3.1:8b
```

4. **Verify installation:**

```bash
ollama list  # Show downloaded models
curl http://localhost:11434/api/tags  # API health check
```

5. **Use in the app:**
   - Go to **RAG Chat** tab
   - Select **Ollama** backend
   - Choose your model from dropdown
   - Click "Initialize RAG Chat"

**Model Recommendations:**
- **Learning/Development**: llama3.2:3b (2GB RAM, fastest)
- **Balanced Quality**: mistral:7b (4.1GB RAM)
- **Production**: llama3.1:8b or llama3.1:70b (requires 40GB RAM)

**Ollama Resources:**
- Official website: https://ollama.com
- Model library: https://ollama.com/library
- Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md

## Usage

### Quick Start

1. **Launch the app:**

```bash
cd learning_app
python app.py
```

2. **Open your browser** to `http://localhost:7860`

3. **Follow the learning path:**
   - Start with **Document Processing** tab
   - Load or upload a document
   - Progress through each tab in order

### Learning Path

#### 1. Document Processing (Chapter 3)

- **Purpose**: Extract and prepare documents for AI processing
- **Actions**:
  - Select a pre-loaded sample or upload your own file
  - Clean text to remove extra whitespace
  - Enable chunking to split documents (recommended: 512 chars)
- **Outputs**: Document preview, metadata, chunk statistics

#### 2. Embeddings Playground (Chapter 4)

- **Purpose**: Understand text embeddings and semantic similarity
- **Actions**:
  - Compare two texts for similarity
  - Test different embedding models
  - View embedding dimensions and generation time
- **Models**:
  - `all-MiniLM-L6-v2`: Fast, 384 dimensions (recommended)
  - `all-mpnet-base-v2`: High quality, 768 dimensions
  - `paraphrase-MiniLM-L6-v2`: Optimized for paraphrases

#### 3. Vector Search Lab (Chapter 5)

- **Purpose**: Experiment with semantic search using ChromaDB
- **Actions**:
  - Load processed documents into vector database
  - Search using natural language queries
  - Adjust `top_k` to control number of results
  - Measure search performance
- **Key Parameters**:
  - `top_k`: Number of results (1-20)
  - Embedding model: Affects search quality

#### 4. Hybrid Retrieval Studio (Chapter 6)

- **Purpose**: Combine lexical + semantic search for best results
- **Actions**:
  - Compare BM25 vs Vector vs Hybrid retrieval
  - Adjust weights to favor lexical or semantic search
  - Enable cross-encoder reranking for production quality
- **Key Parameters**:
  - `bm25_weight`: 0-1 (favor exact keyword matches)
  - `vector_weight`: 0-1 (favor semantic similarity)
  - `reranking`: Enable for highest accuracy (slower)

#### 5. 📊 Visualization Lab (Phase 1 Enhancement)

- **Purpose**: Visualize high-dimensional embeddings and document relationships
- **Actions**:
  - Generate embeddings for loaded documents
  - Visualize embedding space in 2D or 3D
  - Create similarity heatmaps showing document relationships
- **Features**:
  - **Embedding Space**: UMAP or t-SNE dimensionality reduction
  - **Similarity Heatmap**: Visual matrix of document similarities
  - Interactive plots with hover information
- **Key Parameters**:
  - Reduction method: UMAP (faster, global structure) vs t-SNE (local clusters)
  - Dimensions: 2D (easier to interpret) vs 3D (more information)

#### 6. 🔍 Explainability Studio (Phase 1 Enhancement)

- **Purpose**: Understand how retrieval systems make decisions
- **Actions**:
  - Analyze token-level contributions to similarity
  - Break down BM25 scoring by term
  - Explain vector similarity component-wise
- **Features**:
  - **Token Analysis**: See which query terms contribute most
  - **BM25 Breakdown**: Understand TF, IDF, and final scores
  - **Vector Similarity**: Examine cosine similarity calculation
- **Use Cases**:
  - Debug why certain documents match (or don't)
  - Understand semantic vs lexical matching
  - Learn how retrieval algorithms work

#### 7. 🎯 Advanced Retrieval (Phase 2 Enhancement)

- **Purpose**: Intelligent query handling and result optimization
- **Sub-tabs**:
  - **Query Intelligence**: Analyze query intent, complexity, and get optimization suggestions
  - **Multi-Query Search**: Decompose complex queries and execute parallel searches with result fusion
  - **Context & Filtering**: Apply advanced filters including MMR (Maximal Marginal Relevance)
- **Features**:
  - **Intent Classification**: Factual, conceptual, exploratory, comparison
  - **Query Decomposition**: Break complex queries into sub-queries
  - **Query Expansion**: Add synonyms and context
  - **Query Rephrasing**: Generate alternative formulations
  - **Parallel Execution**: Run multiple query variations concurrently
  - **Result Fusion**: Weighted voting, round robin, score aggregation
  - **MMR**: Balance relevance vs diversity
  - **Diversity Filtering**: Remove redundant results
- **Key Parameters**:
  - Query strategy: Decompose, expand, rephrase, hybrid
  - Fusion method: Weighted voting (recommended), round robin, score aggregation
  - MMR lambda: 1.0 (pure relevance) to 0.0 (pure diversity)
- **Use Cases**:
  - Complex queries needing better coverage
  - Production systems requiring quality + diversity
  - Exploratory research avoiding redundancy

#### 8. 💬 RAG Chat (Phase 3 Enhancement)

- **Purpose**: Interactive conversational RAG (Retrieval-Augmented Generation) system
- **Features**:
  - **Multiple LLM Backends**: Local (DistilGPT2), Ollama (local high-quality models), and OpenAI API (GPT-4o, GPT-4.1, o3-pro, o4-mini)
  - **Ollama Support**: Run production-quality open-source models locally (Llama 3.2, Mistral, Phi-3, etc.)
  - **Latest Models**: GPT-4.1 with 1M token context window, GPT-4o-mini most cost-effective
  - **Conversation History**: Multi-turn conversations with automatic pruning
  - **Token Budget Management**: Smart allocation of context window (up to 1M tokens!)
  - **Educational Process Viewer**: See the 6-step RAG process transparently
  - **Source Attribution**: Citations for every response
- **Actions**:
  - Select LLM backend and model
  - Initialize RAG chat system
  - Ask questions and get RAG-enhanced responses
  - View detailed RAG process breakdown
  - Manage conversation history
- **RAG Process Steps**:
  1. **Query Processing**: Analyze and prepare user query
  2. **Document Retrieval**: Fetch relevant documents from vector store
  3. **Context Assembly**: Format context and manage token budget
  4. **Prompt Construction**: Build complete prompt with system/context/history
  5. **Response Generation**: Generate answer using LLM
  6. **Source Attribution**: Link response to source documents
- **Key Parameters**:
  - `backend`: Local (free, CPU), Ollama (high-quality local), or OpenAI (requires API key)
  - `model`: Model selection per backend
  - `top_k`: Number of documents to retrieve (3-5 recommended)
  - `retrieval_method`: BM25, vector, or hybrid
- **LLM Backend Options**:
  - **Local (DistilGPT2)**: Free, fast, CPU-only, educational quality (2K context)
  - **Ollama**: Production-quality open-source models, runs locally, no API costs
    - `llama3.2:3b` - Fastest, 3B params, 2GB RAM, 4K context (recommended)
    - `mistral:7b` - Balanced, 7B params, 4.1GB RAM, 8K context
    - `llama3.1:8b` - High quality, 8B params, 4.7GB RAM, 8K context
    - `gemma2:9b` - Google's model, 9B params, 8K context
    - `llama3.1:70b` - Production, 70B params, 40GB RAM, 8K context
    - Plus phi3:mini, qwen2.5:7b, and mixtral:8x7b
  - **OpenAI**: Cloud API, highest quality, requires API key, costs per token
    - GPT-4o/GPT-4o-mini - 128K context, most cost-effective ($0.15/1M tokens)
    - GPT-4.1/GPT-4.1-mini - 1M context window
    - o3-pro/o4-mini - Advanced reasoning models
- **Token Budget**:
  - System prompt: 10% of context window
  - RAG context: 50% of context window
  - Conversation history: 20% of context window
  - Generation buffer: 20% of context window
- **Use Cases**:
  - Understand how RAG systems work end-to-end
  - Compare local vs API-based LLMs
  - Learn token budget management
  - Explore multi-turn conversational AI
  - See transparent AI decision-making process

## Parameter Guide

### Document Processing

| Parameter | Range | Recommended | Effect |
|-----------|-------|-------------|--------|
| chunk_size | 128-2048 | 512 | Larger = more context, smaller = more precise |
| clean_text | True/False | True | Removes extra whitespace |

### Embeddings

| Model | Dimensions | Speed | Use Case |
|-------|------------|-------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | General purpose |
| all-mpnet-base-v2 | 768 | Slower | Higher quality |
| paraphrase-MiniLM | 384 | Fast | Paraphrase detection |

### Vector Search

| Parameter | Range | Recommended | Effect |
|-----------|-------|-------------|--------|
| top_k | 1-20 | 5 | Number of results |
| model | See above | all-MiniLM-L6-v2 | Search quality |

### Hybrid Retrieval

| Parameter | Range | Recommended | Effect |
|-----------|-------|-------------|--------|
| bm25_weight | 0-1 | 0.5 | Keyword matching strength |
| vector_weight | 0-1 | 0.5 | Semantic matching strength |
| reranking | True/False | False (dev), True (prod) | Accuracy vs speed |

### Weight Guidelines

- **BM25-heavy (0.7/0.3)**: Technical docs, error codes, specific IDs
- **Balanced (0.5/0.5)**: General purpose, mixed queries
- **Vector-heavy (0.3/0.7)**: Conceptual queries, synonym matching

## Pre-loaded Sample Documents

The app includes sample documents in `data/preloaded/`:
- Technical documentation
- Text files for testing
- Sample structured data

To add your own samples:
1. Place files in `learning_app/data/preloaded/`
2. Supported formats: PDF, DOCX, TXT, JSON, CSV
3. Restart the app to see new files

## Architecture

```
learning_app/
├── app.py                       # Main Gradio application
├── modules/                     # Core functionality
│   ├── document_processor.py    # Document extraction & chunking
│   ├── embeddings_engine.py     # Embedding generation
│   ├── vector_store.py          # ChromaDB operations
│   ├── retrieval_pipeline.py    # Hybrid retrieval logic
│   ├── visualization_engine.py  # Phase 1: UMAP/t-SNE & heatmaps
│   ├── explainability_engine.py # Phase 1: Token analysis & BM25 breakdown
│   ├── query_intelligence.py    # Phase 2: Query analysis & optimization
│   ├── multi_query_engine.py    # Phase 2: Query decomposition & fusion
│   ├── advanced_filtering.py    # Phase 2: MMR & diversity filtering
│   ├── llm_manager.py           # Phase 3: LLM backend abstraction
│   ├── context_manager.py       # Phase 3: Token budget management
│   ├── rag_pipeline.py          # Phase 3: RAG orchestration
│   └── conversation_engine.py   # Phase 3: Conversation history
├── utils/                       # Helper functions
│   ├── formatters.py            # Output formatting
│   ├── validators.py            # Input validation
│   └── plot_helpers.py          # Phase 1: Plotly utilities
├── tests/                       # Test suites
│   ├── test_phase1_visualization_explainability.py
│   ├── test_phase2_advanced_retrieval.py
│   └── test_phase3_rag_chat.py
├── run.sh                       # Automated launch script
├── run_tests.sh                 # Interactive test runner
└── data/
    ├── preloaded/               # Sample documents
    └── uploads/                 # User-uploaded files
```

## Testing

The app includes comprehensive test suites to validate all features.

### Interactive Test Runner

**Quick Start:**
```bash
cd learning_app
./run_tests.sh
```

The interactive test runner provides:
- **Menu-driven interface** for easy test selection
- **Run all tests** at once with comprehensive summary
- **Run individual tests** for targeted validation
- **Color-coded output** for clear status reporting
- **Automatic environment setup** and warning suppression

**Available Test Suites:**

1. **Phase 1: Visualization & Explainability**
   - Tests embedding space visualization (UMAP/t-SNE)
   - Tests similarity heatmaps
   - Tests token analysis and explainability features
   - File: `tests/test_phase1_visualization_explainability.py`

2. **Phase 2: Advanced Retrieval**
   - Tests query intelligence and intent classification
   - Tests multi-query decomposition and fusion
   - Tests advanced filtering (MMR, diversity)
   - File: `tests/test_phase2_advanced_retrieval.py`

3. **Phase 3: RAG Chat**
   - Tests LLM manager with multiple backends
   - Tests context manager and token budgets
   - Tests RAG pipeline and conversation engine
   - File: `tests/test_phase3_rag_chat.py`

### Manual Test Execution

```bash
# Activate virtual environment first
source ../.venv/bin/activate

# Set environment variables to suppress warnings
export TOKENIZERS_PARALLELISM=false
export OMP_NESTED=FALSE

# Run specific test
python tests/test_phase3_rag_chat.py

# Run all tests
python tests/test_phase1_visualization_explainability.py
python tests/test_phase2_advanced_retrieval.py
python tests/test_phase3_rag_chat.py
```

### Test Output

Tests provide:
- ✓ **Success indicators** for passed tests
- ✗ **Error messages** for failed tests
- **Performance metrics** (timing, token counts)
- **Detailed summaries** for each test suite

## Troubleshooting

### App won't start

**Issue**: Import errors or missing dependencies

**Solution**:
```bash
# Ensure you're in the project root
cd /path/to/project

# Install all dependencies
uv pip install -r requirements.txt

# Try again
cd learning_app
python app.py
```

### Models downloading slowly

**Issue**: First-time model downloads can be slow

**Solution**: Models are cached after first download. Subsequent launches will be faster.

### Port already in use

**Issue**: Port 7860 is already occupied

**Solution**: Use the automated `run.sh` script which automatically finds available ports:
```bash
cd learning_app
./run.sh
```

### Out of memory errors

**Issue**: Too many documents or large embeddings

**Solution**:
- Reduce chunk_size
- Process fewer documents
- Use smaller embedding model (all-MiniLM-L6-v2)

### Search returns no results

**Issue**: Documents not loaded or query mismatch

**Solution**:
1. Verify documents are loaded (Step 1 in Vector Search Lab)
2. Try different queries
3. Increase top_k value

### Reranking is slow

**Issue**: Cross-encoder reranking is computationally expensive

**Solution**: This is expected. Disable reranking for experimentation, enable for final results.

### Ollama connection errors

**Issue**: "Ollama server not available" or connection refused

**Solution**:
1. Ensure Ollama is installed: `ollama --version`
2. Start Ollama server: `ollama serve`
3. Verify server is running: `curl http://localhost:11434/api/tags`
4. Check firewall isn't blocking port 11434

**Issue**: Model not found error

**Solution**:
```bash
# List installed models
ollama list

# Pull the missing model
ollama pull llama3.2:3b

# Verify model is available
ollama list | grep llama3.2
```

**Issue**: Ollama generation is slow

**Solution**:
- Use smaller models (llama3.2:3b, phi3:mini)
- Reduce max_tokens in generation settings
- Ollama uses CPU by default; GPU acceleration requires compatible hardware
- First generation is slower (model loading), subsequent ones are faster

**Issue**: Out of memory with Ollama

**Solution**:
- Use smaller models: llama3.2:3b (2GB) instead of llama3.1:70b (40GB)
- Close other applications
- Check available RAM: smaller models for <8GB systems

## Extending the App

### Adding New Embedding Models

Edit `modules/embeddings_engine.py`:

```python
AVAILABLE_MODELS = {
    "your-model-name": {
        "dimensions": 512,
        "description": "Your model description",
        "size_mb": 100
    }
}
```

### Adding New Document Types

Edit `modules/document_processor.py`:

```python
def extract_text(self, file_path: str):
    # Add your extraction logic
    elif ext == '.your_ext':
        # Handle your file type
        pass
```

### Customizing UI

Edit `app.py` and modify Gradio components:
- Change themes: `gr.Blocks(theme=gr.themes.Soft())`
- Add components: `gr.Slider()`, `gr.Textbox()`, etc.
- Modify layout: Use `gr.Row()` and `gr.Column()`

## Performance Tips

1. **Use pre-loaded samples** for quick testing
2. **Enable chunking** to handle large documents
3. **Start with small top_k** (3-5) for faster results
4. **Disable reranking** during experimentation
5. **Use smaller models** (all-MiniLM-L6-v2) for speed

## Educational Use

This app is designed for learning. Experiment with:

1. **Different document types**: See how extraction varies
2. **Various chunk sizes**: Find optimal balance
3. **Multiple models**: Compare speed vs quality
4. **Weight combinations**: Discover best hybrid ratios
5. **Query variations**: Test lexical vs semantic matching

## Support

For issues or questions:
- Check the course notebooks for detailed explanations
- Review the Troubleshooting section above
- Examine error messages carefully

## License

This learning app is part of the LinkedIn Learning course materials.

---

**Happy Learning!** Experiment freely and discover how modern AI retrieval systems work.
