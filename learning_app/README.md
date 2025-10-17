# AI Engineering Interactive Learning App

A comprehensive interactive application for learning AI engineering fundamentals through hands-on experimentation.

## Overview

This Gradio-based application provides an interactive environment for exploring core AI engineering concepts covered in the LinkedIn Learning course "Fundamentals of AI Engineering: Principles and Practical Applications."

### Features

- **Document Processing**: Extract and process PDFs, DOCX, TXT, JSON, and CSV files
- **Embeddings Playground**: Compare embedding models and compute semantic similarity
- **Vector Search Lab**: Experiment with ChromaDB for semantic search
- **Hybrid Retrieval Studio**: Combine BM25 + vector search with cross-encoder reranking

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

All dependencies including Gradio 4.19.0 (stable version for compatibility) and sentence-transformers are included in requirements.txt.

2. **Launch the app:**

```bash
cd learning_app
python app.py
```

The app will start on `http://localhost:7860`

**For GitHub Codespaces:** The port will be automatically forwarded. Check the PORTS tab in VS Code to access the app URL.

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
├── app.py                    # Main Gradio application
├── modules/                  # Core functionality
│   ├── document_processor.py # Document extraction & chunking
│   ├── embeddings_engine.py  # Embedding generation
│   ├── vector_store.py       # ChromaDB operations
│   └── retrieval_pipeline.py # Hybrid retrieval logic
├── utils/                    # Helper functions
│   ├── formatters.py         # Output formatting
│   └── validators.py         # Input validation
└── data/
    ├── preloaded/            # Sample documents
    └── uploads/              # User-uploaded files
```

## Troubleshooting

### App won't start

**Issue**: Import errors or missing dependencies

**Solution**:
```bash
# Ensure you're in the project root
cd /path/to/project

# Install all dependencies
uv pip install -r requirements.txt
uv pip install gradio==4.44.1

# Try again
cd learning_app
python app.py
```

### Models downloading slowly

**Issue**: First-time model downloads can be slow

**Solution**: Models are cached after first download. Subsequent launches will be faster.

### Gradio version issues

**Issue**: API schema errors with Gradio 4.44.1

**Solution**: We use Gradio 4.19.0 (stable version) which works reliably in GitHub Codespaces and local environments.

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
