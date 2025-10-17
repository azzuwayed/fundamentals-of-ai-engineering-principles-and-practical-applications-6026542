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

## Key Technologies

**LlamaIndex:** Document readers (PDF/DOCX/Web/CSV/JSON), BM25Retriever, VectorStoreIndex, SentenceSplitter chunking
**Vector/Embeddings:** ChromaDB, SentenceTransformers (`all-MiniLM-L6-v2`)
**LLMs:** Transformers library, PyTorch CPU-only

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
