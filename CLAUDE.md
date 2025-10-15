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

## Notes

- Educational notebooks for software engineers learning AI engineering
- Cells defining functions should print confirmation (e.g., "✓ Function defined successfully!")
- `chapter_4/*.py` exercises are intentionally incomplete (learner exercises)
