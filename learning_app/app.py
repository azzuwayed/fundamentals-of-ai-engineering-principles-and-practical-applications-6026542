"""
AI Engineering Interactive Learning App
A comprehensive learning environment for AI engineering concepts.
"""
import gradio as gr
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import (
    DocumentProcessor, get_preloaded_documents,
    EmbeddingsEngine, format_similarity_score,
    VectorStore, RetrievalPipeline
)
from utils import (
    format_results_table, format_metrics, format_document_preview,
    format_embedding_info, format_comparison_table,
    validate_file_upload, validate_text_input, validate_top_k
)

# Global state
document_processor = DocumentProcessor()
current_documents = []
current_chunks = []
vector_store = None
retrieval_pipeline = None


# ============================================================================
# CHAPTER 3: DOCUMENT PROCESSING TAB
# ============================================================================

def process_document_fn(file, use_preloaded, preloaded_file, clean_text, chunk_size_enabled, chunk_size):
    """Process document and return results."""
    global current_documents, current_chunks

    try:
        # Determine which file to use
        if use_preloaded:
            if not preloaded_file:
                return "Please select a pre-loaded file", "", ""
            file_path = os.path.join("data/preloaded", preloaded_file)
        else:
            if file is None:
                return "Please upload a file", "", ""
            is_valid, error = validate_file_upload(file.name)
            if not is_valid:
                return f"Error: {error}", "", ""
            file_path = file.name

        # Process document
        chunk_size_val = int(chunk_size) if chunk_size_enabled else None
        text, metadata, chunks = document_processor.process_document(
            file_path, clean=clean_text, chunk_size=chunk_size_val
        )

        # Store for other tabs
        current_documents = [text]
        current_chunks = chunks

        # Format output
        preview = format_document_preview(text, max_chars=1000)

        metadata_str = format_metrics(metadata)

        chunks_preview = ""
        if chunks:
            chunks_preview = f"Generated {len(chunks)} chunks\n\n"
            chunks_preview += "First 3 chunks:\n" + "=" * 60 + "\n"
            for i, chunk in enumerate(chunks[:3]):
                chunks_preview += f"\nChunk {i}:\n{chunk['text'][:200]}...\n"
                chunks_preview += f"({chunk['char_count']} chars, {chunk['word_count']} words)\n"

        return preview, metadata_str, chunks_preview

    except Exception as e:
        return f"Error: {str(e)}", "", ""


def get_preloaded_files():
    """Get list of pre-loaded files."""
    return get_preloaded_documents()


# ============================================================================
# CHAPTER 4: EMBEDDINGS PLAYGROUND TAB
# ============================================================================

def generate_similarity_fn(text1, text2, model_name):
    """Compute similarity between two texts."""
    try:
        is_valid1, error1 = validate_text_input(text1, min_length=5)
        if not is_valid1:
            return f"Text 1 error: {error1}", ""

        is_valid2, error2 = validate_text_input(text2, min_length=5)
        if not is_valid2:
            return f"Text 2 error: {error2}", ""

        engine = EmbeddingsEngine(model_name)
        similarity, timings = engine.compute_similarity(text1, text2)

        # Format results
        result = f"{'=' * 60}\n"
        result += f"SIMILARITY ANALYSIS\n"
        result += f"{'=' * 60}\n\n"
        result += f"Model: {model_name}\n"
        result += f"Similarity Score: {format_similarity_score(similarity)}\n\n"
        result += format_metrics(timings)

        # Show embedding info
        emb1, _ = engine.generate_embedding(text1)
        emb_info = format_embedding_info(emb1, model_name)

        return result, emb_info

    except Exception as e:
        return f"Error: {str(e)}", ""


def compare_models_fn(text):
    """Compare different embedding models."""
    try:
        is_valid, error = validate_text_input(text, min_length=5)
        if not is_valid:
            return f"Error: {error}"

        engine = EmbeddingsEngine()
        results = engine.compare_models(text)

        return format_comparison_table(results, "MODEL COMPARISON")

    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# CHAPTER 5: VECTOR SEARCH LAB TAB
# ============================================================================

def load_to_vector_store_fn(model_name):
    """Load current documents into vector store."""
    global vector_store, current_documents

    try:
        if not current_documents:
            return "No documents loaded. Please process a document in the Document Processing tab first."

        vector_store = VectorStore(
            collection_name="learning_app",
            model_name=model_name
        )
        vector_store.clear_collection()

        # Add documents (use chunks if available, otherwise full documents)
        docs_to_add = [chunk['text'] for chunk in current_chunks] if current_chunks else current_documents
        ids = [f"chunk_{i}" for i in range(len(docs_to_add))]

        num_added, add_time = vector_store.add_documents(docs_to_add, ids=ids)

        result = f"{'=' * 60}\n"
        result += f"VECTOR STORE LOADED\n"
        result += f"{'=' * 60}\n\n"
        result += f"Model: {model_name}\n"
        result += f"Documents added: {num_added}\n"
        result += f"Load time: {add_time:.4f}s\n"
        result += f"Throughput: {num_added/add_time:.0f} docs/sec\n"
        result += f"{'=' * 60}\n"

        return result

    except Exception as e:
        return f"Error: {str(e)}"


def search_vector_store_fn(query, top_k):
    """Search the vector store."""
    global vector_store

    try:
        if vector_store is None:
            return "Please load documents into the vector store first.", ""

        is_valid, error = validate_text_input(query, min_length=3)
        if not is_valid:
            return f"Query error: {error}", ""

        is_valid, error = validate_top_k(top_k, max_documents=vector_store.get_count())
        if not is_valid:
            return f"top_k error: {error}", ""

        results, metrics = vector_store.search(query, n_results=top_k)

        # Format results
        results_df = format_results_table(results, show_content=True)
        metrics_str = format_metrics(metrics)

        return results_df, metrics_str

    except Exception as e:
        return f"Error: {str(e)}", ""


# ============================================================================
# CHAPTER 6: HYBRID RETRIEVAL STUDIO TAB
# ============================================================================

def setup_retrieval_pipeline_fn():
    """Set up retrieval pipeline with current documents."""
    global retrieval_pipeline, current_documents

    try:
        if not current_documents:
            return "No documents loaded. Please process a document in the Document Processing tab first."

        # Use a richer set of documents if available
        docs_to_use = [chunk['text'] for chunk in current_chunks[:20]] if current_chunks else current_documents

        retrieval_pipeline = RetrievalPipeline(docs_to_use, chunk_size=512)
        stats = retrieval_pipeline.get_stats()

        result = f"{'=' * 60}\n"
        result += f"RETRIEVAL PIPELINE READY\n"
        result += f"{'=' * 60}\n\n"
        result += format_metrics(stats)

        return result

    except Exception as e:
        return f"Error: {str(e)}"


def compare_retrieval_methods_fn(query, top_k):
    """Compare BM25, vector, and hybrid retrieval."""
    global retrieval_pipeline

    try:
        if retrieval_pipeline is None:
            return "", "", "", "Please set up the retrieval pipeline first."

        is_valid, error = validate_text_input(query, min_length=3)
        if not is_valid:
            return "", "", "", f"Query error: {error}"

        comparison = retrieval_pipeline.compare_methods(query, top_k=top_k)

        # Format each method's results
        bm25_results = format_results_table(comparison['bm25']['results'], show_content=True)
        bm25_time = f"Time: {comparison['bm25']['time']:.4f}s"

        vector_results = format_results_table(comparison['vector']['results'], show_content=True)
        vector_time = f"Time: {comparison['vector']['time']:.4f}s"

        hybrid_results = format_results_table(comparison['hybrid']['results'], show_content=True)
        hybrid_time = f"Time: {comparison['hybrid']['time']:.4f}s"

        summary = f"{'=' * 60}\n"
        summary += f"RETRIEVAL COMPARISON SUMMARY\n"
        summary += f"{'=' * 60}\n\n"
        summary += f"Query: {query}\n\n"
        summary += f"BM25 (Lexical):     {comparison['bm25']['time']:.4f}s\n"
        summary += f"Vector (Semantic):  {comparison['vector']['time']:.4f}s\n"
        summary += f"Hybrid (Combined):  {comparison['hybrid']['time']:.4f}s\n"
        summary += f"{'=' * 60}\n"

        return f"{bm25_results}\n\n{bm25_time}", f"{vector_results}\n\n{vector_time}", f"{hybrid_results}\n\n{hybrid_time}", summary

    except Exception as e:
        return "", "", "", f"Error: {str(e)}"


def hybrid_retrieval_with_weights_fn(query, top_k, bm25_weight, vector_weight, use_reranking):
    """Hybrid retrieval with custom weights and optional reranking."""
    global retrieval_pipeline

    try:
        if retrieval_pipeline is None:
            return "", "Please set up the retrieval pipeline first."

        is_valid, error = validate_text_input(query, min_length=3)
        if not is_valid:
            return "", f"Query error: {error}"

        if use_reranking:
            results, timings = retrieval_pipeline.retrieve_with_reranking(
                query, top_k=top_k,
                retrieval_method="hybrid",
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )
        else:
            results, _, timings = retrieval_pipeline.retrieve_hybrid(
                query, top_k=top_k,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )

        results_df = format_results_table(results, show_content=True)
        metrics_str = format_metrics(timings)

        return results_df, metrics_str

    except Exception as e:
        return "", f"Error: {str(e)}"


# ============================================================================
# BUILD GRADIO INTERFACE
# ============================================================================

with gr.Blocks(title="AI Engineering Learning App", theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # AI Engineering Interactive Learning App

    Experiment with AI engineering concepts from document processing to hybrid retrieval.
    Navigate through the tabs in order for the best learning experience.
    """)

    with gr.Tabs():

        # ====================================================================
        # WELCOME TAB
        # ====================================================================
        with gr.Tab("Welcome & Quick Start"):
            gr.Markdown("""
            ## Welcome to the AI Engineering Learning App!

            This interactive application lets you experiment with core AI engineering concepts:

            ### Learning Path (Follow in Order)

            **1. Document Processing (Chapter 3)**
            - Upload or select sample documents
            - Extract and clean text from PDFs, DOCX, TXT
            - Preview chunking strategies
            - **Start here:** Load a document to use in other tabs

            **2. Embeddings Playground (Chapter 4)**
            - Compare different embedding models
            - Compute semantic similarity between texts
            - Visualize embedding dimensions
            - Understand how text becomes vectors

            **3. Vector Search Lab (Chapter 5)**
            - Load documents into ChromaDB vector store
            - Experiment with semantic search
            - Tune parameters (top_k, models)
            - Measure search performance

            **4. Hybrid Retrieval Studio (Chapter 6)**
            - Compare BM25 vs Vector search
            - Mix strategies with custom weights
            - Apply cross-encoder reranking
            - Build complete retrieval pipelines

            ### Quick Start Guide

            1. **Go to "Document Processing" tab**
            2. **Select a pre-loaded sample** or upload your own document
            3. **Click "Process Document"** to extract text
            4. **Enable chunking** to split text (recommended: 512 chars)
            5. **Move to "Vector Search Lab"** and click "Load to Vector Store"
            6. **Try searching** with different queries
            7. **Explore "Hybrid Retrieval"** to compare methods

            ### Tips for Experimentation

            - Start with pre-loaded samples for quick testing
            - Try different embedding models to see trade-offs
            - Adjust chunk_size to see impact on search quality
            - Compare BM25 vs Vector search for different query types
            - Use reranking for production-quality results

            ### Need Help?

            - Each tab has info sections explaining parameters
            - Hover over inputs for tooltips
            - Error messages will guide you if something goes wrong

            **Ready to start? Go to the Document Processing tab! →**
            """)

        # ====================================================================
        # CHAPTER 3: DOCUMENT PROCESSING
        # ====================================================================
        with gr.Tab("1. Document Processing"):
            gr.Markdown("## Document Processing (Chapter 3)\nExtract, clean, and chunk documents")

            with gr.Row():
                with gr.Column():
                    use_preloaded = gr.Checkbox(label="Use pre-loaded sample", value=True)
                    preloaded_file = gr.Dropdown(
                        choices=get_preloaded_files(),
                        label="Select pre-loaded file",
                        value=get_preloaded_files()[0] if get_preloaded_files() else None
                    )
                    uploaded_file = gr.File(label="Or upload your own file", file_types=[".pdf", ".docx", ".txt", ".json", ".csv"])

                    clean_text = gr.Checkbox(label="Clean text (remove extra whitespace)", value=True)

                    with gr.Row():
                        chunk_enabled = gr.Checkbox(label="Enable chunking", value=True)
                        chunk_size = gr.Slider(minimum=128, maximum=2048, value=512, step=64, label="Chunk size")

                    process_btn = gr.Button("Process Document", variant="primary")

                with gr.Column():
                    preview_output = gr.Textbox(label="Document Preview", lines=15)
                    metadata_output = gr.Textbox(label="Metadata", lines=8)
                    chunks_output = gr.Textbox(label="Chunks Preview", lines=10)

            process_btn.click(
                process_document_fn,
                inputs=[uploaded_file, use_preloaded, preloaded_file, clean_text, chunk_enabled, chunk_size],
                outputs=[preview_output, metadata_output, chunks_output]
            )

            gr.Markdown("""
            ### About Document Processing
            - **PDF/DOCX**: Extracted using LlamaIndex readers
            - **Chunking**: Splits text into manageable pieces for embedding
            - **Chunk size**: Balance between context (larger) and precision (smaller)
            - **Typical range**: 256-1024 characters
            """)

        # ====================================================================
        # CHAPTER 4: EMBEDDINGS PLAYGROUND
        # ====================================================================
        with gr.Tab("2. Embeddings Playground"):
            gr.Markdown("## Embeddings Playground (Chapter 4)\nExperiment with text embeddings and similarity")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Similarity Comparison")
                    text1 = gr.Textbox(label="Text 1", placeholder="Enter first text...", lines=3)
                    text2 = gr.Textbox(label="Text 2", placeholder="Enter second text...", lines=3)
                    embedding_model = gr.Dropdown(
                        choices=list(EmbeddingsEngine.get_available_models().keys()),
                        value="all-MiniLM-L6-v2",
                        label="Embedding Model"
                    )
                    similarity_btn = gr.Button("Compute Similarity", variant="primary")

                with gr.Column():
                    similarity_output = gr.Textbox(label="Similarity Results", lines=10)
                    embedding_info_output = gr.Textbox(label="Embedding Details", lines=10)

            similarity_btn.click(
                generate_similarity_fn,
                inputs=[text1, text2, embedding_model],
                outputs=[similarity_output, embedding_info_output]
            )

            gr.Markdown("---")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Model Comparison")
                    compare_text = gr.Textbox(label="Text to embed", placeholder="Enter text to compare across models...", lines=3)
                    compare_btn = gr.Button("Compare Models")

                with gr.Column():
                    compare_output = gr.Textbox(label="Model Comparison Results", lines=12)

            compare_btn.click(
                compare_models_fn,
                inputs=[compare_text],
                outputs=[compare_output]
            )

            gr.Markdown("""
            ### About Embeddings
            - **all-MiniLM-L6-v2**: Fast, 384 dimensions, general purpose
            - **all-mpnet-base-v2**: Higher quality, 768 dimensions, slower
            - **paraphrase-MiniLM-L6-v2**: Optimized for detecting paraphrases
            - **Similarity score**: 0-1, higher = more similar
            - **> 0.7**: Semantically similar
            """)

        # ====================================================================
        # CHAPTER 5: VECTOR SEARCH LAB
        # ====================================================================
        with gr.Tab("3. Vector Search Lab"):
            gr.Markdown("## Vector Search Lab (Chapter 5)\nSemantic search with ChromaDB")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: Load Documents")
                    vector_model = gr.Dropdown(
                        choices=list(EmbeddingsEngine.get_available_models().keys()),
                        value="all-MiniLM-L6-v2",
                        label="Embedding Model"
                    )
                    load_vector_btn = gr.Button("Load to Vector Store", variant="primary")
                    load_status = gr.Textbox(label="Load Status", lines=8)

            load_vector_btn.click(
                load_to_vector_store_fn,
                inputs=[vector_model],
                outputs=[load_status]
            )

            gr.Markdown("---")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 2: Search")
                    search_query = gr.Textbox(label="Search Query", placeholder="Enter your search query...")
                    top_k_search = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of results (top_k)")
                    search_btn = gr.Button("Search", variant="primary")

                with gr.Column():
                    search_results = gr.Textbox(label="Search Results", lines=15)
                    search_metrics = gr.Textbox(label="Performance Metrics", lines=8)

            search_btn.click(
                search_vector_store_fn,
                inputs=[search_query, top_k_search],
                outputs=[search_results, search_metrics]
            )

            gr.Markdown("""
            ### About Vector Search
            - **Semantic search**: Finds meaning, not just keywords
            - **ChromaDB**: Vector database with automatic embedding
            - **top_k**: Number of most similar results to return
            - **Distance metric**: Cosine similarity (default)
            - **Use case**: When you need conceptual matches
            """)

        # ====================================================================
        # CHAPTER 6: HYBRID RETRIEVAL STUDIO
        # ====================================================================
        with gr.Tab("4. Hybrid Retrieval Studio"):
            gr.Markdown("## Hybrid Retrieval Studio (Chapter 6)\nCombine BM25 + Vector search with reranking")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: Setup Pipeline")
                    setup_pipeline_btn = gr.Button("Setup Retrieval Pipeline", variant="primary")
                    setup_status = gr.Textbox(label="Setup Status", lines=6)

            setup_pipeline_btn.click(
                setup_retrieval_pipeline_fn,
                outputs=[setup_status]
            )

            gr.Markdown("---")

            gr.Markdown("### Step 2: Compare Retrieval Methods")

            with gr.Row():
                compare_query = gr.Textbox(label="Query", placeholder="Enter query to compare methods...")
                compare_top_k = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Results per method")

            compare_btn = gr.Button("Compare Methods", variant="primary")

            with gr.Row():
                with gr.Column():
                    bm25_output = gr.Textbox(label="BM25 Results (Lexical)", lines=10)
                with gr.Column():
                    vector_output = gr.Textbox(label="Vector Results (Semantic)", lines=10)
                with gr.Column():
                    hybrid_output = gr.Textbox(label="Hybrid Results (Combined)", lines=10)

            comparison_summary = gr.Textbox(label="Comparison Summary", lines=6)

            compare_btn.click(
                compare_retrieval_methods_fn,
                inputs=[compare_query, compare_top_k],
                outputs=[bm25_output, vector_output, hybrid_output, comparison_summary]
            )

            gr.Markdown("---")

            gr.Markdown("### Step 3: Custom Hybrid Retrieval")

            with gr.Row():
                with gr.Column():
                    hybrid_query = gr.Textbox(label="Query", placeholder="Enter your query...")
                    hybrid_top_k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of results")
                    bm25_weight = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label="BM25 Weight")
                    vector_weight = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label="Vector Weight")
                    use_reranking = gr.Checkbox(label="Enable Cross-Encoder Reranking", value=False)
                    hybrid_btn = gr.Button("Run Hybrid Retrieval", variant="primary")

                with gr.Column():
                    hybrid_results = gr.Textbox(label="Results", lines=15)
                    hybrid_metrics = gr.Textbox(label="Performance Metrics", lines=8)

            hybrid_btn.click(
                hybrid_retrieval_with_weights_fn,
                inputs=[hybrid_query, hybrid_top_k, bm25_weight, vector_weight, use_reranking],
                outputs=[hybrid_results, hybrid_metrics]
            )

            gr.Markdown("""
            ### About Hybrid Retrieval
            - **BM25**: Lexical search, good for exact terms (e.g., "error code 404")
            - **Vector**: Semantic search, good for concepts (e.g., "connection problems")
            - **Hybrid**: Combines both, captures keywords + meaning
            - **Weights**: Adjust based on your use case
              - High BM25: Technical docs, error codes, IDs
              - High Vector: Conceptual queries, synonyms
              - Balanced: General purpose
            - **Reranking**: Cross-encoder re-scores results (slower but more accurate)
            - **Production tip**: Hybrid + reranking = best quality
            """)

    gr.Markdown("""
    ---
    ### About This App
    Built for the LinkedIn Learning course: *Fundamentals of AI Engineering*

    Explore document processing → embeddings → vector search → hybrid retrieval
    """)


if __name__ == "__main__":
    print("Starting AI Engineering Learning App...")
    print("=" * 60)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
