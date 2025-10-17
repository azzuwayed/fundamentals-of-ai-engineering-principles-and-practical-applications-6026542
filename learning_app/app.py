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
from modules.visualization_engine import VisualizationEngine
from modules.explainability_engine import (
    ExplainabilityEngine,
    format_bm25_table,
    format_similarity_breakdown
)
from utils import (
    format_results_table, format_metrics, format_document_preview,
    format_embedding_info, format_comparison_table,
    validate_file_upload, validate_text_input, validate_top_k,
    create_plotly_config
)

# Global state
document_processor = DocumentProcessor()
current_documents = []
current_chunks = []
vector_store = None
retrieval_pipeline = None

# Phase 1 Enhancement engines
visualization_engine = VisualizationEngine()
explainability_engine = ExplainabilityEngine()
current_embeddings = []  # Store embeddings for visualization
current_model = None  # Current embedding model


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
# TAB 5: VISUALIZATION LAB
# ============================================================================

def generate_embeddings_for_viz_fn(model_name):
    """Generate embeddings for current documents for visualization."""
    global current_embeddings, current_model, current_chunks, current_documents

    try:
        if not current_documents and not current_chunks:
            return "No documents loaded. Please process a document in the Document Processing tab first.", None

        docs_to_embed = [chunk['text'] for chunk in current_chunks[:50]] if current_chunks else current_documents[:50]

        if len(docs_to_embed) < 2:
            return "Need at least 2 documents/chunks for visualization.", None

        # Generate embeddings
        engine = EmbeddingsEngine(model_name)
        embeddings, gen_time = engine.generate_embeddings_batch(docs_to_embed)

        # Store for reuse
        current_embeddings = embeddings
        current_model = model_name

        result = f"✓ Generated embeddings for {len(embeddings)} documents\n"
        result += f"Model: {model_name}\n"
        result += f"Dimensions: {len(embeddings[0])}\n"
        result += f"Time: {gen_time:.4f}s\n"
        result += f"Ready for visualization!"

        return result, None

    except Exception as e:
        return f"Error: {str(e)}", None


def plot_embedding_space_fn(method, dimensions):
    """Create embedding space visualization."""
    global current_embeddings, visualization_engine, current_chunks

    try:
        if not current_embeddings:
            return None

        # Create labels
        if current_chunks:
            labels = [f"Chunk {i+1}" for i in range(len(current_embeddings))]
        else:
            labels = [f"Doc {i+1}" for i in range(len(current_embeddings))]

        # Generate plot
        n_components = 3 if dimensions == "3D" else 2
        fig, timings = visualization_engine.plot_embedding_space(
            current_embeddings,
            labels=labels,
            method=method.lower(),
            n_components=n_components,
            title=f"Embedding Space ({method.upper()}, {dimensions})"
        )

        return fig

    except Exception as e:
        print(f"Error in plot_embedding_space_fn: {e}")
        return None


def plot_similarity_heatmap_fn():
    """Create similarity heatmap."""
    global current_embeddings, visualization_engine, current_chunks

    try:
        if not current_embeddings:
            return None

        # Create labels
        if current_chunks:
            labels = [f"C{i+1}" for i in range(len(current_embeddings))]
        else:
            labels = [f"D{i+1}" for i in range(len(current_embeddings))]

        # Limit to 20 for readability
        embeddings_subset = current_embeddings[:20]
        labels_subset = labels[:20]

        # Generate heatmap
        fig, metadata = visualization_engine.plot_similarity_heatmap(
            embeddings_subset,
            labels=labels_subset,
            title="Document Similarity Matrix"
        )

        return fig

    except Exception as e:
        print(f"Error in plot_similarity_heatmap_fn: {e}")
        return None


# ============================================================================
# TAB 6: EXPLAINABILITY STUDIO
# ============================================================================

def explain_token_similarity_fn(query, document, model_name):
    """Explain token-level contributions to similarity."""
    try:
        is_valid1, error1 = validate_text_input(query, min_length=3)
        if not is_valid1:
            return f"Query error: {error1}", "", None

        is_valid2, error2 = validate_text_input(document, min_length=10)
        if not is_valid2:
            return f"Document error: {error2}", "", None

        # Load model
        model = EmbeddingsEngine(model_name).model

        # Get explanation
        explanation = explainability_engine.explain_token_similarity(
            query, document, model, top_k=10
        )

        # Format results
        result = f"Overall Similarity: {explanation['overall_similarity']:.4f}\n\n"
        result += f"Top Contributing Tokens:\n"
        result += "=" * 60 + "\n"

        for i, token_info in enumerate(explanation['token_contributions'][:10], 1):
            in_doc = "✓" if token_info['in_document'] else "✗"
            result += f"{i}. {token_info['token']}: {token_info['contribution']:.4f} [{in_doc} in doc]\n"

        # Generate highlighted text
        highlighted = explainability_engine.highlight_important_tokens(
            query,
            explanation['token_contributions']
        )

        return result, highlighted, None

    except Exception as e:
        return f"Error: {str(e)}", "", None


def explain_bm25_fn(query, document):
    """Explain BM25 score calculation."""
    try:
        is_valid1, error1 = validate_text_input(query, min_length=3)
        if not is_valid1:
            return f"Query error: {error1}", ""

        is_valid2, error2 = validate_text_input(document, min_length=10)
        if not is_valid2:
            return f"Document error: {error2}", ""

        # Get explanation
        explanation = explainability_engine.explain_bm25_score(query, document)

        # Format results
        result = f"Total BM25 Score: {explanation['total_bm25_score']:.4f}\n\n"
        result += f"Parameters:\n"
        result += f"  k1={explanation['parameters']['k1']}, b={explanation['parameters']['b']}\n"
        result += f"  Doc length: {explanation['parameters']['doc_length']} tokens\n"
        result += f"  Avg doc length: {explanation['parameters']['avg_doc_length']:.0f} tokens\n\n"

        # Create table
        table_html = format_bm25_table(explanation['term_scores'])

        return result, table_html

    except Exception as e:
        return f"Error: {str(e)}", ""


def explain_vector_similarity_fn(query, document, model_name):
    """Explain vector similarity calculation."""
    try:
        is_valid1, error1 = validate_text_input(query, min_length=3)
        if not is_valid1:
            return f"Query error: {error1}", ""

        is_valid2, error2 = validate_text_input(document, min_length=10)
        if not is_valid2:
            return f"Document error: {error2}", ""

        # Load model
        model = EmbeddingsEngine(model_name).model

        # Get explanation
        explanation = explainability_engine.explain_vector_similarity(query, document, model)

        # Format results
        result = f"Cosine Similarity: {explanation['cosine_similarity']:.4f}\n"
        result += f"Interpretation: {explanation['interpretation']}\n\n"
        result += f"Components:\n"
        result += f"  Dot Product: {explanation['dot_product']:.4f}\n"
        result += f"  Query Magnitude: {explanation['query_norm']:.4f}\n"
        result += f"  Document Magnitude: {explanation['doc_norm']:.4f}\n"
        result += f"  Euclidean Distance: {explanation['euclidean_distance']:.4f}\n\n"
        result += f"Top Contributing Dimensions:\n"
        result += "=" * 60 + "\n"

        for i, dim in enumerate(explanation['top_contributing_dimensions'][:5], 1):
            result += f"{i}. Dim {dim['dimension']}: {dim['contribution']:.4f} ({dim['percentage']:.1f}%)\n"

        # Create breakdown HTML
        breakdown_html = format_similarity_breakdown(explanation)

        return result, breakdown_html

    except Exception as e:
        return f"Error: {str(e)}", ""


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

        # ====================================================================
        # TAB 5: VISUALIZATION LAB
        # ====================================================================
        with gr.Tab("5. 📊 Visualization Lab"):
            gr.Markdown("## Visualization Lab (Phase 1 Enhancements)\nVisualize embedding spaces and document similarities")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: Generate Embeddings")
                    viz_model = gr.Dropdown(
                        choices=list(EmbeddingsEngine.get_available_models().keys()),
                        value="all-MiniLM-L6-v2",
                        label="Embedding Model"
                    )
                    generate_emb_btn = gr.Button("Generate Embeddings", variant="primary")
                    emb_status = gr.Textbox(label="Status", lines=5)

            generate_emb_btn.click(
                generate_embeddings_for_viz_fn,
                inputs=[viz_model],
                outputs=[emb_status, gr.State()]
            )

            gr.Markdown("---")

            with gr.Tabs():
                # Sub-tab: Embedding Space Visualization
                with gr.Tab("Embedding Space"):
                    gr.Markdown("### Visualize High-Dimensional Embeddings in 2D/3D")

                    with gr.Row():
                        reduction_method = gr.Radio(
                            choices=["UMAP", "t-SNE"],
                            value="UMAP",
                            label="Dimensionality Reduction Method"
                        )
                        viz_dimensions = gr.Radio(
                            choices=["2D", "3D"],
                            value="2D",
                            label="Visualization Dimensions"
                        )

                    plot_space_btn = gr.Button("Generate Visualization", variant="primary")
                    embedding_space_plot = gr.Plot(label="Embedding Space Visualization")

                    plot_space_btn.click(
                        plot_embedding_space_fn,
                        inputs=[reduction_method, viz_dimensions],
                        outputs=[embedding_space_plot]
                    )

                    gr.Markdown("""
                    **About Embedding Space Visualization:**
                    - **UMAP**: Preserves global structure, faster, good for large datasets
                    - **t-SNE**: Preserves local structure, slower, good for cluster visualization
                    - **2D**: Easier to interpret, faster to compute
                    - **3D**: More information, interactive rotation
                    - **Hover**: See document labels and details
                    - **Use Case**: Understand how documents cluster semantically
                    """)

                # Sub-tab: Similarity Heatmap
                with gr.Tab("Similarity Heatmap"):
                    gr.Markdown("### Document Similarity Matrix")

                    plot_heatmap_btn = gr.Button("Generate Heatmap", variant="primary")
                    heatmap_plot = gr.Plot(label="Similarity Heatmap")

                    plot_heatmap_btn.click(
                        plot_similarity_heatmap_fn,
                        outputs=[heatmap_plot]
                    )

                    gr.Markdown("""
                    **About Similarity Heatmap:**
                    - **Color Scale**: Green (similar) → Yellow (moderate) → Red (dissimilar)
                    - **Diagonal**: Always 1.0 (document compared to itself)
                    - **Symmetry**: Matrix is symmetric (A→B = B→A)
                    - **Click**: See exact similarity scores between document pairs
                    - **Use Case**: Identify duplicate or highly related documents
                    - **Note**: Limited to first 20 documents for readability
                    """)

        # ====================================================================
        # TAB 6: EXPLAINABILITY STUDIO
        # ====================================================================
        with gr.Tab("6. 🔍 Explainability Studio"):
            gr.Markdown("## Explainability Studio (Phase 1 Enhancements)\nUnderstand how retrieval systems make decisions")

            with gr.Tabs():
                # Sub-tab: Token Analysis
                with gr.Tab("Token Analysis"):
                    gr.Markdown("### Token-Level Similarity Contributions")

                    with gr.Row():
                        with gr.Column():
                            token_query = gr.Textbox(label="Query", placeholder="Enter your query...", lines=2)
                            token_document = gr.Textbox(label="Document", placeholder="Enter document text...", lines=5)
                            token_model = gr.Dropdown(
                                choices=list(EmbeddingsEngine.get_available_models().keys()),
                                value="all-MiniLM-L6-v2",
                                label="Embedding Model"
                            )
                            token_analyze_btn = gr.Button("Analyze Tokens", variant="primary")

                        with gr.Column():
                            token_results = gr.Textbox(label="Token Contributions", lines=15)

                    token_highlighted = gr.HTML(label="Highlighted Query (by importance)")

                    token_analyze_btn.click(
                        explain_token_similarity_fn,
                        inputs=[token_query, token_document, token_model],
                        outputs=[token_results, token_highlighted, gr.State()]
                    )

                    gr.Markdown("""
                    **About Token Analysis:**
                    - Shows which query tokens contribute most to similarity
                    - **Contribution**: Higher = more important for matching
                    - **✓ in doc**: Token appears in the document
                    - **Highlighting**: Darker = more important
                    - **Use Case**: Debug why documents match or don't match
                    - **Insight**: Understand semantic vs lexical matching
                    """)

                # Sub-tab: BM25 Breakdown
                with gr.Tab("BM25 Breakdown"):
                    gr.Markdown("### BM25 Score Calculation Explained")

                    with gr.Row():
                        with gr.Column():
                            bm25_query = gr.Textbox(label="Query", placeholder="Enter query...", lines=2)
                            bm25_document = gr.Textbox(label="Document", placeholder="Enter document...", lines=5)
                            bm25_explain_btn = gr.Button("Explain BM25", variant="primary")

                        with gr.Column():
                            bm25_results = gr.Textbox(label="BM25 Score Breakdown", lines=10)

                    bm25_table = gr.HTML(label="Term-by-Term Analysis")

                    bm25_explain_btn.click(
                        explain_bm25_fn,
                        inputs=[bm25_query, bm25_document],
                        outputs=[bm25_results, bm25_table]
                    )

                    gr.Markdown("""
                    **About BM25:**
                    - **TF (Term Frequency)**: How many times term appears in document
                    - **IDF (Inverse Document Frequency)**: Rarity of term across corpus
                    - **k1**: Controls term frequency saturation (default: 1.5)
                    - **b**: Controls document length normalization (default: 0.75)
                    - **Formula**: IDF × (TF × (k1 + 1)) / (TF + k1 × (1 - b + b × doc_len/avg_len))
                    - **Use Case**: Understand why lexical search returns specific results
                    """)

                # Sub-tab: Vector Similarity
                with gr.Tab("Vector Similarity"):
                    gr.Markdown("### Cosine Similarity Component Analysis")

                    with gr.Row():
                        with gr.Column():
                            vec_query = gr.Textbox(label="Query", placeholder="Enter query...", lines=2)
                            vec_document = gr.Textbox(label="Document", placeholder="Enter document...", lines=5)
                            vec_model = gr.Dropdown(
                                choices=list(EmbeddingsEngine.get_available_models().keys()),
                                value="all-MiniLM-L6-v2",
                                label="Embedding Model"
                            )
                            vec_explain_btn = gr.Button("Explain Similarity", variant="primary")

                        with gr.Column():
                            vec_results = gr.Textbox(label="Similarity Breakdown", lines=15)

                    vec_breakdown = gr.HTML(label="Formula Breakdown")

                    vec_explain_btn.click(
                        explain_vector_similarity_fn,
                        inputs=[vec_query, vec_document, vec_model],
                        outputs=[vec_results, vec_breakdown]
                    )

                    gr.Markdown("""
                    **About Vector Similarity:**
                    - **Cosine Similarity**: Measures angle between embedding vectors
                    - **Range**: -1 (opposite) to 1 (identical), usually 0-1 for text
                    - **Dot Product**: Sum of component-wise multiplications
                    - **Magnitude**: Length of embedding vector (L2 norm)
                    - **Formula**: cos(θ) = (A · B) / (||A|| × ||B||)
                    - **Top Dimensions**: Which embedding dimensions contribute most
                    - **Use Case**: Understand semantic similarity at vector level
                    """)

    gr.Markdown("""
    ---
    ### About This App
    Built for the LinkedIn Learning course: *Fundamentals of AI Engineering*

    Explore document processing → embeddings → vector search → hybrid retrieval → visualization → explainability
    """)


if __name__ == "__main__":
    print("Starting AI Engineering Learning App...")
    print("=" * 60)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
