"""
Test script for Phase 1 Enhancements.
Validates visualization and explainability features.
"""
import sys
import os

# Add parent directory (learning_app/) to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress sklearn and UMAP warnings
import warnings
warnings.filterwarnings('ignore', message='.*force_all_finite.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*n_jobs value.*overridden.*', category=UserWarning)

import numpy as np
from sentence_transformers import SentenceTransformer


def test_visualization_engine():
    """Test visualization engine functionality."""
    print("\n" + "=" * 60)
    print("Testing Visualization Engine")
    print("=" * 60)

    try:
        from modules.visualization_engine import VisualizationEngine

        engine = VisualizationEngine()
        print("✓ VisualizationEngine imported successfully")

        # Test data
        embeddings = np.random.rand(10, 384).tolist()  # 10 embeddings, 384 dimensions
        labels = [f"Doc {i+1}" for i in range(10)]

        # Test dimensionality reduction
        print("\n→ Testing dimensionality reduction...")
        reduced_umap, time_umap = engine.reduce_dimensions(embeddings, method="umap", n_components=2)
        print(f"  ✓ UMAP reduction: {reduced_umap.shape} in {time_umap:.4f}s")

        reduced_tsne, time_tsne = engine.reduce_dimensions(embeddings, method="tsne", n_components=2)
        print(f"  ✓ t-SNE reduction: {reduced_tsne.shape} in {time_tsne:.4f}s")

        # Test embedding space plot
        print("\n→ Testing embedding space visualization...")
        fig_2d, timings_2d = engine.plot_embedding_space(embeddings, labels, method="umap", n_components=2)
        print(f"  ✓ 2D plot created: {type(fig_2d).__name__}")

        fig_3d, timings_3d = engine.plot_embedding_space(embeddings, labels, method="umap", n_components=3)
        print(f"  ✓ 3D plot created: {type(fig_3d).__name__}")

        # Test similarity heatmap
        print("\n→ Testing similarity heatmap...")
        heatmap_fig, metadata = engine.plot_similarity_heatmap(embeddings, labels)
        print(f"  ✓ Heatmap created: avg similarity = {metadata['avg_similarity']:.4f}")

        print("\n✓ All visualization tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Visualization engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_explainability_engine():
    """Test explainability engine functionality."""
    print("\n" + "=" * 60)
    print("Testing Explainability Engine")
    print("=" * 60)

    try:
        from modules.explainability_engine import ExplainabilityEngine, format_bm25_table, format_similarity_breakdown

        engine = ExplainabilityEngine()
        print("✓ ExplainabilityEngine imported successfully")

        # Test data
        query = "machine learning algorithms"
        document = "Machine learning algorithms are computational methods that enable systems to learn from data and improve their performance over time."
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Test token similarity
        print("\n→ Testing token similarity explanation...")
        token_exp = engine.explain_token_similarity(query, document, model, top_k=5)
        print(f"  ✓ Overall similarity: {token_exp['overall_similarity']:.4f}")
        print(f"  ✓ Top tokens analyzed: {len(token_exp['token_contributions'])}")

        # Test BM25 explanation
        print("\n→ Testing BM25 explanation...")
        bm25_exp = engine.explain_bm25_score(query, document)
        print(f"  ✓ BM25 score: {bm25_exp['total_bm25_score']:.4f}")
        print(f"  ✓ Terms analyzed: {len(bm25_exp['term_scores'])}")

        # Test BM25 table formatting
        table_html = format_bm25_table(bm25_exp['term_scores'])
        print(f"  ✓ BM25 table HTML generated: {len(table_html)} chars")

        # Test vector similarity
        print("\n→ Testing vector similarity explanation...")
        vec_exp = engine.explain_vector_similarity(query, document, model)
        print(f"  ✓ Cosine similarity: {vec_exp['cosine_similarity']:.4f}")
        print(f"  ✓ Top dimensions: {len(vec_exp['top_contributing_dimensions'])}")

        # Test similarity breakdown formatting
        breakdown_html = format_similarity_breakdown(vec_exp)
        print(f"  ✓ Similarity breakdown HTML generated: {len(breakdown_html)} chars")

        # Test token highlighting
        print("\n→ Testing token highlighting...")
        highlighted = engine.highlight_important_tokens(
            query,
            token_exp['token_contributions']
        )
        print(f"  ✓ Highlighted HTML generated: {len(highlighted)} chars")

        print("\n✓ All explainability tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Explainability engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plot_helpers():
    """Test plot helpers functionality."""
    print("\n" + "=" * 60)
    print("Testing Plot Helpers")
    print("=" * 60)

    try:
        from utils.plot_helpers import (
            create_plotly_config,
            format_hover_data,
            get_color_scale,
            create_annotation_box,
            create_score_gauge
        )

        print("✓ Plot helpers imported successfully")

        # Test plotly config
        print("\n→ Testing plotly config...")
        config = create_plotly_config(filename="test")
        print(f"  ✓ Config created with {len(config)} settings")

        # Test hover data formatting
        print("\n→ Testing hover data formatting...")
        docs = [
            {"title": "Doc 1", "score": 0.85, "length": 500},
            {"title": "Doc 2", "score": 0.72, "length": 300}
        ]
        hover_texts = format_hover_data(docs)
        print(f"  ✓ Formatted {len(hover_texts)} hover texts")

        # Test color scale
        print("\n→ Testing color scales...")
        colors = get_color_scale(10, palette="Viridis", discrete=False)
        print(f"  ✓ Generated {len(colors)} colors")

        # Test annotation box
        print("\n→ Testing annotation box...")
        annotation = create_annotation_box("Test annotation", x=0.5, y=1.0)
        print(f"  ✓ Annotation created with {len(annotation)} properties")

        # Test score gauge
        print("\n→ Testing score gauge...")
        gauge_fig = create_score_gauge(0.75, "Test Score")
        print(f"  ✓ Gauge figure created: {type(gauge_fig).__name__}")

        print("\n✓ All plot helper tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Plot helpers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Module Imports")
    print("=" * 60)

    imports = [
        ("modules.visualization_engine", "VisualizationEngine"),
        ("modules.explainability_engine", "ExplainabilityEngine"),
        ("utils.plot_helpers", "create_plotly_config"),
    ]

    all_passed = True
    for module_name, class_name in imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"✗ {module_name}.{class_name}: {e}")
            all_passed = False

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PHASE 1 ENHANCEMENTS TEST SUITE")
    print("=" * 60)

    results = {
        "Imports": test_imports(),
        "Visualization Engine": test_visualization_engine(),
        "Explainability Engine": test_explainability_engine(),
        "Plot Helpers": test_plot_helpers()
    }

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
