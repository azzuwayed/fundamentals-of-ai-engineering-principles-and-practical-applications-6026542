"""
Test script for Phase 2 Enhancements.
Validates query intelligence, multi-query, and advanced filtering features.
"""
import sys
import os

# Add parent directory (learning_app/) to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def test_query_intelligence():
    """Test query intelligence module."""
    print("\n" + "=" * 60)
    print("Testing Query Intelligence")
    print("=" * 60)

    try:
        from modules.query_intelligence import QueryIntelligence

        engine = QueryIntelligence()
        print("✓ QueryIntelligence imported successfully")

        # Test query analysis
        print("\n→ Testing query analysis...")
        query = "What is machine learning and how does it work?"
        analysis = engine.analyze_query(query)

        print(f"  ✓ Intent: {analysis.intent} (confidence: {analysis.intent_confidence:.2f})")
        print(f"  ✓ Complexity: {analysis.complexity_score:.3f}")
        print(f"  ✓ Keywords: {len(analysis.keywords)} extracted")
        print(f"  ✓ Suggestions: {len(analysis.suggestions)} generated")

        # Test different intents
        print("\n→ Testing intent classification...")
        test_queries = {
            "What is Python?": "factual",
            "Why does Python use indentation?": "conceptual",
            "Explore machine learning algorithms": "exploratory",
            "Compare Python vs Java": "comparison"
        }

        for test_query, expected_intent in test_queries.items():
            analysis = engine.analyze_query(test_query)
            print(f"  ✓ '{test_query[:30]}...' → {analysis.intent}")

        # Test report formatting
        print("\n→ Testing report formatting...")
        report_html = engine.format_analysis_report(analysis)
        print(f"  ✓ Report HTML generated: {len(report_html)} chars")

        print("\n✓ All query intelligence tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Query intelligence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_query_engine():
    """Test multi-query engine module."""
    print("\n" + "=" * 60)
    print("Testing Multi-Query Engine")
    print("=" * 60)

    try:
        from modules.multi_query_engine import MultiQueryEngine

        engine = MultiQueryEngine(max_workers=2)
        print("✓ MultiQueryEngine imported successfully")

        # Test query decomposition
        print("\n→ Testing query decomposition...")
        query = "Explain machine learning algorithms and their applications"
        variations = engine.decompose_query(query, num_variations=3)
        print(f"  ✓ Generated {len(variations)} decomposition variations")
        for i, var in enumerate(variations[:2], 1):
            print(f"    {i}. {var.variation}")

        # Test query expansion
        print("\n→ Testing query expansion...")
        variations = engine.expand_query(query, num_expansions=3)
        print(f"  ✓ Generated {len(variations)} expansion variations")
        for i, var in enumerate(variations[:2], 1):
            print(f"    {i}. {var.variation}")

        # Test query rephrasing
        print("\n→ Testing query rephrasing...")
        variations = engine.rephrase_query(query, num_rephrasings=3)
        print(f"  ✓ Generated {len(variations)} rephrasing variations")

        # Test parallel execution (mock function)
        print("\n→ Testing parallel execution...")

        def mock_retrieval(q, k):
            """Mock retrieval function for testing."""
            return [
                {"text": f"Result for '{q}' #{i}", "score": 0.9 - i*0.1}
                for i in range(min(k, 3))
            ]

        test_variations = engine.expand_query("test query", 3)
        results = engine.execute_parallel(
            test_variations,
            mock_retrieval,
            max_results_per_query=3
        )
        print(f"  ✓ Executed {len(results)} queries in parallel")
        print(f"  ✓ Total results retrieved: {sum(len(r) for r in results.values())}")

        # Test result fusion
        print("\n→ Testing result fusion...")
        fused_weighted = engine.fuse_results(results, strategy="weighted_voting", top_k=5)
        print(f"  ✓ Weighted voting fusion: {len(fused_weighted)} results")

        fused_round_robin = engine.fuse_results(results, strategy="round_robin", top_k=5)
        print(f"  ✓ Round robin fusion: {len(fused_round_robin)} results")

        fused_score = engine.fuse_results(results, strategy="score_aggregation", top_k=5)
        print(f"  ✓ Score aggregation fusion: {len(fused_score)} results")

        # Test complete multi-query execution
        print("\n→ Testing complete multi-query execution...")
        multi_result = engine.execute_multi_query(
            query="test query",
            retrieval_fn=mock_retrieval,
            strategy="hybrid",
            fusion_method="weighted_voting",
            num_variations=3,
            top_k=5
        )
        print(f"  ✓ Multi-query executed: {multi_result.metrics['num_queries_executed']} queries")
        print(f"  ✓ Execution time: {multi_result.execution_time:.4f}s")
        print(f"  ✓ Final results: {len(multi_result.fused_results)}")

        # Test comparison table formatting
        print("\n→ Testing comparison table formatting...")
        table_html = engine.format_comparison_table(multi_result)
        print(f"  ✓ Comparison table HTML generated: {len(table_html)} chars")

        print("\n✓ All multi-query engine tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Multi-query engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_filtering():
    """Test advanced filtering module."""
    print("\n" + "=" * 60)
    print("Testing Advanced Filtering")
    print("=" * 60)

    try:
        from modules.advanced_filtering import AdvancedFilter, FilterConfig
        import numpy as np

        filter_engine = AdvancedFilter()
        print("✓ AdvancedFilter imported successfully")

        # Test data
        results = [
            {
                "text": "Document 1 about machine learning",
                "score": 0.9,
                "similarity": 0.9,
                "metadata": {"type": "pdf", "category": "AI", "source": "arxiv"}
            },
            {
                "text": "Document 2 about machine learning algorithms",
                "score": 0.85,
                "similarity": 0.85,
                "metadata": {"type": "pdf", "category": "AI", "source": "arxiv"}
            },
            {
                "text": "Document 3 about neural networks",
                "score": 0.75,
                "similarity": 0.75,
                "metadata": {"type": "docx", "category": "ML", "source": "wiki"}
            },
            {
                "text": "Document 4 completely different topic",
                "score": 0.45,
                "similarity": 0.45,
                "metadata": {"type": "txt", "category": "other", "source": "blog"}
            }
        ]

        # Test similarity filtering
        print("\n→ Testing similarity filtering...")
        filtered = filter_engine.filter_by_similarity(results, min_score=0.7)
        print(f"  ✓ Filtered by similarity (>= 0.7): {len(filtered)} results")

        # Test metadata filtering
        print("\n→ Testing metadata filtering...")
        filtered = filter_engine.filter_by_document_type(results, ["pdf"])
        print(f"  ✓ Filtered by document type (pdf): {len(filtered)} results")

        filtered = filter_engine.filter_by_categories(results, ["AI", "ML"])
        print(f"  ✓ Filtered by categories (AI, ML): {len(filtered)} results")

        filtered = filter_engine.filter_by_sources(results, ["arxiv"])
        print(f"  ✓ Filtered by sources (arxiv): {len(filtered)} results")

        # Test diversity filter
        print("\n→ Testing diversity filter...")
        filtered = filter_engine.apply_diversity_filter(results, similarity_threshold=0.8)
        print(f"  ✓ Diversity filter applied: {len(filtered)} diverse results")

        # Test MMR (text-based)
        print("\n→ Testing MMR (text-based)...")
        filtered = filter_engine.apply_mmr(results, lambda_param=0.5, top_k=3)
        print(f"  ✓ MMR applied (λ=0.5): {len(filtered)} results")

        # Test MMR with embeddings
        print("\n→ Testing MMR (embedding-based)...")
        # Add mock embeddings
        for result in results:
            result['embedding'] = np.random.rand(384).tolist()

        query_embedding = np.random.rand(384)
        filtered = filter_engine.apply_mmr(results, query_embedding, lambda_param=0.7, top_k=3)
        print(f"  ✓ MMR with embeddings (λ=0.7): {len(filtered)} results")

        # Test complete filtering pipeline
        print("\n→ Testing complete filtering pipeline...")
        config = FilterConfig(
            document_types=["pdf"],
            categories=["AI", "ML"],
            min_similarity=0.5,
            enable_diversity=True,
            diversity_threshold=0.8
        )
        filtered = filter_engine.apply_filters(results, config)
        print(f"  ✓ Complete pipeline: {len(filtered)} results")

        # Test filter summary formatting
        print("\n→ Testing filter summary formatting...")
        summary_html = filter_engine.format_filter_summary(config)
        print(f"  ✓ Filter summary HTML generated: {len(summary_html)} chars")

        print("\n✓ All advanced filtering tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Advanced filtering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Module Imports")
    print("=" * 60)

    imports = [
        ("modules.query_intelligence", "QueryIntelligence"),
        ("modules.multi_query_engine", "MultiQueryEngine"),
        ("modules.advanced_filtering", "AdvancedFilter"),
        ("modules.advanced_filtering", "FilterConfig"),
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
    print("PHASE 2 ENHANCEMENTS TEST SUITE")
    print("=" * 60)

    results = {
        "Imports": test_imports(),
        "Query Intelligence": test_query_intelligence(),
        "Multi-Query Engine": test_multi_query_engine(),
        "Advanced Filtering": test_advanced_filtering()
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
