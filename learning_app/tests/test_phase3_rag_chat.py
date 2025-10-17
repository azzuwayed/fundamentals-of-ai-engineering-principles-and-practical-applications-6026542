"""
Test script for Phase 3: RAG Chat Enhancements.
Validates LLM management, context management, conversation engine, and RAG pipeline.
"""
import sys
import os

# Add parent directory (learning_app/) to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def test_llm_manager():
    """Test LLM manager module."""
    print("\n" + "=" * 60)
    print("Testing LLM Manager")
    print("=" * 60)

    try:
        from modules.llm_manager import LLMManager, LocalLLM, LLMConfig

        # Test factory
        print("\n→ Testing LLM factory...")
        llm = LLMManager.create_llm(backend="local", model_name="distilgpt2")
        print("  ✓ Created local LLM")

        # Test model info
        print("\n→ Testing model info...")
        info = llm.get_model_info()
        print(f"  ✓ Model: {info['name']}")
        print(f"  ✓ Backend: {info['backend']}")
        print(f"  ✓ Context window: {info['context_window']}")

        # Test token counting
        print("\n→ Testing token counting...")
        text = "This is a test sentence for token counting."
        tokens = llm.count_tokens(text)
        print(f"  ✓ Text: '{text}'")
        print(f"  ✓ Tokens: {tokens}")

        # Test generation
        print("\n→ Testing text generation...")
        config = LLMConfig(max_tokens=20, temperature=0.7)
        response = llm.generate("The quick brown fox", config)
        print(f"  ✓ Input: 'The quick brown fox'")
        print(f"  ✓ Generated: '{response.text[:50]}...'")
        print(f"  ✓ Tokens used: {response.tokens_used}")

        # Test available backends
        print("\n→ Testing available backends...")
        backends = LLMManager.get_available_backends()
        print(f"  ✓ Available backends: {len(backends)}")
        for backend in backends:
            print(f"    - {backend['name']}: {backend['description']}")

        print("\n✓ All LLM manager tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ LLM manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_manager():
    """Test context manager module."""
    print("\n" + "=" * 60)
    print("Testing Context Manager")
    print("=" * 60)

    try:
        from modules.context_manager import ContextManager

        # Create context manager
        print("\n→ Creating context manager...")
        cm = ContextManager(
            context_window=1024,
            system_prompt_ratio=0.1,
            context_ratio=0.5,
            history_ratio=0.2,
            generation_ratio=0.2
        )
        print("  ✓ Context manager created")
        print(f"  ✓ Total window: {cm.context_window} tokens")

        # Test budget info
        print("\n→ Testing budget info...")
        budget = cm.get_budget_info()
        print(f"  ✓ System budget: {budget['budgets']['system']}")
        print(f"  ✓ Context budget: {budget['budgets']['context']}")
        print(f"  ✓ History budget: {budget['budgets']['history']}")
        print(f"  ✓ Generation budget: {budget['budgets']['generation']}")

        # Test token counting function (simple mock)
        def simple_token_counter(text):
            return len(text.split())

        # Test allocation with fitting content
        print("\n→ Testing allocation (content fits)...")
        system_prompt = "You are a helpful assistant."
        rag_context = "Document 1: This is content. " * 10
        history = [{"role": "user", "content": "Hello"}]

        truncated_system, truncated_context, truncated_history, stats = cm.allocate_budget(
            system_prompt, rag_context, history, simple_token_counter
        )
        print(f"  ✓ Original total: {stats['original']['total']} tokens")
        print(f"  ✓ Final total: {stats['final']['total']} tokens")
        print(f"  ✓ Truncated: {stats['truncated']}")

        # Test allocation with oversized content
        print("\n→ Testing allocation (content too large)...")
        large_context = "Word " * 1000  # Much larger than budget
        _, truncated_large, _, stats_large = cm.allocate_budget(
            system_prompt, large_context, [], simple_token_counter
        )
        print(f"  ✓ Original context: {stats_large['original']['context']} tokens")
        print(f"  ✓ Final context: {stats_large['final']['context']} tokens")
        print(f"  ✓ Truncated: {stats_large['truncated']}")

        # Test report formatting
        print("\n→ Testing report formatting...")
        report_html = cm.format_allocation_report(stats)
        print(f"  ✓ Report HTML generated: {len(report_html)} chars")

        print("\n✓ All context manager tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Context manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conversation_engine():
    """Test conversation engine module."""
    print("\n" + "=" * 60)
    print("Testing Conversation Engine")
    print("=" * 60)

    try:
        from modules.conversation_engine import ConversationEngine, Message

        # Create engine
        print("\n→ Creating conversation engine...")
        engine = ConversationEngine(max_history_turns=5)
        print(f"  ✓ Engine created with max {engine.max_history_turns} turns")

        # Test message adding
        print("\n→ Testing message management...")
        engine.add_user_message("Hello, how are you?")
        engine.add_assistant_message("I'm doing well, thank you!")
        engine.add_user_message("What's the weather like?")
        print(f"  ✓ Added 3 messages")

        # Test history retrieval
        print("\n→ Testing history retrieval...")
        history = engine.get_history_for_rag()
        print(f"  ✓ History for RAG: {len(history)} messages")
        for msg in history:
            print(f"    - {msg['role']}: {msg['content'][:50]}...")

        # Test statistics
        print("\n→ Testing conversation statistics...")
        stats = engine.get_stats()
        print(f"  ✓ Total messages: {stats['total_messages']}")
        print(f"  ✓ User messages: {stats['user_messages']}")
        print(f"  ✓ Assistant messages: {stats['assistant_messages']}")
        print(f"  ✓ Turns: {stats['turns']}")

        # Test HTML formatting
        print("\n→ Testing HTML formatting...")
        html = engine.get_history_html()
        print(f"  ✓ History HTML: {len(html)} chars")

        stats_html = engine.get_stats_html()
        print(f"  ✓ Stats HTML: {len(stats_html)} chars")

        # Test history pruning
        print("\n→ Testing history pruning...")
        for i in range(15):
            engine.add_user_message(f"Message {i}")
            engine.add_assistant_message(f"Response {i}")

        pruned_history = engine.get_history_for_rag()
        print(f"  ✓ After adding 15 more turns")
        print(f"  ✓ History size: {len(pruned_history)} messages")
        print(f"  ✓ Max allowed: {engine.max_history_turns * 2} messages")
        assert len(pruned_history) <= engine.max_history_turns * 2

        # Test clear
        print("\n→ Testing conversation clear...")
        engine.clear()
        assert len(engine.messages) == 0
        print("  ✓ Conversation cleared")

        # Test export/import
        print("\n→ Testing export/import...")
        engine.add_user_message("Test message")
        exported = engine.export_to_dict()
        print(f"  ✓ Exported: {len(exported)} keys")

        new_engine = ConversationEngine.load_from_dict(exported)
        assert len(new_engine.messages) == 1
        print("  ✓ Imported successfully")

        print("\n✓ All conversation engine tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Conversation engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_pipeline():
    """Test RAG pipeline integration."""
    print("\n" + "=" * 60)
    print("Testing RAG Pipeline")
    print("=" * 60)

    try:
        from modules.rag_pipeline import RAGPipeline
        from modules.llm_manager import LLMManager
        from modules.context_manager import ContextManager

        # Create mock retrieval pipeline
        print("\n→ Creating mock retrieval pipeline...")

        class MockRetrievalPipeline:
            def retrieve(self, query, top_k, method, **kwargs):
                return [
                    {"text": f"Document {i}: {query}", "similarity": 0.9 - i * 0.1}
                    for i in range(top_k)
                ]

            def get_stats(self):
                return {"documents": 10}

        mock_retrieval = MockRetrievalPipeline()
        print("  ✓ Mock retrieval pipeline created")

        # Create LLM and context manager
        print("\n→ Creating LLM and context manager...")
        llm = LLMManager.create_llm("local", "distilgpt2")
        cm = ContextManager(context_window=1024)
        print("  ✓ LLM and context manager created")

        # Create RAG pipeline
        print("\n→ Creating RAG pipeline...")
        rag = RAGPipeline(
            retrieval_pipeline=mock_retrieval,
            llm=llm,
            context_manager=cm
        )
        print("  ✓ RAG pipeline created")

        # Execute RAG
        print("\n→ Executing RAG pipeline...")
        result = rag.execute(
            query="What is machine learning?",
            top_k=3,
            retrieval_method="hybrid"
        )
        print(f"  ✓ Query: {result.query}")
        print(f"  ✓ Response: {result.response[:100]}...")
        print(f"  ✓ Sources: {len(result.sources)}")
        print(f"  ✓ Steps: {len(result.steps)}")
        print(f"  ✓ Duration: {result.total_duration_ms:.1f}ms")
        print(f"  ✓ Tokens used: {result.tokens_used}")

        # Verify steps
        print("\n→ Verifying RAG steps...")
        expected_steps = [
            "Query Processing",
            "Document Retrieval",
            "Context Assembly",
            "Prompt Construction",
            "Response Generation",
            "Source Attribution"
        ]
        for i, step in enumerate(result.steps):
            assert step.name == expected_steps[i]
            print(f"  ✓ Step {i+1}: {step.name} ({step.duration_ms:.1f}ms)")

        # Test HTML formatting
        print("\n→ Testing process HTML formatting...")
        process_html = rag.format_rag_process(result)
        print(f"  ✓ Process HTML: {len(process_html)} chars")

        print("\n✓ All RAG pipeline tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test that all Phase 3 modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Module Imports")
    print("=" * 60)

    imports = [
        ("modules.llm_manager", "LLMManager"),
        ("modules.llm_manager", "LocalLLM"),
        ("modules.llm_manager", "LLMConfig"),
        ("modules.context_manager", "ContextManager"),
        ("modules.context_manager", "TokenBudget"),
        ("modules.rag_pipeline", "RAGPipeline"),
        ("modules.rag_pipeline", "RAGResult"),
        ("modules.conversation_engine", "ConversationEngine"),
        ("modules.conversation_engine", "Message"),
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
    print("PHASE 3: RAG CHAT ENHANCEMENTS TEST SUITE")
    print("=" * 60)

    results = {
        "Imports": test_imports(),
        "LLM Manager": test_llm_manager(),
        "Context Manager": test_context_manager(),
        "Conversation Engine": test_conversation_engine(),
        "RAG Pipeline": test_rag_pipeline()
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
