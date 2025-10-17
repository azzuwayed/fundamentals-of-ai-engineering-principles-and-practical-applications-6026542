"""
Retrieval pipeline module for the learning app (Chapter 6).
Handles hybrid retrieval with BM25, vector search, and reranking.
"""
import time
from typing import List, Dict, Optional, Tuple
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from sentence_transformers import CrossEncoder
import numpy as np


class RetrievalPipeline:
    """Complete hybrid retrieval pipeline with BM25, vector search, and reranking."""

    def __init__(self, documents: List[str], chunk_size: int = 512):
        """
        Initialize retrieval pipeline.

        Args:
            documents: List of document texts
            chunk_size: Size for chunking documents
        """
        # Create LlamaIndex documents
        self.documents = [Document(text=doc, id_=f"doc_{i}") for i, doc in enumerate(documents)]

        # Chunk documents
        splitter = SentenceSplitter(chunk_size=chunk_size)
        self.nodes = splitter.get_nodes_from_documents(self.documents)

        # Initialize BM25 retriever
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.nodes,
            similarity_top_k=10
        )

        # Initialize vector store index
        self.vector_index = VectorStoreIndex(self.nodes)
        self.vector_retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=10
        )

        # Cross-encoder for reranking (lazy loaded)
        self.cross_encoder = None
        self.cross_encoder_model = None

    def retrieve_bm25(self, query: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """
        Retrieve using BM25 (lexical search).

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            Tuple of (results, query time)
        """
        start_time = time.time()

        # Update similarity_top_k for this query
        self.bm25_retriever.similarity_top_k = top_k
        nodes = self.bm25_retriever.retrieve(query)

        query_time = time.time() - start_time

        results = []
        for i, node in enumerate(nodes):
            results.append({
                'rank': i + 1,
                'id': node.node_id,
                'content': node.text,
                'score': node.score if node.score else 0.0
            })

        return results, query_time

    def retrieve_vector(self, query: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """
        Retrieve using vector search (semantic search).

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            Tuple of (results, query time)
        """
        start_time = time.time()

        # Update similarity_top_k for this query
        self.vector_retriever.similarity_top_k = top_k
        nodes = self.vector_retriever.retrieve(query)

        query_time = time.time() - start_time

        results = []
        for i, node in enumerate(nodes):
            results.append({
                'rank': i + 1,
                'id': node.node_id,
                'content': node.text,
                'score': node.score if node.score else 0.0
            })

        return results, query_time

    def retrieve_hybrid(self, query: str, top_k: int = 5,
                       bm25_weight: float = 0.5,
                       vector_weight: float = 0.5) -> Tuple[List[Dict], float, Dict]:
        """
        Retrieve using hybrid approach (BM25 + Vector).

        Args:
            query: Query text
            top_k: Number of results
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores

        Returns:
            Tuple of (results, query time, component times)
        """
        # Get results from both retrievers
        bm25_results, bm25_time = self.retrieve_bm25(query, top_k=top_k * 2)
        vector_results, vector_time = self.retrieve_vector(query, top_k=top_k * 2)

        # Normalize weights
        total_weight = bm25_weight + vector_weight
        bm25_weight = bm25_weight / total_weight
        vector_weight = vector_weight / total_weight

        # Combine scores
        combined_scores = {}

        # Add BM25 scores
        for result in bm25_results:
            doc_id = result['id']
            combined_scores[doc_id] = {
                'content': result['content'],
                'bm25_score': result['score'] * bm25_weight,
                'vector_score': 0.0
            }

        # Add vector scores
        for result in vector_results:
            doc_id = result['id']
            if doc_id in combined_scores:
                combined_scores[doc_id]['vector_score'] = result['score'] * vector_weight
            else:
                combined_scores[doc_id] = {
                    'content': result['content'],
                    'bm25_score': 0.0,
                    'vector_score': result['score'] * vector_weight
                }

        # Calculate final scores and sort
        final_results = []
        for doc_id, scores in combined_scores.items():
            final_score = scores['bm25_score'] + scores['vector_score']
            final_results.append({
                'id': doc_id,
                'content': scores['content'],
                'score': final_score,
                'bm25_component': scores['bm25_score'],
                'vector_component': scores['vector_score']
            })

        # Sort by final score and take top_k
        final_results.sort(key=lambda x: x['score'], reverse=True)
        final_results = final_results[:top_k]

        # Add ranks
        for i, result in enumerate(final_results):
            result['rank'] = i + 1

        total_time = bm25_time + vector_time
        component_times = {
            'bm25_time': bm25_time,
            'vector_time': vector_time,
            'total_time': total_time
        }

        return final_results, total_time, component_times

    def rerank(self, query: str, results: List[Dict],
               model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Tuple[List[Dict], float]:
        """
        Rerank results using cross-encoder.

        Args:
            query: Original query
            results: Results to rerank
            model_name: Cross-encoder model name

        Returns:
            Tuple of (reranked results, rerank time)
        """
        # Lazy load cross-encoder
        if self.cross_encoder is None or self.cross_encoder_model != model_name:
            self.cross_encoder = CrossEncoder(model_name)
            self.cross_encoder_model = model_name

        # Prepare pairs for cross-encoder
        pairs = [[query, result['content']] for result in results]

        start_time = time.time()
        scores = self.cross_encoder.predict(pairs)
        rerank_time = time.time() - start_time

        # Update scores and resort
        reranked_results = []
        for i, result in enumerate(results):
            new_result = result.copy()
            new_result['original_score'] = result['score']
            new_result['score'] = float(scores[i])
            reranked_results.append(new_result)

        # Sort by new scores
        reranked_results.sort(key=lambda x: x['score'], reverse=True)

        # Update ranks
        for i, result in enumerate(reranked_results):
            result['rank'] = i + 1

        return reranked_results, rerank_time

    def retrieve_with_reranking(self, query: str, top_k: int = 5,
                               retrieval_method: str = "hybrid",
                               bm25_weight: float = 0.5,
                               vector_weight: float = 0.5,
                               rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Tuple[List[Dict], Dict]:
        """
        Complete retrieval pipeline with reranking.

        Args:
            query: Query text
            top_k: Number of final results
            retrieval_method: "bm25", "vector", or "hybrid"
            bm25_weight: Weight for BM25 (if hybrid)
            vector_weight: Weight for vector (if hybrid)
            rerank_model: Cross-encoder model for reranking

        Returns:
            Tuple of (final results, timing dict)
        """
        # Initial retrieval (get more candidates for reranking)
        candidate_k = top_k * 3

        if retrieval_method == "bm25":
            candidates, retrieval_time = self.retrieve_bm25(query, top_k=candidate_k)
            component_times = {'bm25_time': retrieval_time}
        elif retrieval_method == "vector":
            candidates, retrieval_time = self.retrieve_vector(query, top_k=candidate_k)
            component_times = {'vector_time': retrieval_time}
        else:  # hybrid
            candidates, retrieval_time, component_times = self.retrieve_hybrid(
                query, top_k=candidate_k,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )

        # Rerank
        reranked, rerank_time = self.rerank(query, candidates, model_name=rerank_model)

        # Take top_k after reranking
        final_results = reranked[:top_k]

        timings = {
            **component_times,
            'rerank_time': rerank_time,
            'total_time': retrieval_time + rerank_time
        }

        return final_results, timings

    def compare_methods(self, query: str, top_k: int = 5) -> Dict[str, any]:
        """
        Compare all retrieval methods side by side.

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            Comparison results with timing
        """
        bm25_results, bm25_time = self.retrieve_bm25(query, top_k=top_k)
        vector_results, vector_time = self.retrieve_vector(query, top_k=top_k)
        hybrid_results, hybrid_time, _ = self.retrieve_hybrid(query, top_k=top_k)

        return {
            'query': query,
            'bm25': {
                'results': bm25_results,
                'time': bm25_time
            },
            'vector': {
                'results': vector_results,
                'time': vector_time
            },
            'hybrid': {
                'results': hybrid_results,
                'time': hybrid_time
            }
        }

    def get_stats(self) -> Dict[str, any]:
        """
        Get pipeline statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'num_documents': len(self.documents),
            'num_chunks': len(self.nodes),
            'avg_chunk_length': np.mean([len(node.text) for node in self.nodes])
        }
