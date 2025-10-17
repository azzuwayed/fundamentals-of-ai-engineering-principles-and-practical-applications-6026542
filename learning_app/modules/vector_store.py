"""
Vector store module for the learning app (Chapter 5).
Handles ChromaDB operations for vector search.
"""
import time
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Handles ChromaDB vector database operations."""

    def __init__(self, collection_name: str = "learning_app",
                 model_name: str = "all-MiniLM-L6-v2",
                 persist_directory: Optional[str] = None):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Embedding model to use
            persist_directory: Directory for persistent storage (None = in-memory)
        """
        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Set up embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        self.collection_name = collection_name
        self.model_name = model_name

        # Create or get collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )

    def add_documents(self, documents: List[str], ids: Optional[List[str]] = None,
                     metadatas: Optional[List[Dict]] = None) -> Tuple[int, float]:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            ids: Optional list of document IDs
            metadatas: Optional list of metadata dicts

        Returns:
            Tuple of (number added, time taken)
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [{"text": doc[:100]} for doc in documents]

        start_time = time.time()
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        add_time = time.time() - start_time

        return len(documents), add_time

    def search(self, query: str, n_results: int = 5,
               where: Optional[Dict] = None,
               where_document: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
        """
        Search for similar documents.

        Args:
            query: Query text
            n_results: Number of results to return
            where: Metadata filter
            where_document: Content filter

        Returns:
            Tuple of (results list, metrics dict)
        """
        start_time = time.time()

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            where_document=where_document
        )

        query_time = time.time() - start_time

        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })

        metrics = {
            "query_time": query_time,
            "num_results": len(formatted_results),
            "collection_size": self.get_count()
        }

        return formatted_results, metrics

    def get_count(self) -> int:
        """
        Get number of documents in collection.

        Returns:
            Number of documents
        """
        return self.collection.count()

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise Exception(f"Error clearing collection: {str(e)}")

    def get_collection_info(self) -> Dict[str, any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection information
        """
        return {
            "name": self.collection_name,
            "model": self.model_name,
            "count": self.get_count(),
            "metadata": self.collection.metadata
        }

    def benchmark_search(self, queries: List[str], n_results: int = 5) -> Dict[str, any]:
        """
        Benchmark search performance.

        Args:
            queries: List of queries to test
            n_results: Number of results per query

        Returns:
            Performance metrics
        """
        query_times = []

        for query in queries:
            _, metrics = self.search(query, n_results=n_results)
            query_times.append(metrics["query_time"])

        avg_time = sum(query_times) / len(query_times)
        min_time = min(query_times)
        max_time = max(query_times)

        return {
            "num_queries": len(queries),
            "avg_query_time": avg_time,
            "min_query_time": min_time,
            "max_query_time": max_time,
            "queries_per_second": 1 / avg_time if avg_time > 0 else 0,
            "total_time": sum(query_times)
        }


class VectorStoreManager:
    """Manages multiple vector store instances."""

    def __init__(self):
        self.stores: Dict[str, VectorStore] = {}

    def create_store(self, name: str, model_name: str = "all-MiniLM-L6-v2",
                    persist_directory: Optional[str] = None) -> VectorStore:
        """
        Create a new vector store.

        Args:
            name: Store name
            model_name: Embedding model
            persist_directory: Persistence directory

        Returns:
            Created VectorStore instance
        """
        store = VectorStore(
            collection_name=name,
            model_name=model_name,
            persist_directory=persist_directory
        )
        self.stores[name] = store
        return store

    def get_store(self, name: str) -> Optional[VectorStore]:
        """
        Get a vector store by name.

        Args:
            name: Store name

        Returns:
            VectorStore instance or None
        """
        return self.stores.get(name)

    def list_stores(self) -> List[str]:
        """
        Get list of store names.

        Returns:
            List of store names
        """
        return list(self.stores.keys())

    def delete_store(self, name: str) -> bool:
        """
        Delete a vector store.

        Args:
            name: Store name

        Returns:
            True if deleted, False if not found
        """
        if name in self.stores:
            del self.stores[name]
            return True
        return False
