"""
Embeddings engine module for the learning app (Chapter 4).
Handles embedding generation and similarity computation.
"""
import time
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
import numpy as np


class EmbeddingsEngine:
    """Handles embedding generation and similarity computation."""

    # Available models with their properties
    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2": {
            "dimensions": 384,
            "description": "Fast and efficient, good for general use",
            "size_mb": 80
        },
        "all-mpnet-base-v2": {
            "dimensions": 768,
            "description": "Higher quality, larger model",
            "size_mb": 420
        },
        "paraphrase-MiniLM-L6-v2": {
            "dimensions": 384,
            "description": "Optimized for paraphrase detection",
            "size_mb": 80
        }
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embeddings engine.

        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.AVAILABLE_MODELS.keys())}")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimensions = self.AVAILABLE_MODELS[model_name]["dimensions"]

    def generate_embedding(self, text: str) -> Tuple[List[float], float]:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Tuple of (embedding vector, generation time in seconds)
        """
        start_time = time.time()
        embedding = self.model.encode(text, convert_to_numpy=True)
        generation_time = time.time() - start_time

        return embedding.tolist(), generation_time

    def generate_embeddings_batch(self, texts: List[str]) -> Tuple[List[List[float]], float]:
        """
        Generate embeddings for multiple texts (batched).

        Args:
            texts: List of input texts

        Returns:
            Tuple of (list of embedding vectors, total generation time)
        """
        start_time = time.time()
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        generation_time = time.time() - start_time

        return [emb.tolist() for emb in embeddings], generation_time

    def compute_similarity(self, text1: str, text2: str) -> Tuple[float, Dict[str, float]]:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Tuple of (similarity score, timing dict)
        """
        # Generate embeddings
        emb1, time1 = self.generate_embedding(text1)
        emb2, time2 = self.generate_embedding(text2)

        # Compute similarity
        similarity = float(util.cos_sim([emb1], [emb2])[0][0])

        timings = {
            "embedding_1_time": time1,
            "embedding_2_time": time2,
            "total_time": time1 + time2
        }

        return similarity, timings

    def compute_similarity_matrix(self, texts: List[str]) -> Tuple[np.ndarray, float]:
        """
        Compute pairwise similarity matrix for multiple texts.

        Args:
            texts: List of texts

        Returns:
            Tuple of (similarity matrix, generation time)
        """
        # Generate all embeddings at once (efficient batching)
        embeddings, gen_time = self.generate_embeddings_batch(texts)

        # Compute pairwise similarities
        similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()

        return similarity_matrix, gen_time

    def compare_models(self, text: str, model_names: List[str] = None) -> List[Dict[str, any]]:
        """
        Compare different embedding models on the same text.

        Args:
            text: Text to embed
            model_names: List of models to compare (None = all available)

        Returns:
            List of comparison results
        """
        if model_names is None:
            model_names = list(self.AVAILABLE_MODELS.keys())

        results = []
        for model_name in model_names:
            try:
                # Create temporary engine for this model
                engine = EmbeddingsEngine(model_name)
                embedding, gen_time = engine.generate_embedding(text)

                results.append({
                    "model": model_name,
                    "dimensions": engine.dimensions,
                    "generation_time_ms": gen_time * 1000,
                    "description": self.AVAILABLE_MODELS[model_name]["description"],
                    "embedding_preview": str(embedding[:5]) + " ..."
                })
            except Exception as e:
                results.append({
                    "model": model_name,
                    "error": str(e)
                })

        return results

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the currently loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "description": self.AVAILABLE_MODELS[self.model_name]["description"],
            "size_mb": self.AVAILABLE_MODELS[self.model_name]["size_mb"]
        }

    @staticmethod
    def get_available_models() -> Dict[str, Dict]:
        """
        Get information about all available models.

        Returns:
            Dictionary mapping model names to their properties
        """
        return EmbeddingsEngine.AVAILABLE_MODELS.copy()


def format_similarity_score(score: float) -> str:
    """
    Format similarity score with interpretation.

    Args:
        score: Similarity score (0-1)

    Returns:
        Formatted string with interpretation
    """
    interpretation = ""
    if score >= 0.9:
        interpretation = "Very High (Near identical)"
    elif score >= 0.7:
        interpretation = "High (Semantically similar)"
    elif score >= 0.5:
        interpretation = "Moderate (Related)"
    elif score >= 0.3:
        interpretation = "Low (Somewhat related)"
    else:
        interpretation = "Very Low (Unrelated)"

    return f"{score:.4f} - {interpretation}"
