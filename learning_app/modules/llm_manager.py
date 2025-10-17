"""
LLM Manager Module for RAG Chat.

Provides multiple LLM backend support with unified interface:
- LocalLLM: CPU-friendly DistilGPT2 (educational, no API costs)
- OpenAILLM: GPT-3.5/GPT-4 (high quality, requires API key)

Includes token counting and budget management for educational purposes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
import warnings


@dataclass
class LLMConfig:
    """Configuration for LLM generation."""

    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None


@dataclass
class LLMResponse:
    """Response from LLM generation."""

    text: str
    tokens_used: int
    model_name: str
    backend: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLM(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """
        Generate response from prompt.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            LLMResponse with generated text and metadata
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """
        Get model information.

        Returns:
            Dictionary with model details
        """
        pass


class LocalLLM(LLM):
    """
    Local LLM using DistilGPT2.

    Advantages:
    - No API costs
    - Works offline
    - Fast on CPU
    - Privacy-friendly

    Limitations:
    - Lower quality than modern LLMs
    - Shorter context window
    - Less instruction-following ability

    Educational value: Shows how local models work
    """

    def __init__(self, model_name: str = "distilgpt2", device: str = "cpu"):
        """
        Initialize local LLM.

        Args:
            model_name: Hugging Face model name
            device: Device for inference (cpu/cuda)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer lazily."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load with low memory footprint
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left'
            )

            # Set pad token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True
            )

            self._model.to(self.device)
            self._model.eval()

        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")

    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """
        Generate response using local model.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            LLMResponse with generated text
        """
        import torch

        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                inputs.input_ids,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask
            )

        # Decode
        full_response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only new tokens (remove prompt)
        generated_text = full_response[len(prompt):].strip()

        # Count tokens
        tokens_used = len(outputs[0])

        return LLMResponse(
            text=generated_text,
            tokens_used=tokens_used,
            model_name=self.model_name,
            backend="local",
            metadata={
                "device": self.device,
                "prompt_tokens": len(inputs.input_ids[0]),
                "generation_tokens": tokens_used - len(inputs.input_ids[0])
            }
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens using local tokenizer."""
        tokens = self._tokenizer.encode(text)
        return len(tokens)

    def get_model_info(self) -> Dict:
        """Get local model information."""
        return {
            "name": self.model_name,
            "backend": "local",
            "device": self.device,
            "context_window": 1024,  # DistilGPT2 context
            "parameters": "82M",
            "architecture": "GPT-2 distilled"
        }


class OpenAILLM(LLM):
    """
    OpenAI API LLM.

    Advantages:
    - High quality responses
    - Large context window
    - Strong instruction following
    - Latest models

    Limitations:
    - Requires API key
    - Costs money per token
    - Requires internet

    Educational value: Shows production RAG patterns
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI LLM.

        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            warnings.warn(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "OpenAI library not installed. Run: pip install openai"
                )

    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """
        Generate response using OpenAI API.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            LLMResponse with generated text
        """
        if not self._client:
            return LLMResponse(
                text="[OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.]",
                tokens_used=0,
                model_name=self.model_name,
                backend="openai",
                metadata={"error": "api_key_missing"}
            )

        try:
            # Call API
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop_sequences
            )

            # Extract response
            generated_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            return LLMResponse(
                text=generated_text,
                tokens_used=tokens_used,
                model_name=self.model_name,
                backend="openai",
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": response.choices[0].finish_reason
                }
            )

        except Exception as e:
            return LLMResponse(
                text=f"[OpenAI API error: {str(e)}]",
                tokens_used=0,
                model_name=self.model_name,
                backend="openai",
                metadata={"error": str(e)}
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            import tiktoken

            # Get encoding for model
            if "gpt-4" in self.model_name:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.model_name:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))

        except Exception:
            # Fallback: rough estimate (1 token ≈ 4 chars)
            return len(text) // 4

    def get_model_info(self) -> Dict:
        """Get OpenAI model information."""
        context_windows = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000
        }

        return {
            "name": self.model_name,
            "backend": "openai",
            "context_window": context_windows.get(self.model_name, 4096),
            "api_configured": self._client is not None,
            "provider": "OpenAI"
        }


class LLMManager:
    """
    Factory for creating LLM backends.

    Provides unified interface for switching between local and API models.
    """

    @staticmethod
    def create_llm(
        backend: str = "local",
        model_name: Optional[str] = None,
        **kwargs
    ) -> LLM:
        """
        Create LLM backend.

        Args:
            backend: "local" or "openai"
            model_name: Model name (optional, uses defaults)
            **kwargs: Additional backend-specific arguments

        Returns:
            LLM instance
        """
        if backend == "local":
            model_name = model_name or "distilgpt2"
            return LocalLLM(model_name=model_name, **kwargs)

        elif backend == "openai":
            model_name = model_name or "gpt-3.5-turbo"
            return OpenAILLM(model_name=model_name, **kwargs)

        else:
            raise ValueError(
                f"Unknown backend: {backend}. Choose 'local' or 'openai'."
            )

    @staticmethod
    def get_available_backends() -> List[Dict]:
        """
        Get list of available backends.

        Returns:
            List of backend information dictionaries
        """
        return [
            {
                "id": "local",
                "name": "Local (DistilGPT2)",
                "description": "CPU-friendly, no API costs, works offline",
                "requires_api_key": False,
                "quality": "Basic",
                "speed": "Fast"
            },
            {
                "id": "openai",
                "name": "OpenAI (GPT-3.5/GPT-4)",
                "description": "High quality, requires API key and credits",
                "requires_api_key": True,
                "quality": "Excellent",
                "speed": "Medium"
            }
        ]

    @staticmethod
    def get_available_models(backend: str) -> List[str]:
        """
        Get available models for backend.

        Args:
            backend: Backend name

        Returns:
            List of model names
        """
        if backend == "local":
            return ["distilgpt2", "gpt2", "gpt2-medium"]

        elif backend == "openai":
            return [
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o"
            ]

        else:
            return []
