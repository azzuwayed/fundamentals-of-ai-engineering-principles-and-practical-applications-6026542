"""
LLM Manager Module for RAG Chat.

Provides multiple LLM backend support with unified interface:
- LocalLLM: CPU-friendly DistilGPT2 (educational, no API costs)
- OpenAILLM: Latest GPT-4o, GPT-4.1, and reasoning models (high quality, requires API key)

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
    generation_time: float = 0.0
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

            # Get encoding for model (use cl100k_base for GPT-4o, GPT-4.1, o-series)
            if "gpt-4o" in self.model_name or "gpt-4.1" in self.model_name or "o3" in self.model_name or "o4" in self.model_name:
                encoding = tiktoken.get_encoding("cl100k_base")
            elif "gpt-4" in self.model_name:
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
            # Legacy models (for backward compatibility)
            "gpt-3.5-turbo": 16385,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            # Current models (2025)
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4.1": 1000000,  # 1M context!
            "gpt-4.1-mini": 1000000,
            # Reasoning models
            "o3-pro": 200000,
            "o4-mini": 128000
        }

        return {
            "name": self.model_name,
            "backend": "openai",
            "context_window": context_windows.get(self.model_name, 128000),
            "api_configured": self._client is not None,
            "provider": "OpenAI"
        }


class OllamaLLM(LLM):
    """
    Ollama local LLM using official ollama Python library.

    Advantages:
    - High quality open-source models (Llama 3, Mistral, Phi, etc.)
    - Runs completely locally
    - No API costs
    - Privacy-friendly
    - Easy model switching
    - Streaming support available

    Limitations:
    - Requires Ollama installation
    - Requires model downloads (can be large)
    - GPU recommended for best performance

    Educational value: Shows production-quality local inference with official library
    """

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama LLM.

        Args:
            model_name: Ollama model name (e.g., "llama3.2:3b", "mistral")
            base_url: Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self._client = None
        self._check_ollama_available()

    def _check_ollama_available(self):
        """Check if Ollama is running using official library."""
        try:
            import ollama

            # Create client for custom base URL
            if self.base_url != "http://localhost:11434":
                self._client = ollama.Client(host=self.base_url)
            else:
                self._client = ollama.Client()

            # Test connection by listing models
            try:
                self._client.list()
            except ollama.ResponseError as e:
                warnings.warn(
                    f"Ollama server not responding properly at {self.base_url}. "
                    f"Error: {e.error}. Make sure Ollama is running: 'ollama serve'"
                )
            except Exception as e:
                warnings.warn(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Error: {str(e)}. Install Ollama from https://ollama.ai and run 'ollama serve'"
                )
        except ImportError:
            warnings.warn(
                "ollama library not installed. Run: pip install ollama"
            )
            self._client = None

    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """
        Generate response using Ollama official library.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            LLMResponse with generated text
        """
        import ollama
        import time

        if not self._client:
            return LLMResponse(
                text="[Ollama client not initialized. Install ollama: pip install ollama]",
                tokens_used=0,
                model_name=self.model_name,
                backend="ollama",
                metadata={"error": "client_not_initialized"}
            )

        try:
            start_time = time.time()

            # Call Ollama generate API with official library
            response = self._client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "num_predict": config.max_tokens,
                    "stop": config.stop_sequences or []
                }
            )

            generation_time = time.time() - start_time

            # Extract response data
            generated_text = response.get("response", "")

            # Get token counts (if provided by Ollama)
            prompt_tokens = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)
            tokens_used = prompt_tokens + completion_tokens

            # If token counts not provided, estimate
            if tokens_used == 0:
                tokens_used = self.count_tokens(prompt) + self.count_tokens(generated_text)

            return LLMResponse(
                text=generated_text,
                tokens_used=tokens_used,
                model_name=self.model_name,
                backend="ollama",
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "generation_time": generation_time,
                    "eval_duration_ms": response.get("eval_duration", 0) / 1_000_000 if response.get("eval_duration") else 0,
                    "total_duration_ms": response.get("total_duration", 0) / 1_000_000 if response.get("total_duration") else 0
                }
            )

        except ollama.ResponseError as e:
            # Better error handling with official library
            error_message = f"Ollama error: {e.error}"
            if "model" in str(e.error).lower():
                error_message += f". Try 'ollama pull {self.model_name}'"

            return LLMResponse(
                text=f"[{error_message}]",
                tokens_used=0,
                model_name=self.model_name,
                backend="ollama",
                metadata={
                    "error": e.error,
                    "status_code": e.status_code if hasattr(e, 'status_code') else None
                }
            )

        except Exception as e:
            return LLMResponse(
                text=f"[Ollama error: {str(e)}. Make sure Ollama is running: 'ollama serve']",
                tokens_used=0,
                model_name=self.model_name,
                backend="ollama",
                metadata={"error": str(e)}
            )

    def generate_stream(self, prompt: str, config: LLMConfig):
        """
        Generate response with streaming (yields chunks as they arrive).

        Args:
            prompt: Input prompt
            config: Generation configuration

        Yields:
            Text chunks as they are generated

        Example:
            ```python
            for chunk in llm.generate_stream("Tell me a story", config):
                print(chunk, end='', flush=True)
            ```
        """
        import ollama

        if not self._client:
            yield "[Ollama client not initialized. Install ollama: pip install ollama]"
            return

        try:
            # Call Ollama generate API with streaming enabled
            stream = self._client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True,
                options={
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "num_predict": config.max_tokens,
                    "stop": config.stop_sequences or []
                }
            )

            # Yield each chunk as it arrives
            for chunk in stream:
                if "response" in chunk:
                    yield chunk["response"]

        except ollama.ResponseError as e:
            error_message = f"Ollama error: {e.error}"
            if "model" in str(e.error).lower():
                error_message += f". Try 'ollama pull {self.model_name}'"
            yield f"[{error_message}]"

        except Exception as e:
            yield f"[Ollama error: {str(e)}. Make sure Ollama is running: 'ollama serve']"

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Ollama's tokenizer.

        Note: Ollama doesn't expose tokenization directly, so we use a rough estimate.
        """
        # Rough estimate: 1 token ≈ 4 characters for most models
        # This is acceptable for educational purposes and budget management
        return len(text) // 4

    def get_model_info(self) -> Dict:
        """Get Ollama model information using official library."""
        import ollama

        if not self._client:
            return {
                "name": self.model_name,
                "backend": "ollama",
                "context_window": 4096,
                "ollama_running": False,
                "note": "Install Ollama from https://ollama.ai and run 'ollama serve'"
            }

        # Try to get actual model info from Ollama
        try:
            models_response = self._client.list()
            # Use attribute access - response.models, not response.get("models")
            models = models_response.models if hasattr(models_response, 'models') else []

            for model in models:
                # Use attribute access - model.model, not model.get("name")
                model_name_full = getattr(model, 'model', '')
                # Match model name (handle both "llama3.2:3b" and "llama3.2")
                if model_name_full.startswith(self.model_name) or self.model_name in model_name_full:
                    # Use attribute access - model.size, not model.get("size")
                    size_bytes = getattr(model, 'size', 0)
                    size_gb = size_bytes / (1024**3)

                    # Estimate context window based on model family
                    context_window = 4096  # Default
                    if "llama3" in self.model_name.lower():
                        context_window = 8192
                    elif "mistral" in self.model_name.lower():
                        context_window = 8192
                    elif "phi" in self.model_name.lower():
                        context_window = 4096
                    elif "gemma" in self.model_name.lower():
                        context_window = 8192

                    # Use attribute access - model.details, not model.get("details")
                    details = getattr(model, 'details', None)

                    # details might also be a typed object, so use getattr
                    if details:
                        family = getattr(details, 'family', 'unknown')
                        format_type = getattr(details, 'format', 'unknown')
                        parameter_size = getattr(details, 'parameter_size', 'unknown')
                        quantization = getattr(details, 'quantization_level', 'unknown')
                    else:
                        family = 'unknown'
                        format_type = 'unknown'
                        parameter_size = 'unknown'
                        quantization = 'unknown'

                    return {
                        "name": self.model_name,
                        "backend": "ollama",
                        "context_window": context_window,
                        "size_gb": f"{size_gb:.1f}",
                        "family": family,
                        "format": format_type,
                        "parameter_size": parameter_size,
                        "quantization": quantization,
                        "ollama_running": True
                    }

        except ollama.ResponseError as e:
            return {
                "name": self.model_name,
                "backend": "ollama",
                "context_window": 4096,
                "ollama_running": False,
                "error": e.error
            }
        except Exception:
            pass

        # Fallback info if model not found but Ollama is running
        return {
            "name": self.model_name,
            "backend": "ollama",
            "context_window": 4096,
            "ollama_running": True,
            "note": f"Model not pulled. Run: ollama pull {self.model_name}"
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
            backend: "local", "ollama", or "openai"
            model_name: Model name (optional, uses defaults)
            **kwargs: Additional backend-specific arguments

        Returns:
            LLM instance
        """
        if backend == "local":
            model_name = model_name or "distilgpt2"
            return LocalLLM(model_name=model_name, **kwargs)

        elif backend == "ollama":
            model_name = model_name or "llama3.2:3b"
            return OllamaLLM(model_name=model_name, **kwargs)

        elif backend == "openai":
            model_name = model_name or "gpt-3.5-turbo"
            return OpenAILLM(model_name=model_name, **kwargs)

        else:
            raise ValueError(
                f"Unknown backend: {backend}. Choose 'local', 'ollama', or 'openai'."
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
                "id": "ollama",
                "name": "Ollama (Llama 3/Mistral/Phi)",
                "description": "Production-quality local models, requires Ollama installation",
                "requires_api_key": False,
                "quality": "Excellent",
                "speed": "Medium"
            },
            {
                "id": "openai",
                "name": "OpenAI (GPT-4o/GPT-4.1/o3-pro)",
                "description": "Latest models with huge context (up to 1M tokens), requires API key",
                "requires_api_key": True,
                "quality": "Excellent",
                "speed": "Fast"
            }
        ]

    @staticmethod
    def get_available_models(backend: str) -> List[str]:
        """
        Get available models for backend.

        For Ollama, returns only installed models from local system.
        For other backends, returns recommended models.

        Args:
            backend: Backend name

        Returns:
            List of model names
        """
        if backend == "local":
            # Return local models with size info
            return [
                "distilgpt2 (82M params, 0.3 GB)",
                "gpt2 (124M params, 0.5 GB)",
                "gpt2-medium (355M params, 1.4 GB)"
            ]

        elif backend == "ollama":
            # Query actual installed models from Ollama
            try:
                import ollama

                client = ollama.Client()
                response = client.list()

                # Use attribute access - response.models (not dict key)
                models = response.models if hasattr(response, 'models') else []

                # Sort by size (smaller models first)
                models_sorted = sorted(models, key=lambda m: getattr(m, 'size', 0))

                # Format model names with sizes for dropdown display
                model_names_sorted = [
                    f"{getattr(m, 'model', '')} ({getattr(m, 'size', 0) / (1024**3):.1f} GB)"
                    for m in models_sorted
                ]

                if model_names_sorted:
                    return model_names_sorted
                else:
                    # No models installed
                    return ["No models installed - run 'ollama pull llama3.2:3b'"]

            except Exception:
                # Ollama not available, return helpful message
                return ["Ollama not available - install from https://ollama.ai"]

        elif backend == "openai":
            return [
                # Recommended models (2025)
                "gpt-4o",              # Best overall - 128K context
                "gpt-4o-mini",         # Most cost-effective - 128K context
                "gpt-4.1",             # Newest with 1M context!
                "gpt-4.1-mini",        # Efficient with 1M context
                # Reasoning models
                "o3-pro",              # Advanced reasoning - 200K context
                "o4-mini",             # Fast reasoning - 128K context
                # Legacy (backward compatibility)
                "gpt-3.5-turbo",       # Legacy - being phased out
                "gpt-4",               # Legacy GPT-4
                "gpt-4-turbo"          # Legacy Turbo
            ]

        else:
            return []
