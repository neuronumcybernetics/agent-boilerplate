# !pip install llama-cpp-python

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Local model configuration (default)
REPO_ID = os.getenv("REPO_ID", "Qwen/Qwen2.5-3B-Instruct-GGUF")
FILENAME = os.getenv("FILENAME", "qwen2.5-3b-instruct-q4_k_m.gguf")

# Cloud API configuration (optional - works with any OpenAI-compatible API)
# Examples: OpenAI, OpenRouter, Together AI, Groq, Ollama with openai compatibility, etc.
# If API_KEY is set, cloud API will be used. Otherwise, local model is used.
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")  # Default to OpenAI
MODEL = os.getenv("MODEL", "gpt-4o-mini")  # Default model

_model = None

class CloudAPIWrapper:
    """Wrapper to provide llama-cpp-python compatible interface for OpenAI-compatible APIs."""

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def create_chat_completion(self, messages, **kwargs):
        """
        Create a chat completion compatible with llama-cpp-python interface.
        Returns a dict with 'choices' and 'usage' keys.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )

        # Convert OpenAI response to llama-cpp-python compatible format
        return {
            "choices": [
                {
                    "message": {
                        "role": response.choices[0].message.role,
                        "content": response.choices[0].message.content
                    }
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }


def get_model():
    global _model

    if _model is None:
        # Use cloud API if API_KEY is set, otherwise use local model
        if API_KEY:
            print(f"Initializing cloud API client with base URL {BASE_URL} and model {MODEL}...")
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=API_KEY,
                    base_url=BASE_URL,
                )
                _model = CloudAPIWrapper(client, MODEL)
                print("Cloud API client initialized.")
            except Exception as e:
                print(f"Failed to initialize cloud API: {e}")
                print("Falling back to local model...")
                _model = _load_local_model()
        else:
            # Use local llama-cpp-python model (default)
            _model = _load_local_model()

    return _model


def _load_local_model():
    """Load local llama-cpp-python model."""
    try:
        from llama_cpp import Llama

        print(f"Loading local model {REPO_ID}/{FILENAME}...")
        llm = Llama.from_pretrained(
            repo_id=REPO_ID,
            filename=FILENAME,
            n_gpu_layers=-1,  # Use GPU if available
            n_ctx=2048,
        )
        print("Local model loaded successfully.")
        return llm

    except ImportError:
        print("ERROR: llama-cpp-python not installed!")
        print("Install it with: pip install llama-cpp-python")
        raise

    except Exception as e:
        print(f"Failed to load local model: {e}")

        # Try to fall back to cloud API if available
        if API_KEY:
            print("Attempting to fall back to cloud API...")
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=API_KEY,
                    base_url=BASE_URL,
                )
                wrapper = CloudAPIWrapper(client, MODEL)
                print("Fallback to cloud API successful.")
                return wrapper
            except Exception as fallback_error:
                print(f"Fallback to cloud API also failed: {fallback_error}")

        raise RuntimeError(
            "Failed to load local model and no fallback available. "
            "Either fix the local model installation or set API_KEY and optionally BASE_URL."
        )


if __name__ == "__main__":
    model = get_model()
    print("Model ready.")

    # Test the model
    try:
        result = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, I am working!' in one sentence."}
            ]
        )
        print("\nTest output:", result["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Test failed: {e}")
