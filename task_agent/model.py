# !pip install llama-cpp-python

from llama_cpp import Llama

REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"

_llm = None

def get_model():
    global _llm
    if _llm is None:
        print(f"Loading model {REPO_ID}...")
        _llm = Llama.from_pretrained(
            repo_id=REPO_ID,
            filename=FILENAME,
            n_gpu_layers=-1,
            n_ctx=2048,
        )
        print("Model loaded.")
    return _llm

if __name__ == "__main__":
    get_model()
    print("Model downloaded and ready.")
