source .venv/bin/activate

# Finding Compatible Models on Hugging Face

The Filter Rules:

1. The Hub: Always start your search inside the mlx-community organization page: huggingface.co/mlx-community. If it is here, it is guaranteed to work on your Mac.
2. Search Terms: Use the search bar on that page and type 4bit and instruct (or it).
3. Avoid "Base" Models: If a model doesn't have "Instruct", "Chat", or "IT" in the name, do not use it for your chatbot. Base models are only trained to autocomplete text, not to answer questions.

# Hardware Limits & Model Selection (M4 Mac with 16GB RAM)

Apple Silicon uses Unified Memory, meaning the CPU and GPU share the same 16GB pool. macOS reserves ~4GB for the operating system, leaving **~11-12GB of usable RAM** for local LLMs and their context cache.

### Model Size Guide (Based on 4-bit Quantization)

A good rule of thumb: 1 Billion parameters at 4-bit compression requires roughly 0.7 GB of RAM.

- **1B - 3B Parameters (e.g., Gemma 2B, Qwen 1.5B):** \* _RAM Usage:_ 1.5GB - 2.5GB.
  - _Verdict:_ Extremely fast, lightweight. Best for background tasks or when running heavy applications (VSCode, Docker) simultaneously.
- **7B - 9B Parameters (e.g., Llama 3 8B, Qwen 2.5 7B):** \* _RAM Usage:_ 4.5GB - 6GB.
  - _Verdict:_ **The Sweet Spot.** Excellent reasoning and coding capabilities. Pushes memory pressure to "Yellow" but leaves enough room for the OS and the AI's short-term memory (KV cache) to function smoothly.
- **12B - 14B Parameters (e.g., Mistral Nemo 12B, Qwen 14B):** \* _RAM Usage:_ 8GB - 9.5GB.
  - _Verdict:_ **The Absolute Limit.** Will work, but heavily restricts multitasking. Long conversations may cause the KV cache to exceed available RAM, leading to severe slowdowns or crashes.

### Quantization Rules

- **4-bit (Recommended):** The gold standard for local Mac usage. Massive reduction in RAM requirements with almost zero noticeable loss in intelligence.
- **8-bit:** Only recommended for models under 3B parameters. On larger models, it will exceed the 16GB RAM limit.
- **fp16 / bf16 (Uncompressed):** Do not run uncompressed models unless they are extremely small (under 1.5B parameters).
