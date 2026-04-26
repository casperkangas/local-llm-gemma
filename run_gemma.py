from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

# 1. Load the model and tokenizer
print("Loading model into memory...")
model, tokenizer = load("mlx-community/gemma-2-2b-it-4bit")

# 2. Define and format your prompt
user_prompt = "Write a short haiku about running an AI locally on a Mac."
messages = [{"role": "user", "content": user_prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

# 3. Create a memory-efficient prompt cache
prompt_cache = make_prompt_cache(
    model,
    max_kv_size=4096  
)

# 4. Generate the response (Removed KV quantization due to Gemma architecture)
print("\nGenerating response with optimized memory...\n")
response = generate(
    model,
    tokenizer,
    prompt=formatted_prompt,
    max_tokens=100,
    prompt_cache=prompt_cache,
    verbose=True
)

print("\n--- AI Response ---")
print(response)

# 5. Clean up
mx.metal.clear_cache()
print("\n[GPU memory cache cleared]")