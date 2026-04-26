import time
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

# Start the total execution timer
start_time = time.time()

# 1. Load the Qwen 2.5 Coder 7B model
print("Loading Qwen 2.5 Coder into memory...")
model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

# 2. Define a coding prompt
user_prompt = "Write a Python function that takes a list of numbers and returns only the prime numbers. Include brief comments."
messages = [{"role": "user", "content": user_prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

# 3. Setup the cache for memory management
# Increased to 8192 to allow for a longer continuous chat history
prompt_cache = make_prompt_cache(model, max_kv_size=8192)

# 4. Generate the response
# DEBUG: print("\nGenerating response with optimized memory...\n")
response = generate(
    model,
    tokenizer,
    prompt=formatted_prompt,
    max_tokens=2000,
    prompt_cache=prompt_cache,
    verbose=True # This triggers the built-in Token/Second metrics
)

# 5. Memory cleanup
mx.clear_cache()
# DEBUG: print("\n[GPU memory cache cleared]")

# Calculate and print total execution time
end_time = time.time()
print(f"\n[Total script execution time: {end_time - start_time:.2f} seconds]")