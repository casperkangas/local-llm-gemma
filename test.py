from mlx_lm import load, stream_generate
import mlx.core as mx

# ==========================================
# CONFIGURATION
# ==========================================

# MODEL_REPO = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"    # 8B
# MODEL_REPO = "mlx-community/gemma-2-9b-it-4bit" # 9 B         # 9B
# MODEL_REPO = "mlx-community/Mistral-Nemo-Instruct-2407-4bit"  # 12B
MODEL_REPO = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"    # 14B

# 1. Load the model
print(f"Loading {MODEL_REPO} into memory...")

# model, tokenizer = load(MODEL_REPO)
model, tokenizer = load(
    MODEL_REPO, 
    tokenizer_config={"fix_mistral_regex": True}
)

# 2. Initialize the chat history
# We start with a completely empty array for maximum model compatibility.
messages = []

print("\n--- Chatbot initialized! Type 'quit' or 'exit' to stop. ---\n")

# 3. Start the interactive loop
while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['quit', 'exit']:
        print("Ending chat...")
        break
        
    messages.append({"role": "user", "content": user_input})
    
    # Format the history using the model's native chat template
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    print("\nAI: ", end="", flush=True)
    
    # 4. Stream the response directly to the terminal
    full_response = ""
    
    for response in stream_generate(
        model, 
        tokenizer, 
        formatted_prompt, 
        max_tokens=2000
    ):
        print(response.text, end="", flush=True)
        full_response += response.text
        
    print("\n") 
    
    # Append the AI's final answer to the history
    messages.append({"role": "assistant", "content": full_response})
    
    # 5. Clean up memory after every single turn
    mx.clear_cache()

print("\n[GPU memory cache cleared]")