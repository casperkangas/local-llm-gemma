from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

# 1. Load the model
print("Loading Qwen 2.5 Coder into memory...")
model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

# 2. Setup the cache for memory management
# Increased to 8192 to allow for a longer continuous chat history
prompt_cache = make_prompt_cache(model, max_kv_size=8192)

# 3. Initialize the chat history
# We start with a "system" prompt to tell the AI how to behave
messages = [
    {"role": "system", "content": "You are an expert coding assistant. Don't waste tokens on pleasantries. Provide concise, accurate code answers to the user's questions."}
]

print("\n--- Chatbot initialized! Type 'quit' or 'exit' to stop. ---\n")

# 4. Start the interactive loop
while True:
    # Get user input from the terminal
    user_input = input("You: ")
    
    # Check if the user wants to close the program
    if user_input.lower() in ['quit', 'exit']:
        print("Ending chat...")
        break
        
    # Append the user's message to the conversation history
    messages.append({"role": "user", "content": user_input})
    
    # Format the entire history so the AI remembers previous questions
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    print("\nAI: ", end="", flush=True)
    
    # 5. Stream the response
    full_response = ""
    for response in stream_generate(
        model, 
        tokenizer, 
        formatted_prompt, 
        max_tokens=2000,
        prompt_cache=prompt_cache
    ):
        # Intercept and remove the stop token before printing
        clean_text = response.text.replace("<|im_end|>", "")
        
        print(clean_text, end="", flush=True)
        full_response += clean_text
        
    print("\n") # Add a blank line when the AI finishes speaking
    
    # Append the AI's final answer to the history so it has context for the next loop
    messages.append({"role": "assistant", "content": full_response})

# 6. Clean up when the loop breaks
mx.clear_cache()
print("\n[GPU memory cache cleared]")