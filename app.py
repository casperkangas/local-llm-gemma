import streamlit as st
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

# 1. Page Configuration
# This sets the browser tab title and icon
st.set_page_config(page_title="Local Mac LLM", page_icon="🤖")
st.title("💻 Local Qwen Coder")

# 2. Cache the Model Loading
# @st.cache_resource is CRITICAL. It tells Streamlit to run this function ONCE 
# when the app starts, and keep the result (the 4.5GB model) in RAM. 
# Without this, it would try to reload the model every time you send a message.
@st.cache_resource
def get_model():
    print("Loading model into unified memory...")
    return load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

model, tokenizer = get_model()

# 3. Initialize Session State (The App's Memory)
# We use st.session_state to remember variables across script reruns.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are an expert coding assistant. Don't waste tokens on pleasantries. Provide concise, accurate code answers to the user's questions."}]

if "prompt_cache" not in st.session_state:
    st.session_state.prompt_cache = make_prompt_cache(model, max_kv_size=8192)

# 4. Display the Conversation History
# Loop through the memory and draw the chat bubbles on the screen
for msg in st.session_state.messages:
    if msg["role"] != "system": # We hide the system prompt from the UI
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 5. The Chat Input Box
# This creates the text box at the bottom. If the user types something, the code block runs.
if user_input := st.chat_input("Ask Qwen a coding question..."):
    
    # Add user message to memory and draw it on screen
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 6. Generate and Stream the AI Response
    with st.chat_message("assistant"):
        formatted_prompt = tokenizer.apply_chat_template(st.session_state.messages, add_generation_prompt=True)
        
        # We wrap our MLX stream in a generator function so Streamlit can type it out live
        def stream_parser():
            for response in stream_generate(
                model, 
                tokenizer, 
                formatted_prompt, 
                max_tokens=2000, 
                prompt_cache=st.session_state.prompt_cache
            ):
                # Apply our previous fix for the ChatML stop token
                yield response.text.replace("<|im_end|>", "")
        
        # st.write_stream automatically handles the typewriter effect and returns the full text
        full_response = st.write_stream(stream_parser())
        
        # Save the final text to memory
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Free up GPU math cache
        mx.clear_cache()