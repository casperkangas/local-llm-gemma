import streamlit as st
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

# 1. Page Configuration
st.set_page_config(page_title="Local Mac LLM", page_icon="🤖")

# 2. Define Your Model Library (The Menu)
AVAILABLE_MODELS = {
    "Qwen 2.5 Coder (7B)": {
        "repo_id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "system_prompt": "You are Qwen, a highly advanced local AI coding assistant. You only provide clean, well-documented code."
    },
    "Gemma 2 (2B)": {
        "repo_id": "mlx-community/gemma-2-2b-it-4bit",
        "system_prompt": None  # Gemma does not support system roles
    }
}

# 3. The Sidebar UI
st.sidebar.title("⚙️ Model Settings")
selected_model_name = st.sidebar.selectbox("Choose an AI:", list(AVAILABLE_MODELS.keys()))

# --- THE FIX IS HERE ---
# We grab the dictionary for the selected model, then extract the specific pieces
selected_config = AVAILABLE_MODELS[selected_model_name]
selected_repo_id = selected_config["repo_id"]
selected_system_prompt = selected_config["system_prompt"]
# -----------------------

st.title(f"💻 Chatting with {selected_model_name.split(' ')[0]}")

# 4. Dynamic Model Loading
@st.cache_resource(show_spinner="Loading model into Mac RAM...")
def get_model(repo_id):
    print(f"\n[System] Loading {repo_id}...")
    return load(repo_id)

# 5. Safe Context Switching & Memory Management
if "current_model_id" not in st.session_state or st.session_state.current_model_id != selected_repo_id:
    st.cache_resource.clear()
    st.session_state.current_model_id = selected_repo_id
    
    # --- SECOND PART OF THE FIX ---
    # We use the custom system prompt we pulled from the dictionary above!
    if selected_system_prompt is None:
        st.session_state.messages = [] # Gemma
    else:
        st.session_state.messages = [
            {"role": "system", "content": selected_system_prompt}
        ]
    # ------------------------------
    
    model, tokenizer = get_model(selected_repo_id)
    st.session_state.prompt_cache = make_prompt_cache(model, max_kv_size=8192)
    
    mx.clear_cache()
    print("[System] Switched models. Old memory wiped completely.")
else:
    model, tokenizer = get_model(selected_repo_id)

# 6. Display the Conversation History
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 7. The Chat Input Box & Generation
if user_input := st.chat_input(f"Message {selected_model_name}..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        formatted_prompt = tokenizer.apply_chat_template(st.session_state.messages, add_generation_prompt=True)
        
        def stream_parser():
            for response in stream_generate(
                model, 
                tokenizer, 
                formatted_prompt, 
                max_tokens=2000, 
                prompt_cache=st.session_state.prompt_cache
            ):
                yield response.text.replace("<|im_end|>", "")
        
        full_response = st.write_stream(stream_parser())
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        mx.clear_cache()