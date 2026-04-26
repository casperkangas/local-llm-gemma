import streamlit as st
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

# 1. Page Configuration
st.set_page_config(page_title="Local Mac LLM", page_icon="🤖")

AVAILABLE_MODELS = {
    "Qwen 2.5 Coder (7B)": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "Gemma 2 (2B)": "mlx-community/gemma-2-2b-it-4bit"
}

st.sidebar.title("⚙️ Model Settings")
selected_model_name = st.sidebar.selectbox("Choose an AI:", list(AVAILABLE_MODELS.keys()))
selected_repo_id = AVAILABLE_MODELS[selected_model_name]

st.title(f"💻 Chatting with {selected_model_name.split(' ')[0]}")

# 2. Dynamic Model Loading
@st.cache_resource(show_spinner="Loading model into Mac RAM...")
def get_model(repo_id):
    print(f"\n[System] Loading {repo_id}...")
    return load(repo_id)

# 3. Safe Context Switching & Memory Management
if "current_model_id" not in st.session_state or st.session_state.current_model_id != selected_repo_id:
    # FIX 1: Destroy Streamlit's cache IMMEDIATELY to drop the old model from RAM
    st.cache_resource.clear()
    
    st.session_state.current_model_id = selected_repo_id
    
    # FIX 2: Handle strict template rules. Gemma crashes if given a "system" role.
    if "Gemma" in selected_model_name:
        st.session_state.messages = [] # Gemma must start with a user message
    else:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful local AI assistant running on a Mac."}
        ]
    
    # CRITICAL: We load the model AFTER clearing the cache, otherwise we instantly delete what we just loaded
    model, tokenizer = get_model(selected_repo_id)
    st.session_state.prompt_cache = make_prompt_cache(model, max_kv_size=8192)
    
    mx.clear_cache()
    print("[System] Switched models. Old memory wiped completely.")
else:
    # If the dropdown hasn't changed, just grab the active model normally
    model, tokenizer = get_model(selected_repo_id)

# 4. Display the Conversation History
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 5. The Chat Input Box & Generation
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