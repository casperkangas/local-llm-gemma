import streamlit as st
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx
import gc

# 1. Page Configuration
st.set_page_config(page_title="Local Mac LLM", page_icon="🤖")

# 2. Define Your Model Library (The Menu)
AVAILABLE_MODELS = {
    "Coder Qwen (7B)": {
        "repo_id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "system_prompt": "Your name is Goofy. You are a highly advanced AI coding assistant. You only provide clean, well-documented code."
    },
    "Scholar Qwen (7B)": {
        "repo_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "system_prompt": "Your name is Casper. You are a highly intelligent AI tutor. The user you are talking to is named Julianna. Always be encouraging, helpful, and clear when helping her with her schoolwork, but don't give false information and make sure everything is correct. If you don't know the answer to something, say you don't know but offer to help her find the answer."
    }
}

# 3. App Routing (Determine which screen to show)
if "app_stage" not in st.session_state:
    st.session_state.app_stage = "setup"

# ==========================================
# STAGE 1: THE SETUP SCREEN
# ==========================================
if st.session_state.app_stage == "setup":
    st.title("🤖 Welcome to Local AI")
    st.write("Please select an AI assistant to launch for this session.")
    
    selected_model_name = st.selectbox("Available Models:", list(AVAILABLE_MODELS.keys()))
    
    if st.button("Launch Assistant", type="primary"):
        # Save the selection to memory and move to the chat stage
        st.session_state.selected_model_name = selected_model_name
        st.session_state.config = AVAILABLE_MODELS[selected_model_name]
        st.session_state.app_stage = "chat"
        st.rerun() # Forces the app to refresh and load Stage 2

# ==========================================
# STAGE 2: THE CHAT INTERFACE
# ==========================================
elif st.session_state.app_stage == "chat":
    
    selected_model_name = st.session_state.selected_model_name
    config = st.session_state.config
    
    # --- NEW: Set a safe token limit for your Mac (e.g., 8192) ---
    MAX_CONTEXT_TOKENS = 8192
    
    # 1. Model Loading Engine
    @st.cache_resource(show_spinner="Loading model into Mac RAM... This takes a few seconds.")
    def get_model(repo_id):
        return load(repo_id)

    model, tokenizer = get_model(config["repo_id"])
    
    # 3. Initialize Chat Memory & Context Counter
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": config["system_prompt"]}]
        st.session_state.total_tokens = len(tokenizer.encode(config["system_prompt"]))

    # --- NEW: Auto-Clear Safeguard ---
    if st.session_state.total_tokens >= MAX_CONTEXT_TOKENS:
        st.warning(f"⚠️ Memory limit reached ({MAX_CONTEXT_TOKENS} tokens). Clearing chat to prevent crashing.")
        st.session_state.messages = [{"role": "system", "content": config["system_prompt"]}]
        st.session_state.total_tokens = len(tokenizer.encode(config["system_prompt"]))

    # 2. Safe Sidebar Controls & Memory Monitor
    st.sidebar.title("⚙️ Session Info")
    st.sidebar.success(f"**Active AI:**\n{selected_model_name}")
    
    # --- NEW: Visual Memory Monitor ---
    st.sidebar.markdown("### 📊 Memory Monitor")
    usage_percent = min(st.session_state.total_tokens / MAX_CONTEXT_TOKENS, 1.0)
    st.sidebar.progress(usage_percent, text=f"Context: {st.session_state.total_tokens} / {MAX_CONTEXT_TOKENS} tokens")
    
    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state.messages = [{"role": "system", "content": config["system_prompt"]}]
        st.session_state.total_tokens = len(tokenizer.encode(config["system_prompt"]))
        st.rerun()
    
    if st.sidebar.button("🛑 End Session & Choose New AI"):
        st.cache_resource.clear()
        gc.collect()
        mx.clear_cache()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.title(f"💻 Chatting with {selected_model_name.split(' ')[0]}")

    # 4. Display the Conversation History
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # --- NEW: Show token count under assistant messages ---
                if msg["role"] == "assistant" and "token_count" in msg:
                    st.caption(f"⚡ {msg['token_count']} tokens generated")

    # 5. The Chat Input Box & Generation
    if user_input := st.chat_input("Type your message..."):
        
        # Calculate user input tokens
        user_tokens = len(tokenizer.encode(user_input))
        st.session_state.total_tokens += user_tokens
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            
            formatted_prompt = tokenizer.apply_chat_template(
                st.session_state.messages, 
                add_generation_prompt=True
            )
            
            def stream_parser():
                for response in stream_generate(
                    model, 
                    tokenizer, 
                    formatted_prompt, 
                    max_tokens=2000
                ):
                    yield response.text.replace("<|im_end|>", "")
            
            full_response = st.write_stream(stream_parser())
            
            # --- NEW: Calculate generated tokens and save them ---
            generated_tokens = len(tokenizer.encode(full_response))
            st.session_state.total_tokens += generated_tokens
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "token_count": generated_tokens # Save this so we can display it later
            })
            
            mx.clear_cache()
            st.rerun() # Refresh to update the sidebar progress bar