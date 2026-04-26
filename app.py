import streamlit as st
from mlx_lm import load, stream_generate
import mlx.core as mx
import gc
import os
import signal

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
    },
    "WIP WIP WIP Coder Qwen (14B)": {
        "repo_id": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
        "system_prompt": "Your name is Goofy. You are a highly advanced AI coding assistant. You only provide clean, well-documented code."
    },
    "Gemma (9B)": {
        "repo_id": "mlx-community/gemma-2-9b-it-4bit",
        "system_prompt": None
    },
    "Mistral (12B)": {
        "repo_id": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        "system_prompt": "Your name is Goofy. You are a highly advanced AI coding assistant. You only provide clean, well-documented code."
    },
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
        st.session_state.selected_model_name = selected_model_name
        st.session_state.config = AVAILABLE_MODELS[selected_model_name]
        
        # 1. Wipe old memory and set up the new system prompt
        if st.session_state.config["system_prompt"] is not None:
            st.session_state.messages = [{"role": "system", "content": st.session_state.config["system_prompt"]}]
        else:
            st.session_state.messages = []
            
        # 2. Reset the token counter to 0
        st.session_state.total_tokens = 0
        # -------------------------------
        
        st.session_state.app_stage = "chat"
        st.rerun()

# ==========================================
# STAGE 2: THE CHAT INTERFACE
# ==========================================
elif st.session_state.app_stage == "chat":
    
    selected_model_name = st.session_state.selected_model_name
    config = st.session_state.config
    
    # --- Two-Tier Memory Limits ---
    SOFT_LIMIT = 8192
    HARD_LIMIT = 16384
    
    # 1. Model Loading Engine
    @st.cache_resource(show_spinner="Loading model into Mac RAM... This takes a few seconds.")
    def get_model(repo_id, is_mistral):
        # If it is Mistral, load it with the regex fix
        if is_mistral:
            return load(
                repo_id, 
                tokenizer_config={"fix_mistral_regex": True}
            )
        # If it is any other model (Llama, Qwen, Gemma), load it normally
        else:
            return load(repo_id)

    # We check the name of the selected model to see if "Mistral" is in it
    check_if_mistral = "Mistral" in selected_model_name
    
    # We pass both the repo link AND our True/False check into the function
    model, tokenizer = get_model(config["repo_id"], check_if_mistral)
    
    # 3. Calculate System Tokens (Safely handled after tokenizer is loaded)
    if st.session_state.total_tokens == 0 and config["system_prompt"] is not None:
        st.session_state.total_tokens = len(tokenizer.encode(config["system_prompt"]))

    # --- Two-Tier Auto-Clear Safeguard ---
    if st.session_state.total_tokens >= HARD_LIMIT:
        st.error(f"🚨 Critical memory limit reached ({HARD_LIMIT} tokens). Chat auto-cleared to prevent Mac crash.")
        st.session_state.messages = [{"role": "system", "content": config["system_prompt"]}] if config["system_prompt"] is not None else []
        st.session_state.total_tokens = len(tokenizer.encode(config["system_prompt"])) if config["system_prompt"] is not None else 0
    elif st.session_state.total_tokens >= SOFT_LIMIT:
        st.warning(f"⚠️ Memory warning ({st.session_state.total_tokens} / {HARD_LIMIT} tokens). The chat is getting long. Consider clearing history soon to maintain performance!")
        
    # 3. Safe Sidebar Controls & Memory Monitor
    st.sidebar.title("⚙️ Session Info")
    st.sidebar.success(f"**Active AI:**\n{selected_model_name}")
    
    # --- Visual Memory Monitor ---
    st.sidebar.markdown("### 📊 Memory Monitor")
    
    # Calculate percentage based on the Hard Limit (max 1.0 to avoid UI errors)
    usage_percent = min(st.session_state.total_tokens / HARD_LIMIT, 1.0)
    
    # Dynamic status indicator
    if st.session_state.total_tokens >= HARD_LIMIT:
        status_text = "🔴 **Critical:** Limit Reached"
    elif st.session_state.total_tokens >= SOFT_LIMIT:
        status_text = "🟠 **Warning:** Soft Limit Exceeded"
    else:
        status_text = "🟢 **Status:** Healthy"
        
    st.sidebar.write(status_text)
    st.sidebar.progress(usage_percent, text=f"Context: {st.session_state.total_tokens} / {HARD_LIMIT} tokens")
    
    # --- Clear Chat Button ---
    if st.sidebar.button("🗑️ Clear Chat History", type="secondary"):
        if config["system_prompt"] is not None:
            st.session_state.messages = [{"role": "system", "content": config["system_prompt"]}]
            st.session_state.total_tokens = len(tokenizer.encode(config["system_prompt"]))
        else:
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            
        st.rerun()
    
    if st.sidebar.button("🛑 End Session & Choose New AI"):
        st.cache_resource.clear()
        gc.collect()
        mx.clear_cache()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
        
    st.sidebar.markdown("---") # Adds a clean visual divider line
    
    # --- Master Kill Switch ---
    if st.sidebar.button("🔌 Power Off Server (Close App)", type="primary"):
        st.cache_resource.clear()
        gc.collect()
        mx.clear_cache()
        
        # 2. Find this specific Streamlit process and kill it
        current_process_id = os.getpid()
        os.kill(current_process_id, signal.SIGTERM)

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
                try:
                    for response in stream_generate(
                        model, 
                        tokenizer, 
                        formatted_prompt, 
                        max_tokens=5000
                    ):
                        yield response.text.replace("<|im_end|>", "").replace("<end_of_turn>", "")
                except Exception as e:
                    st.error(f"Failed to generate response: {e}")
                    st.stop()
            
            full_response = st.write_stream(stream_parser())
            
            generated_tokens = len(tokenizer.encode(full_response))
            st.session_state.total_tokens += generated_tokens
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "token_count": generated_tokens # Save this so we can display it later
            })
            
            mx.clear_cache()
            st.rerun() # Refresh to update the sidebar progress bar