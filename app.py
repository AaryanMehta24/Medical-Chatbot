import streamlit as st
import requests

# ----------------------------
# Config & UI Styling
# ----------------------------
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="centered"
)

# ----------------------------
# Custom CSS for Light Theme & Highlighted Chat
# ----------------------------
st.markdown("""
<style>
/* General App Background */
.stApp {
    background-color: #f0f4f8; /* soft light background */
    color: #000000;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Chat Title */
h1 {
    color: #007BFF;
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 0.2rem;
}

/* Chat Description */
h3 {
    color: #555555;
    text-align: center;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* Chat Messages */
.stChatMessage {
    border-radius: 16px;
    padding: 12px 20px;
    margin-bottom: 12px;
    max-width: 75%;
    line-height: 1.5;
    font-size: 1rem;
    word-wrap: break-word;
}

/* User Messages */
.stChatMessage.user {
    background-color: #d1e7ff; /* light blue highlight */
    color: #000000;
    align-self: flex-end;
}

/* Assistant Messages */
.stChatMessage.assistant {
    background-color: #fff3cd; /* light yellow highlight */
    color: #000000;
    border: 1px solid #ffeeba;
    align-self: flex-start;
}

/* Chat Input Box */
.stChatInput {
    padding: 10px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border: 1px solid #c0c0c0;
    background-color: #ffffff;
    color: #000000;
}

/* Scrollable Chat Area */
[data-testid="stVerticalBlock"] {
    max-height: 600px;
    overflow-y: auto;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Page Title & Description
# ----------------------------
st.title("🩺 Medical Chatbot")
st.markdown(
    "Welcome to the **Medical Chatbot**! 🩺\n\n"
    "Ask any medical-related questions, and I’ll provide helpful guidance "
    "based on medical knowledge. Remember, I am **not a doctor**, and this is "
    "for educational purposes only.",
    unsafe_allow_html=True
)

# ----------------------------
# Session State (for storing messages)
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Display Chat History
# ----------------------------
for msg in st.session_state.messages:
    role_class = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(msg["role"]):
        st.markdown(f'<div class="{role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

# ----------------------------
# User Input
# ----------------------------
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(f'<div class="user">{user_input}</div>', unsafe_allow_html=True)

    # Call FastAPI endpoint
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"question": user_input},
            timeout=60
        )
        response.raise_for_status()
        answer = response.json()["answer"]
    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    # Append assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(f'<div class="assistant">{answer}</div>', unsafe_allow_html=True)
