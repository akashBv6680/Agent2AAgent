import streamlit as st
import google.generativeai as genai
import os

# --- 1. Configuration and API Key Setup ---
# Set the page title and layout
st.set_page_config(page_title="Gemini Agent Assistant", layout="wide")
st.title("🤖 Gemini Agent Assistant (Simulated A2A)")
st.subheader("Leveraging the Gemini API for Conversational AI")

# Get API Key from Streamlit Secrets or Environment Variable
# In Streamlit Cloud, you'd set this as a secret in the 'secrets.toml' file.
# 
try:
    # Use st.secrets for secure deployment on Streamlit Cloud
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    # Fallback for local testing if you set an environment variable
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("🚨 GEMINI_API_KEY not found. Please set it in your environment variables or Streamlit secrets.")
    st.stop()

# Configure the Gemini client
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# Define the model to use
MODEL_NAME = 'gemini-2.5-flash'  # Excellent for general chat and speed

# Define the Knowledge Base (KB) via System Instruction
DATA_SCIENCE_KB = (
    "You are a helpful Data Science and AI agent. Your responses should be grounded in Data Science "
    "concepts, machine learning, and statistical analysis. Keep your answers concise, informative, "
    "and professional. Do not answer questions outside of Data Science, AI, or general knowledge."
)

# --- 2. State Initialization and Functions ---

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize a new chat session with the model and system instruction
if "chat" not in st.session_state:
    try:
        model = genai.GenerativeModel(
            model=MODEL_NAME,
            system_instruction=DATA_SCIENCE_KB
        )
        st.session_state.chat = model.start_chat(history=st.session_state.messages)
    except Exception as e:
        st.error(f"Error initializing chat model: {e}")
        st.stop()


# Function to display message popups (left for user, right for assistant)
def display_chat_history():
    for message in st.session_state.messages:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            # Apply custom CSS for right-alignment of user message pop-up
            if role == "user":
                # NOTE: This uses an HTML/CSS hack for right-justification, which might break 
                # with Streamlit updates, but is a common workaround.
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: flex-end;'>
                        <div style='background-color: #007AFF; color: white; border-radius: 10px; padding: 10px;'>
                            {message["content"]}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(message["content"])

# --- 3. Sidebar for Discovery and Information ---
with st.sidebar:
    st.header("Agent Information")
    st.markdown("---")
    
    # Discovery Agent Feature (Simulated)
    st.markdown("### 🔍 Available Agent Models")
    st.info(
        f"**Primary Agent Model:** `{MODEL_NAME}`\n"
        f"This is a powerful agent capable of chat and function calling (tool use)."
    )
    st.markdown(
        """
        *Note: In a true A2A system, this would list different specialized agents 
        (e.g., 'HR Agent', 'Finance Agent') based on their 'AgentCard' discovery.*
        """
    )

    # Knowledge Base Description
    st.markdown("### 🧠 Knowledge Base")
    st.markdown(f"The current agent is primed with a **Data Science** Knowledge Base via the following instruction:")
    st.code(DATA_SCIENCE_KB, language="text")
    st.markdown("---")
    st.write("Current Session History Length:", len(st.session_state.messages))

# --- 4. Main Chat Interface ---

# Display all existing messages
display_chat_history()

# Chat input widget
if prompt := st.chat_input("Ask a Data Science question..."):
    # 1. Display user message immediately (uses custom HTML/CSS for right-align)
    with st.chat_message("user"):
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end;'>
                <div style='background-color: #007AFF; color: white; border-radius: 10px; padding: 10px;'>
                    {prompt}
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # 2. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Get assistant response
    with st.chat_message("assistant"):
        # Use a placeholder to stream the response
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Send message and stream the response
            response_stream = st.session_state.chat.send_message(prompt, stream=True)

            for chunk in response_stream:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌") # Typying effect
            
            message_placeholder.markdown(full_response) # Final message

            # 4. Add assistant response to history (uses default left-align)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Gemini API Call Error: {e}")
            full_response = "Sorry, I encountered an error while processing your request."
            st.session_state.messages.append({"role": "assistant", "content": full_response})
