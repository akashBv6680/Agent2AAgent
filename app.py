import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv() 

# 1. Configure the API Key
# Use os.environ.get for deployment environments (like Streamlit Cloud)
gemini_api_key = os.environ.get("GEMINI_API_KEY")

if not gemini_api_key:
    # Stop if the key is not available
    st.error("GEMINI_API_KEY environment variable not set. Please set it in secrets or .env file.")
    st.stop()

# Configure the Generative AI client
genai.configure(api_key=gemini_api_key)

# 2. Initialize the Generative Model
# NOTE: Using the positional argument syntax required for google-generativeai==0.8.5
try:
    # Pass "gemini-pro" as a positional argument!
    model = genai.GenerativeModel("gemini-pro") 
except Exception as e:
    st.error(f"Error initializing chat model: {e}")
    st.stop()
    
# 3. Streamlit App Setup
st.title("Simple Streamlit Gemini Chat")

# Initialize chat history in Streamlit session state
if "chat" not in st.session_state:
    # Use the model's start_chat method
    st.session_state.chat = model.start_chat()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get model response
    with st.spinner("Gemini is thinking..."):
        # Send message to the chat session
        response = st.session_state.chat.send_message(prompt)
        
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response.text)
        
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response.text})
