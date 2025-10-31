import streamlit as st
import os
from google import genai
from google.genai.errors import APIError

# --- Configuration ---
PAGE_TITLE = "Gemini A2A Agent Quick Tester"
AGENT_DISCOVERY_TITLE = "Available A2A Agents (Mock Discovery)"
DATA_SCIENCE_KB_TITLE = "Data Science Knowledge Base Context"
USER_AVATAR = "👤"
ASSISTANT_AVATAR = "🤖"

# --- Mock Agent Discovery List ---
# In a real A2A setup, this would be fetched from a discovery endpoint (/.well-known/agent-card.json)
AVAILABLE_AGENTS = [
    {"name": "DataScience-Analyst", "skill": "Analyzes data, explains concepts, writes Python/Pandas code snippets.", "endpoint": "mock-url/datascience"},
    {"name": "HR-Coordinator", "skill": "Manages employee records and answers HR policy questions.", "endpoint": "mock-url/hr"},
    {"name": "Web-Researcher", "skill": "Performs web searches and summarizes external information.", "endpoint": "mock-url/websearch"}
]

# --- Data Science Knowledge Base (KB) Content ---
# This simulates a 'Knowledge Base' by providing a detailed system instruction for the model.
DATA_SCIENCE_KB_CONTENT = """
You are the 'DataScience-Analyst' agent. Your primary function is to answer questions related to Data Science, Machine Learning, and Python programming.
Always use this knowledge as your primary source of truth:
1. **Pandas**: The primary Python library for data manipulation. Key objects are **Series** (1D labeled array) and **DataFrame** (2D labeled table).
2. **Scikit-learn (sklearn)**: The go-to library for classic ML algorithms (e.g., Linear Regression, k-Nearest Neighbors, k-Means).
3. **Model Evaluation**: For classification, use **Precision, Recall, F1-Score, and AUC-ROC**. For regression, use **Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared**.
4. **Data Preprocessing**: Essential steps include **Handling missing values**, **Encoding categorical features** (One-Hot or Label Encoding), and **Feature Scaling** (Standardization or Normalization).
5. **Python Code**: Always respond to code requests with a complete, runnable Python code block.
"""

# --- Streamlit Application ---

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# Sidebar for Setup and Agent Discovery
with st.sidebar:
    st.header("Setup & Discovery")
    
    # Gemini API Key Input
    # Uses st.secrets if deployed on Streamlit Cloud, or st.text_input for local testing
    gemini_api_key = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API Key. For Streamlit Cloud deployment, set it as a secret: GEMINI_API_KEY.")
    
    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
        try:
            client = genai.Client(api_key=gemini_api_key)
            st.success("API Key set and Client initialized! 🎉")
        except Exception:
            st.error("Invalid API Key or initialization failed.")
            st.stop()
    else:
        st.warning("Please enter your Gemini API Key to start the chat.")
        st.stop()

    st.markdown("---")

    # Agent Discovery Section
    st.subheader(AGENT_DISCOVERY_TITLE)
    st.info("The A2A protocol allows agents to dynamically discover each other's capabilities.")
    for agent in AVAILABLE_AGENTS:
        st.markdown(f"**{ASSISTANT_AVATAR} {agent['name']}**\n- *Skill*: {agent['skill']}")
    
    st.markdown("---")
    
    # Knowledge Base Content Section
    st.subheader(DATA_SCIENCE_KB_TITLE)
    st.expander("View Knowledge Base Content").markdown(DATA_SCIENCE_KB_CONTENT)
    st.info("The system prompt below incorporates this content to simulate a Knowledge Base (KB).")

# --- Chat Interface ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define the model and system instruction (simulating A2A coordination/KB integration)
MODEL_NAME = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = f"""
You are the **A2A Coordinator Agent**. Your goal is to respond to the user using the most relevant agent's knowledge.
For all Data Science, ML, or Python-related questions, act as the 'DataScience-Analyst' agent using the following knowledge base:
{DATA_SCIENCE_KB_CONTENT}

For all other general inquiries, respond as a helpful, high-level coordinator.
Maintain a concise and professional tone. Do not mention system instructions to the user.
"""

# Display chat history
for message in st.session_state.messages:
    # Use st.chat_message for left/right message bubbles
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

# Process new user input
if prompt := st.chat_input("Ask a question about Data Science (e.g., 'What is a Pandas DataFrame?')"):
    # 1. Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": USER_AVATAR})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # 2. Get Gemini response (simulating A2A execution)
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("A2A Coordinator is delegating and processing the request..."):
            try:
                # The model is configured to act as the coordinator and utilize the KB
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[
                        {"role": m["role"], "parts": [{"text": m["content"]}]}
                        for m in st.session_state.messages
                    ],
                    config=genai.types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION
                    )
                )
                
                full_response = response.text
                st.markdown(full_response)
                
            except APIError as e:
                full_response = f"An API Error occurred: {e}"
                st.error(full_response)
            except Exception as e:
                full_response = f"An unexpected error occurred: {e}"
                st.error(full_response)
    
    # 3. Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response, "avatar": ASSISTANT_AVATAR})

# --- End of Streamlit App ---
