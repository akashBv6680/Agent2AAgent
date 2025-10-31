import streamlit as st
import os
import time
from google import genai
from google.genai.errors import APIError

# --- Configuration & Constants ---
PAGE_TITLE = "Gemini A2A Agent Dashboard"
USER_AVATAR = "👤"
ASSISTANT_AVATAR = "🤖"
ANALYST_AVATAR = "🔬"
GEMINI_MODEL_ROLE = "model"

# --- Agent Profiles ---
AGENT_PROFILES = {
    "Coordinator": {
        "name": "Coordinator-Agent",
        "avatar": ASSISTANT_AVATAR,
        "model": "gemini-2.5-flash",
        "instruction": "You are the Coordinator-Agent. Your primary goal is to manage the user's overall task. You receive input from the user and from the DataScience-Analyst. If the user asks a non-Data Science question (like asking for a status), answer it. If the user asks a Data Science question, you will delegate the task to the Analyst. Respond by acknowledging the task and indicating it is delegated.",
        "status": "Available",
    },
    "Analyst": {
        "name": "DataScience-Analyst",
        "avatar": ANALYST_AVATAR,
        "model": "gemini-2.5-flash",
        "instruction": """
            You are the specialized DataScience-Analyst. Your task is to process input specifically related to Data Science, Machine Learning, and Python/Pandas.
            Always use this knowledge as your primary source:
            1. Pandas: The primary Python library for data manipulation. Key objects are **Series** and **DataFrame**.
            2. Scikit-learn (sklearn): The go-to library for classic ML algorithms.
            3. Data Preprocessing: Essential steps include **Handling missing values**, **Encoding categorical features**, and **Feature Scaling**.
            4. Python Code: Always respond to code requests with a complete, runnable Python code block.
            """,
        "status": "Available",
    }
}

# --- Utility Functions ---

def init_session_state():
    """Initializes all necessary session state variables."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = None
    
    # History for the left (Coordinator/Main) chat pane
    if "main_chat_history" not in st.session_state:
        st.session_state.main_chat_history = [] 

    # History for the right (Analyst/Secondary) chat pane
    if "analyst_chat_history" not in st.session_state:
        st.session_state.analyst_chat_history = []

    # Dynamic status of agents
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = {
            "Coordinator": "Available",
            "Analyst": "Available"
        }
    
    # The piece of information being passed between agents
    if "data_transfer" not in st.session_state:
        st.session_state.data_transfer = None


def generate_agent_response(agent_key, history_key, system_instruction, prompt_text):
    """Generates response using the Gemini API for a specific agent/history."""
    
    # 1. Update Agent Status to Working
    st.session_state.agent_status[agent_key] = "Working"
    st.rerun() # Rerun to visually update the status indicator immediately
    
    client = st.session_state.api_client
    
    # Prepare history for the API call (maps Streamlit 'assistant' to Gemini 'model')
    gemini_contents = [
        {
            "role": GEMINI_MODEL_ROLE if m["role"] == "assistant" else m["role"],
            "parts": [{"text": m["content"]}]
        }
        for m in st.session_state[history_key]
    ]

    # Add the current prompt as the last user message
    gemini_contents.append({"role": "user", "parts": [{"text": prompt_text}]})
    
    try:
        response = client.models.generate_content(
            model=AGENT_PROFILES[agent_key]["model"],
            contents=gemini_contents,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        
        # 2. Update Agent Status to Available/Delegated
        if agent_key == "Coordinator" and "data science" in prompt_text.lower():
            # If the Coordinator delegates, its status should reflect that
            st.session_state.agent_status[agent_key] = "Delegated"
        else:
            st.session_state.agent_status[agent_key] = "Available"
        
        return response.text
        
    except APIError as e:
        st.session_state.agent_status[agent_key] = "Error"
        return f"An API Error occurred for {agent_key}: {e}"
    except Exception as e:
        st.session_state.agent_status[agent_key] = "Error"
        return f"An unexpected error occurred for {agent_key}: {e}"

# --- Main Application Logic ---

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

init_session_state()

# --- Sidebar for Setup ---
with st.sidebar:
    st.header("Setup & Status")
    
    gemini_api_key = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API Key.")
    
    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
        try:
            st.session_state.api_client = genai.Client(api_key=gemini_api_key)
            st.success("API Client Initialized! 🎉")
        except Exception as e:
            st.error(f"Invalid API Key or initialization failed: {e}")
            st.stop()
    else:
        st.warning("Please enter your Gemini API Key to start.")
        st.stop()

    st.markdown("---")
    st.subheader("Agent Availability")
    
    # Display dynamic status for each agent
    for key, profile in AGENT_PROFILES.items():
        status = st.session_state.agent_status[key]
        emoji = "🟢" if status == "Available" else ("🟡" if status == "Working" else ("🔵" if status == "Delegated" else "🔴"))
        st.markdown(f"{profile['avatar']} **{profile['name']}**:\n{emoji} **{status}**")
        if status == "Working":
            st.progress(50, text="Performing task...")
            # Simulate work for the visual effect
            time.sleep(0.1) 
    
    st.markdown("---")
    st.subheader("Cross-Agent Data Transfer")
    st.code(st.session_state.data_transfer or "None", language="json")


# --- Main Content: Two Columns for Two Chat Interfaces ---
col_main, col_analyst = st.columns(2)

# =========================================================================
# LEFT COLUMN: COORDINATOR AGENT (Main User Chat)
# =========================================================================
with col_main:
    st.subheader(f"{AGENT_PROFILES['Coordinator']['name']} Chat")
    
    # Display Chat History
    main_chat_container = st.container(height=400)
    for message in st.session_state.main_chat_history:
        with main_chat_container.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    # User Input
    if main_prompt := st.chat_input("Ask the Coordinator a question..."):
        
        # 1. Add user message to history and display
        st.session_state.main_chat_history.append({"role": "user", "content": main_prompt, "avatar": USER_AVATAR})
        with main_chat_container.chat_message("user", avatar=USER_AVATAR):
            st.markdown(main_prompt)
        
        # 2. Get Coordinator response
        with main_chat_container.chat_message("assistant", avatar=AGENT_PROFILES['Coordinator']['avatar']):
            with st.spinner(f"{AGENT_PROFILES['Coordinator']['name']} is processing..."):
                coordinator_response = generate_agent_response(
                    "Coordinator",
                    "main_chat_history",
                    AGENT_PROFILES['Coordinator']['instruction'],
                    main_prompt
                )
                st.markdown(coordinator_response)
        
        # 3. Add Coordinator response to history
        st.session_state.main_chat_history.append({"role": "assistant", "content": coordinator_response, "avatar": AGENT_PROFILES['Coordinator']['avatar']})
        
        # 4. Check for Delegation (A2A Logic)
        if st.session_state.agent_status["Coordinator"] == "Delegated":
            # Pass the original user prompt as the data transfer to the Analyst
            st.session_state.data_transfer = {"task": "Analyze Data Science Request", "prompt": main_prompt}
            
            # Immediately trigger the Analyst
            st.rerun() 


# =========================================================================
# RIGHT COLUMN: DATA SCIENCE ANALYST (Delegated Task Chat)
# =========================================================================
with col_analyst:
    st.subheader(f"{AGENT_PROFILES['Analyst']['name']} Task Status")
    
    # Display Analyst Chat History
    analyst_chat_container = st.container(height=400)
    for message in st.session_state.analyst_chat_history:
        with analyst_chat_container.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    # Handle incoming delegated task from the Coordinator
    if st.session_state.data_transfer:
        task_info = st.session_state.data_transfer
        analyst_prompt = f"TASK: {task_info['task']}. INPUT: {task_info['prompt']}"
        original_user_prompt = task_info['prompt']
        
        # Reset transfer state to avoid re-triggering
        st.session_state.data_transfer = None 
        
        # 1. Add task delegation as a 'user' message to the Analyst's history
        st.session_state.analyst_chat_history.append({"role": "user", "content": f"Delegated Task:\n\n`{original_user_prompt}`", "avatar": AGENT_PROFILES['Coordinator']['avatar']})
        
        with analyst_chat_container.chat_message("user", avatar=AGENT_PROFILES['Coordinator']['avatar']):
            st.markdown(f"**Delegated Task from Coordinator:**\n\n`{original_user_prompt}`")
        
        # 2. Get Analyst response
        with analyst_chat_container.chat_message("assistant", avatar=AGENT_PROFILES['Analyst']['avatar']):
            with st.spinner(f"{AGENT_PROFILES['Analyst']['name']} is processing..."):
                analyst_response = generate_agent_response(
                    "Analyst",
                    "analyst_chat_history",
                    AGENT_PROFILES['Analyst']['instruction'],
                    analyst_prompt
                )
                st.markdown(analyst_response)

        # 3. Add Analyst response to history
        st.session_state.analyst_chat_history.append({"role": "assistant", "content": analyst_response, "avatar": AGENT_PROFILES['Analyst']['avatar']})
        
        # 4. Agent-to-Agent Feedback (Analyst sends result back to Coordinator)
        # We put the Analyst's final result into the Coordinator's history to continue the conversation thread
        st.session_state.main_chat_history.append({"role": "assistant", "content": f"**[Analyst Report]**\n\n{analyst_response}", "avatar": AGENT_PROFILES['Analyst']['avatar']})
        st.session_state.agent_status["Coordinator"] = "Available" # Coordinator is free again
        st.rerun() # Rerun to display the final result in the Coordinator chat
