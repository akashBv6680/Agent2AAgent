import streamlit as st
import os

# --- CRITICAL IMPORTS ---
# Hub is now imported from the separate 'langchain_hub' package.
from langchain import hub 
from langchain.agents import AgentExecutor, create_react_agent

# Components are imported from their respective modular packages
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


# --- Configuration and Setup ---

st.set_page_config(page_title="Streamlit LangChain Agent Chat", layout="wide")
st.title("🤖 LangChain Agent with Search")

# Streamlit Cloud uses st.secrets to manage API keys.
# Ensure you have a section named [secrets] with openai_api_key in your secrets.toml
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Please set the `OPENAI_API_KEY` in your Streamlit Cloud secrets.")
    st.stop()
openai_api_key = st.secrets["OPENAI_API_KEY"]


# --- LangChain Agent Initialization (Cached) ---

@st.cache_resource
def get_agent_executor(api_key):
    """Initializes and returns the LangChain Agent Executor."""
    
    # 1. Initialize the LLM
    llm = ChatOpenAI(
        temperature=0, 
        streaming=True, 
        api_key=api_key, 
        model="gpt-3.5-turbo"
    )

    # 2. Define the tools
    tools = [
        DuckDuckGoSearchRun(name="DuckDuckGo Search"),
    ]

    # 3. Pull the prompt template from LangChain Hub
    # The 'hub' object must be imported from the correct package (langchain-hub)
    prompt = hub.pull("hwchase17/react")

    # 4. Create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)

    # 5. Create the Agent Executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    
    return agent_executor

# Initialize the agent once
agent_executor = get_agent_executor(openai_api_key)


# --- Session State and Chat History Display ---

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm a search-enabled AI assistant. Ask me anything!"}
    ]

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- Main Logic: React to User Input ---

if prompt := st.chat_input("Ask a question to the agent..."):
    # 1. Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Display assistant thinking and response with streaming
    with st.chat_message("assistant"):
        # Create a container for streamed output
        st_container = st.empty()
        
        # Initialize the callback handler to stream agent thoughts to the container
        st_callback = StreamlitCallbackHandler(st_container)
        
        final_response = ""
        try:
            # Invoke the agent executor with the callback
            response = agent_executor.invoke(
                {"input": prompt},
                {"callbacks": [st_callback]}
            )
            # The final output is captured after the agent finishes
            final_response = response["output"]
            
        except Exception as e:
            final_response = f"An error occurred: {e}"
            st_container.error(final_response)

        # 3. Update the container with the final, complete response
        st_container.markdown(final_response)
        
        # 4. Add the final response to chat history
        st.session_state.messages.append({"role": "assistant", "content": final_response})
