import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_openai import OpenAI
from langchain.callbacks import StreamlitCallbackHandler
import os

# --- Configuration and Setup ---

# Set page configuration
st.set_page_config(page_title="Streamlit LangChain Agent Chat", layout="wide")
st.title("🤖 LangChain Agent Chat with Streaming")
st.caption("This agent uses the DuckDuckGo Search tool.")

# Load API Key (Assuming OpenAI key is set in Streamlit secrets or env var)
# For the agent to run, you must have an OPENAI_API_KEY set.
try:
    openai_api_key = os.environ.get("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
except (KeyError, AttributeError):
    st.error("Please set the `OPENAI_API_KEY` in your environment or `secrets.toml` file.")
    st.stop()


# --- LangChain Agent Initialization (Cached) ---

@st.cache_resource
def get_agent_executor(api_key):
    """Initializes and returns the LangChain Agent Executor."""
    # 1. Initialize the LLM (must support streaming)
    # Note: Using `OpenAI` or `ChatOpenAI` from langchain_openai
    llm = OpenAI(temperature=0, streaming=True, api_key=api_key)

    # 2. Load the tools the agent will use
    tools = load_tools(["ddg-search"]) # DuckDuckGo Search

    # 3. Get the prompt to use - ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    # 4. Create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)

    # 5. Create the Agent Executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor

agent_executor = get_agent_executor(openai_api_key)


# --- Session State Management ---

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm an AI assistant. How can I help you today?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Logic: React to User Input ---

if prompt := st.chat_input("Ask a question to the agent..."):
    # 1. Display user message in chat message container and add to history
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Display assistant thinking and response
    with st.chat_message("assistant"):
        # Create an empty container where the streaming response will be written
        st_container = st.container()
        
        # Initialize the StreamlitCallbackHandler to stream output to the container
        # This is crucial for seeing the agent's thought process and final answer live.
        st_callback = StreamlitCallbackHandler(st_container)
        
        # Invoke the agent executor
        # The agent's thinking and final answer will be streamed via the callback.
        try:
            response = agent_executor.invoke(
                {"input": prompt},
                {"callbacks": [st_callback]}
            )
            # The final output is in response["output"]
            final_response = response["output"]
            
        except Exception as e:
            final_response = f"An error occurred: {e}"
            st_container.error(final_response)

        # 3. The `st_container` has already been updated by the callback.
        # We ensure the final answer is captured in the message history.
        st.session_state.messages.append({"role": "assistant", "content": final_response})
