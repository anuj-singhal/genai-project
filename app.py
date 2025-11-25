"""
On-Prem GenAI Workbench
Main Streamlit Application
"""

import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="On-Prem GenAI Workbench",
    page_icon="🤖",
    layout="wide"
)

from config import Config
from llm_interface import LLMInterface

def initialize_session_state():
    """Initialize session state variables"""
    config = Config()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = config.DEFAULT_MODEL
    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = config.DEFAULT_SYSTEM_PROMPT
    if 'initial_knowledge' not in st.session_state:
        st.session_state.initial_knowledge = ""
    if 'temperature' not in st.session_state:
        st.session_state.temperature = config.TEMPERATURE
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = config.MAX_TOKENS
    if 'current_input' not in st.session_state:
        st.session_state.current_input = ""

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Application header with reduced spacing
    st.markdown("<h1 style='margin-top: -50px;'>🤖 On-Prem GenAI Workbench</h1>", unsafe_allow_html=True)
    st.markdown("*A self-service GenAI platform for enterprise LLM interactions*")
    
    # Create the main interface
    interface = LLMInterface()
    interface.render()

if __name__ == "__main__":
    main()