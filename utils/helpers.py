"""
Utility functions for On-Prem GenAI Workbench
"""

import streamlit as st
from typing import Dict, List, Any
import json
import time

class ChatUtils:
    """Utilities for chat functionality"""
    
    @staticmethod
    def format_message_for_display(message: Dict[str, str]) -> str:
        """Format a message for display in the chat interface"""
        role_emoji = "👤" if message["role"] == "user" else "🤖"
        return f"{role_emoji} **{message['role'].title()}**: {message['content']}"
    
    @staticmethod
    def export_chat_history(messages: List[Dict[str, str]]) -> str:
        """Export chat history as JSON string"""
        return json.dumps(messages, indent=2)
    
    @staticmethod
    def import_chat_history(json_str: str) -> List[Dict[str, str]]:
        """Import chat history from JSON string"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            st.error("Invalid JSON format for chat history")
            return []
    
    @staticmethod
    def calculate_tokens_estimate(text: str) -> int:
        """Rough estimation of tokens (1 token ≈ 4 characters)"""
        return len(text) // 4


class SessionStateManager:
    """Manager for Streamlit session state"""
    
    @staticmethod
    def initialize_defaults():
        """Initialize default session state values"""
        defaults = {
            'messages': [],
            'scenario': 'scenario_1',
            'model_selected': False,
            'current_model': None,
            'system_prompt': None,
            'temperature': 0.7,
            'max_tokens': 4096
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def reset_chat_state():
        """Reset chat-related session state"""
        st.session_state.messages = []
        st.session_state.model_selected = False
        st.session_state.current_model = None
    
    @staticmethod
    def get_state_value(key: str, default: Any = None) -> Any:
        """Safely get a value from session state"""
        return st.session_state.get(key, default)
    
    @staticmethod
    def set_state_value(key: str, value: Any):
        """Set a value in session state"""
        st.session_state[key] = value


class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_info_box(title: str, content: str, type: str = "info"):
        """Render an information box"""
        if type == "info":
            st.info(f"**{title}**\n\n{content}")
        elif type == "success":
            st.success(f"**{title}**\n\n{content}")
        elif type == "warning":
            st.warning(f"**{title}**\n\n{content}")
        elif type == "error":
            st.error(f"**{title}**\n\n{content}")
    
    @staticmethod
    def render_metric_row(metrics: Dict[str, Any]):
        """Render a row of metrics"""
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics.items()):
            with col:
                st.metric(label=label, value=value)
    
    @staticmethod
    def render_progress_indicator(message: str = "Processing..."):
        """Render a progress indicator"""
        with st.spinner(message):
            time.sleep(0.5)  # Small delay for visual feedback


# Future utility classes will be added here for:
# - DatabaseUtils (for Scenario 3)
# - DocumentUtils (for Scenario 2)
# - EmbeddingUtils (for Scenario 2)