"""
Unified LLM Interface for On-Prem GenAI Workbench
Handles Direct LLM interaction with RAG Documents extension
"""

import streamlit as st
from openai import OpenAI
from typing import List, Dict, Generator, Optional
import time
import tiktoken
from config import Config
from utils import ChatUtils, SessionStateManager
from rag import DocumentProcessor, VectorStore, RAGAgent

class LLMInterface:
    """Unified interface for LLM interactions with RAG support"""
    
    def __init__(self):
        """Initialize the LLM Interface handler"""
        self.config = Config()
        self.client = None
        self._initialize_client()
        self.tokenizer = None
        self._initialize_tokenizer()
        
        # Initialize RAG components
        self._initialize_rag_components()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        if self.config.validate_api_key():
            self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for token counting"""
        try:
            # Use cl100k_base encoding for GPT-4 and GPT-3.5 models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
    
    def _initialize_rag_components(self):
        """Initialize RAG components in session state"""
        if 'document_processor' not in st.session_state:
            st.session_state.document_processor = DocumentProcessor(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                chunking_strategy=self.config.CHUNKING_STRATEGY
            )
        
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = VectorStore(
                persist_directory=self.config.CHROMA_PERSIST_DIRECTORY,
                collection_name="rag_documents"
            )
        
        if 'rag_agent' not in st.session_state:
            st.session_state.rag_agent = RAGAgent(
                vector_store=st.session_state.vector_store,
                llm_client=self.client
            )
        
        if 'processed_documents' not in st.session_state:
            st.session_state.processed_documents = []
        
        if 'rag_enabled' not in st.session_state:
            st.session_state.rag_enabled = False
        
        if 'embedding_type' not in st.session_state:
            st.session_state.embedding_type = self.config.DEFAULT_EMBEDDING_TYPE
        
        if 'embedding_model' not in st.session_state:
            st.session_state.embedding_model = self.config.DEFAULT_EMBEDDING_MODEL
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback to approximate count (1 token ≈ 4 characters)
            return len(text) // 4
    
    def render(self):
        """Render the unified LLM interface"""
        # Check API key first
        if not self.config.validate_api_key():
            self._render_api_key_input()
            return
        
        # Create layout with sidebar and main content
        self._render_sidebar()
        self._render_main_content()
    
    def _render_api_key_input(self):
        """Render API key input when not configured"""
        st.error("⚠️ OpenAI API Key not found! Please set the OPENAI_API_KEY environment variable.")
        st.info("You can set it using: `export OPENAI_API_KEY='your-api-key-here'`")
        
        # Allow user to input API key temporarily
        api_key_input = st.text_input("Or enter your OpenAI API Key here:", type="password")
        if api_key_input:
            self.config.OPENAI_API_KEY = api_key_input
            self.client = OpenAI(api_key=api_key_input)
            st.success("API Key set temporarily for this session!")
            st.rerun()
    
    def _render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model selection
            st.subheader("🤖 Select OpenAI Model")
            current_model = st.selectbox(
                "Choose model:",
                options=self.config.OPENAI_MODELS,
                index=self.config.OPENAI_MODELS.index(st.session_state.current_model),
                key="model_selector",
                help="Choose the OpenAI model for interaction"
            )
            
            # Update model in session state if changed
            if current_model != st.session_state.current_model:
                st.session_state.current_model = current_model
                st.rerun()
            
            # Display model info
            model_info = self.config.get_model_info(current_model)
            st.caption(f"ℹ️ {model_info['description']}")
            st.caption(f"Context Window: {model_info['context_window']:,} tokens")
            
            st.divider()
            
            # Advanced settings (visible, not in expander)
            st.subheader("⚙️ Advanced Settings")
            
            # Temperature slider
            temperature = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=self.config.MAX_TEMPERATURE,
                value=st.session_state.temperature,
                step=0.1,
                key="temperature_slider",
                help="Controls randomness in responses (0=deterministic, 1=creative)"
            )
            
            # Update temperature if changed
            if temperature != st.session_state.temperature:
                st.session_state.temperature = temperature
            
            # Max tokens input
            max_tokens = st.number_input(
                "Max Tokens:",
                min_value=100,
                max_value=4096,
                value=st.session_state.max_tokens,
                step=100,
                key="max_tokens_input",
                help="Maximum tokens in response"
            )
            
            # Update max tokens if changed
            if max_tokens != st.session_state.max_tokens:
                st.session_state.max_tokens = max_tokens
            
            st.divider()
            
            # Prompt Examples
            st.subheader("📚 Example Prompts")
            if st.button("🎓 Academic Assistant", use_container_width=True):
                self._load_example_prompt("academic")
            if st.button("💻 Code Helper", use_container_width=True):
                self._load_example_prompt("code")
            if st.button("📊 Data Analyst", use_container_width=True):
                self._load_example_prompt("analyst")
            if st.button("✍️ Creative Writer", use_container_width=True):
                self._load_example_prompt("writer")
            if st.button("🏢 Business Consultant", use_container_width=True):
                self._load_example_prompt("business")
            
            st.divider()
            
            # Action buttons
            st.subheader("🎯 Actions")
            
            if st.button("🆕 Start New Chat Session", use_container_width=True):
                st.session_state.messages = []
                st.success("✅ New chat session started!")
                st.rerun()
            
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            # Display session info
            st.divider()
            st.subheader("📊 Session Info")
            st.caption(f"**Messages:** {len(st.session_state.messages)}")
            
            # Calculate total tokens
            if st.session_state.messages:
                total_input_tokens = 0
                total_output_tokens = 0
                
                for msg in st.session_state.messages:
                    tokens = self.count_tokens(msg['content'])
                    if msg['role'] == 'user':
                        total_input_tokens += tokens
                    else:
                        total_output_tokens += tokens
                
                st.metric("Input Tokens", f"{total_input_tokens:,}")
                st.metric("Output Tokens", f"{total_output_tokens:,}")
                st.metric("Total Tokens", f"{total_input_tokens + total_output_tokens:,}")
            
            # RAG status
            if st.session_state.rag_enabled:
                st.divider()
                st.subheader("📄 RAG Status")
                stats = st.session_state.vector_store.get_collection_stats()
                st.success("🔍 RAG Mode Active")
                st.caption(f"**Documents:** {len(st.session_state.processed_documents)}")
                st.caption(f"**Total Chunks:** {stats.get('total_chunks', 0)}")
    
    def _load_example_prompt(self, example_type: str):
        """Load example prompts based on type"""
        examples = {
            "academic": {
                "system": "You are an academic assistant specializing in research and education. Provide detailed, well-structured responses with proper citations when relevant. Use academic language while remaining accessible.",
                "knowledge": "Field of expertise: Computer Science, AI, and Machine Learning\nTarget audience: Graduate students and researchers\nPreferred citation style: APA\nFocus areas: Deep learning, NLP, Computer Vision"
            },
            "code": {
                "system": "You are an expert programming assistant. Provide clean, efficient, and well-commented code. Follow best practices and explain your implementation choices.",
                "knowledge": "Primary languages: Python, JavaScript, SQL\nFrameworks: React, Django, FastAPI, Streamlit\nDevelopment environment: VS Code\nCoding style: PEP 8 for Python, ESLint for JavaScript\nFocus: Production-ready code with error handling"
            },
            "analyst": {
                "system": "You are a data analyst expert. Provide insights backed by data, suggest appropriate visualizations, and explain statistical concepts clearly. Focus on actionable recommendations.",
                "knowledge": "Tools: Python (pandas, numpy, scikit-learn), SQL, Tableau\nDomains: E-commerce, Finance, Healthcare\nAnalysis types: Predictive modeling, A/B testing, Time series\nBusiness KPIs: Revenue, Customer retention, Conversion rates"
            },
            "writer": {
                "system": "You are a creative writing assistant. Help with storytelling, character development, and various writing styles. Provide constructive feedback and creative suggestions.",
                "knowledge": "Genres: Science fiction, Fantasy, Mystery, Contemporary fiction\nWriting styles: First-person narrative, Third-person omniscient\nTarget audience: Young adults and general fiction readers\nFocus: Character-driven narratives with strong dialogue"
            },
            "business": {
                "system": "You are a business consultant with expertise in strategy and operations. Provide practical, actionable advice based on industry best practices. Focus on ROI and measurable outcomes.",
                "knowledge": "Industries: Technology, Retail, Healthcare, Finance\nExpertise: Digital transformation, Process optimization, Change management\nFrameworks: SWOT, Porter's Five Forces, Lean Six Sigma\nMetrics: ROI, NPV, Market share, Customer satisfaction"
            }
        }
        
        if example_type in examples:
            st.session_state.system_prompt = examples[example_type]["system"]
            st.session_state.initial_knowledge = examples[example_type]["knowledge"]
            st.success(f"✅ Loaded {example_type.title()} example prompts!")
            st.rerun()
    
    def _render_main_content(self):
        """Render the main content area"""
        # Prompts Section (Always visible at top)
        with st.container():
            st.markdown("### 📝 Prompts Configuration")
            
            # Create two columns for prompts
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # System Prompt
                system_prompt = st.text_area(
                    "**System Prompt** - Define the assistant's behavior:",
                    value=st.session_state.system_prompt,
                    height=120,
                    key="system_prompt_input",
                    help="This sets the overall behavior and context for the AI assistant"
                )
                
                # Update system prompt if changed
                if system_prompt != st.session_state.system_prompt:
                    st.session_state.system_prompt = system_prompt
            
            with col2:
                # Initial Knowledge/Context
                initial_knowledge = st.text_area(
                    "**Initial Knowledge/Context** - Add background information (optional):",
                    value=st.session_state.initial_knowledge,
                    height=120,
                    key="initial_knowledge_input",
                    placeholder="Add any background information, context, or knowledge that should be considered in the conversation...",
                    help="Optional: Add any initial knowledge or context that the AI should be aware of"
                )
                
                # Update initial knowledge if changed
                if initial_knowledge != st.session_state.initial_knowledge:
                    st.session_state.initial_knowledge = initial_knowledge
        
        # Divider
        st.divider()
        
        # RAG Document Knowledge Base Section
        self._render_rag_section()
        
        # Divider between configuration and chat
        st.divider()
        
        # Chat Interface Section
        chat_col1, chat_col2 = st.columns([4, 1])
        
        with chat_col1:
            st.markdown("### 💬 Chat Interface")
            if st.session_state.rag_enabled:
                st.success("🔍 RAG Mode Active - Using document knowledge")
        
        with chat_col2:
            # Real-time token counter for current input
            if 'current_input' not in st.session_state:
                st.session_state.current_input = ""
            
            current_tokens = self.count_tokens(st.session_state.current_input)
            st.markdown(f"<div style='text-align: right; color: #666; font-size: 14px;'>Current Input: {current_tokens} tokens</div>", unsafe_allow_html=True)
        
        # Create a container for the chat messages with fixed height
        chat_container = st.container(height=400)
        
        with chat_container:
            # Display chat messages
            self._display_chat_messages()
        
        # Chat input (outside the scrollable container)
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Store the current input for token counting
            st.session_state.current_input = user_input
            
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message immediately
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)
            
            # Get AI response
            with chat_container:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    
                    # Check if RAG is enabled
                    if st.session_state.rag_enabled and st.session_state.processed_documents:
                        # Use RAG agent for response
                        with st.spinner("🔍 Searching documents..."):
                            rag_response = st.session_state.rag_agent.process_query(
                                query=user_input,
                                system_prompt=st.session_state.system_prompt,
                                initial_knowledge=st.session_state.initial_knowledge,
                                conversation_history=st.session_state.messages[:-1],
                                model=st.session_state.current_model,
                                temperature=st.session_state.temperature,
                                max_tokens=st.session_state.max_tokens,
                                top_k=st.session_state.get('rag_top_k', 5),
                                similarity_threshold=st.session_state.get('rag_similarity_threshold', 0.3)
                            )
                        
                        # Display response with sources
                        full_response = rag_response.answer
                        
                        if rag_response.sources:
                            full_response += "\n\n📚 **Sources:**\n"
                            for source in rag_response.sources:
                                full_response += f"- {source['filename']} (Similarity: {source['similarity']:.1%})\n"
                        
                        message_placeholder.markdown(full_response)
                    else:
                        # Regular streaming response
                        full_response = ""
                        for chunk in self._get_ai_response_stream(user_input):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                        
                        # Final update without cursor
                        message_placeholder.markdown(full_response)
            
            # Add assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.current_input = ""
            st.rerun()
    
    def _render_rag_section(self):
        """Render the RAG document knowledge base section"""
        with st.expander("📄 **Document Knowledge Base** (Click to expand)", expanded=st.session_state.rag_enabled):
            # Embedding configuration
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Embedding type selection
                embedding_type = st.selectbox(
                    "Embedding Type:",
                    options=["sentence-transformer", "openai"],
                    index=0 if st.session_state.embedding_type == "sentence-transformer" else 1,
                    help="Choose embedding model type (Sentence Transformers are free, OpenAI requires API credits)"
                )
                
                if embedding_type != st.session_state.embedding_type:
                    st.session_state.embedding_type = embedding_type
            
            with col2:
                # Model selection based on type
                if embedding_type == "openai":
                    models = list(self.config.EMBEDDING_MODELS["openai"].keys())
                    model_descriptions = self.config.EMBEDDING_MODELS["openai"]
                else:
                    models = list(self.config.EMBEDDING_MODELS["sentence-transformer"].keys())
                    model_descriptions = self.config.EMBEDDING_MODELS["sentence-transformer"]
                
                embedding_model = st.selectbox(
                    "Embedding Model:",
                    options=models,
                    format_func=lambda x: f"{x} - {model_descriptions[x]}",
                    index=models.index(st.session_state.embedding_model) if st.session_state.embedding_model in models else 0
                )
                
                if embedding_model != st.session_state.embedding_model:
                    st.session_state.embedding_model = embedding_model
                    # Update vector store embedding model
                    try:
                        if embedding_type == "openai":
                            st.session_state.vector_store.set_embedding_model(
                                embedding_type="openai",
                                model_name=embedding_model,
                                api_key=self.config.OPENAI_API_KEY
                            )
                        else:
                            st.session_state.vector_store.set_embedding_model(
                                embedding_type="sentence-transformer",
                                model_name=embedding_model
                            )
                    except Exception as e:
                        st.error(f"Error setting embedding model: {str(e)}")
            
            st.divider()
            
            # File upload section
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=['pdf', 'txt', 'docx', 'md'],
                accept_multiple_files=True,
                help="Upload PDF, TXT, DOCX, or Markdown files"
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("📥 Process & Add Documents", use_container_width=True, disabled=not uploaded_files):
                    if uploaded_files:
                        # Set embedding model if not set
                        if not st.session_state.vector_store.embedding_model:
                            try:
                                if st.session_state.embedding_type == "openai":
                                    st.session_state.vector_store.set_embedding_model(
                                        embedding_type="openai",
                                        model_name=st.session_state.embedding_model,
                                        api_key=self.config.OPENAI_API_KEY
                                    )
                                else:
                                    st.session_state.vector_store.set_embedding_model(
                                        embedding_type="sentence-transformer",
                                        model_name=st.session_state.embedding_model
                                    )
                            except Exception as e:
                                st.error(f"Error setting embedding model: {str(e)}")
                                return
                        
                        # Process documents
                        with st.spinner("Processing documents..."):
                            processed_docs = st.session_state.document_processor.process_multiple_files(uploaded_files)
                            
                            if processed_docs:
                                # Add to vector store
                                success = st.session_state.vector_store.add_documents(
                                    documents=[],  # Not used in current implementation
                                    processed_docs=processed_docs
                                )
                                
                                if success:
                                    st.session_state.processed_documents.extend(processed_docs)
                                    st.session_state.rag_enabled = True
                                    st.success(f"✅ Processed {len(processed_docs)} documents successfully!")
                                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear All Documents", use_container_width=True):
                    if st.session_state.vector_store.clear_all():
                        st.session_state.processed_documents = []
                        st.session_state.rag_enabled = False
                        st.success("✅ Cleared all documents")
                        st.rerun()
            
            # Display current documents
            if st.session_state.processed_documents:
                st.divider()
                st.subheader("📚 Current Documents")

                for idx, doc in enumerate(st.session_state.processed_documents):
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 0.5])
                    with col1:
                        st.text(f"📄 {doc.filename}")
                    with col2:
                        st.text(f"{doc.total_chunks} chunks")
                    with col3:
                        st.text(f"~{doc.total_tokens:,} tokens")
                    with col4:
                        if st.button("🗑️", key=f"delete_doc_{idx}", help=f"Delete {doc.filename}"):
                            if st.session_state.vector_store.delete_document(doc.doc_id):
                                st.session_state.processed_documents.pop(idx)
                                if not st.session_state.processed_documents:
                                    st.session_state.rag_enabled = False
                                st.rerun()
                
                # Document statistics
                stats = st.session_state.document_processor.get_document_stats(st.session_state.processed_documents)
                st.divider()
                st.markdown(f"**Total:** {stats['total_documents']} documents, {stats['total_chunks']} chunks, ~{stats['total_tokens']:,} tokens")
            
            # RAG Settings
            if st.session_state.rag_enabled:
                st.divider()
                st.subheader("⚙️ RAG Settings")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    top_k = st.number_input(
                        "Top K Chunks:",
                        min_value=1,
                        max_value=20,
                        value=st.session_state.get('rag_top_k', 5),
                        help="Number of most relevant chunks to retrieve"
                    )
                    st.session_state.rag_top_k = top_k
                
                with col2:
                    similarity_threshold = st.slider(
                        "Similarity Threshold:",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.get('rag_similarity_threshold', 0.3),
                        step=0.1,
                        help="Minimum similarity score for retrieved chunks"
                    )
                    st.session_state.rag_similarity_threshold = similarity_threshold
    
    def _display_chat_messages(self):
        """Display chat messages from session state"""
        if not st.session_state.messages:
            st.info("👋 Start a conversation by typing a message below!")
            return
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def _get_ai_response_stream(self, user_input: str) -> Generator[str, None, None]:
        """Get streaming response from OpenAI API"""
        try:
            # Prepare messages
            messages = []
            
            # Add system prompt
            system_content = st.session_state.system_prompt
            
            # Append initial knowledge to system prompt if provided
            if st.session_state.initial_knowledge:
                system_content += f"\n\nAdditional Context:\n{st.session_state.initial_knowledge}"
            
            messages.append({"role": "system", "content": system_content})
            
            # Add conversation history
            for msg in st.session_state.messages[:-1]:  # Exclude the last user message we just added
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Make streaming API call
            stream = self.client.chat.completions.create(
                model=st.session_state.current_model,
                messages=messages,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                top_p=self.config.TOP_P,
                stream=True
            )
            
            # Yield chunks as they come
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.error(f"❌ Error getting AI response: {str(e)}")
            
            # Check if it's an API key error
            if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                st.error("Please check your OpenAI API key.")
            
            yield error_msg