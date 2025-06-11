import os
import streamlit as st
from load_documents import get_document_chunks
from vector import VectorStore
from rag_chain import LegalMate
import time

# Set page config
st.set_page_config(
    page_title="LegalMate - Indian Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "legal_mate" not in st.session_state:
    st.session_state.legal_mate = None

def initialize_legal_mate():
    """Initialize the LegalMate system."""
    try:
        with st.spinner("Loading documents and initializing LegalMate..."):
            # Get document chunks
            chunks = get_document_chunks()
            
            # Initialize vector store
            vector_store = VectorStore()
            
            # Create or load vector store
            if not os.path.exists("chroma_db"):
                st.info("Creating vector store for the first time...")
                vector_store.create_vector_store(chunks)
            else:
                st.info("Loading existing vector store...")
                vector_store.load_vector_store()
                
            # Initialize LegalMate
            st.info("Initializing LLM...")
            legal_mate = LegalMate(vector_store)
            return legal_mate
    except Exception as e:
        st.error(f"Error initializing LegalMate: {str(e)}")
        return None

def main():
    # Title and description
    st.title("⚖️ LegalMate")
    st.markdown("""
    Welcome to LegalMate, your local legal assistant for Indian law and constitution!
    Ask questions about the Indian Constitution, fundamental rights, duties, laws, and legal remedies.
    """)
    
    # Initialize LegalMate if not already initialized
    if st.session_state.legal_mate is None:
        st.session_state.legal_mate = initialize_legal_mate()
    
    if st.session_state.legal_mate is None:
        st.error("""
        Failed to initialize LegalMate. Please ensure:
        1. Ollama is installed and running
        2. The llama3.2:3b model is pulled (run: `ollama pull llama3.2:3b`)
        3. The Indian Constitution PDF is in the data directory
        """)
        return
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your legal question"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.legal_mate.get_response(prompt)
                    st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main() 