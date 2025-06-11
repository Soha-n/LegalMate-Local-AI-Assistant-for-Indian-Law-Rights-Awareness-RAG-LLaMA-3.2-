# LegalMate - Local Legal Assistant

LegalMate is a Retrieval-Augmented Generation (RAG) chatbot that helps answer legal questions related to the Indian Constitution, fundamental rights, duties, laws, and legal remedies. It runs entirely on your local machine without requiring any external APIs.

## Prerequisites

1. Python 3.8 or higher
2. Ollama installed and running locally
3. The llama2:3b model pulled in Ollama

## Setup

1. Install Ollama from [https://ollama.ai](https://ollama.ai)

2. Pull the required model:

```bash
ollama pull llama2:3b
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Download the Indian Constitution PDF:

```bash
# Create a data directory
mkdir data
# Download the constitution PDF (you'll need to manually download this)
# Place it in the data directory as 'indian_constitution.pdf'
```

## Running the Application

1. Start the Streamlit interface:

```bash
streamlit run main.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Features

- Local RAG-based legal assistant
- Uses the Indian Constitution as knowledge base
- No external API dependencies
- Simple and intuitive interface
- Fallback to base model when context is insufficient

## Project Structure

- `load_documents.py`: Document loading and processing
- `vector.py`: Vector store setup and management
- `rag_chain.py`: RAG chain implementation
- `main.py`: Main application and interface
- `data/`: Directory containing the Indian Constitution PDF
