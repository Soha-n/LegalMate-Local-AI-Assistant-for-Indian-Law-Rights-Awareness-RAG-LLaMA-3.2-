import os
from typing import List
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_pdf(file_path: str) -> List[Document]:
    """
    Load and split a PDF document into chunks.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        List[Document]: List of document chunks
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at {file_path}")
    
    # Load the PDF
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_document_chunks() -> List[Document]:
    """
    Get document chunks from the Indian Constitution PDF.
    
    Returns:
        List[Document]: List of document chunks
    """
    pdf_path = os.path.join("data", "indian_constitution.pdf")
    return load_pdf(pdf_path) 