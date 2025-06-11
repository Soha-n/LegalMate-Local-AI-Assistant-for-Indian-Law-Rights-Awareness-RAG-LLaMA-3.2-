from typing import List
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.llms import Ollama
from langchain.schema import Document
from vector import VectorStore
import time

class LegalMate:
    def __init__(self, vector_store: VectorStore):
        """
        Initialize LegalMate with vector store and LLM.
        
        Args:
            vector_store (VectorStore): Initialized vector store instance
        """
        self.vector_store = vector_store
        
        # Initialize LLM with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.llm = Ollama(
                    model="llama3.2:3b",
                    temperature=0.5,
                    timeout=120  # Increase timeout for larger responses
                )
                # Test the LLM
                self.llm("test")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to initialize LLM after {max_retries} attempts: {str(e)}")
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(2)
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are LegalMate, a helpful legal assistant for Indian citizens.
            Answer the user's legal query using the following legal context.
            If the context is not relevant or insufficient, provide a general legal awareness response.
            Always be clear, accurate, and helpful in your responses.
            Format your response in a clear, structured way.
            
            Context: {context}
            """),
            ("human", "{question}")
        ])
        
        # Create the chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )
        
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format the retrieved documents into a context string.
        
        Args:
            documents (List[Document]): List of retrieved documents
            
        Returns:
            str: Formatted context string
        """
        if not documents:
            return "No relevant context found. Providing general legal awareness response."
        return "\n\n".join([doc.page_content for doc in documents])
        
    def get_response(self, query: str) -> str:
        """
        Get a response for a legal query.
        
        Args:
            query (str): User's legal question
            
        Returns:
            str: Generated response
        """
        try:
            # Retrieve relevant documents
            documents = self.vector_store.similarity_search(query)
            
            # Format context
            context = self._format_context(documents)
            
            # Generate response
            response = self.chain.run(
                context=context,
                question=query
            )
            
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your query: {str(e)}. Please try again or rephrase your question." 
