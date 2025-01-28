from pathlib import Path
from typing import List, Dict, Optional
import warnings
import logging
import PyPDF2
import chromadb
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.api.types import Documents, Embeddings
from chromadb.config import Settings

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
logging.getLogger("torch.classes").setLevel(logging.ERROR)

class PDFProcessor:
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.embedding_model = SentenceTransformer(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
    def extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF with better error handling."""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with improved chunking strategy."""
        return self.text_splitter.split_text(text)

    def generate_embeddings(self, chunks: List[str]) -> Embeddings:
        """Generate embeddings with batching for better performance."""
        try:
            return self.embedding_model.encode(chunks, convert_to_tensor=True).tolist()
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            raise

class ChromaDBManager:
    def __init__(self):
        """Initialize ChromaDB with HTTP client."""
        self.client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            settings=Settings(
                chroma_api_impl="rest",
                chroma_server_host="localhost",
                chroma_server_http_port=8000
            )
        )

    def store_embeddings(self, collection_name: str, chunks: List[str], 
                         embeddings: Embeddings) -> None:
        """Store embeddings with metadata and error handling."""
        try:
            # Get or create collection
            try:
                collection = self.client.get_collection(name=collection_name)
            except:
                collection = self.client.create_collection(name=collection_name)
            
            # Add chunks in batches for better performance
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_end = min(i + batch_size, len(chunks))
                collection.add(
                    documents=chunks[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    ids=[f"chunk_{j}" for j in range(i, batch_end)],
                    metadatas=[{"index": j, "source": "pdf"} for j in range(i, batch_end)]
                )
        except Exception as e:
            st.error(f"Error storing embeddings: {str(e)}")
            raise

    def query_similar(self, collection_name: str, question: str, 
                      embedding_model: SentenceTransformer, k: int = 3) -> Optional[Documents]:
        """Query similar documents with improved context retrieval."""
        try:
            collection = self.client.get_collection(collection_name)
            question_embedding = embedding_model.encode([question]).tolist()
            
            results = collection.query(
                query_embeddings=question_embedding,
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            if results and results['documents']:
                return " ".join(results['documents'][0])
            return None
        except Exception as e:
            st.error(f"Error querying database: {str(e)}")
            return None

class QASystem:
    def __init__(self, model_name: str = "llama3.2"):
        self.llm = OllamaLLM(model=model_name)

    def get_answer(self, question: str, context: str) -> str:
        """Get answer with improved prompt engineering."""
        prompt = f"""Given the context provided, answer the following question concisely, accurately, and interactively.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response if isinstance(response, str) else response.get('text', 
                "Error: Unable to generate response")
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return f"Error generating response: {str(e)}"

# Helper function for refining collection names
def generate_refined_collection_name(document_name: str) -> str:
    """Generate a refined collection name based on the document name."""
    refined_name = document_name.replace('.pdf', '')  # Remove .pdf
    refined_name = refined_name[:50]  # Limit to 50 characters
    refined_name = refined_name.replace(' ', '_').replace('-', '_')  # Replace spaces and dashes with underscores
    return f"pdf_{refined_name}"

def main():
    st.set_page_config(page_title="PDF Q&A System", layout="wide")
    
    st.title("📚 Advanced PDF Question Answering System")

    st.markdown("""
    Upload a PDF file and ask questions about its content. 
    The system will analyze the document and provide relevant answers.
    """)

    # Initialize components
    try:
        pdf_processor = PDFProcessor()
        chroma_manager = ChromaDBManager()
        qa_system = QASystem()
    except Exception as e:
        st.error(f"""Error initializing system components: {str(e)}
        
        Please make sure the ChromaDB server is running. You can start it with:
        ```
        chroma run --path /path/to/your/data
        ```
        """)
        return

    # File upload with progress
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            # Save and process PDF
            pdf_path = Path("temp") / uploaded_file.name
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract and process text
            text = pdf_processor.extract_text(pdf_path)
            chunks = pdf_processor.chunk_text(text)
            
            # Generate embeddings and store
            with st.spinner("Generating embeddings..."):
                embeddings = pdf_processor.generate_embeddings(chunks)
                collection_name = generate_refined_collection_name(uploaded_file.name)
                chroma_manager.store_embeddings(collection_name, chunks, embeddings)

            st.success(f"\u2705 Processed {len(chunks)} sections from the PDF")

        # Q&A Interface
        st.subheader("Ask Questions")
        question = st.text_input("Enter your question about the document:")

        if question:
            with st.spinner("Searching for answer..."):
                context = chroma_manager.query_similar(
                    collection_name, 
                    question, 
                    pdf_processor.embedding_model
                )
                
                if context:
                    answer = qa_system.get_answer(question, context)
                    
                    # Display answer with formatting
                    st.markdown("### Answer")
                    st.markdown(answer)
                    
                    # Option to show source context
                    if st.checkbox("Show source context"):
                        st.markdown("### Source Context")
                        st.markdown(context)
                else:
                    st.warning("No relevant context found in the document.")

if __name__ == "__main__":
    main()
