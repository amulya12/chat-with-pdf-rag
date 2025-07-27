import streamlit as st
import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Configure HuggingFace API
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

class PDFChatApp:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.conversation_history = []
        
    def initialize_embeddings(self):
        """Initialize the embedding model"""
        try:
            # Using Google's text-embedding-gecko for embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            logger.info("Embeddings model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            st.error(f"Error initializing embeddings: {e}")
            return False
        return True
    
    def load_pdfs(self, uploaded_files):
        """Load and process PDF files"""
        if not uploaded_files:
            return False
            
        try:
            documents = []
            
            for uploaded_file in uploaded_files:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Load PDF
                loader = PyPDFLoader(tmp_file_path)
                pdf_documents = loader.load()
                documents.extend(pdf_documents)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
            logger.info(f"Loaded {len(documents)} documents from {len(uploaded_files)} PDF files")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} text chunks")
            
            # Create vector store
            if self.embeddings:
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                logger.info("Vector store created successfully")
                return True
            else:
                st.error("Embeddings model not initialized")
                return False
                
        except Exception as e:
            logger.error(f"Error loading PDFs: {e}")
            st.error(f"Error loading PDFs: {e}")
            return False
    
    def initialize_qa_chain(self):
        """Initialize the question-answering chain"""
        try:
            if not self.vectorstore:
                st.error("Please upload PDF files first")
                return False
            

            
            # Create retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Initialize LLM (using Gemini if available, otherwise HuggingFace)
            if GOOGLE_API_KEY:
                # Using Gemini
                model = genai.GenerativeModel('gemini-2.0-flash')
                llm = self._create_gemini_llm_wrapper(model)
            elif HUGGINGFACE_API_KEY:
                # Using HuggingFace
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
                    model_kwargs={"temperature": 0.5, "max_length": 512}
                )
            else:
                st.error("No API key found. Please set GOOGLE_API_KEY or HUGGINGFACE_API_KEY in your .env file")
                return False
            
            # Create a custom QA chain that handles the output properly
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
            
            # Create a simple prompt template
            template = """Use the following context to answer the question at the end.

Context: {context}

Question: {question}

Answer:"""
            
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template
            )
            
            # Create a simple LLM chain
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            
            # Create a custom QA function
            def qa_with_sources(question: str):
                # Get relevant documents
                docs = retriever.get_relevant_documents(question)
                context = "\n".join([doc.page_content for doc in docs])
                
                # Generate answer
                result = llm_chain.invoke({"context": context, "question": question})
                
                return {
                    "answer": result["text"],
                    "source_documents": docs
                }
            
            self.qa_chain = qa_with_sources
            
            logger.info("QA chain initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing QA chain: {e}")
            st.error(f"Error initializing QA chain: {e}")
            return False
    
    def _create_gemini_llm_wrapper(self, model):
        """Create a wrapper for Gemini model to work with LangChain"""
        from langchain.llms.base import LLM
        from typing import Any, List, Optional
        
        class GeminiLLM(LLM):
            model: Any
            
            @property
            def _llm_type(self) -> str:
                return "gemini"
            
            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                try:
                    # Handle both string and dict inputs from the chain
                    if isinstance(prompt, dict):
                        # Extract the question from the chain input
                        if "question" in prompt:
                            question = prompt["question"]
                        else:
                            question = str(prompt)
                    else:
                        question = prompt
                    
                    response = self.model.generate_content(question)
                    return response.text
                except Exception as e:
                    logger.error(f"Error calling Gemini: {e}")
                    return f"Error generating response: {e}"
            
            @property
            def _identifying_params(self) -> dict:
                return {"model": "gemini-2.0-flash"}
        
        return GeminiLLM(model=model)
    
    def ask_question(self, question: str):
        """Ask a question and get response"""
        try:
            if not self.qa_chain:
                return "Please initialize the QA chain first by uploading PDF files."
            
            # Get response from QA chain
            result = self.qa_chain(question)
            
            # Extract answer and sources
            answer = result["answer"]
            source_documents = result.get("source_documents", [])
            
            # Format sources
            sources = []
            for doc in source_documents:
                if hasattr(doc, 'metadata') and doc.metadata:
                    source_info = f"Page {doc.metadata.get('page', 'N/A')} from {doc.metadata.get('source', 'Unknown')}"
                    sources.append(source_info)
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"Error asking question: {e}")
            return f"Error processing your question: {e}", []

def main():
    st.set_page_config(
        page_title="Chat with PDF - RAG System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Chat with PDF - RAG System")
    st.markdown("Upload multiple PDF documents and ask questions about their content!")
    
    # Initialize session state
    if 'pdf_chat_app' not in st.session_state:
        st.session_state.pdf_chat_app = PDFChatApp()
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📄 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with"
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} PDF file(s)")
            
            if st.button("🔄 Process PDFs"):
                with st.spinner("Processing PDFs..."):
                    # Initialize embeddings
                    if st.session_state.pdf_chat_app.initialize_embeddings():
                        # Load and process PDFs
                        if st.session_state.pdf_chat_app.load_pdfs(uploaded_files):
                            # Initialize QA chain
                            if st.session_state.pdf_chat_app.initialize_qa_chain():
                                st.success("✅ PDFs processed successfully! You can now ask questions.")
                            else:
                                st.error("❌ Failed to initialize QA chain")
                        else:
                            st.error("❌ Failed to process PDFs")
                    else:
                        st.error("❌ Failed to initialize embeddings")
        
        # API Key configuration
        st.header("🔑 API Configuration")
        st.markdown("""
        **Required API Keys:**
        - **Google API Key** (for Gemini LLM) - Set in .env file as `GOOGLE_API_KEY`
        - **HuggingFace API Key** (alternative) - Set in .env file as `HUGGINGFACE_API_KEY`
        
        Get your API keys from:
        - [Google AI Studio](https://makersuite.google.com/app/apikey)
        - [HuggingFace](https://huggingface.co/settings/tokens)
        """)
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display conversation history
    for i, (question, answer, sources) in enumerate(st.session_state.conversation_history):
        with st.expander(f"Q: {question[:50]}...", expanded=False):
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Answer:** {answer}")
            if sources:
                st.markdown("**Sources:**")
                for source in sources:
                    st.markdown(f"- {source}")
    
    # Question input with form for better handling
    with st.form("question_form"):
        question = st.text_input(
            "Ask a question about your PDFs:",
            placeholder="e.g., What are the main topics discussed in the documents?",
            key="question_input"
        )
        
        submit_button = st.form_submit_button("🚀 Ask Question", type="primary")
    
    if submit_button:
        if not question.strip():
            st.warning("Please enter a question.")
        elif not st.session_state.pdf_chat_app.qa_chain:
            st.error("Please upload and process PDF files first.")
        else:
            with st.spinner("Generating answer..."):
                answer, sources = st.session_state.pdf_chat_app.ask_question(question)
                
                # Add to conversation history
                st.session_state.conversation_history.append((question, answer, sources))
                
                # Display the answer
                st.markdown("### Answer:")
                st.markdown(answer)
                
                if sources:
                    st.markdown("### Sources:")
                    for source in sources:
                        st.markdown(f"- {source}")
                
                # Rerun to refresh the interface
                st.rerun()
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built using Streamlit, LangChain, and RAG</p>
        <p>Upload PDFs, ask questions, and get intelligent answers!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 