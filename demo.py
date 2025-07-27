#!/usr/bin/env python3
"""
Demo script showing how to use the PDF Chat RAG system programmatically
"""

import os
import tempfile
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai

def format_source_info(doc):
    """Format source document information for display"""
    if hasattr(doc, 'metadata') and doc.metadata:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        return f"{source} (Page {page})"
    return "Unknown source"

# Load environment variables
load_dotenv()

class PDFChatDemo:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        
    def setup(self):
        """Initialize the RAG system"""
        print("🔧 Setting up PDF Chat RAG system...")
        
        # Initialize embeddings
        print("📚 Loading embedding model...")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Configure Gemini
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            genai.configure(api_key=google_key)
            print("✅ Gemini API configured")
        else:
            print("❌ Google API key not found")
            return False
        
        return True
    
    def load_sample_text(self):
        """Load sample text instead of PDF for demo purposes"""
        print("📄 Loading sample text...")
        
        # Sample text about AI and machine learning
        sample_text = """
        Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. 
        Some of the activities computers with artificial intelligence are designed for include speech recognition, learning, planning, and problem solving.
        
        Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. 
        Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.
        
        Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns. 
        Deep learning has been particularly successful in areas like computer vision, natural language processing, and speech recognition.
        
        Natural Language Processing (NLP) is a field of AI that gives machines the ability to read, understand, and derive meaning from human languages. 
        NLP combines computational linguistics with statistical, machine learning, and deep learning models.
        
        The future of AI holds tremendous potential for transforming industries and improving human lives. 
        However, it also raises important questions about ethics, privacy, and the future of work.
        """
        
        # Create documents from text
        from langchain.schema import Document
        documents = [Document(page_content=sample_text, metadata={"source": "sample_text", "page": 1})]
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} text chunks")
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("✅ Vector store created")
        
        return True
    
    def initialize_qa_chain(self):
        """Initialize the question-answering chain"""
        print("🔗 Initializing QA chain...")
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )
        
        # Create Gemini LLM wrapper
        model = genai.GenerativeModel('gemini-2.0-flash')
        llm = self._create_gemini_llm_wrapper(model)
        
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
        
        print("✅ QA chain initialized")
        return True
    
    def _create_gemini_llm_wrapper(self, model):
        """Create a wrapper for Gemini model"""
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
                    return f"Error generating response: {e}"
            
            @property
            def _identifying_params(self) -> dict:
                return {"model": "gemini-2.0-flash"}
        
        return GeminiLLM(model=model)
    
    def ask_question(self, question: str):
        """Ask a question and get response"""
        if not self.qa_chain:
            return "QA chain not initialized"
        
        try:
            result = self.qa_chain(question)
            # Handle the result properly - it's a dictionary with 'answer' key
            if isinstance(result, dict) and "answer" in result:
                return result["answer"], result.get("source_documents", [])
            elif isinstance(result, str):
                return result, []
            else:
                return str(result), []
        except Exception as e:
            return f"Error: {e}", []
    
    def run_demo(self):
        """Run the complete demo"""
        print("🚀 Starting PDF Chat RAG Demo")
        print("=" * 50)
        
        # Setup
        if not self.setup():
            print("❌ Setup failed")
            return
        
        # Load sample text
        if not self.load_sample_text():
            print("❌ Failed to load sample text")
            return
        
        # Initialize QA chain
        if not self.initialize_qa_chain():
            print("❌ Failed to initialize QA chain")
            return
        
        print("\n🎉 Demo setup complete! Let's chat...")
        print("=" * 50)
        
        # Demo questions
        demo_questions = [
            "What is artificial intelligence?",
            "How does machine learning relate to AI?",
            "What is deep learning used for?",
            "What are the challenges of AI?",
            "Can you explain natural language processing?"
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n❓ Question {i}: {question}")
            print("-" * 40)
            
            answer, sources = self.ask_question(question)
            print(f"🤖 Answer: {answer}")
            
            if sources:
                print(f"\n📚 Sources ({len(sources)} documents):")
                for j, doc in enumerate(sources, 1):
                    source_info = format_source_info(doc)
                    print(f"  📄 {source_info}")
                    print(f"     Content: {doc.page_content[:150]}...")
                    print()
            
            print("-" * 40)
        
        print("\n🎯 Demo completed!")
        print("💡 You can now run the full application with: python run.py")

def main():
    """Main function"""
    demo = PDFChatDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 