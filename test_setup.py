#!/usr/bin/env python3
"""
Test script to verify the PDF Chat RAG application setup
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    packages = [
        ("streamlit", "Streamlit"),
        ("langchain", "LangChain"),
        ("PyPDF2", "PyPDF2"),
        ("faiss", "FAISS"),
        ("google.generativeai", "Google Generative AI"),
        ("langchain_google_genai", "LangChain Google GenAI")
    ]
    
    failed_imports = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_api_keys():
    """Test if API keys are configured"""
    print("\n🔑 Testing API keys...")
    
    load_dotenv()
    
    google_key = os.getenv("GOOGLE_API_KEY")
    huggingface_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if google_key:
        print(f"✅ Google API Key: {google_key[:10]}...")
    else:
        print("❌ Google API Key not found")
    
    if huggingface_key:
        print(f"✅ HuggingFace API Key: {huggingface_key[:10]}...")
    else:
        print("❌ HuggingFace API Key not found")
    
    if not google_key and not huggingface_key:
        print("⚠️  No API keys found! Please set up your .env file")
        return False
    
    return True

def test_embedding_model():
    """Test if embedding model can be loaded"""
    print("\n🧠 Testing embedding model...")
    
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Test embedding
        test_text = "Hello, world!"
        embedding = embeddings.embed_query(test_text)
        
        print(f"✅ Embedding model loaded successfully")
        print(f"   Embedding dimension: {len(embedding)}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return False

def test_gemini_api():
    """Test Google Gemini API connection"""
    print("\n🤖 Testing Google Gemini API...")
    
    load_dotenv()
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_key:
        print("⚠️  Skipping Gemini test - no API key")
        return False
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=google_key)
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Hello, this is a test.")
        
        print("✅ Gemini API connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Gemini API: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 PDF Chat RAG Setup Test")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test API keys
    api_keys_ok = test_api_keys()
    
    # Test embedding model
    embedding_ok = test_embedding_model()
    
    # Test Gemini API
    gemini_ok = test_gemini_api()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 40)
    print(f"Package Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"API Keys: {'✅ PASS' if api_keys_ok else '❌ FAIL'}")
    print(f"Embedding Model: {'✅ PASS' if embedding_ok else '❌ FAIL'}")
    print(f"Gemini API: {'✅ PASS' if gemini_ok else '❌ FAIL'}")
    
    if all([imports_ok, api_keys_ok, embedding_ok]):
        print("\n🎉 All tests passed! Your setup is ready.")
        print("🚀 You can now run: python run.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("📖 Check the README.md for setup instructions.")

if __name__ == "__main__":
    main() 