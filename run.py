#!/usr/bin/env python3
"""
Simple script to run the PDF Chat RAG application
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    try:
        # Check if requirements are installed
        print("🚀 Starting PDF Chat RAG Application...")
        
        # Check if .env file exists
        if not os.path.exists('.env'):
            print("⚠️  Warning: .env file not found!")
            print("📝 Please create a .env file with your API keys:")
            print("   Copy env_example.txt to .env and add your API keys")
            print("   Get your Google API key from: https://makersuite.google.com/app/apikey")
            print()
        
        # Run the Streamlit app
        print("🌐 Starting Streamlit server...")
        print("📱 The app will open in your browser at http://localhost:8501")
        print("🛑 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("💡 Make sure you have installed all requirements:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 