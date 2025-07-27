"""
Utility functions for the PDF Chat RAG application
"""

import os
import tempfile
from typing import List, Dict, Any
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

def validate_pdf_file(file_path: str) -> bool:
    """
    Validate if a file is a valid PDF
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        bool: True if valid PDF, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception as e:
        logger.error(f"Error validating PDF file {file_path}: {e}")
        return False

def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extract metadata from filename
    
    Args:
        filename: Name of the file
        
    Returns:
        Dict containing metadata
    """
    return {
        "source": filename,
        "file_type": "pdf",
        "filename": os.path.basename(filename)
    }

def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might cause issues
    text = text.replace('\x00', '')
    
    return text

def create_temp_file(content: bytes, suffix: str = '.pdf') -> str:
    """
    Create a temporary file with given content
    
    Args:
        content: File content as bytes
        suffix: File extension
        
    Returns:
        Path to the temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(content)
    temp_file.close()
    return temp_file.name

def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up temporary file
    
    Args:
        file_path: Path to the temporary file to delete
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary file {file_path}: {e}")

def format_source_info(document: Document) -> str:
    """
    Format source information for display
    
    Args:
        document: LangChain Document object
        
    Returns:
        Formatted source string
    """
    if not hasattr(document, 'metadata') or not document.metadata:
        return "Unknown source"
    
    metadata = document.metadata
    source = metadata.get('source', 'Unknown')
    page = metadata.get('page', 'N/A')
    
    # Extract filename from source path
    filename = os.path.basename(source) if source != 'Unknown' else 'Unknown'
    
    return f"Page {page} from {filename}"

def chunk_text_by_sentences(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Split text into chunks based on sentences
    
    Args:
        text: Text to split
        max_chunk_size: Maximum size of each chunk
        
    Returns:
        List of text chunks
    """
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return 0.0

def validate_api_keys() -> Dict[str, bool]:
    """
    Validate if required API keys are set
    
    Returns:
        Dict with API key validation status
    """
    google_key = os.getenv("GOOGLE_API_KEY")
    huggingface_key = os.getenv("HUGGINGFACE_API_KEY")
    
    return {
        "google_api_key": bool(google_key),
        "huggingface_api_key": bool(huggingface_key),
        "has_any_key": bool(google_key or huggingface_key)
    } 