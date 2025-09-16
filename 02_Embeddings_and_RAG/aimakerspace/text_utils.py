"""
Text processing utilities for document loading and text splitting.

This module provides classes for loading various document types (text files, PDFs)
and splitting them into manageable chunks for RAG applications.
"""

import os
from typing import List, Union
import pdfplumber


class BaseDocumentLoader:
    """Base class for document loaders."""
    
    def __init__(self, path: str, encoding: str = "utf-8"):
        """
        Initialize the document loader.
        
        Args:
            path: Path to file or directory
            encoding: Text encoding for text files
        """
        self.documents = []
        self.path = path
        self.encoding = encoding
    
    def load_documents(self) -> List[str]:
        """Load documents and return them as a list of strings."""
        self.load()
        return self.documents
    
    def load(self):
        """Load documents. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load method")


class PDFFileLoader(BaseDocumentLoader):
    """Loader for PDF files using pdfplumber for robust text extraction."""
    
    def __init__(self, path: str):
        """
        Initialize PDF file loader.
        
        Args:
            path: Path to PDF file or directory containing PDF files
        """
        super().__init__(path)
    
    def load(self):
        """Load PDF files from path."""
        if os.path.isdir(self.path):
            self._load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self._load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .pdf file."
            )
    
    def _load_file(self):
        """Load text from a single PDF file."""
        try:
            with pdfplumber.open(self.path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():  # Only add non-empty documents
                    self.documents.append(text.strip())
        except Exception as e:
            raise ValueError(f"Failed to load PDF file {self.path}: {str(e)}")
    
    def _load_directory(self):
        """Load all PDF files from a directory."""
        pdf_files = []
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {self.path}")
        
        for pdf_file in pdf_files:
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    if text.strip():  # Only add non-empty documents
                        self.documents.append(text.strip())
            except Exception as e:
                print(f"Warning: Failed to load PDF {pdf_file}: {str(e)}")
                continue


class TextFileLoader(BaseDocumentLoader):
    """Loader for text files only."""
    
    def __init__(self, path: str, encoding: str = "utf-8"):
        """
        Initialize text file loader.
        
        Args:
            path: Path to file or directory
            encoding: Text encoding for text files
        """
        super().__init__(path, encoding)
    
    def load(self):
        """Load text files from path."""
        if os.path.isdir(self.path):
            self._load_directory()
        elif os.path.isfile(self.path):
            if self.path.endswith(".txt"):
                self._load_file()
            else:
                raise ValueError(
                    "Provided file is not a .txt file. Use PDFFileLoader for PDF files."
                )
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )
    
    def _load_file(self):
        """Load a single text file."""
        try:
            with open(self.path, "r", encoding=self.encoding) as f:
                content = f.read()
                if content.strip():  # Only add non-empty documents
                    self.documents.append(content)
        except Exception as e:
            raise ValueError(f"Failed to load text file {self.path}: {str(e)}")
    
    def _load_directory(self):
        """Load all .txt files from a directory."""
        text_files = []
        
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    text_files.append(file_path)
        
        if not text_files:
            raise ValueError(f"No .txt files found in directory: {self.path}")
        
        # Load text files
        for text_file in text_files:
            try:
                with open(text_file, "r", encoding=self.encoding) as f:
                    content = f.read()
                    if content.strip():
                        self.documents.append(content)
            except Exception as e:
                print(f"Warning: Failed to load text file {text_file}: {str(e)}")
                continue


class CharacterTextSplitter:
    """Splits text into overlapping chunks for RAG applications."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        if chunk_size <= chunk_overlap:
            raise ValueError("Chunk size must be greater than chunk overlap")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split(self, text: str) -> List[str]:
        """
        Split a single text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop if chunk_overlap >= chunk_size
            if start >= len(text):
                break
        
        return chunks
    
    def split_texts(self, texts: List[str]) -> List[str]:
        """
        Split multiple texts into chunks.
        
        Args:
            texts: List of texts to split
            
        Returns:
            List of all text chunks
        """
        all_chunks = []
        for text in texts:
            chunks = self.split(text)
            all_chunks.extend(chunks)
        return all_chunks




if __name__ == "__main__":
    # Example usage
    print("Testing text utilities...")
    
    # Test text file loading
    try:
        text_loader = TextFileLoader("data/PMarcaBlogs.txt")
        text_docs = text_loader.load_documents()
        print(f"✅ Text file loaded: {len(text_docs)} document(s)")
    except Exception as e:
        print(f"❌ Text loading failed: {e}")
    
    # Test PDF file loading
    try:
        pdf_loader = PDFFileLoader("data/2A_Forbes.pdf")
        pdf_docs = pdf_loader.load_documents()
        print(f"✅ PDF file loaded: {len(pdf_docs)} document(s)")
    except Exception as e:
        print(f"❌ PDF loading failed: {e}")
    
    
    # Test text splitting
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    test_text = "This is a test document. " * 50  # Create a longer text
    chunks = splitter.split(test_text)
    print(f"✅ Text splitting works: {len(chunks)} chunks")
    print(f"   First chunk: {chunks[0][:100]}...")