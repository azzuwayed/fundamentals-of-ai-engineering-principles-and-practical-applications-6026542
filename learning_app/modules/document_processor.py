"""
Document processing module for the learning app (Chapter 3).
Handles document extraction, cleaning, and chunking.
"""
import os
import re
from typing import Optional, Tuple, List, Dict
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader, DocxReader
import json


class DocumentProcessor:
    """Handles document extraction, cleaning, and chunking."""

    def __init__(self):
        self.pdf_reader = PDFReader()
        self.docx_reader = DocxReader()

    def extract_text(self, file_path: str) -> Tuple[str, Dict[str, any]]:
        """
        Extract text from document.

        Args:
            file_path: Path to document file

        Returns:
            Tuple of (extracted_text, metadata)
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        metadata = {
            "filename": os.path.basename(file_path),
            "file_type": ext,
            "file_size_kb": os.path.getsize(file_path) / 1024
        }

        try:
            if ext == '.pdf':
                documents = self.pdf_reader.load_data(file_path)
                text = "\n\n".join([doc.text for doc in documents])
            elif ext == '.docx':
                documents = self.docx_reader.load_data(file_path)
                text = "\n\n".join([doc.text for doc in documents])
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = json.dumps(data, indent=2)
            elif ext == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path)
                text = df.to_string()
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            metadata["char_count"] = len(text)
            metadata["word_count"] = len(text.split())

            return text, metadata

        except Exception as e:
            raise Exception(f"Error extracting text from {file_path}: {str(e)}")

    def clean_text(self, text: str, remove_extra_whitespace: bool = True,
                    remove_special_chars: bool = False) -> str:
        """
        Clean extracted text.

        Args:
            text: Text to clean
            remove_extra_whitespace: Remove extra whitespace
            remove_special_chars: Remove special characters

        Returns:
            Cleaned text
        """
        cleaned = text

        if remove_extra_whitespace:
            # Replace multiple spaces with single space
            cleaned = re.sub(r' +', ' ', cleaned)
            # Replace multiple newlines with double newline
            cleaned = re.sub(r'\n\n+', '\n\n', cleaned)
            # Strip leading/trailing whitespace
            cleaned = cleaned.strip()

        if remove_special_chars:
            # Keep alphanumeric, spaces, and basic punctuation
            cleaned = re.sub(r'[^\w\s.,!?;:()\-]', '', cleaned)

        return cleaned

    def chunk_text(self, text: str, chunk_size: int = 512,
                   chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Create LlamaIndex document
        document = Document(text=text)

        # Use SentenceSplitter for intelligent chunking
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        nodes = splitter.get_nodes_from_documents([document])

        chunks = []
        for i, node in enumerate(nodes):
            chunks.append({
                "chunk_id": i,
                "text": node.text,
                "char_count": len(node.text),
                "word_count": len(node.text.split())
            })

        return chunks

    def process_document(self, file_path: str, clean: bool = True,
                        chunk_size: Optional[int] = None) -> Tuple[str, Dict, List]:
        """
        Complete document processing pipeline.

        Args:
            file_path: Path to document
            clean: Whether to clean text
            chunk_size: If provided, chunk the text

        Returns:
            Tuple of (text, metadata, chunks)
        """
        # Extract
        text, metadata = self.extract_text(file_path)

        # Clean
        if clean:
            text = self.clean_text(text)

        # Chunk
        chunks = []
        if chunk_size:
            chunks = self.chunk_text(text, chunk_size=chunk_size)
            metadata["num_chunks"] = len(chunks)

        return text, metadata, chunks


def load_preloaded_document(filename: str, data_dir: str = "data/preloaded") -> str:
    """
    Load a pre-loaded sample document.

    Args:
        filename: Name of the file to load
        data_dir: Directory containing pre-loaded files

    Returns:
        Full path to the document
    """
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pre-loaded document not found: {filename}")
    return file_path


def get_preloaded_documents(data_dir: str = "data/preloaded") -> List[str]:
    """
    Get list of available pre-loaded documents.

    Args:
        data_dir: Directory containing pre-loaded files

    Returns:
        List of filenames
    """
    if not os.path.exists(data_dir):
        return []

    files = []
    for filename in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, filename)):
            _, ext = os.path.splitext(filename)
            if ext.lower() in ['.pdf', '.docx', '.txt', '.json', '.csv']:
                files.append(filename)

    return sorted(files)
