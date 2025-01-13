from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Union

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.document_loaders import (
    PDFMinerLoader,
    PythonLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from .vector_store import VectorStoreManager

class DocumentIngester:
    """Handles document ingestion with support for multiple file types."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        text_splitter: Optional[TextSplitter] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable = len,
        metadata_extractors: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize the document ingester."""
        self.vector_store = vector_store
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )
        
        self.loader_map = {
            ".txt": TextLoader,
            ".pdf": PDFMinerLoader,
            ".md": UnstructuredMarkdownLoader,
            ".py": PythonLoader,
            ".sh": TextLoader,
            ".bash": TextLoader,
        }
        
        self.metadata_extractors = metadata_extractors or {}
    
    def _get_loader(self, file_path: Union[str, Path]) -> Callable:
        """Get the appropriate loader for a file."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix not in self.loader_map:
            raise ValueError(f"Unsupported file type: {suffix}")
        return self.loader_map[suffix]
    
    def _extract_metadata(
        self,
        file_path: Union[str, Path],
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract and combine metadata for a document."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        metadata = {
            "source": str(path),
            "file_type": suffix.lstrip("."),
            "file_name": path.name,
        }
        
        # Extract additional metadata if available
        if suffix in self.metadata_extractors:
            try:
                extracted = self.metadata_extractors[suffix](str(path))
                metadata.update(extracted)
            except Exception as e:
                print(f"Error extracting metadata from {path}: {str(e)}")
        
        # Add custom metadata if provided
        if custom_metadata:
            metadata.update(custom_metadata)
            
        return metadata
    
    def ingest_document(
        self,
        file_path: str,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Ingest a single document, extract metadata, and split into chunks."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get loader and load document
        loader = self._get_loader(path)(str(path))
        documents = loader.load()
        
        # Extract and add metadata
        metadata = self._extract_metadata(path, custom_metadata)
        for doc in documents:
            doc.metadata.update(metadata)
        
        return self.text_splitter.split_documents(documents)
    
    def ingest_documents(
        self,
        paths: List[str],
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Ingest multiple documents with metadata extraction."""
        all_documents = []
        for path in paths:
            try:
                doc_chunks = self.ingest_document(path, custom_metadata)
                all_documents.extend(doc_chunks)
            except Exception as e:
                print(f"Error ingesting {path}: {str(e)}")
        return all_documents
    
    def add_metadata_extractor(
        self,
        file_extension: str,
        extractor: Callable[[str], Dict[str, Any]],
    ):
        """Add a metadata extractor for a specific file type."""
        self.metadata_extractors[file_extension.lower()] = extractor
    
    def set_text_splitter(self, text_splitter: TextSplitter):
        """Update the text splitter."""
        self.text_splitter = text_splitter
    
    def ingest_and_store(
        self,
        paths: List[str],
        custom_metadata: Optional[Dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
        additional_metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """Ingest documents and add them directly to the vector store."""
        if self.vector_store is None:
            raise ValueError("No vector store configured. Please provide a vector store during initialization.")
        
        # First ingest the documents
        documents = self.ingest_documents(paths, custom_metadata)
        
        # Update document metadatas if provided
        if additional_metadatas:
            for doc, metadata in zip(documents, additional_metadatas):
                doc.metadata.update(metadata)
        
        # Add to vector store
        self.vector_store.add_documents(documents, ids=ids) 