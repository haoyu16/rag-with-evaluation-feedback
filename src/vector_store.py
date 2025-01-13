from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain_community.vectorstores import Chroma

class VectorStoreManager:
    """Manages vector store operations."""
    
    def __init__(
        self,
        vector_store_cls: Optional[Type[VectorStore]] = None,
        persist_directory: str = "./vector_store",
        embedding_function: Optional[Embeddings] = None,
        collection_name: str = "documents",
        **vector_store_kwargs: Any,
    ):
        """Initialize the vector store manager."""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.embedding_function = embedding_function or OpenAIEmbeddings()
        self.collection_name = collection_name
        self.vector_store_cls = vector_store_cls or Chroma
        self.vector_store_kwargs = vector_store_kwargs
        
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> VectorStore:
        """Initialize the vector store with appropriate arguments."""
        init_kwargs = {
            "embedding_function": self.embedding_function,
            **self.vector_store_kwargs
        }
        
        if issubclass(self.vector_store_cls, Chroma):
            init_kwargs.update({
                "persist_directory": str(self.persist_directory),
                "collection_name": self.collection_name,
            })
        
        return self.vector_store_cls(**init_kwargs)
    
    def _prepare_search_kwargs(
        self,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare search kwargs based on vector store capabilities."""
        search_kwargs = {"k": k}
        if filter is not None and hasattr(self.vector_store, "similarity_search_with_filter"):
            search_kwargs["filter"] = filter
        return search_kwargs
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add documents to the vector store."""
        if hasattr(self.vector_store, "add_documents"):
            self.vector_store.add_documents(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )
        else:
            # Fallback for vector stores that only support from_documents
            self.vector_store = self.vector_store_cls.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                ids=ids,
                metadatas=metadatas,
                **self.vector_store_kwargs
            )
        
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Perform similarity search."""
        search_kwargs = self._prepare_search_kwargs(k, filter)
        if filter is not None and hasattr(self.vector_store, "similarity_search_with_filter"):
            return self.vector_store.similarity_search_with_filter(query, **search_kwargs)
        return self.vector_store.similarity_search(query, **search_kwargs)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """Perform similarity search with relevance scores."""
        search_kwargs = self._prepare_search_kwargs(k, filter)
        if hasattr(self.vector_store, "similarity_search_with_score"):
            return self.vector_store.similarity_search_with_score(query, **search_kwargs)
        else:
            # Fallback for vector stores that don't support scores
            documents = self.similarity_search(query, k=k, filter=filter)
            return [(doc, 1.0) for doc in documents]
    
    def delete_collection(self):
        """Delete the entire collection if supported by the vector store."""
        if hasattr(self.vector_store, "delete_collection"):
            self.vector_store.delete_collection()
            self.vector_store = self._initialize_vector_store() 