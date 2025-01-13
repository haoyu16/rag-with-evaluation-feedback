from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.evaluation.retrieval import (
    ContextRelevancyEvaluator,
    QueryRelevanceEvaluator,
)
from langchain.schema import Document
from langchain.callbacks import StreamingStdOutCallbackHandler

from .graph_rag import create_rag_graph, RAGState, run_rag_pipeline

def create_example_documents() -> List[Document]:
    """Create example documents for testing."""
    return [
        Document(
            page_content="RAG (Retrieval-Augmented Generation) is a technique that combines retrieval and generation.\n"
                        "It helps ground language models with external knowledge sources.\n"
                        "This approach reduces hallucination and improves response accuracy.",
            metadata={"source": "rag_intro.txt"},
        ),
        Document(
            page_content="Key benefits of RAG:\n"
                        "1. Improved accuracy through factual grounding\n"
                        "2. Reduced hallucination by using retrieved context\n"
                        "3. Dynamic knowledge updates without model retraining\n"
                        "4. Better transparency and explainability",
            metadata={"source": "rag_benefits.txt"},
        ),
        Document(
            page_content="RAG implementation steps:\n"
                        "1. Index documents in a vector store\n"
                        "2. Retrieve relevant context for each query\n"
                        "3. Provide context to the language model\n"
                        "4. Generate response using the context",
            metadata={"source": "rag_implementation.txt"},
        ),
    ]

def setup_vector_store() -> Chroma:
    """Set up and populate vector store."""
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="rag_example",
        embedding_function=embeddings,
        persist_directory="./data/chroma_db",
    )
    
    # Add documents
    documents = create_example_documents()
    vector_store.add_documents(documents)
    
    return vector_store

def print_state(state: RAGState, stage: str = ""):
    """Print pipeline state in a formatted way."""
    if stage:
        print(f"\n=== {stage} ===")
    
    if state["iteration"] > 0:
        print(f"\nIteration {state['iteration']} of {state['max_iterations']}")
    
    if state["retrieved_documents"]:
        print("\nRetrieved Documents:")
        for i, doc in enumerate(state["retrieved_documents"], 1):
            print(f"\n[Document {i}]")
            print(f"Content: {doc.page_content[:200]}...")
            if "retrieval_scores" in state["metadata"]:
                print(f"Score: {state['metadata']['retrieval_scores'][i-1]:.4f}")
        if "current_k" in state["metadata"]:
            print(f"\nCurrent k: {state['metadata']['current_k']}")
    
    if state["response"]:
        print("\nGenerated Response:")
        print(state["response"])
        if "current_temperature" in state["metadata"]:
            print(f"\nCurrent Temperature: {state['metadata']['current_temperature']:.2f}")
    
    if state["retrieval_metrics"]:
        print("\nRetrieval Metrics:")
        print(f"Average Relevance: {state['retrieval_metrics']['average_relevance']:.4f}")
        print(f"Precision: {state['retrieval_metrics']['precision']:.4f}")
        print(f"Context Precision: {state['retrieval_metrics']['context_precision']:.4f}")
        if state["retrieval_metrics"].get("metadata"):
            print("\nRetrieval Metadata:")
            for key, value in state["retrieval_metrics"]["metadata"].items():
                print(f"{key}: {value}")
    
    if state["generation_metrics"]:
        print("\nGeneration Metrics:")
        print(f"Relevance Score: {state['generation_metrics']['relevance_score']:.4f}")
        print(f"Factual Consistency: {state['generation_metrics']['factual_consistency']:.4f}")
        print(f"Answer Completeness: {state['generation_metrics']['answer_completeness']:.4f}")
        print(f"Context Utilization: {state['generation_metrics']['context_utilization']:.4f}")
        if state["generation_metrics"].get("metadata"):
            print("\nGeneration Metadata:")
            for key, value in state["generation_metrics"]["metadata"].items():
                print(f"{key}: {value}")
    
    if state["combined_score"]:
        print(f"\nCombined Score (40% retrieval, 60% generation): {state['combined_score']:.4f}")

def main():
    """Run example RAG pipeline with LangGraph."""
    print("Setting up RAG pipeline...")
    
    # Initialize components
    vector_store = setup_vector_store()
    llm = ChatOpenAI(
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    
    # Create pipeline graph with feedback
    graph = create_rag_graph(
        vector_store=vector_store,
        llm=llm,
        retrieval_evaluator=ContextRelevancyEvaluator(),
        generation_evaluator=QueryRelevanceEvaluator(),
        k=2,
        max_iterations=2,  # Allow up to 2 feedback iterations
    )
    
    # Example queries
    queries = [
        "What is RAG and what are its benefits?",
        "How do you implement RAG?",
        "How does RAG help with hallucination?",
    ]
    
    # Run queries
    for i, query in enumerate(queries, 1):
        print(f"\n\nQuery {i}: {query}")
        print("=" * 50)
        
        # Run pipeline with feedback
        state = run_rag_pipeline(
            query=query,
            graph=graph,
            max_iterations=2,
        )
        
        # Print final results
        print("\n=== Final Results ===")
        print_state(state)
        
        # Print improvement summary if multiple iterations occurred
        if state["iteration"] > 1:
            print("\n=== Improvement Summary ===")
            print(f"Number of iterations: {state['iteration']}")
            print(f"Final k value: {state['metadata']['current_k']}")
            print(f"Final temperature: {state['metadata']['current_temperature']:.2f}")
            print(f"Final combined score: {state['combined_score']:.4f}")

if __name__ == "__main__":
    main() 