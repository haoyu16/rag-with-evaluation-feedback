from typing import Dict, List, Any, TypedDict, Optional, Union, Callable
from functools import wraps

from langchain.schema import Document
from langchain.chat_models.base import BaseChatModel
from langchain.vectorstores.base import VectorStore
from langchain.schema.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import CheckpointAt

from ..evaluation import RAGEvaluator, RetrievalMetrics, GenerationMetrics
from .prompts import (
    DEFAULT_RAG_SYSTEM_TEMPLATE,
    DEFAULT_RAG_QUESTION_TEMPLATE,
    CONCISE_RAG_SYSTEM_TEMPLATE,
    CONCISE_RAG_QUESTION_TEMPLATE,
    DETAILED_RAG_SYSTEM_TEMPLATE,
    DETAILED_RAG_QUESTION_TEMPLATE,
)

class RAGState(TypedDict):
    """State for the RAG pipeline."""
    query: str  # The user's query
    retrieved_documents: List[Document]  # Documents retrieved from the vector store
    context: str  # Combined context from retrieved documents
    response: str  # Generated response
    retrieval_metrics: Dict[str, Any]  # Metrics from retrieval evaluation
    generation_metrics: Dict[str, Any]  # Metrics from generation evaluation
    combined_score: float  # Combined evaluation score
    metadata: Dict[str, Any]  # Additional metadata and tracking information
    iteration: int  # Current iteration number
    max_iterations: int  # Maximum number of iterations allowed

def handle_errors(method: Callable) -> Callable:
    """Decorator for handling errors in node operations."""
    @wraps(method)
    def wrapper(self, state: RAGState, *args, **kwargs) -> RAGState:
        try:
            return method(self, state, *args, **kwargs)
        except Exception as e:
            error_type = method.__name__.replace("_", " ")
            state["metadata"][f"{error_type}_error"] = str(e)
            return state
    return wrapper

class BaseNode:
    """Base class for RAG pipeline nodes."""
    
    def adapt_parameters(self, state: RAGState) -> None:
        """Adapt node parameters based on evaluation feedback."""
        pass
    
    def process_iteration(self, state: RAGState) -> None:
        """Process iteration-specific logic."""
        if state["iteration"] > 0:
            self.adapt_parameters(state)

class RetrievalNode(BaseNode):
    """Node for document retrieval."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 4,
        min_relevance_score: float = 0.7,
    ):
        """Initialize retrieval node."""
        self.vector_store = vector_store
        self.initial_k = k
        self.k = k
        self.min_relevance_score = min_relevance_score
    
    def adapt_parameters(self, state: RAGState) -> None:
        """Adapt retrieval parameters based on evaluation feedback."""
        if not state.get("retrieval_metrics"):
            return
            
        avg_relevance = state["retrieval_metrics"].get("average_relevance", 0)
        precision = state["retrieval_metrics"].get("precision", 0)
        
        if avg_relevance < self.min_relevance_score:
            self.k = min(self.k + 2, self.initial_k + 4)
        elif precision < 0.5:
            self.k = max(self.k - 1, 2)
    
    @handle_errors
    def __call__(
        self,
        state: RAGState,
    ) -> RAGState:
        """Retrieve relevant documents for the query."""
        self.process_iteration(state)
        
        results = self.vector_store.similarity_search_with_score(
            state["query"],
            k=self.k,
        )
        
        docs = [doc for doc, _ in results]
        scores = [float(score) for _, score in results]
        
        state["retrieved_documents"] = docs
        state["context"] = "\n\n".join(doc.page_content for doc in docs)
        state["metadata"]["retrieval_scores"] = scores
        state["metadata"]["current_k"] = self.k
        
        return state

class GenerationNode(BaseNode):
    """Node for response generation."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        system_template: str = DEFAULT_RAG_SYSTEM_TEMPLATE,
        question_template: str = DEFAULT_RAG_QUESTION_TEMPLATE,
        min_context_relevance: float = 0.7,
        temperature: Optional[float] = None,
    ):
        """Initialize generation node."""
        self.llm = llm
        self.system_template = system_template
        self.question_template = question_template
        self.min_context_relevance = min_context_relevance
        self.temperature = temperature or (llm.temperature if hasattr(llm, "temperature") else 0.7)
        if hasattr(self.llm, "temperature"):
            self.llm.temperature = self.temperature
    
    def adapt_parameters(self, state: RAGState) -> None:
        """Adapt generation parameters based on evaluation feedback."""
        if not state.get("generation_metrics"):
            return
            
        factual_consistency = state["generation_metrics"].get("factual_consistency", 0)
        context_utilization = state["generation_metrics"].get("context_utilization", 0)
        
        if factual_consistency < 0.7 or context_utilization < 0.6:
            self.temperature = max(self.temperature - 0.1, 0.1)
            if hasattr(self.llm, "temperature"):
                self.llm.temperature = self.temperature
    
    @handle_errors
    def __call__(
        self,
        state: RAGState,
    ) -> RAGState:
        """Generate response using retrieved context."""
        self.process_iteration(state)
        
        if state.get("retrieval_metrics", {}).get("average_relevance", 1.0) < self.min_context_relevance:
            context_note = "\nNote: The available context might not be highly relevant to your question. "
            context_note += "I'll provide the best answer possible while noting any uncertainties."
            state["context"] += context_note
        
        system_message = SystemMessage(
            content=self.system_template.format(context=state["context"] or "No context available.")
        )
        human_message = HumanMessage(
            content=self.question_template.format(question=state["query"])
        )
        
        messages = [system_message, human_message]
        response = self.llm.generate([messages])
        
        state["response"] = response.generations[0][0].text
        state["metadata"]["generation_info"] = response.llm_output or {}
        state["metadata"]["current_temperature"] = self.temperature
        
        return state

class EvaluationNode(BaseNode):
    """Node for pipeline evaluation."""
    
    def __init__(
        self,
        evaluator: RAGEvaluator,
        max_iterations: int = 2,
        relevance_threshold: float = 0.7,
    ):
        """Initialize evaluation node."""
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.relevance_threshold = relevance_threshold
    
    def should_continue(self, state: RAGState) -> bool:
        """Determine if the pipeline should continue iterating."""
        if state["iteration"] >= state["max_iterations"]:
            return False
            
        retrieval_score = state.get("retrieval_metrics", {}).get("average_relevance", 0)
        generation_score = state.get("generation_metrics", {}).get("relevance_score", 0)
        
        return (
            (retrieval_score < self.relevance_threshold or generation_score < self.relevance_threshold)
            and state["iteration"] < self.max_iterations
        )
    
    @handle_errors
    def __call__(
        self,
        state: RAGState,
    ) -> RAGState:
        """Evaluate retrieval and generation performance."""
        eval_results = self.evaluator.evaluate_rag_pipeline(
            query=state["query"],
            response=state["response"],
            retrieved_documents=state["retrieved_documents"],
        )
        
        state["retrieval_metrics"] = eval_results["retrieval_metrics"].__dict__
        state["generation_metrics"] = eval_results["generation_metrics"].__dict__
        state["combined_score"] = eval_results["combined_score"]
        state["metadata"].update(eval_results["metadata"])
        
        state["iteration"] += 1
        
        return state

def create_rag_graph(
    vector_store: VectorStore,
    llm: BaseChatModel,
    evaluator: RAGEvaluator,
    k: int = 4,
    max_iterations: int = 2,
    min_relevance_score: float = 0.7,
    temperature: Optional[float] = None,
) -> StateGraph:
    """Create a RAG pipeline graph with feedback loops."""
    # Create workflow graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    retrieval_node = RetrievalNode(
        vector_store=vector_store,
        k=k,
        min_relevance_score=min_relevance_score,
    )
    generation_node = GenerationNode(
        llm=llm,
        temperature=temperature,
    )
    evaluation_node = EvaluationNode(
        evaluator=evaluator,
        max_iterations=max_iterations,
        relevance_threshold=min_relevance_score,
    )
    
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("generate", generation_node)
    workflow.add_node("evaluate", evaluation_node)
    
    # Define conditional edge
    def should_continue(state: RAGState) -> str:
        """Determine next node based on evaluation results."""
        if evaluation_node.should_continue(state):
            return "retrieve"  # Continue with feedback loop
        return END  # End pipeline
    
    # Add edges with feedback loop
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "retrieve": "retrieve",  # Feedback loop
            END: END,  # End pipeline
        }
    )
    
    # Add checkpoints for observability
    workflow.add_checkpoint("after_retrieval", CheckpointAt("retrieve"))
    workflow.add_checkpoint("after_generation", CheckpointAt("generate"))
    workflow.add_checkpoint("after_evaluation", CheckpointAt("evaluate"))
    
    return workflow.compile()

def run_rag_pipeline(
    query: str,
    graph: StateGraph,
    max_iterations: int = 2,
) -> RAGState:
    """Run the RAG pipeline on a query."""
    # Initialize state with proper typing
    initial_state: RAGState = {
        "query": query,
        "retrieved_documents": [],
        "context": "",
        "response": "",
        "retrieval_metrics": {},
        "generation_metrics": {},
        "combined_score": 0.0,
        "metadata": {},
        "iteration": 0,
        "max_iterations": max_iterations,
    }
    
    # Run pipeline
    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        # Handle pipeline errors gracefully
        final_state = initial_state.copy()
        final_state["metadata"]["pipeline_error"] = str(e)
    
    return final_state 