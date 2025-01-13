from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.evaluation import StringEvaluator
from langchain.evaluation.criteria import (
    CriteriaEvalChain,
    LabeledCriteriaEvalChain,
)
from langchain.evaluation.retrievers import (
    ContextRelevancyEvaluator,
    QueryRelevanceEvaluator,
    ContextQueryRelevanceEvaluator,
)

@dataclass
class BaseEvaluationMetrics:
    """Base class for evaluation metrics."""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalMetrics(BaseEvaluationMetrics):
    """Metrics for evaluating retrieval performance."""
    relevance_scores: List[float]
    average_relevance: float
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    context_precision: float = 0.0

@dataclass
class GenerationMetrics(BaseEvaluationMetrics):
    """Metrics for evaluating generation performance."""
    relevance_score: float
    factual_consistency: float
    answer_completeness: float
    context_utilization: float

class RetrievalEvaluator:
    """Evaluates retrieval performance."""
    
    def __init__(
        self,
        relevancy_evaluator: StringEvaluator,
        relevance_threshold: float = 0.7,
        strict_mode: bool = False,
    ):
        """Initialize retrieval evaluator."""
        self.relevancy_evaluator = relevancy_evaluator
        self.relevance_threshold = relevance_threshold
        self.strict_mode = strict_mode
    
    def _evaluate_single_document(self, query: str, document: Document) -> float:
        """Evaluate relevance of a single document."""
        return self.relevancy_evaluator.evaluate_strings(
            prediction=document.page_content,
            reference=query,
        ).score
    
    def evaluate_relevance(
        self,
        query: str,
        documents: List[Document],
    ) -> List[float]:
        """Evaluate relevance of retrieved documents."""
        if not documents:
            return []
            
        return [self._evaluate_single_document(query, doc) for doc in documents]
    
    def calculate_precision_recall(
        self,
        relevance_scores: List[float],
        ground_truth: Optional[List[Document]] = None,
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        if not relevance_scores:
            return 0.0, 0.0, 0.0
            
        if ground_truth is None:
            # Without ground truth, use relevance threshold
            relevant_count = sum(1 for score in relevance_scores if score >= self.relevance_threshold)
            precision = relevant_count / len(relevance_scores)
            recall = 1.0 if relevant_count > 0 else 0.0
        else:
            # With ground truth, compare retrieved docs to ground truth
            total_relevant = len(ground_truth)
            if self.strict_mode:
                retrieved_relevant = sum(1 for score in relevance_scores if score >= self.relevance_threshold)
                precision = retrieved_relevant / len(relevance_scores)
                recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0
            else:
                precision = sum(relevance_scores) / len(relevance_scores)
                recall = sum(relevance_scores) / total_relevant if total_relevant > 0 else 0.0
        
        # Calculate F1 score
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
            
        return precision, recall, f1_score
    
    def evaluate_context_precision(
        self,
        query: str,
        documents: List[Document],
    ) -> float:
        """Evaluate how precisely the retrieved context matches the query."""
        if not documents:
            return 0.0
            
        combined_context = "\n\n".join(doc.page_content for doc in documents)
        return self.relevancy_evaluator.evaluate_strings(
            prediction=combined_context,
            reference=query,
        ).score
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_documents: List[Document],
        ground_truth: Optional[List[Document]] = None,
    ) -> RetrievalMetrics:
        """Evaluate retrieval performance."""
        relevance_scores = self.evaluate_relevance(query, retrieved_documents)
        average_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        precision, recall, f1_score = self.calculate_precision_recall(relevance_scores, ground_truth)
        context_precision = self.evaluate_context_precision(query, retrieved_documents)
        
        return RetrievalMetrics(
            relevance_scores=relevance_scores,
            average_relevance=average_relevance,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            context_precision=context_precision,
            metadata={
                "num_documents": len(retrieved_documents),
                "has_ground_truth": ground_truth is not None,
                "relevance_threshold": self.relevance_threshold,
                "strict_mode": self.strict_mode,
            },
        )

class RAGEvaluator:
    """Evaluator for RAG pipeline performance."""
    
    EVAL_CRITERIA = {
        "relevance": "The response directly addresses the question and provides relevant information.",
        "factual_consistency": "The response is factually consistent with the provided context and doesn't make claims unsupported by the context.",
        "completeness": "The response covers all important aspects of the question using the available context.",
        "context_utilization": "The response effectively uses the provided context to support its claims.",
    }
    
    def __init__(
        self,
        relevancy_evaluator: Optional[StringEvaluator] = None,
        generation_evaluator: Optional[ChatOpenAI] = None,
        relevance_threshold: float = 0.7,
        strict_mode: bool = False,
    ):
        """Initialize the RAG evaluator."""
        # Initialize evaluators
        self.retrieval_evaluator = RetrievalEvaluator(
            relevancy_evaluator=relevancy_evaluator or ContextRelevancyEvaluator(),
            relevance_threshold=relevance_threshold,
            strict_mode=strict_mode,
        )
        self.generation_evaluator = generation_evaluator or ChatOpenAI(temperature=0)
        
        # Initialize evaluation chains
        self.criteria_chain = CriteriaEvalChain.from_llm(
            llm=self.generation_evaluator,
            criteria=self.EVAL_CRITERIA,
        )
        self.labeled_criteria_chain = LabeledCriteriaEvalChain.from_llm(
            llm=self.generation_evaluator,
            criteria=self.EVAL_CRITERIA,
        )
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_documents: List[Document],
        ground_truth: Optional[List[Document]] = None,
    ) -> RetrievalMetrics:
        """Evaluate retrieval performance."""
        return self.retrieval_evaluator.evaluate_retrieval(
            query=query,
            retrieved_documents=retrieved_documents,
            ground_truth=ground_truth,
        )
    
    def evaluate_generation(
        self,
        query: str,
        response: str,
        context_docs: List[Document],
        reference_answer: Optional[str] = None,
    ) -> GenerationMetrics:
        """Evaluate the quality of a generated response."""
        context = "\n\n".join(doc.page_content for doc in context_docs)
        
        # Evaluate using criteria chain
        eval_result = self.criteria_chain.evaluate_strings(
            prediction=response,
            input=query,
            reference=reference_answer,
            context=context,
        )
        
        # Get detailed feedback if reference is available
        feedback = {}
        if reference_answer:
            labeled_eval = self.labeled_criteria_chain.evaluate_strings(
                prediction=response,
                input=query,
                reference=reference_answer,
                context=context,
            )
            feedback["detailed_feedback"] = labeled_eval.get("reasoning", "")
        
        scores = eval_result.get("scores", {})
        return GenerationMetrics(
            relevance_score=float(scores.get("relevance", 0.0)),
            factual_consistency=float(scores.get("factual_consistency", 0.0)),
            answer_completeness=float(scores.get("completeness", 0.0)),
            context_utilization=float(scores.get("context_utilization", 0.0)),
            metadata={
                "raw_scores": scores,
                "feedback": feedback,
                "num_context_docs": len(context_docs),
                "context_length": len(context),
            },
        )
    
    def evaluate_rag_pipeline(
        self,
        query: str,
        response: str,
        retrieved_documents: Union[List[Document], List[tuple[Document, float]]],
        ground_truth_texts: Optional[List[str]] = None,
        reference_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate both retrieval and generation components."""
        # Extract documents
        docs = [doc for doc, _ in retrieved_documents] if isinstance(retrieved_documents[0], tuple) else retrieved_documents
        
        # Convert ground truth texts to documents if provided
        ground_truth = None if not ground_truth_texts else [
            Document(page_content=text) for text in ground_truth_texts
        ]
        
        # Evaluate components
        retrieval_metrics = self.evaluate_retrieval(query, docs, ground_truth)
        generation_metrics = self.evaluate_generation(query, response, docs, reference_answer)
        
        # Calculate combined score
        combined_score = (
            retrieval_metrics.average_relevance +
            generation_metrics.relevance_score +
            generation_metrics.factual_consistency
        ) / 3
        
        return {
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "combined_score": combined_score,
            "metadata": {
                "num_documents": len(docs),
                "has_ground_truth": ground_truth_texts is not None,
                "has_reference": reference_answer is not None,
            },
        }
    
    def evaluate_batch_pipeline(
        self,
        queries: List[str],
        responses: List[str],
        retrieved_docs_list: List[Union[List[Document], List[tuple[Document, float]]]],
        ground_truth_texts: Optional[List[List[str]]] = None,
        reference_answers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate pipeline performance across multiple queries."""
        all_metrics = [
            self.evaluate_rag_pipeline(
                query=query,
                response=response,
                retrieved_documents=retrieved_docs,
                ground_truth_texts=ground_truth_texts[i] if ground_truth_texts else None,
                reference_answer=reference_answers[i] if reference_answers else None,
            )
            for i, (query, response, retrieved_docs) in enumerate(zip(queries, responses, retrieved_docs_list))
        ]
        
        # Aggregate metrics
        num_queries = len(queries)
        return {
            "average_combined_score": sum(m["combined_score"] for m in all_metrics) / num_queries,
            "average_retrieval_relevance": sum(
                m["retrieval_metrics"].average_relevance for m in all_metrics
            ) / num_queries,
            "average_generation_relevance": sum(
                m["generation_metrics"].relevance_score for m in all_metrics
            ) / num_queries,
            "average_factual_consistency": sum(
                m["generation_metrics"].factual_consistency for m in all_metrics
            ) / num_queries,
            "average_answer_completeness": sum(
                m["generation_metrics"].answer_completeness for m in all_metrics
            ) / num_queries,
            "queries_evaluated": num_queries,
            "per_query_metrics": [
                {
                    "query": query,
                    "combined_score": metrics["combined_score"],
                    "retrieval_metrics": metrics["retrieval_metrics"],
                    "generation_metrics": metrics["generation_metrics"],
                }
                for query, metrics in zip(queries, all_metrics)
            ],
        } 