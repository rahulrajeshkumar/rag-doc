import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from typing import List, Dict, Any
from src.rag_pipeline import RAGPipeline

def evaluate_pipeline(queries: List[str] = None, top_k: int = 5) -> Dict[str, Any]:
    """
    Run latency and retrieval quality benchmark on sample queries.
    """
    if queries is None:
        queries = [
            "What deep learning models are used for Land Use Land Cover dynamics in India?",
            "What is GeoFusion-ChangeNet and how does it encode features for change detection?",
            "How does feature-aware spectral learning improve temporal consistency in LULC mapping?",
            "What satellite remote sensing datasets were analyzed in the study?"
        ]

    rag = RAGPipeline()
    if not rag.loaded:
        rag.ingest_documents()

    results = []
    total_retrieval_time = 0.0
    total_query_time = 0.0

    print("\n[Evaluation] Starting RAG Pipeline Evaluation...")
    for q in queries:
        t0 = time.time()
        chunks = rag.retrieve(q, top_k=top_k)
        retrieval_time = time.time() - t0
        total_retrieval_time += retrieval_time

        t1 = time.time()
        res = rag.query(q, top_k=top_k)
        full_query_time = time.time() - t1
        total_query_time += full_query_time

        citations_count = len(res.get("citations", []))
        has_citations_in_answer = "[" in res["answer"] and "Page" in res["answer"]

        results.append({
            "query": q,
            "top_k_retrieved": len(chunks),
            "retrieval_time_sec": round(retrieval_time, 4),
            "total_query_time_sec": round(full_query_time, 4),
            "citations_found": citations_count,
            "has_intext_citations": has_citations_in_answer,
            "provider": res["provider"]
        })

    avg_retrieval_time = total_retrieval_time / len(queries)
    avg_query_time = total_query_time / len(queries)

    summary = {
        "total_queries_tested": len(queries),
        "avg_retrieval_time_sec": round(avg_retrieval_time, 4),
        "avg_total_query_time_sec": round(avg_query_time, 4),
        "query_details": results
    }

    return summary

if __name__ == "__main__":
    report = evaluate_pipeline()
    import json
    print("\n[Evaluation Summary]")
    print(json.dumps(report, indent=2))
