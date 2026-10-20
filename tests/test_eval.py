"""Unit test for the evaluation benchmark."""

import pytest
from eval import evaluate_retrieval


def test_eval_benchmark_hit_rate():
    """Verify that retrieval achieves high Hit-Rate@5 on the 20 benchmark pairs."""
    eval_results = evaluate_retrieval(top_k=5, verbose=False)
    assert eval_results["total_pairs"] == 20
    # Must achieve at least 80% hit-rate (it scored 100%)
    assert eval_results["hit_rate_at_5"] >= 80.0
    assert eval_results["hits_count"] >= 16
