"""Deterministic API-layer tests only — status codes, routing, and that the
EvaluationRun row is created immediately. None of these wait for the
background benchmark run to actually finish (that drives the real graph,
which needs a live LLM) — same convention as tests/api/test_analysis.py.
"""

import pytest


@pytest.mark.asyncio
async def test_run_evaluation_returns_202_with_a_run_id(client):
    resp = await client.post(
        "/api/v1/evaluation/run",
        json={"label": "api-test-run", "dataset_path": "evaluation/datasets/benchmark.json"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "RUNNING"
    assert body["run_id"]


@pytest.mark.asyncio
async def test_run_appears_in_results_immediately_even_before_it_finishes(client):
    resp = await client.post(
        "/api/v1/evaluation/run",
        json={"label": "api-test-run-2", "dataset_path": "evaluation/datasets/benchmark.json"},
    )
    run_id = resp.json()["run_id"]

    results_resp = await client.get(f"/api/v1/evaluation/results?run_id={run_id}")
    assert results_resp.status_code == 200
    results = results_resp.json()
    assert len(results) == 1
    assert results[0]["run_id"] == run_id
    assert results[0]["label"] == "api-test-run-2"
