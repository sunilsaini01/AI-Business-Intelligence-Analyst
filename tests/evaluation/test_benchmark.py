import pytest

from app.evaluation.benchmark import load_benchmark
from app.evaluation.metrics import jaccard_similarity, overall_task_success


def test_load_benchmark_reads_seed_case():
    cases = load_benchmark("evaluation/datasets/benchmark.json")
    assert any(c["id"] == "bi-004" for c in cases)


def test_load_benchmark_has_all_five_phase8_cases():
    cases = load_benchmark("evaluation/datasets/benchmark.json")
    ids = {c["id"] for c in cases}
    assert ids == {"bi-001", "bi-002", "bi-004", "bi-005", "bi-006"}


def test_load_benchmark_every_case_has_a_type_discriminated_ground_truth():
    cases = load_benchmark("evaluation/datasets/benchmark.json")
    valid_types = {"category_values", "top_category", "period_comparison_with_contribution", "trend_bounds"}
    for case in cases:
        gt = case.get("ground_truth")
        assert gt is not None, f"{case['id']} missing ground_truth"
        assert gt["type"] in valid_types


def test_load_benchmark_diagnostic_case_matches_provided_ground_truth():
    cases = load_benchmark("evaluation/datasets/benchmark.json")
    bi_004 = next(c for c in cases if c["id"] == "bi-004")
    gt = bi_004["ground_truth"]
    assert gt["baseline_value"] == 161445.80
    assert gt["current_value"] == 150633.02
    assert gt["percentage_change"] == -6.7
    assert gt["dominant_contributor"]["group"] == "Enterprise"
    assert gt["dominant_contributor"]["change"] == -10610.84
    assert gt["dominant_contributor"]["pct_of_total_change"] == 74.4


def test_load_benchmark_rejects_a_case_missing_required_fields(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text('[{"question": "no id here"}]', encoding="utf-8")
    with pytest.raises(ValueError):
        load_benchmark(str(bad_file))


def test_jaccard_similarity_exact_match_is_one():
    assert jaccard_similarity({"sql_agent", "analysis_agent"}, {"sql_agent", "analysis_agent"}) == 1.0


def test_jaccard_similarity_partial_overlap():
    assert jaccard_similarity({"a", "b"}, {"a", "c"}) == 1 / 3


def test_overall_score_matches_manual_calc():
    deterministic = [1.0, 0.8, 0.9]
    judge = [4 / 5, 3 / 5]
    expected = 0.70 * (sum(deterministic) / 3) + 0.30 * (sum(judge) / 2)
    assert overall_task_success(deterministic, judge) == expected
