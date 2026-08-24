from app.tools.column_classifier import classify_columns


def test_classifies_dimension_and_metric():
    rows = [{"region_name": "North", "customer_count": 89}, {"region_name": "South", "customer_count": 80}]
    period, dimensions, value = classify_columns(rows)
    assert period is None
    assert dimensions == ["region_name"]
    assert value == "customer_count"


def test_classifies_period_and_metric():
    rows = [{"month": "2026-06", "total_revenue": 100.0}, {"month": "2026-07", "total_revenue": 70.0}]
    period, dimensions, value = classify_columns(rows)
    assert period == "month"
    assert dimensions == []
    assert value == "total_revenue"


def test_classifies_period_dimension_and_metric_together():
    rows = [{"segment": "Enterprise", "region": "North", "month": "2026-06", "revenue": 10.0}]
    period, dimensions, value = classify_columns(rows)
    assert period == "month"
    assert set(dimensions) == {"segment", "region"}
    assert value == "revenue"


def test_id_columns_are_ignored():
    rows = [{"customer_id": 1, "region_id": 2, "revenue": 5.0}]
    period, dimensions, value = classify_columns(rows)
    assert dimensions == []
    assert value == "revenue"


def test_integer_month_column_recognized_as_period():
    rows = [{"month": 6, "revenue": 10.0}, {"month": 7, "revenue": 8.0}]
    period, dimensions, value = classify_columns(rows)
    assert period == "month"
    assert value == "revenue"


def test_high_cardinality_column_excluded_from_dimensions():
    rows = [{"name": f"company-{i}", "revenue": float(i)} for i in range(60)]
    period, dimensions, value = classify_columns(rows)
    assert "name" not in dimensions


def test_no_numeric_column_returns_no_value():
    rows = [{"region": "North", "status": "active"}]
    _period, _dimensions, value = classify_columns(rows)
    assert value is None


def test_empty_rows_returns_all_none():
    assert classify_columns([]) == (None, [], None)


def test_metric_name_hint_prefers_revenue_over_generic_numeric():
    rows = [{"category": "Software", "some_id_number": 42, "revenue": 100.0}]
    _period, _dimensions, value = classify_columns(rows)
    assert value == "revenue"
