import plotly.graph_objects as go

from chart_builder import build_chart_figure, validate_chart


def _chart(**overrides) -> dict:
    base = {
        "chart_type": "bar",
        "title": "Revenue by Region",
        "storage_path": "",
        "spec_json": {
            "title": "Revenue by Region",
            "x_axis": "region",
            "y_axis": "revenue",
            "data": [{"label": "North", "value": 100.0}, {"label": "South", "value": 80.0}],
            "limitations": [],
        },
    }
    base.update(overrides)
    return base


# --- validate_chart ----------------------------------------------------------


def test_valid_chart_passes_validation():
    ok, reason = validate_chart(_chart())
    assert ok is True
    assert reason is None


def test_unknown_chart_type_rejected():
    ok, reason = validate_chart(_chart(chart_type="pie_of_the_future"))
    assert ok is False
    assert "chart_type" in reason


def test_missing_spec_json_rejected():
    ok, reason = validate_chart(_chart(spec_json=None))
    assert ok is False


def test_nonempty_storage_path_rejected_never_treated_as_a_file_path():
    ok, reason = validate_chart(_chart(storage_path="/etc/passwd"))
    assert ok is False
    assert "storage_path" in reason


def test_not_a_dict_rejected_gracefully():
    ok, reason = validate_chart("not a chart")  # type: ignore[arg-type]
    assert ok is False


# --- build_chart_figure -------------------------------------------------------


def test_bar_chart_builds_a_plotly_figure():
    result = build_chart_figure(_chart(chart_type="bar"))
    assert result["kind"] == "figure"
    assert isinstance(result["figure"], go.Figure)


def test_horizontal_bar_chart_builds_a_plotly_figure():
    result = build_chart_figure(_chart(chart_type="horizontal_bar"))
    assert result["kind"] == "figure"


def test_line_and_area_charts_build_a_plotly_figure():
    for chart_type in ("line", "area"):
        result = build_chart_figure(_chart(chart_type=chart_type))
        assert result["kind"] == "figure"


def test_pie_chart_builds_a_plotly_figure():
    result = build_chart_figure(_chart(chart_type="pie"))
    assert result["kind"] == "figure"


def test_scatter_chart_uses_x_y_points():
    scatter = _chart(
        chart_type="scatter",
        spec_json={"title": "t", "data": [{"x": 1, "y": 2}, {"x": 3, "y": 4}], "limitations": []},
    )
    result = build_chart_figure(scatter)
    assert result["kind"] == "figure"


def test_kpi_chart_returns_a_single_metric_not_a_plotly_figure():
    kpi = _chart(chart_type="kpi", spec_json={"title": "Total", "data": [{"label": "Total", "value": 42}]})
    result = build_chart_figure(kpi)
    assert result == {"kind": "kpi", "label": "Total", "value": 42}


def test_table_chart_returns_raw_rows_not_a_plotly_figure():
    table = _chart(
        chart_type="table",
        spec_json={"title": "Raw rows", "data": [{"a": 1, "b": 2}], "limitations": []},
    )
    result = build_chart_figure(table)
    assert result["kind"] == "table"
    assert result["data"] == [{"a": 1, "b": 2}]


def test_invalid_chart_returns_none_never_raises():
    assert build_chart_figure(_chart(chart_type="not_a_real_type")) is None
    assert build_chart_figure(_chart(storage_path="/tmp/x")) is None


def test_malformed_data_points_return_none_never_raise():
    """A data point missing the expected key ('value') must degrade this
    ONE chart, never raise out to the caller."""
    broken = _chart(chart_type="bar", spec_json={"title": "t", "data": [{"label": "North"}], "limitations": []})
    assert build_chart_figure(broken) is None


def test_empty_data_list_does_not_crash():
    empty = _chart(chart_type="bar", spec_json={"title": "t", "data": [], "limitations": []})
    result = build_chart_figure(empty)
    assert result["kind"] == "figure"  # an empty bar chart is still a valid (empty) figure
