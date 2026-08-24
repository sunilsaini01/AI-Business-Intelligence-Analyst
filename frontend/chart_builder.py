"""Pure chart-spec validation + Plotly figure construction (Phase 12) for
the chart records served by GET /analysis/{id}/charts (Phase 7,
app/tools/chart_selector.py). No `streamlit` import, so this is testable
with plain pytest — only frontend/components/charts.py (a thin wrapper)
actually calls into Streamlit to draw the result.

The chart TYPE was already decided server-side; this module never
re-derives which chart type fits the data (Sec "Phase 7 Visualization" —
do not infer chart types from raw analysis data), it only maps the given
`chart_type` to a Plotly figure, or reports the record as unusable.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

_KNOWN_CHART_TYPES = {"kpi", "table", "bar", "horizontal_bar", "line", "area", "pie", "scatter"}


def validate_chart(chart: dict[str, Any]) -> tuple[bool, str | None]:
    """Returns (ok, reason). Never raises — a malformed chart record must
    degrade the ONE chart to "Visualization unavailable.", never take down
    the rest of the report.
    """
    if not isinstance(chart, dict):
        return False, "Chart record is not an object."
    chart_type = chart.get("chart_type")
    if chart_type not in _KNOWN_CHART_TYPES:
        return False, f"Unknown chart_type: {chart_type!r}."
    spec = chart.get("spec_json")
    if not isinstance(spec, dict):
        return False, "Missing spec_json."
    # `storage_path` is a Phase 0-era vestige of the Chart DB model/API
    # schema (built assuming a rendered file) — Plotly specs travel as JSON
    # instead (Sec 2 tech decision, app/graph/state.py::ChartRecord's own
    # docstring), and no file-serving endpoint exists at all. A nonempty
    # value here is never treated as a local path to open — there is no
    # contract under which that would be safe or correct (Sec "Do NOT
    # access arbitrary filesystem paths returned by the API").
    if chart.get("storage_path"):
        return False, "storage_path is set but the API does not serve files by path; ignoring it."
    return True, None


def build_chart_figure(chart: dict[str, Any]) -> dict[str, Any] | None:
    """Returns one of:
    - {"kind": "figure", "figure": <plotly.graph_objects.Figure>} for the
      Plotly-rendered types (bar/horizontal_bar/line/area/pie/scatter)
    - {"kind": "kpi", "label": ..., "value": ...} for a single-metric card
    - {"kind": "table", "title": ..., "data": [...]} for tabular fallback
    - None if the record is invalid or its data points are malformed

    Wrapping everything in a dict (rather than returning bare Figures)
    keeps this function's return type uniform and keeps the plotly.graph_objects
    import contained to this one module — frontend/components/charts.py just
    pattern-matches on `kind`, it never touches Plotly's API directly.
    """
    ok, _ = validate_chart(chart)
    if not ok:
        return None

    spec = chart["spec_json"]
    data = spec.get("data") or []
    title = spec.get("title") or chart.get("title", "")
    chart_type = chart["chart_type"]

    if chart_type == "kpi":
        point = data[0] if data else {"label": title, "value": None}
        return {"kind": "kpi", "label": point.get("label", title), "value": point.get("value")}
    if chart_type == "table":
        return {"kind": "table", "title": title, "data": data}

    try:
        if chart_type in ("bar", "horizontal_bar"):
            labels = [d["label"] for d in data]
            values = [d["value"] for d in data]
            horizontal = chart_type == "horizontal_bar"
            fig = go.Figure(
                go.Bar(
                    x=values if horizontal else labels,
                    y=labels if horizontal else values,
                    orientation="h" if horizontal else "v",
                )
            )
        elif chart_type in ("line", "area"):
            labels = [d["label"] for d in data]
            values = [d["value"] for d in data]
            fig = go.Figure(
                go.Scatter(
                    x=labels, y=values, mode="lines+markers", fill="tozeroy" if chart_type == "area" else None
                )
            )
        elif chart_type == "pie":
            fig = go.Figure(go.Pie(labels=[d["label"] for d in data], values=[d["value"] for d in data]))
        elif chart_type == "scatter":
            fig = go.Figure(go.Scatter(x=[d["x"] for d in data], y=[d["y"] for d in data], mode="markers"))
        else:
            return None
    except (KeyError, TypeError):
        return None  # a data point missing an expected key -> unusable, not a crash

    fig.update_layout(title=title, xaxis_title=spec.get("x_axis"), yaxis_title=spec.get("y_axis"))
    return {"kind": "figure", "figure": fig}
