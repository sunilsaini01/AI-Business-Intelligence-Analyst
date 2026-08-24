"""Streamlit rendering for one chart record — a thin wrapper around
frontend/chart_builder.py's pure validation/figure-construction. Never lets
one malformed chart take down the rest of the report page (Sec "If a chart
fails: do not fail the whole report").
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from chart_builder import build_chart_figure, validate_chart


def render_chart(chart: dict[str, Any]) -> None:
    ok, reason = validate_chart(chart)
    if not ok:
        st.caption("Visualization unavailable.")
        return

    try:
        result = build_chart_figure(chart)
        if result is None:
            st.caption("Visualization unavailable.")
            return
        if result["kind"] == "kpi":
            st.metric(label=result["label"], value=result["value"])
        elif result["kind"] == "table":
            st.caption(result["title"])
            st.dataframe(result["data"], use_container_width=True)
        elif result["kind"] == "figure":
            st.plotly_chart(result["figure"], use_container_width=True)
        else:
            st.caption("Visualization unavailable.")
    except Exception:  # noqa: BLE001 — a rendering failure must degrade this ONE chart, never the page
        st.caption("Visualization unavailable.")
        return

    spec = chart.get("spec_json") or {}
    for limitation in spec.get("limitations") or []:
        st.caption(f"Note: {limitation}")
