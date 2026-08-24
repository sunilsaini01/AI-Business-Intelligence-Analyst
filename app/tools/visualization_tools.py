"""Deterministic Plotly/Matplotlib rendering for the Visualization Agent
(Sec 2, Sec 5). Phase 6-7. The LLM picks *which* chart type; rendering itself
is code, never pixels touched by a prompt.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def render_bar_chart(labels: list[str], values: list[float], title: str) -> dict[str, Any]:
    fig = go.Figure(data=[go.Bar(x=labels, y=values)])
    fig.update_layout(title=title)
    return fig.to_dict()
