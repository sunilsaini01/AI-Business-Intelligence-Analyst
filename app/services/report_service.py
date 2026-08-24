"""Report formatting/export helpers (e.g. PDF/Matplotlib export) that don't
belong in the Report Agent itself. Currently the API reads AnalysisReport
rows directly (see api/routes/analysis.py, reports.py) — this module is the
seam for anything heavier (PDF export, email delivery) added later.
"""

from __future__ import annotations
