"""Phase 12: makes frontend/*.py importable as flat top-level modules
(`import api_client`, `from progress import ...`) from inside the `api`
container's pytest run — exactly matching how they're imported in
production, where `streamlit run app.py` runs from the `frontend/`
directory itself (its own directory is on `sys.path`, not a parent
`frontend` package). Only the pure modules (no `streamlit` import) are
ever imported by these tests — see frontend/app.py's module docstring for
which ones those are.
"""

from __future__ import annotations

import sys
from pathlib import Path

_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))
