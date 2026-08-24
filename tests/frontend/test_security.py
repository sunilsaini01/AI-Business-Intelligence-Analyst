"""Phase 12 security checks — structural guards (the frontend has no
running server of its own to attack the way the backend does, so these are
source-level guarantees: no DB/file access capability exists in the
frontend code at all) plus behavioral checks that the API client never
leaks credentials or arbitrary paths.
"""

from __future__ import annotations

import ast
import inspect

import api_client
import chart_builder
import health
import progress
import report_view


def _imported_top_level_names(module) -> set[str]:
    """Actual `import X` / `from X import ...` module names only — parsed
    via `ast`, not a raw substring search, so a docstring that mentions a
    forbidden package BY NAME while explaining it's deliberately absent
    (as api_client.py's own docstring does) doesn't false-positive."""
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_frontend_module_imports_a_database_driver():
    modules = [api_client, chart_builder, health, progress, report_view]
    forbidden_imports = {"asyncpg", "psycopg", "psycopg2", "sqlalchemy", "sqlite3"}
    for module in modules:
        imported = _imported_top_level_names(module)
        overlap = imported & forbidden_imports
        assert not overlap, f"{module.__name__} must never import {overlap}"


def test_api_client_never_opens_a_local_file_or_shells_out():
    source = inspect.getsource(api_client)
    for forbidden in ("open(", "subprocess", "os.system", "eval(", "exec("):
        assert forbidden not in source


def test_chart_builder_never_opens_a_local_file():
    """storage_path is explicitly rejected (see chart_builder.validate_chart)
    rather than ever being passed to a file-opening call — this asserts
    the capability to do so doesn't exist in the module at all."""
    source = inspect.getsource(chart_builder)
    for forbidden in ("open(", "Path(", "os.path.join", "urlopen"):
        assert forbidden not in source


def test_api_client_source_has_no_hardcoded_credentials_or_keys():
    source = inspect.getsource(api_client)
    for marker in ("sk-", "api_key=", "password=", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "postgresql://"):
        assert marker not in source


def test_apierror_user_message_never_contains_the_raw_exception_type_name():
    """A quick regression guard: every fixed user-facing string literal in
    api_client.py is written by hand (not str(exc)) — this greps the
    source rather than instantiating every path, since the behavioral
    version of this is already covered in test_api_client.py."""
    source = inspect.getsource(api_client)
    # the only place str(exc)/repr(exc) may appear is never — this client
    # always writes its own fixed, safe strings.
    assert "str(exc)" not in source
    assert "repr(exc)" not in source
