"""Deployment smoke test (final deployment phase). Lightweight, deliberately
not a test-framework file (no pytest, no fixtures) — a single script you run
by hand against any running stack, local or deployed:

    python scripts/smoke_test.py --base-url https://aibia-api.onrender.com

Default `--base-url` is http://localhost:8010 (the local docker-compose
port). Exercises exactly the chain docs/deployment.md's deploy checklist
calls for: health -> readiness -> register -> login -> analyze -> poll ->
report -> charts. Prints PASS/FAIL per step and exits non-zero on the first
failure — never disguises a broken step as a pass. Makes one real analysis
call, so it costs one real LLM call against whatever provider the target
server is configured with.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid

import httpx


def _check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8010")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--poll-timeout", type=float, default=120.0)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    client = httpx.Client(base_url=base, timeout=args.timeout)

    print(f"Smoke testing {base}\n")

    resp = client.get("/api/v1/health")
    _check("GET /api/v1/health", resp.status_code == 200, f"status={resp.status_code}")

    resp = client.get("/api/v1/health/ready")
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    _check(
        "GET /api/v1/health/ready",
        resp.status_code == 200 and body.get("ready") is True,
        f"status={resp.status_code} body={body}",
    )

    email = f"smoke-{uuid.uuid4()}@example.com"
    password = "smoke-test-correct-horse-battery"
    resp = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    _check("POST /api/v1/auth/register", resp.status_code == 201, f"status={resp.status_code}")

    resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    _check("POST /api/v1/auth/login", resp.status_code == 200, f"status={resp.status_code}")
    token = resp.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"

    resp = client.get("/api/v1/analysis/00000000-0000-0000-0000-000000000000")
    _check("GET unknown analysis id -> 404", resp.status_code == 404, f"status={resp.status_code}")

    no_auth_resp = httpx.get(f"{base}/api/v1/reports", timeout=args.timeout)
    _check("GET /api/v1/reports without a token -> 401", no_auth_resp.status_code == 401, f"status={no_auth_resp.status_code}")

    resp = client.post("/api/v1/analyze", json={"question": "How many customers do we have per region?"})
    _check("POST /api/v1/analyze", resp.status_code == 202, f"status={resp.status_code}")
    analysis_id = resp.json()["analysis_id"]

    resp = client.get(f"/api/v1/analysis/{analysis_id}/report")
    _check(
        "GET report before DONE -> 409",
        resp.status_code == 409,
        f"status={resp.status_code} (a very fast run may have already finished — not itself a failure)",
    )

    deadline = time.monotonic() + args.poll_timeout
    final_status = None
    while time.monotonic() < deadline:
        resp = client.get(f"/api/v1/analysis/{analysis_id}/status")
        _check("GET /api/v1/analysis/{id}/status", resp.status_code == 200, f"status={resp.status_code}")
        final_status = resp.json()["status"]
        if final_status in ("DONE", "FAILED"):
            break
        time.sleep(2)
    _check(f"Analysis reached a terminal state within {args.poll_timeout:.0f}s", final_status in ("DONE", "FAILED"), f"last status={final_status}")
    _check("Analysis completed successfully (DONE, not FAILED)", final_status == "DONE", f"status={final_status}")

    resp = client.get(f"/api/v1/analysis/{analysis_id}/report")
    _check("GET /api/v1/analysis/{id}/report", resp.status_code == 200, f"status={resp.status_code}")
    report = resp.json()
    for field in ("executive_summary", "key_findings", "evidence", "recommendations", "confidence", "limitations"):
        _check(f"report contains '{field}'", field in report)

    resp = client.get(f"/api/v1/analysis/{analysis_id}/charts")
    _check("GET /api/v1/analysis/{id}/charts", resp.status_code == 200, f"status={resp.status_code}")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
