"""Live end-to-end smoke check: runs one real research query over the WebSocket.

Unlike the pytest suite, this deliberately hits the network — arXiv, Semantic
Scholar, and your configured LLM — so it verifies the things a mocked test cannot.
Expect it to take 10-60 seconds and to burn a few LLM tokens if a key is set.

    cd backend
    python scripts/smoke_e2e.py
    python scripts/smoke_e2e.py --query "Explain attention mechanisms in transformers"

Exits non-zero if the run does not complete, so it is usable in CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.llm_provider import resolve_provider_config  # noqa: E402

DEFAULT_QUERY = "Find recent advances in multimodal deepfake detection and summarize major approaches."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    args = parser.parse_args()

    config = resolve_provider_config()
    print(f"App:            {settings.app_name}")
    print(f"LLM provider:   {config.provider} (model {config.model})")
    print(f"LLM configured: {config.usable}")
    if not config.usable:
        print("  -> summaries will come from the local deterministic fallback")
    print(f"Query:          {args.query}\n")

    client = TestClient(app)
    result = None

    with client.websocket_connect("/ws/research") as socket:
        socket.send_json({"query": args.query})
        while True:
            message = socket.receive_json()
            kind = message.get("type")

            if kind == "progress":
                print(f"  [{message['stage']:<12}] {message['status']:<9} {message['message']}")
            elif kind == "error":
                print(f"\nFAILED: {message['message']}", file=sys.stderr)
                return 1
            elif kind == "result":
                result = message["result"]
                break

    if not result:
        print("\nFAILED: stream ended without a result", file=sys.stderr)
        return 1

    print("\n--- Result ---")
    print(f"run_id:        {result['run_id']}")
    print(f"status:        {result['status']}")
    print(f"sources:       {len(result['sources'])}")
    print(f"critic_score:  {result['critic_score']} ({result.get('critic_method')})")
    print(f"claim_checks:  {len(result.get('claim_checks') or [])}")
    print(f"\nsummary:\n{result['summary']}")

    problems = []
    if result["status"] != "complete":
        problems.append(f"status is {result['status']}")
    if not result["sources"]:
        problems.append("no sources retrieved (network blocked, or both search APIs are down)")
    if not result.get("report_markdown"):
        problems.append("no report markdown produced")

    if problems:
        print("\nFAILED:", "; ".join(problems), file=sys.stderr)
        return 1

    print(f"\nOK — download the report at /reports/{result['run_id']}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
