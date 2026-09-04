"""Run the grounded circuit-analysis agent through an OpenAI-compatible API.

Credentials come from OPENAI_BASE_URL and OPENAI_API_KEY, or from the local,
gitignored .aggregator_credentials.json file beside this script. Command-line
values override both. The endpoint must implement /chat/completions tool
calling, including tool_call ids on assistant responses.

From the repository root:
    pixi run -e qc python qstudy/run_agent_aggregator.py --model openai/gpt-5.4-mini
"""
import argparse
import json
import os
import sys

from agentic_analyst import DeutschJozsa, run_agent_aggregator

CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), ".aggregator_credentials.json")


DEFAULT_QUESTION = (
    "Compare the resource cost of the '0' and 'xor' oracles for this "
    "Deutsch-Jozsa instance, and tell me specifically where the "
    "difference comes from."
)


def load_credentials():
    """Load local aggregator credentials without overriding environment values."""
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE) as f:
            credentials = json.load(f)
        os.environ.setdefault("OPENAI_BASE_URL", credentials.get("base_url", ""))
        os.environ.setdefault("OPENAI_API_KEY", credentials.get("api_key", ""))


def main():
    load_credentials()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Aggregator model id")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"),
                        help="Defaults to OPENAI_BASE_URL")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                        help="Defaults to OPENAI_API_KEY; never printed")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if not args.base_url or not args.api_key:
        parser.error("--base-url/--api-key (or OPENAI_BASE_URL/OPENAI_API_KEY) are required")

    try:
        answer, trace = run_agent_aggregator(
            args.base_url, args.api_key, DeutschJozsa(3), args.question,
            model=args.model, max_steps=args.max_steps, max_tokens=args.max_tokens,
            temperature=args.temperature, timeout=args.timeout,
        )
    except Exception as e:
        print(f"Aggregator agent failed: {e}", file=sys.stderr)
        return 1

    print("\n--- Final answer ---")
    print(answer)
    print(f"\nExecuted {len(trace)} tool call(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
