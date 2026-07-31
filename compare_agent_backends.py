"""
Compare Claude vs a local Ollama model's agentic tool-use *planning* on
the same open-ended question, using agentic_analyst.py's run_agent /
run_agent_ollama.

Unlike compare_backends.py (which scores narration text against a
hand-written reference), there's no single "correct" trace here -- the
useful comparison is the trace itself: how many steps did it take, did it
simulate before inspecting, did it use the comparison tool, did anything
error out. evaluate_trace() below is a first, cheap pass at that; the
richer thing to add once there's more than one question is a rubric that
also judges whether the *final answer* is actually grounded in what the
trace retrieved.

Usage:
    python compare_agent_backends.py
    python compare_agent_backends.py --ollama-model qwen2.5:7b
    python compare_agent_backends.py --skip-anthropic
    python compare_agent_backends.py --skip-ollama
"""
import argparse
import sys

from agentic_analyst import DeutschJozsa, run_agent, run_agent_ollama
from ollama_client import resolve_base_url

DEFAULT_QUESTION = (
    "Compare the resource cost of the '0' and 'xor' oracles for this "
    "Deutsch-Jozsa instance, and tell me specifically where the "
    "difference comes from."
)


def evaluate_trace(trace):
    errors = [t for t in trace if isinstance(t[2], dict) and "error" in t[2]]
    oracles_simulated = sorted({
        t[1].get("oracle") for t in trace
        if t[0] == "run_circuit" and "error" not in t[2]
    })
    return {
        "steps": len(trace),
        "errors": len(errors),
        "oracles_simulated": oracles_simulated,
        "used_compare_tool": any(t[0] == "compare_resource_cost" for t in trace),
    }


def build_anthropic_client():
    try:
        import anthropic
    except ImportError:
        return None, "anthropic package not installed"
    try:
        return anthropic.Anthropic(), None
    except Exception as e:
        return None, f"could not create client (check ANTHROPIC_API_KEY): {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--anthropic-model", default="claude-sonnet-5")
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    parser.add_argument("--ollama-url", default=None,
                         help="defaults to OLLAMA_API_BASE env var, else http://localhost:11434")
    parser.add_argument("--skip-anthropic", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    print(f"Question: {args.question}\n")

    anthropic_result = None
    if not args.skip_anthropic:
        client, err = build_anthropic_client()
        if err:
            print(f"[Anthropic skipped: {err}]", file=sys.stderr)
        else:
            print("=" * 40, "ANTHROPIC", "=" * 40)
            text, trace = run_agent(
                client, DeutschJozsa(3), args.question,
                model=args.anthropic_model, max_steps=args.max_steps,
            )
            anthropic_result = (text, trace)
            print()

    ollama_result = None
    if not args.skip_ollama:
        print("=" * 40, "OLLAMA", "=" * 40)
        try:
            text, trace = run_agent_ollama(
                resolve_base_url(args.ollama_url), DeutschJozsa(3), args.question,
                model=args.ollama_model, max_steps=args.max_steps,
            )
            ollama_result = (text, trace)
        except Exception as e:
            print(f"[Ollama skipped: {e}]", file=sys.stderr)
        print()

    if anthropic_result is None and ollama_result is None:
        print("Neither backend was available -- nothing to compare.", file=sys.stderr)
        sys.exit(1)

    print("=" * 90)
    print(f"{'Metric':<22} {'Anthropic':<30} {'Ollama':<30}")
    print("-" * 90)
    a_eval = evaluate_trace(anthropic_result[1]) if anthropic_result else None
    o_eval = evaluate_trace(ollama_result[1]) if ollama_result else None
    for key in ("steps", "errors", "oracles_simulated", "used_compare_tool"):
        a_val = a_eval[key] if a_eval else "skipped"
        o_val = o_eval[key] if o_eval else "skipped"
        print(f"{key:<22} {str(a_val):<30} {str(o_val):<30}")
    print("=" * 90)

    if anthropic_result:
        print("\n--- Anthropic final answer ---")
        print(anthropic_result[0])
    if ollama_result:
        print("\n--- Ollama final answer ---")
        print(ollama_result[0])


if __name__ == "__main__":
    main()
