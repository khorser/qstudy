"""
Surveys every model an OpenAI-compatible aggregator (pseudonym "ModelProxy"
in this repo's notes -- no real hostname is ever printed or committed here)
reports via GET /models: sends an identity-probe question to each ("What is
your exact model name/version, and who trained you?"), and optionally runs
the same grounded-narration comparison compare_backends.py does against
DeutschJozsa(3), scored against the hand-written reference.

Every prompt/response pair is saved to a timestamped JSON transcript file
under aggregator_transcripts/ (gitignored -- these are working notes, not
committed until their fate is decided) so results can be revisited and
compared later without re-spending API calls -- see
agentic_testing_notes.md's "Testing frontier models via a third-party
aggregator" section for why the identity check matters (a prior run found
most claude-*-labeled endpoints on this aggregator self-identifying as a
different product entirely, "Kiro").

Credentials: reads OPENAI_BASE_URL / OPENAI_API_KEY from the environment,
or -- if present -- from a local, gitignored .aggregator_credentials.json
next to this file:
    {"base_url": "https://example.com/v1", "api_key": "sk-..."}
Never printed, and never included in the saved transcript.

Usage:
    pixi run -e qc python aggregator_survey.py                    # identity probe only, all listed models
    pixi run -e qc python aggregator_survey.py --models claude-fable-5,gpt-5.4
    pixi run -e qc python aggregator_survey.py --narrate           # also run the narration comparison (more tokens/cost)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit

from aggregator_client import AggregatorClient, list_aggregator_models
from headless_facts import compute_all_slice_facts
from ai_narrator import explain_slice, score_coverage, DJ_BV_REFERENCE

CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), ".aggregator_credentials.json")
TRANSCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "aggregator_transcripts")

IDENTITY_PROBE = (
    "What is your exact model name/version, and who trained you? Be as "
    "specific as you can, and say if you are not certain of some detail."
)


# Copied from compare_backends.py -- keep in sync if that copy changes.
class DeutschJozsa:
    """Determine whether a boolean oracle f: {0,1}^n -> {0,1} is constant or balanced using a single query."""

    def __init__(self, n):
        self.n = n

    def f_0(self):
        return QuantumCircuit(self.n + 1)

    def f_xor(self):
        c = self.f_0()
        c.cx(list(range(self.n)), self.n)
        return c

    def get_circuit(self, f, label=""):
        x = QuantumRegister(self.n, name="x")
        y = AncillaRegister(1, name="y")
        r = ClassicalRegister(self.n, name="r")
        c = QuantumCircuit(x, y, r)
        c.x(y)
        c.barrier(label="init")
        c.h(x)
        c.h(y)
        c.barrier(label="prep")
        c.append(f().to_gate(label=f"$U_{{{label}}}$"), [*x, *y])
        c.barrier(label="apply")
        c.h(x)
        c.barrier(label="done")
        c.measure(x, r)
        return c


def load_credentials():
    """Populates OPENAI_BASE_URL / OPENAI_API_KEY from a local, gitignored
    JSON file if they aren't already in the environment. Never reads the
    values back out for printing/logging."""
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE) as f:
            creds = json.load(f)
        os.environ.setdefault("OPENAI_BASE_URL", creds.get("base_url", ""))
        os.environ.setdefault("OPENAI_API_KEY", creds.get("api_key", ""))


def probe_identity(client, model, max_tokens=300, temperature=None):
    entry = {"model": model, "probe": IDENTITY_PROBE}
    try:
        extra = {}
        if temperature is not None:
            extra["temperature"] = temperature
        resp = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": IDENTITY_PROBE}],
            **extra,
        )
        entry["response"] = "".join(b.text for b in resp.content if b.type == "text")
    except Exception as e:
        entry["error"] = str(e)
    return entry


def narrate(client, model, all_facts, max_tokens=300, **extra):
    rows = {}
    for label, facts in all_facts.items():
        try:
            narration = explain_slice(client, facts, model=model, max_tokens=max_tokens, **extra)
            scored = score_coverage(label, narration)
            rows[label] = {
                "narration": narration,
                "coverage": scored["coverage"],
                "missing": scored["missing"],
            }
        except Exception as e:
            rows[label] = {"narration": f"[ERROR: {e}]", "coverage": None, "missing": None}
    return rows


def ensure_output_directory(out_path):
    """Create the parent directory for an output path, when it has one."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", help="comma-separated model ids (default: everything the aggregator lists)")
    parser.add_argument("--narrate", action="store_true",
                         help="also run the DeutschJozsa narration comparison per model (more tokens/cost)")
    parser.add_argument("--max-tokens", type=int, default=300,
                         help="Used for narration, and as the identity probe's "
                              "budget too unless --identity-max-tokens overrides "
                              "it. Reasoning models can burn this whole budget on "
                              "hidden chain-of-thought before any visible text --"
                              "raise it if you see empty/truncated/garbled output.")
    parser.add_argument("--identity-max-tokens", type=int, default=None,
                         help="Overrides --max-tokens for just the identity probe. "
                              "Rarely needed -- mainly for keeping identity-probe "
                              "cost down while still raising --max-tokens for "
                              "--narrate, or vice versa.")
    parser.add_argument("--temperature", type=float, default=None,
                         help="Omit to use the aggregator's own default.")
    parser.add_argument("--out", default=None,
                         help="transcript JSON path (default: timestamped, under aggregator_transcripts/)")
    args = parser.parse_args()

    load_credentials()

    request_extra = {}
    if args.temperature is not None:
        request_extra["temperature"] = args.temperature

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        try:
            models = list_aggregator_models()
        except Exception as e:
            print(f"Could not list aggregator models: {e}", file=sys.stderr)
            sys.exit(1)

    if not models:
        print("Aggregator returned no models.", file=sys.stderr)
        sys.exit(1)

    print(f"Surveying {len(models)} model(s): {', '.join(models)}\n")

    client = AggregatorClient()
    identity_max_tokens = args.identity_max_tokens if args.identity_max_tokens is not None else args.max_tokens

    all_facts = None
    if args.narrate:
        dj = DeutschJozsa(3)
        qc = dj.get_circuit(dj.f_xor, "xor")
        all_facts = compute_all_slice_facts(qc, "DeutschJozsa", DeutschJozsa.__doc__)

    results = []
    for model in models:
        print(f"--- {model} ---")
        identity = probe_identity(client, model, max_tokens=identity_max_tokens,
                                  **request_extra)
        if "error" in identity:
            print(f"  identity probe error: {identity['error']}")
        else:
            print(f"  identity: {identity['response'][:200]}")
        entry = {"model": model, "identity_probe": identity}

        if args.narrate:
            rows = narrate(client, model, all_facts, max_tokens=args.max_tokens, **request_extra)
            entry["narration"] = rows
            covs = [r["coverage"] for r in rows.values() if r["coverage"] is not None]
            if covs:
                print(f"  narration coverage (avg over {len(covs)} slices): {sum(covs) / len(covs):.2f}")
        results.append(entry)
        print()

    out_path = args.out or os.path.join(
        TRANSCRIPTS_DIR, f"aggregator_survey_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    ensure_output_directory(out_path)
    with open(out_path, "w") as f:
        json.dump({
            "identity_probe_question": IDENTITY_PROBE,
            "narrate": args.narrate,
            "reference": DJ_BV_REFERENCE if args.narrate else None,
            "results": results,
        }, f, indent=2)
    print(f"Saved full transcripts for {len(results)} model(s) to {out_path}")

    if args.narrate:
        print("\n" + "=" * 100)
        print(f"{'Model':<30} {'Avg coverage':<14}")
        print("-" * 100)
        for entry in results:
            covs = [r["coverage"] for r in entry["narration"].values() if r["coverage"] is not None]
            avg = f"{sum(covs) / len(covs):.2f}" if covs else "n/a"
            print(f"{entry['model']:<30} {avg:<14}")
        print("=" * 100)


if __name__ == "__main__":
    main()
