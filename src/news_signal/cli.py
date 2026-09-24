"""One-event shadow classifier; no broker connectivity."""

import argparse
from datetime import datetime, timezone
import json
import sys

from .backends import FixtureBackend, LayaBackend, seal_checkpoint
from .core import Refusal, classify, load_json, validate_news


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal-model", help="Hash a materialized local checkpoint; writes JSON to stdout")
    seal.add_argument("--checkpoint", required=True)
    run = sub.add_parser("classify")
    run.add_argument("--input", required=True)
    run.add_argument("--backend", choices=["fixture", "laya"], required=True)
    run.add_argument("--checkpoint")
    run.add_argument("--manifest")
    run.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    run.add_argument("--gpu-uuid")
    run.add_argument("--as-of", help="Explicit replay clock, never a claim of live observation")
    run.add_argument("--max-age-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        if args.command == "seal-model":
            result = seal_checkpoint(args.checkpoint)
        else:
            event = load_json(args.input)
            as_of = args.as_of or datetime.now(timezone.utc).isoformat()
            validate_news(event, as_of, args.max_age_seconds)
            if args.backend == "fixture":
                backend = FixtureBackend()
            else:
                if not args.checkpoint or not args.manifest:
                    raise Refusal("CHECKPOINT_AND_MANIFEST_REQUIRED")
                # Upstream diagnostics use stdout. Keep the CLI evidence stream valid JSON.
                from contextlib import redirect_stdout
                with redirect_stdout(sys.stderr):
                    backend = LayaBackend(args.checkpoint, load_json(args.manifest), args.device, args.gpu_uuid)
            from contextlib import redirect_stdout
            with redirect_stdout(sys.stderr):
                result = classify(event, backend, as_of, args.max_age_seconds)
            result["clock_mode"] = "REPLAY" if args.as_of else "WALL_CLOCK"
            from .core import digest
            result.pop("receipt_sha256")
            result["receipt_sha256"] = digest(result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (Refusal, ImportError, OSError, ValueError) as e:
        print(json.dumps({"schema": "news_refusal.v1", "status": "REFUSED", "reason": str(e), "execution_authorized": False}))
        return 2
    except RuntimeError:
        print(json.dumps({"schema": "news_refusal.v1", "status": "REFUSED", "reason": "BACKEND_RUNTIME_ERROR", "execution_authorized": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
