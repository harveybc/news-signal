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

    registry = sub.add_parser("classify-registry",
                              help="Classify through the installed m5phet.providers entry point and the M5PHET Registry")
    registry.add_argument("--input", required=True)
    registry.add_argument("--task", default="news_relevance_eurusd.v1")
    registry.add_argument("--as-of", help="Explicit replay clock, never a claim of live observation")
    registry.add_argument("--max-age-seconds", type=int, default=900)
    registry.add_argument("--store", help="Directory of the durable shadow store; omitted means do not persist")

    replay = sub.add_parser("replay", help="Read the persisted shadow store back; loads no model and runs no inference")
    replay.add_argument("--store", required=True)

    providers = sub.add_parser("providers", help="What the installed m5phet.providers group offers in this environment")
    args = parser.parse_args(argv)
    try:
        if args.command == "seal-model":
            result = seal_checkpoint(args.checkpoint)
        elif args.command == "providers":
            from .application import discover
            found, report = discover()
            result = {"schema": "news_providers.v1", "discovery": report,
                      "registered": found.names(),
                      "capabilities": {name: found.capabilities(name) for name in found.names()}}
        elif args.command == "replay":
            from .application import replay as replay_store
            result = replay_store(args.store)
        elif args.command == "classify-registry":
            from contextlib import redirect_stdout
            from .application import classify_file
            from .shadow import ShadowStore
            as_of = args.as_of or datetime.now(timezone.utc).isoformat()
            with redirect_stdout(sys.stderr):
                outcome = classify_file(args.input, task_id=args.task, as_of=as_of,
                                        max_age_seconds=args.max_age_seconds,
                                        store=ShadowStore(args.store) if args.store else None)
            result = {k: v for k, v in outcome.items() if k != "request"}
            result["clock_mode"] = "REPLAY" if args.as_of else "WALL_CLOCK"
            if result.get("status") != "SHADOW_ONLY":
                print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
                return 2
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
