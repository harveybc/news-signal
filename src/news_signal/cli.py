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

    ask = sub.add_parser("ask", help="Ask your OWN question of one news item, through the same registry path")
    ask.add_argument("--input", required=True)
    ask.add_argument("--question", required=True, help="the question, in your words; it is encoded with the news")
    ask.add_argument("--name", default="answer", help="the name this answer is returned under")
    ask.add_argument("--option", action="append", default=[], metavar="NAME=DESCRIPTION",
                     help="repeatable; order is preserved and is part of the task identity")
    ask.add_argument("--schema", default="choice")
    ask.add_argument("--as-of", help="Explicit replay clock, never a claim of live observation")
    ask.add_argument("--max-age-seconds", type=int, default=900)
    ask.add_argument("--store")
    ask.add_argument("--print-question", action="store_true",
                     help="print the exact question sent to the model and its identity, and ask nothing")

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
        elif args.command == "ask":
            from contextlib import redirect_stdout
            from .application import classify_event
            from .core import load_json as _load
            from .question import ad_hoc_task
            from .shadow import ShadowStore
            options = []
            for pair in args.option:
                name, sep, description = pair.partition("=")
                if not sep:
                    raise Refusal(f"OPTION_SYNTAX: expected NAME=DESCRIPTION, got {pair!r}")
                options.append((name, description))
            spec = {"name": args.name, "question": args.question, "options": options, "schema": args.schema}
            built = ad_hoc_task(**spec)
            if args.print_question:
                result = {"schema": "news_question.v1", "task_id": built["task_id"],
                          "questions_as_sent": built["questions"],
                          "option_order": {k: list(q["criteria"]) for k, q in built["questions"].items()},
                          "nothing_was_asked_of_the_model": True}
            else:
                as_of = args.as_of or datetime.now(timezone.utc).isoformat()
                with redirect_stdout(sys.stderr):
                    outcome = classify_event(_load(args.input), task_id=built["task_id"], question_spec=spec,
                                             as_of=as_of, max_age_seconds=args.max_age_seconds,
                                             store=ShadowStore(args.store) if args.store else None)
                result = {k: v for k, v in outcome.items() if k != "request"}
                result["clock_mode"] = "REPLAY" if args.as_of else "WALL_CLOCK"
                if result.get("status") != "SHADOW_ONLY":
                    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
                    return 2
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
