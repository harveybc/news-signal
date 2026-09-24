#!/usr/bin/env python3
"""The bounded pilot: the whole public path over the sealed corpus, once, with its cost measured.

It loads the model once and answers every row, because the acceptance is about the path and its numbers, not about how many
times a process can start. Cold and warm latency are separated, the refusals are exercised in the same process as the
answers, and everything it writes is a file another process can read back without a model.

    python3 tools/run_pilot.py --corpus examples/eurusd/corpus.json --out DIR [--store DIR] [--label first]
"""

import argparse
import json
import warnings
from pathlib import Path
import resource
import time

from news_signal.application import classify_event, discover, peak_rss_bytes
from news_signal.core import digest, load_json
from news_signal.provider import PROVIDER_NAME
from news_signal.shadow import ShadowStore


def gpu_state():
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    properties = torch.cuda.get_device_properties(0)
    return {"cuda_available": True, "device_uuid": str(properties.uuid), "device_name": properties.name,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "total_bytes": int(properties.total_memory),
            "attribution": "MEASURED"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--task", default="news_relevance_eurusd.v1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--store")
    parser.add_argument("--label", default="run")
    args = parser.parse_args(argv)

    corpus_path = Path(args.corpus)
    corpus = json.loads(corpus_path.read_text())
    root = corpus_path.parent
    out = Path(args.out)
    (out / "receipts").mkdir(parents=True, exist_ok=True)
    store = ShadowStore(args.store) if args.store else None

    wall_started = time.perf_counter()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    registry, report = discover()
    if PROVIDER_NAME not in registry.names():
        raise SystemExit(f"the provider was not discovered through the installed entry points: {report}")

    rows, refusals, latencies = [], [], []
    caught = warnings.catch_warnings(record=True)
    seen_warnings = caught.__enter__()
    warnings.simplefilter("always")
    for entry in corpus["items"] + corpus["refusals"]:
        event = load_json(root / entry["file"])
        started = time.perf_counter()
        outcome = classify_event(event, task_id=args.task, as_of=corpus["as_of"], registry=registry,
                                 report=report, store=store)
        elapsed = time.perf_counter() - started
        receipt = outcome["receipt"]
        name = Path(entry["file"]).stem + ("" if entry in corpus["items"] else "_refusal")
        (out / "receipts" / f"{name}.json").write_text(
            json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=1, sort_keys=True))
        record = {"file": entry["file"], "event_id": entry["event_id"], "category": entry["category"],
                  "status": receipt["status"], "m5phet_status": receipt["m5phet_status"],
                  "call_seconds": elapsed, "latency": receipt["latency"],
                  "stored": outcome["stored"]}
        if receipt["status"] == "SHADOW_ONLY":
            record["label"] = receipt["features"]["relevance"]["label"]
            record["probabilities"] = receipt["features"]["relevance"]["uncalibrated_probabilities"]
            latencies.append(receipt["latency"]["inference_seconds"])
            rows.append(record)
        else:
            record["why"] = receipt.get("why")
            record["refusal_reasons"] = receipt.get("refusal_reasons")
            refusals.append(record)

    caught.__exit__(None, None, None)
    cpu_ended = resource.getrusage(resource.RUSAGE_SELF)
    ordered = sorted(latencies)
    first = rows[0]["latency"] if rows else {}
    summary = {
        "schema": "news_pilot.v1",
        "label": args.label,
        "task_id": args.task,
        "corpus_sha256": digest(corpus),
        "discovery": report,
        "provider_capabilities": registry.capabilities(PROVIDER_NAME),
        "model_identity": (registry.get(PROVIDER_NAME).identity_for(
            (registry.capabilities(PROVIDER_NAME).get("known_states") or [None])[0])),
        "counts": {"answered": len(rows), "refused": len(refusals),
                   "declared_items": len(corpus["items"]), "declared_refusals": len(corpus["refusals"])},
        "latency_seconds": {
            "cold_load": first.get("load_seconds"),
            "first_inference": rows[0]["latency"]["inference_seconds"] if rows else None,
            "warm_inference_median": ordered[len(ordered) // 2] if ordered else None,
            "warm_inference_min": ordered[0] if ordered else None,
            "warm_inference_max": ordered[-1] if ordered else None,
            "wall_total": time.perf_counter() - wall_started,
        },
        "cpu_seconds": {"user": cpu_ended.ru_utime - cpu_started.ru_utime,
                        "system": cpu_ended.ru_stime - cpu_started.ru_stime},
        "peak_rss_bytes": peak_rss_bytes(),
        "gpu": gpu_state(),
        # upstream conditions declared during load belong in the record: this checkpoint reports invalid temperatures, which
        # is exactly why the confidence field travels as uncalibrated
        "runtime_warnings": sorted({f"{w.category.__name__}: {w.message}" for w in seen_warnings}),
        "rows": rows,
        "refusals": refusals,
        "reading": ("latency and memory are this host's, measured in this process. The labels are the model's; their "
                    "quality is not established here."),
    }
    (out / f"pilot_{args.label}.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=1, sort_keys=True))
    print(json.dumps({k: summary[k] for k in ("counts", "latency_seconds", "cpu_seconds", "peak_rss_bytes", "gpu")},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
