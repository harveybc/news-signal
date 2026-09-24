#!/usr/bin/env python3
"""CL09 parity for user-authored questions, and the CL10 per-error table, in one bounded pass.

Two things are produced here and they answer different questions. The first is whether this wrapper changes what the model is
given or returns when the QUESTION comes from the user rather than from a preset -- measured against the independent witness,
exactly as for the preset. The second is a diagnosis of where the relevance answers are wrong and what the model actually saw
when it got them: the serialized question, the option order, and the token accounting of head, options and state.

    python3 tools/run_questions.py --corpus examples/eurusd/corpus.json --questions Q.json --out DIR
"""

import argparse
import json
from pathlib import Path
import time

from news_signal.application import classify_event, discover, peak_rss_bytes
from news_signal.core import digest, load_json
from news_signal.provider import PROVIDER_NAME
from news_signal.question import ad_hoc_task, question_identity
from news_signal.shadow import ShadowStore
from news_signal.tasks import questions as task_questions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--questions", required=True, help="a JSON list of question specs written by a person")
    parser.add_argument("--out", required=True)
    parser.add_argument("--store")
    parser.add_argument("--preset-task", default="news_relevance_eurusd.v1")
    args = parser.parse_args(argv)

    corpus_path = Path(args.corpus)
    corpus = json.loads(corpus_path.read_text())
    root = corpus_path.parent
    out = Path(args.out)
    (out / "receipts").mkdir(parents=True, exist_ok=True)
    store = ShadowStore(args.store) if args.store else None
    specs = json.loads(Path(args.questions).read_text())

    registry, report = discover()
    if PROVIDER_NAME not in registry.names():
        raise SystemExit(f"the provider was not discovered through the installed entry points: {report}")

    # --- CL09: each user question over every corpus item ---------------------------------------------------------------
    answers, latencies = [], []
    for spec in specs:
        built = ad_hoc_task(**spec)
        for entry in corpus["items"]:
            event = load_json(root / entry["file"])
            started = time.perf_counter()
            outcome = classify_event(event, task_id=built["task_id"], question_spec=spec, as_of=corpus["as_of"],
                                     registry=registry, report=report, store=store)
            elapsed = time.perf_counter() - started
            receipt = outcome["receipt"]
            name = f"{spec['name']}__{Path(entry['file']).stem}"
            (out / "receipts" / f"{name}.json").write_text(
                json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=1, sort_keys=True))
            row = {"question": spec["name"], "task_id": built["task_id"], "file": entry["file"],
                   "event_id": entry["event_id"], "input_sha256": entry["input_sha256"],
                   "status": receipt["status"], "call_seconds": elapsed}
            if receipt["status"] == "SHADOW_ONLY":
                feature = receipt["features"][spec["name"]]
                row["label"] = feature["label"]
                row["probabilities"] = feature["uncalibrated_probabilities"]
                row["sdk_answer"] = feature["sdk_answer"]
                latencies.append(receipt["latency"]["inference_seconds"])
            answers.append(row)

    # --- CL10: the preset's errors, with what the model was actually given ------------------------------------------------
    provider = registry.get(PROVIDER_NAME)
    backend = provider.identity_for((registry.capabilities(PROVIDER_NAME).get("known_states") or [None])[0])
    preset_questions = task_questions(args.preset_task)
    errors, preset_rows = [], []
    for entry in corpus["items"]:
        event = load_json(root / entry["file"])
        outcome = classify_event(event, task_id=args.preset_task, as_of=corpus["as_of"], registry=registry,
                                 report=report, store=store)
        receipt = outcome["receipt"]
        if receipt["status"] != "SHADOW_ONLY":
            continue
        feature = receipt["features"]["relevance"]
        budget = provider.last_budget
        row = {"file": entry["file"], "event_id": entry["event_id"], "category": entry["category"],
               "gold": entry["label"], "predicted": feature["label"],
               "correct": entry["label"] == feature["label"],
               "probabilities": feature["uncalibrated_probabilities"],
               "confidence_field": feature["sdk_answer"].get("confidence"),
               "headline": event["headline"], "body": event["body"],
               "token_budget": budget}
        preset_rows.append(row)
        if not row["correct"]:
            errors.append(row)

    ordered = sorted(latencies)
    summary = {
        "schema": "news_questions_pilot.v1",
        "cl09": {
            "questions": [{"name": s["name"], "task_id": ad_hoc_task(**s)["task_id"],
                           "question": s["question"], "option_order": [o[0] for o in s["options"]],
                           "identity_sha256": question_identity(ad_hoc_task(**s)["questions"])} for s in specs],
            "distinct_task_ids": len({ad_hoc_task(**s)["task_id"] for s in specs}),
            "rows": answers,
            "latency_seconds": {"median": ordered[len(ordered) // 2] if ordered else None,
                                "min": ordered[0] if ordered else None, "max": ordered[-1] if ordered else None},
        },
        "cl10": {
            "preset_task": args.preset_task,
            "questions_as_sent": preset_questions,
            "option_order": {k: list(q["criteria"]) for k, q in preset_questions.items()},
            "scored_rows": len(preset_rows),
            "errors": errors,
            "rows": preset_rows,
            "reading": ("the probabilities are the SDK's uncalibrated head outputs. They are not the probability that the "
                        "label is correct, and 13 rows cannot establish a cause; this table is where to look, not a finding"),
        },
        "model_identity": backend,
        "peak_rss_bytes": peak_rss_bytes(),
        "discovery": report,
    }
    try:
        import torch
        if torch.cuda.is_available():
            summary["gpu"] = {"device_uuid": str(torch.cuda.get_device_properties(0).uuid),
                              "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                              "attribution": "MEASURED"}
    except ImportError:
        pass
    (out / "questions_pilot.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=1, sort_keys=True))
    print(json.dumps({"cl09_questions": summary["cl09"]["distinct_task_ids"],
                      "cl09_rows": len(answers),
                      "cl10_scored": len(preset_rows), "cl10_errors": len(errors),
                      "latency": summary["cl09"]["latency_seconds"], "gpu": summary.get("gpu")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
