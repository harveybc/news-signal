#!/usr/bin/env python3
"""CL07: score predictions against the labels that were sealed before anything ran.

The labels are the corpus's, never the model's own answers. Every number below is reported with the count it rests on,
because on a bank this small a rate is a description of thirteen items and nothing more. The status line travels with the
report so a reader cannot mistake it for a measurement of business quality.

    python3 tools/score_corpus.py --corpus examples/eurusd/corpus.json \
        --wrapper receipts/*.json --direct direct.json --out score.json
"""

import argparse
import glob
import json


def key(entry):
    """Rows are identified by the news identity AND its category: a duplicate and its revision share an event_id."""
    return f"{entry['event_id']}:{entry['category']}"


def score(corpus, predictions):
    labels = sorted({entry["label"] for entry in corpus["items"] if entry["label"]})
    confusion = {truth: {predicted: 0 for predicted in labels} for truth in labels}
    unpredicted, foreign = [], []
    for entry in corpus["items"]:
        identity = key(entry)
        predicted = predictions.get(identity)
        if predicted is None:
            unpredicted.append(identity)
            continue
        if predicted not in labels:
            foreign.append({"row": identity, "predicted": predicted})
            continue
        confusion[entry["label"]][predicted] += 1
    scored = sum(sum(row.values()) for row in confusion.values())
    per_class = {}
    for label in labels:
        true_positive = confusion[label][label]
        predicted_positive = sum(confusion[t][label] for t in labels)
        actual_positive = sum(confusion[label].values())
        precision = true_positive / predicted_positive if predicted_positive else None
        recall = true_positive / actual_positive if actual_positive else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": actual_positive, "predicted": predicted_positive,
                            "true_positive": true_positive}
    return {"schema": "news_corpus_score.v1",
            "status": corpus["status"],
            "task_id": corpus["task_id"],
            "rows_declared": len(corpus["items"]),
            "rows_scored": scored,
            "rows_unpredicted": unpredicted,
            "rows_with_foreign_label": foreign,
            "coverage": scored / len(corpus["items"]) if corpus["items"] else 0.0,
            "class_counts": corpus["class_counts"],
            "confusion": confusion,
            "per_class": per_class,
            "relevant_class": per_class.get("related"),
            "macro_f1": sum(c["f1"] for c in per_class.values()) / len(per_class) if per_class else 0.0,
            "accuracy": (sum(confusion[l][l] for l in labels) / scored) if scored else 0.0,
            "reading": ("the labels are author-written; this describes thirteen items and is not an estimate of business "
                        "accuracy, a calibration or a power calculation")}


def predictions_from_wrapper(paths, corpus):
    """Map each receipt onto the corpus row whose bytes it was computed from, so no result is scored against another item."""
    by_input = {}
    for entry in corpus["items"]:
        by_input.setdefault(entry["input_sha256"], []).append(entry)
    out = {}
    for path in paths:
        receipt = json.loads(open(path, encoding="utf-8").read())
        if receipt.get("status") != "SHADOW_ONLY":
            continue
        for entry in by_input.get(receipt["input_sha256"], []):
            out[key(entry)] = receipt["features"]["relevance"]["label"]
    return out


def predictions_from_direct(path, corpus):
    report = json.loads(open(path, encoding="utf-8").read())
    by_input = {}
    for entry in corpus["items"]:
        by_input.setdefault(entry["input_sha256"], []).append(entry)
    out = {}
    for single in report["single"]:
        for entry in by_input.get(single["input_sha256"], []):
            out[key(entry)] = single["response"]["answers"]["relevance"]["choice"]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--wrapper", nargs="*", default=[])
    parser.add_argument("--direct")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    corpus = json.loads(open(args.corpus, encoding="utf-8").read())
    if "input_sha256" not in (corpus["items"][0] if corpus["items"] else {}):
        raise SystemExit("the corpus must carry each row's input_sha256; regenerate it with tools/make_corpus.py")
    paths = [p for pattern in args.wrapper for p in sorted(glob.glob(pattern))]
    result = {"schema": "news_corpus_score_pair.v1", "same_rows": True}
    if paths:
        result["wrapper"] = score(corpus, predictions_from_wrapper(paths, corpus))
    if args.direct:
        result["direct_sdk"] = score(corpus, predictions_from_direct(args.direct, corpus))
    if "wrapper" in result and "direct_sdk" in result:
        result["identical_labels"] = result["wrapper"]["confusion"] == result["direct_sdk"]["confusion"]
        result["same_rows"] = result["wrapper"]["rows_scored"] == result["direct_sdk"]["rows_scored"]
    text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=1, sort_keys=True)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    main()
