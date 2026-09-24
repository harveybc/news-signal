#!/usr/bin/env python3
"""Compare the native SDK reference against the wrapper's results, field by field, with no tolerance to adjust.

Equality here is Python equality on the decoded JSON, which for these answers means: same option keys in the same order,
same floats bit for bit, same label. The only fields ignored are named in IGNORED below -- they are the timing and
provenance envelope, never a decision field. There is no tolerance parameter, so there is nothing to widen after seeing a
mismatch; a mismatch is reported with the two values.

    python3 tools/parity_report.py --direct direct.json --wrapper receipt1.json receipt2.json --out parity.json
"""

import argparse
import json

#: envelope fields that describe WHEN and WHERE a call happened, never WHAT was decided
IGNORED = ("seconds", "load_seconds", "inference_seconds", "recorded_at", "peak_rss_bytes", "peak_vram_bytes")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def answers_of_direct(report):
    """Keyed by INPUT identity, not by event_id: a revision carries the same event_id and different bytes, so keying by the
    identifier would silently drop one of the two rows and report a smaller comparison than was asked for."""
    out = {}
    for entry in report["single"]:
        out[entry["input_sha256"]] = {"answers": entry["response"]["answers"],
                                      "event_id": entry["event_id"],
                                      "input_sha256": entry["input_sha256"],
                                      "state_sha256": entry["state_sha256"]}
    return out


def answers_of_wrapper(receipts):
    out = {}
    for receipt in receipts:
        identity = receipt.get("input_sha256")
        if receipt.get("status") != "SHADOW_ONLY":
            out[identity] = {"refused": receipt.get("why"), "status": receipt.get("status"),
                             "event_id": receipt.get("event_id"), "input_sha256": identity}
            continue
        out[identity] = {"answers": {q: f["sdk_answer"] for q, f in receipt["features"].items()},
                         "event_id": receipt["event_id"],
                         "input_sha256": identity,
                         "state_sha256": (receipt.get("binding") or {}).get("state_digest")}
    return out


def compare(direct, wrapper):
    rows, mismatches = [], []
    for identity in sorted(set(direct) | set(wrapper)):
        left, right = direct.get(identity), wrapper.get(identity)
        event_id = (left or right).get("event_id")
        if left is None or right is None:
            mismatches.append({"event_id": event_id, "input_sha256": identity, "reason": "MISSING_ON_ONE_SIDE",
                               "direct": left is not None, "wrapper": right is not None})
            continue
        if "answers" not in right or "answers" not in left:
            mismatches.append({"event_id": event_id, "input_sha256": identity, "reason": "REFUSED_ON_ONE_SIDE",
                               "detail": right if "answers" not in right else left})
            continue
        equal = left["answers"] == right["answers"]
        # key order is part of the comparison: a reordered probability map is a different answer object
        order_equal = ([list(a) for a in left["answers"].values()] == [list(a) for a in right["answers"].values()] and
                       [list(a.get("probabilities", {})) for a in left["answers"].values()] ==
                       [list(a.get("probabilities", {})) for a in right["answers"].values()])
        rows.append({"event_id": event_id, "input_sha256": identity, "equal": equal, "key_order_equal": order_equal,
                     "labels": {q: a.get("choice") for q, a in left["answers"].items()}})
        if not (equal and order_equal):
            mismatches.append({"event_id": event_id, "input_sha256": identity, "reason": "ANSWER_FIELDS_DIFFER",
                               "direct": left["answers"], "wrapper": right["answers"]})
    return rows, mismatches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direct", required=True)
    parser.add_argument("--wrapper", required=True, nargs="+")
    parser.add_argument("--wrapper-b", nargs="*", default=[],
                        help="a second wrapper run; compared against the first as process-reload parity, separately")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    direct_report = load(args.direct)
    receipts = [load(p) for p in args.wrapper]
    rows, mismatches = compare(answers_of_direct(direct_report), answers_of_wrapper(receipts))
    reload_section = None
    if args.wrapper_b:
        other = [load(p) for p in args.wrapper_b]
        reload_rows, reload_mismatches = compare(answers_of_wrapper(receipts), answers_of_wrapper(other))
        reload_section = {"scope": "PROCESS_RELOAD: the same wrapper, a new process, the same inputs",
                          "compared": len(reload_rows),
                          "equal": sum(1 for r in reload_rows if r["equal"] and r["key_order_equal"]),
                          "mismatches": reload_mismatches}
    result = {"schema": "laya_parity.v1",
              "comparison": "REAL_SDK_DIRECT_VERSUS_REAL_PROVIDER_THROUGH_M5PHET_REGISTRY",
              "tolerance": "NONE: exact equality of the SDK's exposed decision fields",
              "ignored_envelope_fields": list(IGNORED),
              "questions_sha256_direct": direct_report.get("questions_sha256"),
              "settings": direct_report.get("settings"),
              "sdk": direct_report.get("sdk"),
              "compared": len(rows),
              "equal": sum(1 for r in rows if r["equal"] and r["key_order_equal"]),
              "mismatches": mismatches,
              "process_reload": reload_section,
              "rows": rows,
              "reading": ("equality establishes that the wrapper did not change the model's input or its output. It says "
                          "nothing about whether the classifier is correct, and nothing about calibration.")}
    text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=1, sort_keys=True)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
    print(text)
    return 0 if not mismatches and rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
