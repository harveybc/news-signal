#!/usr/bin/env python3
"""Single versus batched versus permuted-batch passes of the SDK itself.

This is a separate experiment and it is reported separately. The wrapper answers one state per call, so nothing here is part
of the wrapper-parity result. Rows are aligned by POSITION, because several rows of this corpus deliberately share an
event_id: keying by identifier would compare a revision against its original and invent a difference.
"""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direct", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    report_in = json.loads(open(args.direct, encoding="utf-8").read())
    single = [s["response"]["answers"] for s in report_in["single"]]
    sections = {}
    if "batch" in report_in:
        sections["batch"] = [r["response"]["answers"] for r in report_in["batch"]["responses"]]
    if "permuted" in report_in:
        restored = [None] * len(single)
        for position, index in enumerate(report_in["permuted"]["order"]):
            restored[index] = report_in["permuted"]["responses"][position]["response"]["answers"]
        sections["permuted_batch"] = restored

    def measure(left, right):
        labels = maps = 0
        deltas = [0.0]
        for a, b in zip(left, right):
            for question in a:
                labels += a[question]["choice"] == b[question]["choice"]
                maps += a[question]["probabilities"] == b[question]["probabilities"]
                deltas += [abs(a[question]["probabilities"][k] - b[question]["probabilities"][k])
                           for k in a[question]["probabilities"]]
        return {"rows": len(left), "labels_equal": labels, "probability_maps_exactly_equal": maps,
                "max_abs_probability_delta": max(deltas)}

    out = {"schema": "laya_batch_experiment.v1",
           "scope": "A PROPERTY OF THE SDK's BATCHED PATH, NOT OF THE WRAPPER",
           "single_call_seconds": [s["seconds"] for s in report_in["single"]],
           "settings": report_in.get("settings")}
    for name, other in sections.items():
        out[f"single_versus_{name}"] = {**measure(single, other),
                                        "seconds": report_in[name.replace("permuted_batch", "permuted")].get("seconds")}
    if len(sections) == 2:
        out["batch_versus_permuted_batch"] = measure(sections["batch"], sections["permuted_batch"])
    out["reading"] = ("equal labels with unequal probability maps means the batched path is numerically different, not that "
                      "the wrapper changed anything; the upstream README describes a smaller difference than observed here")
    text = json.dumps(out, indent=1, sort_keys=True)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
