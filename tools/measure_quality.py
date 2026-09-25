#!/usr/bin/env python3
"""WP09: measure this package's classification quality on an independently labelled corpus, through the real checkpoint.

    python3 tools/measure_quality.py \
        --calendar <economic_calendar.csv> \
        --out-dir  ~/.local/state/m5phet/wp09-quality-20260925 \
        --evaluation-src <M5PHET checkout>/evaluation/src

Four phases, each of which writes what the next one reads, so an interrupted run resumes instead of restarting:

1. **corpus** — `news_signal.quality_corpus` builds a balanced, seeded, year-stratified sample of economic-calendar
   releases labelled by the calendar's own country field. The manifest and the rows are written to `--out-dir`, never
   into a repository: a corpus in git becomes a corpus somebody edits after seeing the scores.
2. **seal** — the evaluation protocol is declared and the corpus is sealed BEFORE anything is asked of the model, so a
   relabelled or trimmed corpus is detectable afterwards rather than deniable.
3. **run** — every row goes through the classification envelope to the private worker, one question per row, on the same
   transport the workbench uses. Each answer is checked to have come from the real backend and appended to a checkpoint
   file as it arrives. A fixture answer aborts the run by name.
4. **score** — `m5phet_evaluation` computes the confusion, the per-class numbers, macro-F1, accuracy and the
   majority-class baseline on the same scored rows; this tool adds the keyword baseline as a second arm on the same
   sealed rows, the calibration of the uncalibrated probabilities, and the closure table.

Nothing here tunes a prompt. The question's wording is fixed in `quality_corpus` and travels into the manifest, because
Laya encodes the question with the text: a second wording would be a second measurement and would have to be reported
as one.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from news_signal.quality_corpus import (ASSET, CLASSES, QUESTION_INSTRUCTIONS, QUESTION_NAME, QUESTION_OPTIONS,
                                        build_corpus, keyword_prediction)
from news_signal.quality_eval import (QUALITY_SCHEMA, brier, check_model_answer, closure_table, reliability,
                                     skill_from_scores)

#: the split is frozen at a declared instant and not at the clock, so two runs of this tool share one protocol digest
#: and their reports are comparable instead of merely similar
FROZEN_AT = "2026-09-25T00:00:00Z"

#: below this the report flags itself UNDERPOWERED. Declared in advance, as the protocol requires
MINIMUM_ROWS = 300

#: what a comparable literature value would have to be measured on. Named, so `NOT_CARRIED` is a statement with content
LITERATURE_NOTE = ("NOT_CARRIED: no published macro-F1 was found that is matched to this protocol -- zero-shot, three "
                   "classes, economic-calendar release lines, labels from the calendar's country field. Published "
                   "zero-shot financial-text numbers are sentiment or relevance tasks on other corpora, so quoting one "
                   "here would put an unmatched number in the comparison column.")


def load_evaluation(evaluation_src):
    """Import the evaluation package from the checkout the operator names. Nothing about its location is hardcoded."""
    src = Path(evaluation_src).expanduser().resolve()
    if not (src / "m5phet_evaluation").is_dir():
        raise SystemExit(f"EVALUATION_SRC_NOT_FOUND: {src} does not contain m5phet_evaluation")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from m5phet_evaluation import freeze, protocol, report, scoring
    comparison = None
    candidate = src.parent / "compare_stages.py"
    if candidate.is_file():
        spec = importlib.util.spec_from_file_location("wp09_compare_stages", candidate)
        comparison = importlib.util.module_from_spec(spec)
        # a dataclass declared inside a module loaded by path needs that module registered first
        sys.modules[spec.name] = comparison
        spec.loader.exec_module(comparison)
    return protocol, freeze, scoring, report, comparison


def build_engine():
    """The workbench's own engine, so this measurement takes exactly the route the product takes."""
    from m5phet.web.engine import Engine
    engine = Engine()
    if not engine.remote or not engine.remote_command:
        raise SystemExit("NO_WORKER_CONFIGURED: the classification worker is not configured in this environment; a "
                         "measurement run must not fall back to the in-process fixture")
    return engine


def envelope_for(event):
    """One envelope, one news item, one declared `choice` question with the three declared options."""
    return {"schema": "m5phet.task.questions.v1", "area": "classification", "as_of": event["received_at"],
            "state": {"news": event, "asset": ASSET, "language": "en"},
            "questions": {QUESTION_NAME: {"type": "choice", "instructions": QUESTION_INSTRUCTIONS,
                                          "options": [list(option) for option in QUESTION_OPTIONS]}}}


def ask(engine, event):
    """Send one envelope to the worker and return the answer, bound to the request it answered."""
    from m5phet.questions import digest as task_digest, validate_task
    task = envelope_for(event)
    response = engine._remote({"action": "task", "task": task, "data": None})
    if response.get("request_sha256") != task_digest(validate_task(task)):
        raise SystemExit(f"UNBOUND_WORKER_RESULT: the worker's answer for {event['event_id']!r} is not bound to the "
                         f"envelope that was sent")
    if response.get("execution_authorized") is not False:
        raise SystemExit("EXECUTION_AUTHORITY_CLAIMED: an answer declared execution authority")
    return response


def run_corpus(engine, items, answers_path, *, corpus_id, sleep=0.0):
    """Ask every unanswered row, appending each answer as it arrives so an interruption costs one row, not the run."""
    done = {}
    if answers_path.exists():
        for line in answers_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("corpus_id") != corpus_id:
                raise SystemExit(f"CHECKPOINT_FOREIGN_CORPUS: {answers_path} holds answers for corpus "
                                 f"{record.get('corpus_id')!r}, not {corpus_id!r}")
            done[record["event_id"]] = record
    pending = [entry for entry in items if entry["event"]["event_id"] not in done]
    print(f"{len(done)} answered already, {len(pending)} to ask", flush=True)
    started = time.time()
    with answers_path.open("a") as stream:
        for index, entry in enumerate(pending, start=1):
            event = entry["event"]
            response = ask(engine, event)
            answer = check_model_answer((response.get("answers") or {}).get(QUESTION_NAME), row=event["event_id"])
            record = {"corpus_id": corpus_id, "event_id": event["event_id"], "truth": entry["label"],
                      "predicted": answer["label"], "probabilities": answer["uncalibrated_probabilities"],
                      "backend": answer["backend"], "state_ref": response.get("state_ref"),
                      "request_sha256": response.get("request_sha256"),
                      "response_sha256": (answer.get("provenance") or {}).get("response_sha256"),
                      "latency_ms": response.get("latency_ms")}
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            done[event["event_id"]] = record
            if index % 25 == 0 or index == len(pending):
                rate = (time.time() - started) / index
                print(f"  {index}/{len(pending)} asked, {rate:.2f} s/row, "
                      f"{(len(pending) - index) * rate / 60:.1f} min left", flush=True)
            if sleep:
                time.sleep(sleep)
    return done


def calibration_of(records, order):
    """Top-label reliability, expected calibration error and the multiclass Brier score of the uncalibrated head."""
    rows = [(record["probabilities"][record["predicted"]], record["predicted"] == record["truth"])
            for record in records]
    diagram = reliability(rows)
    score = brier([record["probabilities"] for record in records], [record["truth"] for record in records], order)
    return {"reliability": diagram, "expected_calibration_error": diagram["expected_calibration_error"],
            "brier": score, "brier_definition": ("multiclass Brier, summed over classes: mean over rows of "
                                                 "sum_c (p_c - 1{truth=c})^2, range [0, 2]"),
            "statement": ("The provider declares these probabilities uncalibrated. These numbers measure by how much; "
                          "nothing here recalibrates them.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calendar", required=True, help="the economic calendar CSV (headerless)")
    parser.add_argument("--out-dir", required=True, help="where the corpus, the seal, the answers and the reports go")
    parser.add_argument("--evaluation-src", required=True, help="<M5PHET checkout>/evaluation/src")
    parser.add_argument("--rows-per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--frozen-at", default=FROZEN_AT)
    parser.add_argument("--build-only", action="store_true", help="build and seal the corpus, ask the model nothing")
    parser.add_argument("--score-only", action="store_true", help="score the checkpointed answers, ask nothing more")
    args = parser.parse_args()

    protocol_mod, freeze_mod, scoring_mod, report_mod, comparison = load_evaluation(args.evaluation_src)
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    manifest, items = build_corpus(args.calendar, seed=args.seed, rows_per_class=args.rows_per_class)
    (out / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (out / "corpus_rows.jsonl").open("w") as stream:
        for entry in items:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
    print(f"corpus {manifest['corpus_id'][:16]}: {manifest['counts']['per_class']}", flush=True)

    population = tuple(entry["event"]["event_id"] for entry in items)
    truth = {entry["event"]["event_id"]: entry["label"] for entry in items}
    declared = protocol_mod.EvaluationProtocol(
        family="classification",
        population=population,
        label_source=manifest["label_source"],
        label_producer=manifest["label_producer"],
        # the package's vocabulary for a label produced outside the system under test. The manifest carries the word
        # the work package uses, INDEPENDENT_LABELS, and the two say the same thing: no author here wrote a label
        label_provenance="INDEPENDENT_SYSTEM",
        annotation_rules=(
            f"No row was annotated. The label is the calendar's own country field under the declared mapping "
            f"{json.dumps(manifest['mapping'], sort_keys=True)}.",
            f"Releases by euro-area member states {sorted(manifest['excluded_countries'])} are excluded, not "
            f"relabelled: {manifest['exclusion_reason']}",
            f"The text the model reads is built by template {manifest['template']['version']} from the row's own "
            f"description, actual, consensus, previous, unit and expected volatility; the country is never inserted.",
            f"The sample is seeded ({manifest['seed']}), balanced ({manifest['rows_per_class']} per class) and "
            f"stratified over the calendar's years.",
        ),
        ambiguity_adjudication=(
            "No ambiguity was adjudicated. A release line that occurs under two different labels is dropped entirely "
            "rather than assigned to one of them; a line that repeats under one label is kept once. Contestable "
            "countries are excluded by the declared rule above."),
        split=(("holdout", population),),
        split_frozen_at=args.frozen_at,
        split_frozen_by="news_signal.quality_corpus (deterministic, seeded; the checkpoint is zero-shot and was fitted "
                        "on none of these rows)",
        metrics=("macro_f1", "accuracy", "per_class_f1", "confusion", "expected_calibration_error", "brier"),
        baseline="majority_class",
        minimum_rows=MINIMUM_ROWS,
    )
    seal = freeze_mod.seal_corpus(truth, protocol=declared, sealed_at=args.frozen_at)
    (out / "protocol.json").write_text(json.dumps(declared.to_dict(), indent=2, sort_keys=True) + "\n")
    seal.write(out / "corpus_seal.json")
    print(f"protocol {declared.digest[:12]}  seal {seal.seal[:12]}  rows {seal.row_count}", flush=True)
    if args.build_only:
        return 0

    answers_path = out / "answers.jsonl"
    if args.score_only:
        records = {}
        for line in answers_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["event_id"]] = record
    else:
        records = run_corpus(build_engine(), items, answers_path, corpus_id=manifest["corpus_id"])
    missing = [row for row in population if row not in records]
    if missing:
        raise SystemExit(f"INCOMPLETE_RUN: {len(missing)} of {len(population)} rows have no answer, first {missing[0]!r}")

    ordered = [records[row] for row in population]
    laya = {row: records[row]["predicted"] for row in population}
    texts = {entry["event"]["event_id"]: entry["event"]["body"] for entry in items}
    keyword = {row: keyword_prediction(texts[row]) for row in population}

    reports = {}
    for stage, predictions in (("laya_zero_shot", laya), ("keyword_baseline", keyword)):
        metric_sets = [scoring_mod.score_classification(protocol=declared, seal=seal, truth=truth,
                                                        predictions=predictions, labels=CLASSES)]
        if stage == "laya_zero_shot":
            metric_sets.append(scoring_mod.MetricSet(
                name="calibration", family="classification", population=population,
                values=calibration_of(ordered, CLASSES),
                counts={"scored_rows": len(population), "declared_rows": len(population)},
                baseline=None,
                notes=("Calibration is measured, never applied: the provider's probabilities stay uncalibrated.",)))
        built = report_mod.build_report(protocol=declared, seal=seal, metric_sets=metric_sets)
        payload = json.loads(built.to_json())
        annotations = {"stage": stage, "target": "economy_named_in_release", "horizon": "none (single item)",
                       "scale": "macro-F1 over three classes (euro_area, united_states, other), higher is better",
                       "literature": {"value": None, "metric": "macro_f1", "scale": "three-class zero-shot",
                                      "source": LITERATURE_NOTE}}
        if comparison is not None:
            payload = comparison.annotate(built, **annotations)
        else:
            payload.update(annotations)
        path = out / f"report_{stage}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        reports[stage] = (built, metric_sets[0], path)
        print(f"{stage}: macro_f1 {metric_sets[0].values['macro_f1']:.6f}  "
              f"accuracy {metric_sets[0].values['accuracy']:.6f}  "
              f"majority macro_f1 {metric_sets[0].baseline['macro_f1']:.6f}", flush=True)

    rows = []
    for stage, (_built, metrics, _path) in reports.items():
        naive = metrics.baseline
        computed = skill_from_scores(metrics.values["macro_f1"], naive["macro_f1"])
        rows.append({"arm": stage, "metric": "macro_f1",
                     "scale": "three classes, unweighted mean of per-class F1",
                     "n": metrics.counts["scored_rows"], "model_error": computed["model_error"],
                     "naive_name": f"{naive['name']} ({naive['class']}), same rows",
                     "naive_error": computed["naive_error"], "skill": computed["skill"],
                     "skill_status": computed["status"], "literature_value": "NOT_CARRIED",
                     "literature_source": LITERATURE_NOTE.split(":")[0],
                     "comparability": "COMPARABLE: same sealed rows, same metric, same naive reference"})
    laya_metrics = reports["laya_zero_shot"][1]
    keyword_metrics = reports["keyword_baseline"][1]
    rows.append({"arm": "laya_zero_shot vs keyword_baseline", "metric": "macro_f1",
                 "scale": "three classes, unweighted mean of per-class F1",
                 "n": laya_metrics.counts["scored_rows"],
                 "model_error": 1.0 - laya_metrics.values["macro_f1"],
                 "naive_name": "keyword_baseline (country name verbatim), same rows",
                 "naive_error": 1.0 - keyword_metrics.values["macro_f1"],
                 "skill": skill_from_scores(laya_metrics.values["macro_f1"],
                                            keyword_metrics.values["macro_f1"])["skill"],
                 "skill_status": skill_from_scores(laya_metrics.values["macro_f1"],
                                                   keyword_metrics.values["macro_f1"])["status"],
                 "literature_value": "NOT_CARRIED", "literature_source": LITERATURE_NOTE.split(":")[0],
                 "comparability": "COMPARABLE: same sealed rows, same metric"})
    table = closure_table(rows, title=f"WP09 closure -- corpus {manifest['corpus_id'][:16]}, "
                                      f"seal {seal.seal[:12]}, {seal.row_count} rows")
    (out / "closure_table.md").write_text(table)
    calibration = reports["laya_zero_shot"][0].metric_sets[1].values
    summary = {"corpus_id": manifest["corpus_id"], "protocol_digest": declared.digest, "corpus_seal": seal.seal,
               "rows": seal.row_count, "per_class": manifest["counts"]["per_class"],
               "laya": {"macro_f1": laya_metrics.values["macro_f1"], "accuracy": laya_metrics.values["accuracy"],
                        "per_class": laya_metrics.values["per_class"], "confusion": laya_metrics.values["confusion"],
                        "majority_baseline": laya_metrics.baseline,
                        "expected_calibration_error": calibration["expected_calibration_error"],
                        "brier": calibration["brier"]},
               "keyword_baseline": {"macro_f1": keyword_metrics.values["macro_f1"],
                                    "accuracy": keyword_metrics.values["accuracy"],
                                    "per_class": keyword_metrics.values["per_class"]},
               "closure_rows": rows, "headline": reports["laya_zero_shot"][0].headline()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # WP09's done-when: the catalog carries the quality with the corpus it was measured on, or says NOT_MEASURED. The
    # record is a file the operator points NEWS_SIGNAL_QUALITY at, so deleting it takes the claim down with it
    laya_skill = skill_from_scores(laya_metrics.values["macro_f1"], laya_metrics.baseline["macro_f1"])
    (out / "quality.json").write_text(json.dumps({
        "schema": QUALITY_SCHEMA,
        "corpus_id": manifest["corpus_id"],
        "n": laya_metrics.counts["scored_rows"],
        "macro_f1": laya_metrics.values["macro_f1"],
        "accuracy": laya_metrics.values["accuracy"],
        "calibration": {"status": "UNCALIBRATED",
                        "expected_calibration_error": calibration["expected_calibration_error"],
                        "brier": calibration["brier"], "bins": calibration["reliability"]["bin_count"],
                        "statement": calibration["statement"]},
        "protocol_digest": declared.digest,
        "corpus_seal": seal.seal,
        "label_provenance": f"{manifest['label_provenance']} ({declared.label_provenance} in the evaluation package's "
                            f"vocabulary): {manifest['label_source']}",
        "naive": {"majority_class": laya_metrics.baseline["macro_f1"],
                  "keyword_baseline": keyword_metrics.values["macro_f1"], "metric": "macro_f1", "same_rows": True},
        "skill": laya_skill,
        "question": manifest["question"],
        "reading": ("macro-F1 on 450 sealed, independently labelled rows of one task: which economy a calendar release "
                    "line names. It is not a general claim about this classifier on other tasks or other texts."),
    }, indent=2, sort_keys=True) + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
