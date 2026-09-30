"""WP09: what is computed from the worker's answers, and the one thing that invalidates the whole run.

The scoring itself belongs to `m5phet_evaluation` — confusion, per-class precision/recall/F1, macro-F1, accuracy and the
majority-class baseline on the same scored rows — and this module does not reimplement any of it. What lives here is
what that package does not compute: the gate that refuses a run answered by a declared non-model, the calibration of the
uncalibrated probabilities, and the closure table the owner requires of every quality claim.

**The gate.** A fixture answers instantly, deterministically and plausibly, which is exactly why a fixture run looks like
a measurement. `check_model_answer` refuses by name on the first answer that does not come from the real checkpoint, and
the refusal names the row, so a run cannot end with a number nobody can trace to a model.

**The calibration.** The provider states on every answer that its probabilities are uncalibrated. This module measures
by how much: the top-label reliability in ten fixed bins, the expected calibration error over those bins, and the
multiclass Brier score summed over classes. Nothing here recalibrates anything; a measurement of miscalibration is not a
repair of it.

**The skill.** The owner's closure table wants a skill, and skill is defined against an error. Macro-F1 is a score, so
the error is declared once, here, as `1 - macro_f1`, and the skill is `(naive_error - model_error) / naive_error` on the
same rows. The declaration travels in the table, because a skill whose error definition is not written down is a ratio
of two numbers nobody can reproduce.

**The checkpoint.** A quality claim that does not name the checkpoint it was computed against can be quoted beside a
different one. The versioned contract below refuses that before any score. The identifier is the one the sealed manifest
already declares (`laya-checkpoint:` plus the manifest's own sha256, the same string `state_ref_for` publishes). Nothing
here hashes a weight file, opens `NEWS_SIGNAL_CHECKPOINT`, or computes a macro-F1.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

from . import core
from .core import Refusal

#: the bin edges of the reliability diagram: ten fixed bins over [0, 1], declared so two runs bin identically
BIN_COUNT = 10

#: how the closure table's error is obtained from a score. Declared once; quoted in the table
ERROR_DEFINITION = "error = 1 - macro_f1"

BRIER_DEFINITION = "multiclass Brier, summed over classes: mean over rows of sum_c (p_c - 1{truth=c})^2, range [0, 2]"


class FixtureRun(Refusal):
    """The run was answered by something other than the real checkpoint. A number from here is never reported."""


def check_model_answer(answer, *, row):
    """Refuse by name unless this answer came from the real Laya backend. The first offending row aborts the run."""
    if not isinstance(answer, dict):
        raise FixtureRun(f"NO_ANSWER: row {row!r} carries no answer object")
    if answer.get("status") != "OK":
        raise FixtureRun(f"ANSWER_NOT_OK: row {row!r} answered {answer.get('status')!r}: {answer.get('why')!r}")
    backend = answer.get("backend")
    if backend != "laya":
        raise FixtureRun(f"FIXTURE_BACKEND_REFUSED: row {row!r} was answered by backend {backend!r}, not 'laya'; a "
                         f"measurement answered by a declared non-model measures nothing")
    if answer.get("non_model_fixture"):
        raise FixtureRun(f"NON_MODEL_FIXTURE: row {row!r} is marked as fixture output by the provider itself")
    return answer


def reliability(rows, *, bins=BIN_COUNT):
    """Top-label reliability and the expected calibration error over `bins` fixed bins of equal width.

    `rows` is a sequence of `(confidence, correct)`. A bin holds the confidences in `[lo, hi)`, the last one closing on
    1.0 so a probability of exactly 1 is never dropped. Empty bins are reported with zero count and no numbers, because
    a bin with no rows has neither an accuracy nor a mean confidence."""
    if bins < 1:
        raise Refusal("BIN_COUNT: at least one bin is required")
    entries = [(float(confidence), bool(correct)) for confidence, correct in rows]
    if not entries:
        raise Refusal("NO_ROWS: reliability needs at least one scored row")
    width = 1.0 / bins
    table, total = [], len(entries)
    ece = 0.0
    for index in range(bins):
        low, high = index * width, (index + 1) * width
        if index == bins - 1:
            held = [entry for entry in entries if low <= entry[0] <= high]
        else:
            held = [entry for entry in entries if low <= entry[0] < high]
        count = len(held)
        if count:
            accuracy = sum(1 for _, correct in held if correct) / count
            confidence = sum(value for value, _ in held) / count
            ece += (count / total) * abs(accuracy - confidence)
        else:
            accuracy = confidence = None
        table.append({"bin": [round(low, 4), round(high, 4)], "count": count,
                      "mean_confidence": confidence, "accuracy": accuracy})
    return {"bins": table, "expected_calibration_error": ece, "rows": total, "bin_count": bins}


def brier(probabilities, truth, classes):
    """The multiclass Brier score, summed over classes. A row whose probabilities omit a declared class is refused."""
    names = list(classes)
    if not names:
        raise Refusal("CLASSES: at least one declared class is required")
    rows = list(probabilities)
    if not rows or len(rows) != len(truth):
        raise Refusal("BRIER_ROWS: one probability vector and one truth label per row is required")
    total = 0.0
    for vector, label in zip(rows, truth):
        if set(vector) != set(names):
            raise Refusal(f"BRIER_LABELS: probabilities over {sorted(vector)} do not cover the declared {sorted(names)}")
        total += sum((float(vector[name]) - (1.0 if name == label else 0.0)) ** 2 for name in names)
    return total / len(rows)


def skill_from_scores(model_macro_f1, naive_macro_f1):
    """`(naive_error - model_error) / naive_error` with `error = 1 - macro_f1`. A zero naive error is UNDEFINED."""
    model_error = 1.0 - float(model_macro_f1)
    naive_error = 1.0 - float(naive_macro_f1)
    if naive_error == 0:
        return {"error_definition": ERROR_DEFINITION, "model_error": model_error, "naive_error": naive_error,
                "skill": None, "status": "UNDEFINED: the naive reference makes no error on these rows"}
    return {"error_definition": ERROR_DEFINITION, "model_error": model_error, "naive_error": naive_error,
            "skill": (naive_error - model_error) / naive_error, "status": "OK"}


def _number(value, decimals=6):
    return "NOT_COMPUTED" if value is None else f"{float(value):.{decimals}f}"


def closure_table(rows, *, title, decimals=6):
    """The owner's closure table as Markdown: one row per arm, every column present, nothing inferred.

    Each entry of `rows` declares `arm`, `metric`, `scale`, `n`, `model_error`, `naive_name`, `naive_error`, `skill`,
    `literature` and `comparability`. A missing literature value is `NOT_CARRIED`, which is a statement about this
    report and not about the world."""
    header = ("| arm | metric | scale | n | model error | naive reference | naive error | skill | literature value | "
              "literature source | comparability |")
    lines = [f"### {title}", "",
             f"Error definition: `{ERROR_DEFINITION}`. Numbers are rendered with {decimals} decimals, fixed.", "",
             header, "|" + "---|" * 11]
    for row in rows:
        lines.append("| " + " | ".join([
            str(row["arm"]), str(row["metric"]), str(row["scale"]), str(row["n"]),
            _number(row["model_error"], decimals), str(row["naive_name"]), _number(row["naive_error"], decimals),
            _number(row["skill"], decimals) if row.get("skill") is not None else str(row.get("skill_status", "UNDEFINED")),
            str(row.get("literature_value", "NOT_CARRIED")), str(row.get("literature_source", "NOT_CARRIED")),
            str(row["comparability"]),
        ]) + " |")
    return "\n".join(lines) + "\n"


#: the shape of the record the catalog publishes, and the environment variable that names the file holding it.
#: v1 is the historical claim. It has no checkpoint identity, and this loader does not invent one for it.
QUALITY_SCHEMA = "news_signal.quality.v1"
QUALITY_ENV = "NEWS_SIGNAL_QUALITY"

#: the versioned contract. A record under it names the checkpoint it was computed against. v1 is not this contract.
QUALITY_CONTRACT_VERSION = "news_signal.quality.v2"

#: the sealed manifest's own schema. Identity is read from a manifest that already carries it.
MANIFEST_SCHEMA = "news_checkpoint.v1"

#: `provider.state_ref_for` publishes this same string from the manifest sha256, before any weights are loaded.
CHECKPOINT_IDENTITY_PREFIX = "laya-checkpoint:"

_DECLARED_IDENTITY = re.compile(r"laya-checkpoint:[0-9a-f]{64}\Z")

#: a row may already carry the identifier under the contract's name or under the provider's state ref. Both are the
#: manifest's declaration. A bare file hash is neither.
_IDENTITY_KEYS = ("checkpoint_identity", "state_ref")

#: what the catalog says when no measurement has been made. A missing measurement is stated, never left blank
NOT_MEASURED = "NOT_MEASURED"

#: every field a published quality claim must carry. `corpus_id` heads the list on purpose: a macro-F1 whose corpus
#: nobody can name is a number without a population, and WP09's rule is that no number travels without its corpus id.
#: `checkpoint_identity` is not in this list: a historical v1 claim did not carry it, and requiring it here would stamp
#: one on. The versioned contract checks it on its own.
QUALITY_FIELDS = ("schema", "corpus_id", "n", "macro_f1", "calibration", "protocol_digest", "corpus_seal",
                  "label_provenance", "naive", "skill")

#: score fields this module does not report. They belong to `m5phet_evaluation`. Copying one out of a callback would
#: make a number computed beside the contract look like a measurement this contract produced.
_SCORES_NOT_EMITTED = ("macro_f1", "accuracy", "f1", "per_class", "confusion", "brier", "expected_calibration_error")


def parse_checkpoint_identity(value):
    """The manifest's declared identifier, or None. A hash of weight bytes is not rewritten into this form."""
    if isinstance(value, str) and _DECLARED_IDENTITY.fullmatch(value):
        return value
    return None


def checkpoint_identity_from_manifest(manifest):
    """The identifier the sealed manifest already declares. No weight file is opened and no digest is computed.

    A path is refused rather than read: `NEWS_SIGNAL_CHECKPOINT` points at weights, and hashing them would be a new
    identity, not the one the manifest recorded when it was sealed. A declared non-model names no checkpoint."""
    if isinstance(manifest, (str, Path)):
        raise Refusal("CHECKPOINT_IDENTITY_ABSENT: a filesystem path is not a checkpoint identity; the sealed manifest "
                      "declares the identifier and no weight file is read")
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise Refusal("CHECKPOINT_IDENTITY_ABSENT: no sealed news_checkpoint.v1 manifest declares the checkpoint; "
                      "nothing is scored and no weight file is read")
    if manifest.get("kind") == "NON_MODEL_FIXTURE":
        raise Refusal("CHECKPOINT_IDENTITY_ABSENT: a declared non-model names no checkpoint; nothing is scored")
    # `files` is not walked. The sha256 below is the manifest's own field, used verbatim.
    identity = parse_checkpoint_identity(CHECKPOINT_IDENTITY_PREFIX + str(manifest.get("sha256") or ""))
    if identity is None:
        raise Refusal("CHECKPOINT_IDENTITY_ABSENT: the manifest declares no sha256 identity; a weight file is not "
                      "hashed in its place and nothing is scored")
    return identity


def _row_label(row, index):
    if isinstance(row, dict):
        for key in ("row", "event_id"):
            if row.get(key) not in (None, ""):
                return row[key]
    return index


def _identities_on(row):
    """The declared identities on one row. None means the row does not carry the contract's identifier."""
    if not isinstance(row, dict):
        return None
    found = []
    for key in _IDENTITY_KEYS:
        if key not in row or row[key] in (None, ""):
            continue
        parsed = parse_checkpoint_identity(row[key])
        if parsed is None:
            return None
        found.append(parsed)
    return found


def require_checkpoint_identity(rows):
    """The one checkpoint a batch was computed against, or a refusal before any score.

    The first row that names nothing, or that names something other than the checkpoint the batch already bound,
    aborts the batch. Two identities are not averaged and not reported side by side as one number. The return value
    is the identifier only: this function emits no F1."""
    materialized = list(rows)
    if not materialized:
        raise Refusal("CHECKPOINT_IDENTITY_ABSENT: the batch has no row that names a checkpoint; nothing is scored "
                      "and no F1 is emitted")
    chosen = None
    for index, row in enumerate(materialized):
        where = _row_label(row, index)
        found = _identities_on(row)
        if not found:
            raise Refusal(f"CHECKPOINT_IDENTITY_ABSENT: row {where!r} names no checkpoint identity declared by the "
                          f"manifest contract; scoring is refused and no F1 is emitted")
        if any(item != found[0] for item in found):
            raise Refusal(f"CHECKPOINT_IDENTITY_MIXED: row {where!r} names {found[0]!r} and {found[1]!r}; identities "
                          f"are not blended and no F1 is emitted")
        identity = found[0]
        if chosen is None:
            chosen = identity
        elif identity != chosen:
            raise Refusal(f"CHECKPOINT_IDENTITY_MIXED: row {where!r} names {identity!r}, not {chosen!r}; a batch is "
                          f"one checkpoint or it is not scored, and no F1 is emitted")
    return chosen


def quality_contract_report(rows, score=None):
    """A versioned contract record for one batch. `score` is called only after every row names the same checkpoint.

    `score`, when given, is the caller's scorer. This function does not compute confusion, macro-F1, accuracy or a
    baseline, and it drops those keys if the callback returns them, so a number cannot be mistaken for one this
    contract measured. The record echoes the checkpoint identity. A refusal raises and returns nothing."""
    materialized = list(rows)
    identity = require_checkpoint_identity(materialized)
    report = {"schema": QUALITY_CONTRACT_VERSION, "version": QUALITY_CONTRACT_VERSION,
              "checkpoint_identity": identity, "n": len(materialized)}
    if score is None:
        return report
    measured = score(materialized)
    if not isinstance(measured, dict):
        raise Refusal("QUALITY_MEASUREMENT_REFUSED: the scorer returned nothing this contract can bind; no F1 is emitted")
    for key, value in measured.items():
        if key in ("schema", "version", "checkpoint_identity") or key in _SCORES_NOT_EMITTED:
            continue
        report[key] = value
    return report


def _as_versioned_contract(payload, where):
    """The versioned contract, or a refusal. The payload is not modified: a historical record stays as it was."""
    if not isinstance(payload, dict) or payload.get("schema") != QUALITY_CONTRACT_VERSION:
        found = payload.get("schema") if isinstance(payload, dict) else type(payload).__name__
        raise Refusal(f"NOT_A_VERSIONED_CONTRACT: {where} is {found!r}, not {QUALITY_CONTRACT_VERSION}; a historical "
                      f"record is not given a checkpoint identity it did not carry")
    if payload.get("version") != QUALITY_CONTRACT_VERSION:
        raise Refusal(f"NOT_A_VERSIONED_CONTRACT: {where} declares version {payload.get('version')!r}, not "
                      f"{QUALITY_CONTRACT_VERSION}")
    if parse_checkpoint_identity(payload.get("checkpoint_identity")) is None:
        raise Refusal(f"CHECKPOINT_IDENTITY_ABSENT: {where} names no checkpoint identity; the record is not a "
                      f"versioned quality contract and no F1 is emitted")
    return payload


def load_quality_contract(payload):
    """Accept only a versioned contract record. Historical quality claims are refused, not upgraded."""
    return _as_versioned_contract(payload, "record")


def quality_record(environ=None):
    """The measured quality this installation may publish, or `NOT_MEASURED`.

    The record is a file the operator points at, not a constant in the source, so a number can never outlive the run
    that produced it: delete the file and the catalog says NOT_MEASURED again. A file that exists but does not carry
    every declared field -- the corpus id above all -- is refused rather than published in part, because a partial
    quality claim reads exactly like a complete one.

    A historical `news_signal.quality.v1` record is published as it was written. It names no checkpoint, and none is
    added. A `news_signal.quality.v2` record is the versioned contract and is published only when it already names
    the checkpoint. Either way this function does not score."""
    env = os.environ if environ is None else environ
    path = env.get(QUALITY_ENV)
    if not path:
        return NOT_MEASURED
    payload = core.load_json(Path(path).expanduser())
    if isinstance(payload, dict) and payload.get("schema") == QUALITY_CONTRACT_VERSION:
        _as_versioned_contract(payload, repr(path))
        missing = [field for field in QUALITY_FIELDS if payload.get(field) in (None, "")]
        if missing:
            raise Refusal(f"INCOMPLETE_QUALITY_RECORD: {path!r} carries no {missing}; a quality claim without its "
                          f"corpus, its counts and its naive reference is not a measurement")
        return payload
    if not isinstance(payload, dict) or payload.get("schema") != QUALITY_SCHEMA:
        raise Refusal(f"UNKNOWN_QUALITY_SCHEMA: {path!r} is not a {QUALITY_SCHEMA} record")
    missing = [field for field in QUALITY_FIELDS if payload.get(field) in (None, "")]
    if missing:
        raise Refusal(f"INCOMPLETE_QUALITY_RECORD: {path!r} carries no {missing}; a quality claim without its corpus, "
                      f"its counts and its naive reference is not a measurement")
    return payload
