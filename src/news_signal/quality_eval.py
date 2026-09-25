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
"""

from __future__ import annotations

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
