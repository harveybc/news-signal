"""CL09 / NL01: a question the user wrote, not an alias for a task we shipped.

The distinction this file defends: Laya encodes the question, its options and the news TOGETHER, so a question is an input to
the model, not a label on a preset. Everything that reaches the encoder therefore belongs to the task's identity -- the
words, the option names, their descriptions and their order. Two different questions must produce two different identities
and two different SDK inputs, and neither may be quietly answered by the EURUSD preset.

Parity against real weights is measured by the pilot, not here. These tests establish the contract and the refusals.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from m5phet.runtime import Registry, Status, run

from news_signal.application import classify_event
from news_signal.core import Refusal, digest
from news_signal.provider import LayaNewsProvider, request_for, state_ref_for
from news_signal.question import AD_HOC_PREFIX, MAX_OPTIONS, ad_hoc_task, build_question
from news_signal.shadow import ShadowStore
from news_signal.tasks import questions as task_questions

from test_classification_first import AS_OF, CHECKPOINT_SHA, StubLaya, TASK, corpus_events

ROOT = Path(__file__).resolve().parents[1]

RATE_CUT = {"name": "rate_cut_risk", "question": "Does this news make a near-term euro-area rate cut more likely?",
            "options": [("more_likely", "the news points to easier policy"),
                        ("less_likely", "the news points to tighter policy"),
                        ("unclear", "the news does not bear on the policy path")]}

TONE = {"name": "market_tone", "question": "What tone does this news take about the euro area's economy?",
        "options": [("optimistic", "expects improvement"), ("pessimistic", "expects deterioration"),
                    ("neutral", "neither")]}


@pytest.fixture
def stub(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "news_checkpoint.v1", "sha256": CHECKPOINT_SHA, "files": {}}))
    backend = StubLaya()
    provider = LayaNewsProvider(environ={"NEWS_SIGNAL_MANIFEST": str(manifest), "NEWS_SIGNAL_DEVICE": "cpu"},
                                backend_factory=lambda config, state: backend)
    registry = Registry()
    registry.register(provider)
    return registry, backend


def ask(registry, event, spec, store=None, as_of=AS_OF):
    built = ad_hoc_task(**spec)
    return classify_event(event, task_id=built["task_id"], question_spec=spec, as_of=as_of,
                          registry=registry, store=store)


# --- two genuinely different questions over identical news ----------------------------------------------------------------

def test_two_different_questions_over_the_same_news_are_two_tasks(stub, tmp_path):
    registry, backend = stub
    event = corpus_events()["00_relevant"]
    first = ask(registry, event, RATE_CUT, ShadowStore(tmp_path / "s"))
    second = ask(registry, event, TONE, ShadowStore(tmp_path / "s"))
    assert first["status"] == second["status"] == "SHADOW_ONLY"
    assert first["receipt"]["task_id"] != second["receipt"]["task_id"]
    assert first["receipt"]["task_sha256"] != second["receipt"]["task_sha256"]
    assert first["receipt"]["input_sha256"] == second["receipt"]["input_sha256"], "the news is the same news"
    assert sorted(first["receipt"]["features"]) == ["rate_cut_risk"]
    assert sorted(second["receipt"]["features"]) == ["market_tone"]
    # what actually reached the model differs, not only the label on it
    asked = [tuple(q) for _state, q in backend.seen]
    assert ("rate_cut_risk",) in asked and ("market_tone",) in asked


def test_changing_a_criterion_changes_the_identity_and_the_model_input(stub):
    registry, backend = stub
    event = corpus_events()["00_relevant"]
    original = ask(registry, event, RATE_CUT)
    nudged = json.loads(json.dumps(RATE_CUT))
    nudged["options"][0][1] = "the news points to easier policy soon"
    changed = ask(registry, event, nudged)
    assert original["receipt"]["task_id"] != changed["receipt"]["task_id"], "a rubric is part of the model's input"
    sent = [json.dumps(r["question"]["questions"], sort_keys=True)
            for r in (original["receipt"], changed["receipt"])]
    assert sent[0] != sent[1]


def test_reordering_the_options_is_a_different_task(stub):
    registry, _backend = stub
    event = corpus_events()["00_relevant"]
    forward = ask(registry, event, RATE_CUT)
    reversed_spec = dict(RATE_CUT, options=list(reversed(RATE_CUT["options"])))
    backward = ask(registry, event, reversed_spec)
    assert forward["receipt"]["task_id"] != backward["receipt"]["task_id"]
    assert (forward["receipt"]["question"]["option_order"]["rate_cut_risk"]
            != backward["receipt"]["question"]["option_order"]["rate_cut_risk"])


def test_the_same_question_always_recovers_the_same_identity():
    assert ad_hoc_task(**RATE_CUT)["task_id"] == ad_hoc_task(**json.loads(json.dumps(RATE_CUT)))["task_id"]
    assert ad_hoc_task(**RATE_CUT)["task_id"].startswith(AD_HOC_PREFIX + ":")


def test_a_novel_question_is_never_answered_by_the_preset(stub):
    """The failure mode this forbids: mapping anything unrecognised onto the EURUSD relevance task and answering that."""
    registry, backend = stub
    event = corpus_events()["00_relevant"]
    outcome = ask(registry, event, RATE_CUT)
    assert outcome["receipt"]["features"].keys() == {"rate_cut_risk"}
    assert "relevance" not in outcome["receipt"]["features"]
    assert outcome["receipt"]["question"]["origin"] == "USER_AUTHORED_QUESTION"
    assert outcome["receipt"]["question"]["sha256"] != digest(task_questions(TASK))
    for _state, asked in backend.seen:
        assert asked == ("rate_cut_risk",)


def test_the_preset_still_works_and_is_labelled_as_one(stub):
    registry, _backend = stub
    outcome = classify_event(corpus_events()["00_relevant"], task_id=TASK, as_of=AS_OF, registry=registry)
    assert outcome["status"] == "SHADOW_ONLY"
    assert outcome["receipt"]["question"]["origin"] == "PRESET_TASK"
    assert outcome["receipt"]["task_id"] == TASK


# --- identity cannot be detached from the question it names ----------------------------------------------------------------

def test_an_identity_cannot_be_attached_to_a_different_question(stub):
    """Two ways in, both closed: building such a request refuses, and a hand-forged one refuses inside the runtime."""
    registry, backend = stub
    event = corpus_events()["00_relevant"]
    with pytest.raises(Refusal) as exc:
        request_for(event, task_id=ad_hoc_task(**RATE_CUT)["task_id"], state_ref=state_ref_for(CHECKPOINT_SHA),
                    request_id="forged", as_of=AS_OF, question_spec=TONE)
    assert "QUESTION_IDENTITY_MISMATCH" in str(exc.value)
    forged = request_for(event, task_id=ad_hoc_task(**TONE)["task_id"], state_ref=state_ref_for(CHECKPOINT_SHA),
                         request_id="forged", as_of=AS_OF, question_spec=TONE)
    forged["task_id"] = ad_hoc_task(**RATE_CUT)["task_id"]           # the label now names another question
    result = run(forged, registry)
    assert result["status"] == Status.INVALID_INPUT
    assert "QUESTION_IDENTITY_MISMATCH" in json.dumps(result["outputs"])
    assert backend.seen == [], "a forged identity never reaches the model"


def test_a_user_question_without_its_spec_is_refused(stub):
    registry, backend = stub
    event = corpus_events()["00_relevant"]
    request = request_for(event, task_id=TASK, state_ref=state_ref_for(CHECKPOINT_SHA),
                          request_id="bare", as_of=AS_OF)
    request["task_id"] = ad_hoc_task(**RATE_CUT)["task_id"]
    result = run(request, registry)
    assert result["status"] == Status.INVALID_INPUT
    assert "QUESTION_SPEC_REQUIRED" in json.dumps(result["outputs"])
    assert backend.seen == []


def test_an_unknown_task_kind_is_refused_by_the_state(stub):
    registry, backend = stub
    event = corpus_events()["00_relevant"]
    request = request_for(event, task_id=TASK, state_ref=state_ref_for(CHECKPOINT_SHA),
                          request_id="kind", as_of=AS_OF)
    request["task_id"] = "adhoc.question.v2:" + "0" * 64
    result = run(request, registry)
    assert result["status"] == Status.INVALID_INPUT
    assert "compatible task kind" in (result["why"] or "")
    assert backend.seen == []


# --- refusals, before anything is encoded -----------------------------------------------------------------------------------

@pytest.mark.parametrize("spec,fragment", [
    (dict(RATE_CUT, schema="free_text"), "UNSUPPORTED_ANSWER_SCHEMA"),
    (dict(RATE_CUT, question=""), "QUESTION_FIELD_REQUIRED"),
    (dict(RATE_CUT, question="x" * 601), "QUESTION_FIELD_TOO_LONG"),
    (dict(RATE_CUT, options=[("only_one", "a question with one answer is not a question")]), "QUESTION_OPTION_COUNT"),
    (dict(RATE_CUT, options=[(f"o{i}", "d") for i in range(MAX_OPTIONS + 1)]), "QUESTION_OPTION_COUNT"),
    (dict(RATE_CUT, options=[("Same", "a"), ("same ", "b")]), "DUPLICATE_OPTION"),
    (dict(RATE_CUT, options=[("a", ""), ("b", "d")]), "QUESTION_FIELD_REQUIRED"),
    (dict(RATE_CUT, options="not a mapping"), "QUESTION_OPTIONS_REQUIRED"),
    (dict(RATE_CUT, options=[("a", "x" * 401), ("b", "d")]), "QUESTION_FIELD_TOO_LONG"),
])
def test_a_malformed_question_is_refused_by_name(spec, fragment):
    with pytest.raises(Refusal) as exc:
        build_question(**spec)
    assert fragment in str(exc.value)


def test_a_question_that_would_overflow_the_shared_budget_is_refused_not_truncated():
    spec = dict(RATE_CUT, options=[(f"option_{i}", "d" * 399) for i in range(11)])
    with pytest.raises(Refusal) as exc:
        build_question(**spec)
    assert "QUESTION_BUDGET_EXCEEDED" in str(exc.value)
    assert "encoded together with the news" in str(exc.value)


def test_a_malformed_question_never_loads_the_model(stub):
    registry, backend = stub
    with pytest.raises(Refusal):
        ask(registry, corpus_events()["00_relevant"], dict(RATE_CUT, schema="free_text"))
    assert backend.seen == []


# --- the public command, and what survives a restart --------------------------------------------------------------------------

def test_the_public_command_asks_the_question_and_prints_what_was_sent():
    args = [sys.executable, "-m", "news_signal", "ask", "--input", str(ROOT / "examples/eurusd/events/00_relevant.json"),
            "--question", RATE_CUT["question"], "--name", RATE_CUT["name"],
            *[f"--option={n}={d}" for n, d in RATE_CUT["options"]], "--print-question"]
    done = subprocess.run(args, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    printed = json.loads(done.stdout)
    assert printed["task_id"] == ad_hoc_task(**RATE_CUT)["task_id"]
    assert printed["option_order"]["rate_cut_risk"] == [n for n, _d in RATE_CUT["options"]]
    assert printed["nothing_was_asked_of_the_model"] is True


def test_an_answer_to_a_user_question_survives_a_new_process(stub, tmp_path):
    registry, _backend = stub
    store_dir = tmp_path / "store"
    outcome = ask(registry, corpus_events()["00_relevant"], RATE_CUT, ShadowStore(store_dir))
    assert outcome["stored"]["disposition"] == "STORED"
    done = subprocess.run([sys.executable, "-m", "news_signal", "replay", "--store", str(store_dir)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["records"] == 1 and report["actionable_events"] == 1 and report["integrity_failures"] == []
    kept = ShadowStore(store_dir).all_records()[0]
    assert kept["task_id"] == ad_hoc_task(**RATE_CUT)["task_id"]
    assert kept["question"]["questions"]["rate_cut_risk"]["instructions"] == RATE_CUT["question"]
    assert kept["execution_authorized"] is False
