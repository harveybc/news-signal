"""The unified question envelope on the native area: named `choice` questions in, named answers out, one SDK call.

These tests drive `m5phet.questions.run_task` with the real Registry, the real provider and the stub SDK from
`test_classification_first`. They establish the CONTRACT -- per-question refusal by name, one backend call for a whole set,
the state wrapped and validated the way the typed path validates it, the SDK's numbers untouched -- and nothing about what
the real weights would answer.
"""

import json

import pytest

from m5phet.questions import (MALFORMED_QUESTION, STATE_REQUIRED, TASK_SCHEMA, UNSUPPORTED_QUESTION_TYPE, catalog,
                              run_task)
from m5phet.runtime import Registry

from news_signal.core import digest
from news_signal.provider import LayaNewsProvider, PROVIDER_NAME, state_ref_for
from news_signal.question import build_question

from test_classification_first import AS_OF, CHECKPOINT_SHA, StubLaya, corpus_events

RELEVANCE = {"type": "choice",
             "instructions": "Does this news item bear directly on the euro or the US dollar?",
             "options": [["related", "ECB or Fed policy, euro-area or US inflation, employment, growth or trade data"],
                         ["unrelated", "No direct economic bearing on EUR or USD"],
                         ["unclear", "Not enough evidence to decide"]]}
TONE = {"type": "choice",
        "instructions": "What is the tone of the item for the named asset?",
        "options": [["hawkish", "Tighter policy or stronger currency"], ["dovish", "Looser policy or weaker currency"],
                    ["neutral", "No directional reading"]]}


def envelope(state, **questions):
    return {"schema": TASK_SCHEMA, "area": "classification", "state": state, "questions": questions, "as_of": AS_OF}


@pytest.fixture
def stub_registry(manifest_file):
    backend = StubLaya()
    provider = LayaNewsProvider(environ={"NEWS_SIGNAL_MANIFEST": str(manifest_file), "NEWS_SIGNAL_DEVICE": "cpu"},
                                backend_factory=lambda config, manifest: backend)
    registry = Registry()
    registry.register(provider)
    return registry, provider, backend


@pytest.fixture
def manifest_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "news_checkpoint.v1", "sha256": CHECKPOINT_SHA, "files": {}}))
    return path


def test_the_provider_declares_the_native_area_and_the_choice_type(stub_registry):
    registry, provider, _ = stub_registry
    cat = catalog(registry)
    assert cat["classification"]["provider"] == PROVIDER_NAME
    assert cat["classification"]["question_types"] == {"choice": {"required": ["options"], "optional": ["instructions"]}}
    assert provider.capabilities()["question_types"] == cat["classification"]["question_types"]


def test_two_choice_questions_in_one_envelope_are_one_backend_call(stub_registry):
    registry, provider, backend = stub_registry
    event = corpus_events()["00_relevant"]
    out = run_task(envelope({"news": event}, relevancia=RELEVANCE, tono=TONE), registry)
    assert out["provider"] == PROVIDER_NAME and out["answered"] == 2 and out["refused"] == 0, out["answers"]
    assert list(out["answers"]) == ["relevancia", "tono"]
    assert len(backend.seen) == 1 and provider.calls == 1, "the set conditions the encoding: one set, one call"
    state, sent = backend.seen[0]
    assert sent == ("relevancia", "tono") and json.loads(state)["body"] == event["body"]
    assert out["answers"]["relevancia"]["label"] == "related"
    assert out["answers"]["tono"]["type"] == "choice" and out["answers"]["tono"]["status"] == "OK"
    assert out["state_ref"] == state_ref_for(CHECKPOINT_SHA)
    assert out["execution_authorized"] is False
    for answer in out["answers"].values():
        assert answer["calibration"] == "UNCALIBRATED" and answer["uncertainty"] == "UNCALIBRATED_CLASS_PROBABILITIES"
        assert "confidence" not in {k for k in answer if k != "sdk_answer"}
        assert answer["population"]["event_ids"] == [event["event_id"]]


def test_the_stubs_numbers_pass_through_untouched(stub_registry):
    registry, _, backend = stub_registry
    event = corpus_events()["00_relevant"]
    out = run_task(envelope({"news": event}, relevancia=RELEVANCE, tono=TONE), registry)
    questions = {}
    for name, question in (("relevancia", RELEVANCE), ("tono", TONE)):
        questions.update(build_question(name=name, question=question["instructions"], options=question["options"]))
    native = backend.predict(json.dumps({k: event[k] for k in ("asset", "headline", "body")}, sort_keys=True,
                                        ensure_ascii=False, separators=(",", ":")), questions)["answers"]
    for name in ("relevancia", "tono"):
        answer = out["answers"][name]
        assert answer["sdk_answer"] == native[name], "the wrapper must not round, reorder or renormalize the SDK's answer"
        assert answer["uncalibrated_probabilities"] == native[name]["probabilities"]
        assert list(answer["uncalibrated_probabilities"]) == list(native[name]["probabilities"])
        assert answer["label"] == native[name]["choice"] and answer["probability_decimals"] == 4
    assert out["answers"]["relevancia"]["provenance"]["questions_sha256"] == digest(questions)


def test_a_malformed_question_is_refused_by_name_while_the_other_is_answered(stub_registry):
    registry, _, backend = stub_registry
    duplicate = dict(TONE, options=[["hawkish", "Tighter"], ["Hawkish ", "Tighter, again"], ["dovish", "Looser"]])
    out = run_task(envelope({"news": corpus_events()["00_relevant"]}, relevancia=RELEVANCE, tono=duplicate), registry)
    assert out["answers"]["tono"]["status"] == "REFUSED"
    assert out["answers"]["tono"]["refusal"] == MALFORMED_QUESTION and "DUPLICATE_OPTION" in out["answers"]["tono"]["why"]
    assert out["answers"]["tono"]["type"] == "choice" and "label" not in out["answers"]["tono"]
    assert out["answers"]["relevancia"]["status"] == "OK" and out["answered"] == 1 and out["refused"] == 1
    assert len(backend.seen) == 1 and backend.seen[0][1] == ("relevancia",), "the refused question never reached the SDK"


@pytest.mark.parametrize("bad, marker", [
    (dict(TONE, instructions="   "), "QUESTION_FIELD_REQUIRED"),
    ({"type": "choice", "options": TONE["options"]}, "QUESTION_FIELD_REQUIRED"),
    (dict(TONE, options=[["only", "one option"]]), "QUESTION_OPTION_COUNT"),
    (dict(TONE, options=[["hawkish", "x" * 401], ["dovish", "Looser"]]), "QUESTION_FIELD_TOO_LONG"),
])
def test_every_gate_of_the_question_compiler_refuses_by_name(stub_registry, bad, marker):
    registry, _, _ = stub_registry
    out = run_task(envelope({"news": corpus_events()["00_relevant"]}, relevancia=RELEVANCE, tono=bad), registry)
    assert out["answers"]["tono"]["refusal"] == MALFORMED_QUESTION and marker in out["answers"]["tono"]["why"]
    assert out["answers"]["relevancia"]["status"] == "OK"


def test_an_option_the_pinned_sdk_would_cut_is_refused_by_name_and_never_sent(stub_registry):
    registry, provider, backend = stub_registry
    # the stub tokenizer counts words; 49 words is over the SDK's 48-token option limit and under the character limits
    long = dict(TONE, options=[["hawkish", " ".join(["w"] * 49)], ["dovish", "Looser"]])
    out = run_task(envelope({"news": corpus_events()["00_relevant"]}, relevancia=RELEVANCE, tono=long), registry)
    assert out["answers"]["tono"]["refusal"] == MALFORMED_QUESTION
    assert "TOKEN_BUDGET_EXCEEDED" in out["answers"]["tono"]["why"]
    assert out["answers"]["relevancia"]["status"] == "OK" and backend.seen[0][1] == ("relevancia",)
    assert provider.last_budget_checked is True and provider.last_budget["tono"]["fits"] is False


def test_an_undeclared_type_never_reaches_the_provider(stub_registry):
    registry, _, backend = stub_registry
    out = run_task(envelope({"news": corpus_events()["00_relevant"]}, relevancia=RELEVANCE,
                            n={"type": "point_forecast", "horizon": 3}), registry)
    assert out["answers"]["n"]["refusal"] == UNSUPPORTED_QUESTION_TYPE and out["answers"]["relevancia"]["status"] == "OK"
    assert backend.seen[0][1] == ("relevancia",)


def test_text_news_is_wrapped_into_a_manual_context_event(stub_registry):
    registry, provider, backend = stub_registry
    text = "The central bank raised rates by 25 basis points citing persistent inflation."
    task = envelope({"asset": "EURUSD", "language": "en", "news": text}, relevancia=RELEVANCE)
    task["as_of"] = None                      # text has no clocks of its own, so it is judged at the present clock
    out = run_task(task, registry)
    assert out["answers"]["relevancia"]["status"] == "OK", out["answers"]
    assert out["answers"]["relevancia"]["label"] == "related"
    state = json.loads(backend.seen[0][0])
    assert state == {"asset": "EURUSD", "headline": "Manual context", "body": text}
    population = out["answers"]["relevancia"]["population"]
    assert population["asset"] == "EURUSD" and population["event_ids"][0].startswith("manual:")
    assert provider.last_inference["event_id"] == population["event_ids"][0]


def test_text_news_may_travel_as_data_and_cannot_be_replayed_at_a_historical_clock(stub_registry):
    registry, _, backend = stub_registry
    text = "Inflation surprised to the upside."
    task = {"schema": TASK_SCHEMA, "area": "classification", "state": {"asset": "EURUSD"},
            "questions": {"relevancia": RELEVANCE}}
    out = run_task(task, registry, data=text)
    assert out["answers"]["relevancia"]["status"] == "OK" and json.loads(backend.seen[0][0])["body"] == text
    replay = run_task(dict(task, as_of=AS_OF), registry, data=text)
    assert replay["answers"]["relevancia"]["refusal"] == STATE_REQUIRED
    assert "HISTORICAL_REPLAY_NEEDS_EVENT" in replay["answers"]["relevancia"]["why"]
    assert len(backend.seen) == 1


def test_a_wrong_language_event_is_refused_before_the_sdk(stub_registry):
    registry, provider, backend = stub_registry
    event = dict(corpus_events()["00_relevant"], language="es")
    out = run_task(envelope({"news": event}, relevancia=RELEVANCE, tono=TONE), registry)
    for name in ("relevancia", "tono"):
        assert out["answers"][name]["status"] == "REFUSED"
        assert out["answers"][name]["refusal"] == STATE_REQUIRED and "LANGUAGE_NOT_VALIDATED" in out["answers"][name]["why"]
        assert "label" not in out["answers"][name] and "uncalibrated_probabilities" not in out["answers"][name]
    assert backend.seen == [] and provider.calls == 0 and out["refused"] == 2


def test_stale_and_missing_news_are_refused_with_the_same_policy_as_the_typed_path(stub_registry):
    registry, _, backend = stub_registry
    event = corpus_events()["00_relevant"]
    stale = run_task(dict(envelope({"news": event}, relevancia=RELEVANCE), as_of="2026-09-24T18:00:00Z"), registry)
    assert "STALE_NEWS" in stale["answers"]["relevancia"]["why"]
    late = run_task(envelope({"news": event, "max_age_seconds": 86400}, relevancia=RELEVANCE), registry)
    assert late["answers"]["relevancia"]["status"] == "OK"
    empty = run_task(envelope({"asset": "EURUSD"}, relevancia=RELEVANCE), registry)
    assert empty["answers"]["relevancia"]["refusal"] == STATE_REQUIRED and "NEWS_REQUIRED" in empty["answers"]["relevancia"]["why"]
    assert len(backend.seen) == 1


def test_the_typed_single_question_path_is_unchanged_by_the_envelope(stub_registry):
    from m5phet.runtime import Status, run
    from news_signal.provider import request_for
    registry, provider, backend = stub_registry
    event = corpus_events()["00_relevant"]
    result = run(request_for(event, task_id="news_relevance_eurusd.v1", state_ref=state_ref_for(CHECKPOINT_SHA),
                             request_id="r1", as_of=AS_OF), registry)
    assert result["status"] == Status.OK, result["why"]
    out = run_task(envelope({"news": event}, relevancia=RELEVANCE), registry)
    assert out["answers"]["relevancia"]["status"] == "OK" and provider.calls == 2 and len(backend.seen) == 2


def test_every_answer_carries_the_exact_wording_it_was_scored_with(stub_registry):
    """Retsu (2026-09-24): the English sentence scored euro_area 0.9666 and the Spanish one 0.9605 on the same news.
    Laya encodes the question text with the item, so a rewording is a different input; the answer must say which one."""
    registry, _, backend = stub_registry
    event = corpus_events()["00_relevant"]
    english = dict(RELEVANCE, instructions="Is this news relevant to EURUSD?")
    spanish = dict(RELEVANCE, instructions="¿Es esta noticia relevante para EURUSD?")
    first = run_task(envelope({"news": event}, relevancia=english), registry)
    second = run_task(envelope({"news": event}, relevancia=spanish), registry)
    for out, question in ((first, english), (second, spanish)):
        answer = out["answers"]["relevancia"]
        assert answer["instructions"] == question["instructions"]
        assert answer["options"] == list(question["options"])
        assert "rewording" in answer["wording"]
    # two wordings are two question sets: the SDK saw two different inputs and the provenance says so
    assert first["answers"]["relevancia"]["provenance"]["questions_sha256"] != \
        second["answers"]["relevancia"]["provenance"]["questions_sha256"]
    assert len(backend.seen) == 2
