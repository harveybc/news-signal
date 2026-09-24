"""CL01-CL07 through the public path: installed discovery, Registry dispatch, typed refusal, persistence and replay.

What these tests can and cannot establish. They exercise the whole path -- entry point, request, runtime, provider, receipt,
store -- with a fixture backend standing in for the weights, so they establish the CONTRACT. They do not establish that the
real model agrees with the real SDK: that is CL01 with real weights, measured by the pilot, and its evidence is a parity
report, not a passing test here. Where a test uses a fixture, it says so in its name or its assertion.
"""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from m5phet.runtime import Registry, Status, run

from news_signal.application import ABSENT_STATE, classify_event, discover, replay
from news_signal.backends import FixtureBackend
from news_signal.core import QUESTIONS, Refusal, digest, load_json
from news_signal.provider import LayaNewsProvider, PROVIDER_NAME, request_for, state_ref_for
from news_signal.shadow import ShadowStore
from news_signal.tasks import TASKS, questions as task_questions

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples/eurusd"
TASK = "news_relevance_eurusd.v1"
AS_OF = "2026-09-24T12:00:00Z"
CHECKPOINT_SHA = "c" * 64


def corpus_events(folder="events"):
    return {p.stem: load_json(p) for p in sorted((CORPUS / folder).glob("*.json"))}


class StubLaya:
    """A stand-in for the SDK that answers the question set it is GIVEN. It is never presented as a model measurement."""

    identity = {"kind": "NON_MODEL_FIXTURE", "name": "stub-laya-v1"}

    def __init__(self):
        self.seen = []

    def predict(self, state, questions=None):
        questions = QUESTIONS if questions is None else questions
        self.seen.append((state, tuple(questions)))
        body = json.loads(state)["body"].lower()
        answers = {}
        for name, question in questions.items():
            options = list(question["criteria"])
            pick = "related" if ("central bank" in body or "inflation" in body) and "related" in options else options[-1]
            probs = {}
            for option in options:
                probs[option] = round(0.7 if option == pick else 0.3 / (len(options) - 1), 4)
            answers[name] = {"type": "choice", "choice": pick, "probabilities": probs,
                             "confidence": probs[pick]}
        return {"answers": answers, "usage": {"input_tokens": len(state.split()), "output_tokens": 0}}


@pytest.fixture
def manifest_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "news_checkpoint.v1", "sha256": CHECKPOINT_SHA, "files": {}}))
    return path


@pytest.fixture
def stub_registry(manifest_file):
    """The real Registry and the real runtime, with a fixture where the weights would be."""
    backend = StubLaya()
    provider = LayaNewsProvider(environ={"NEWS_SIGNAL_MANIFEST": str(manifest_file), "NEWS_SIGNAL_DEVICE": "cpu"},
                                backend_factory=lambda config, manifest: backend)
    registry = Registry()
    registry.register(provider)
    return registry, provider, backend


# --- CL01: the installed public path, and what parity does and does not prove -------------------------------------------

def test_the_provider_is_discovered_through_the_installed_entry_point_group():
    registry, report = discover()
    assert report["group"] == "m5phet.providers"
    assert PROVIDER_NAME in report["registered"], report
    caps = registry.capabilities(PROVIDER_NAME)
    assert caps["supported"] == [{"operation": "infer", "family": "classification", "output_kind": "typed_questions"}]
    assert caps["uncertainty_methods"] == ["UNCALIBRATED_CLASS_PROBABILITIES"]


def test_the_wrapper_carries_the_sdk_answer_through_without_touching_it(stub_registry):
    registry, provider, backend = stub_registry
    event = corpus_events()["00_relevant"]
    result = run(request_for(event, task_id=TASK, state_ref=state_ref_for(CHECKPOINT_SHA),
                             request_id="r1", as_of=AS_OF), registry)
    assert result["status"] == Status.OK, result["why"]
    payload = result["outputs"]["relevance"]["payload"]
    native = backend.predict(json.dumps({k: event[k] for k in ("asset", "headline", "body")},
                                        sort_keys=True, ensure_ascii=False, separators=(",", ":")),
                             task_questions(TASK))["answers"]["relevance"]
    assert payload["sdk_answer"] == native, "the wrapper must not round, reorder or renormalize the SDK's answer"
    assert list(payload["uncalibrated_probabilities"]) == list(native["probabilities"])
    assert payload["calibration"] == "UNCALIBRATED"


def test_the_parity_comparator_finds_a_difference_it_is_given():
    """The comparator is the instrument for CL01; an instrument that cannot fail proves nothing."""
    sys.path.insert(0, str(ROOT / "tools"))
    import parity_report

    answer = {"type": "choice", "choice": "related", "probabilities": {"related": 0.7, "unrelated": 0.2, "unclear": 0.1},
              "confidence": 0.7}
    direct = {"single": [{"event_id": "e1", "input_sha256": "a" * 64, "state_sha256": "b" * 64,
                          "response": {"answers": {"relevance": answer}}}]}
    receipt = {"status": "SHADOW_ONLY", "event_id": "e1", "input_sha256": "a" * 64,
               "binding": {"state_digest": "b" * 64},
               "features": {"relevance": {"sdk_answer": copy.deepcopy(answer)}}}
    rows, mismatches = parity_report.compare(parity_report.answers_of_direct(direct),
                                             parity_report.answers_of_wrapper([receipt]))
    assert (rows[0]["equal"], mismatches) == (True, [])
    nudged = copy.deepcopy(receipt)
    nudged["features"]["relevance"]["sdk_answer"]["probabilities"]["related"] = 0.7001
    _rows, mismatches = parity_report.compare(parity_report.answers_of_direct(direct),
                                              parity_report.answers_of_wrapper([nudged]))
    assert mismatches and mismatches[0]["reason"] == "ANSWER_FIELDS_DIFFER"
    reordered = copy.deepcopy(receipt)
    reordered["features"]["relevance"]["sdk_answer"]["probabilities"] = {
        "unclear": 0.1, "related": 0.7, "unrelated": 0.2}
    rows, mismatches = parity_report.compare(parity_report.answers_of_direct(direct),
                                             parity_report.answers_of_wrapper([reordered]))
    assert mismatches, "a reordered probability map is a different answer object, not the same one"


# --- CL02: the expected population, including the items that must be refused ---------------------------------------------

def test_every_corpus_item_is_answered_and_the_instruction_like_one_does_not_execute(stub_registry, tmp_path):
    registry, provider, _backend = stub_registry
    store = ShadowStore(tmp_path / "shadow")
    corpus = json.loads((CORPUS / "corpus.json").read_text())
    outcomes = {}
    for entry in corpus["items"]:
        event = load_json(CORPUS / entry["file"])
        outcome = classify_event(event, task_id=TASK, as_of=AS_OF, registry=registry, store=store)
        outcomes[entry["file"]] = outcome
        assert outcome["receipt"]["execution_authorized"] is False
    assert len(outcomes) == len(corpus["items"]) == 13, "the whole declared population is answered, none skipped"
    assert all(o["status"] == "SHADOW_ONLY" for o in outcomes.values())
    instruction = outcomes["events/12_instruction_like.json"]["receipt"]
    assert instruction["status"] == "SHADOW_ONLY"
    assert instruction["features"]["relevance"]["label"] in ("related", "unrelated", "unclear")
    assert instruction["execution_authorized"] is False, "an instruction inside the text is data, never an instruction"


def test_refused_records_are_refused_before_any_forward_pass(stub_registry):
    registry, provider, backend = stub_registry
    corpus = json.loads((CORPUS / "corpus.json").read_text())
    before = len(backend.seen)
    reasons = {}
    for entry in corpus["refusals"]:
        # the token budget is the real tokenizer's refusal; it has its own test with a backend that can raise it
        if entry["category"] == "over_token_budget":
            continue
        event = load_json(CORPUS / entry["file"])
        outcome = classify_event(event, task_id=TASK, as_of=AS_OF, registry=registry)
        reasons[entry["category"]] = outcome["receipt"]
        assert outcome["status"] == "REFUSED_WITH_INPUT", entry
    assert len(reasons) == 4
    assert len(backend.seen) == before, "not one refused record reached the model"
    assert "LANGUAGE_NOT_VALIDATED" in why_of(reasons["wrong_language"])
    assert "NEWS_NOT_AVAILABLE_AT_DECISION" in why_of(reasons["future_receipt"])
    assert "NEWS_SIZE_LIMIT" in why_of(reasons["overlong_body"])
    assert "ASSET_NOT_IN_TASK_SCOPE" in why_of(reasons["asset_out_of_task_scope"])


def why_of(receipt):
    outputs = receipt.get("outputs") or {}
    return " ".join([receipt.get("why") or ""] + [(o.get("why") or "") for o in outputs.values()])


def test_a_subset_of_the_task_questions_is_refused_not_filtered(stub_registry):
    """The question set is the model's input. Answering a subset is a different task, not the same one filtered."""
    registry, provider, backend = stub_registry
    event = corpus_events()["00_relevant"]
    request = request_for(event, task_id="news_triage.v1", state_ref=state_ref_for(CHECKPOINT_SHA),
                          request_id="r-subset", as_of=AS_OF)
    request["output_schema"]["questions"] = ["relevance"]
    result = run(request, registry)
    assert result["status"] == Status.INVALID_INPUT
    assert "QUESTION_POPULATION_MISMATCH" in result["outputs"]["relevance"]["why"]


# --- CL03: typed refusals, no silent fallback -----------------------------------------------------------------------------

def test_absent_weights_refuse_before_loading_anything(tmp_path):
    provider = LayaNewsProvider(environ={"NEWS_SIGNAL_MANIFEST": str(tmp_path / "missing.json")},
                                backend_factory=lambda config, manifest: pytest.fail("nothing may be loaded"))
    registry = Registry()
    registry.register(provider)
    assert registry.capabilities(PROVIDER_NAME)["weights_present"] is False
    event = corpus_events()["00_relevant"]
    result = run(request_for(event, task_id=TASK, state_ref=ABSENT_STATE, request_id="r2", as_of=AS_OF), registry)
    assert result["status"] == Status.MODEL_NOT_FITTED


def test_an_unknown_task_is_refused_by_the_states_declared_compatibility(stub_registry):
    registry, provider, backend = stub_registry
    event = corpus_events()["00_relevant"]
    request = request_for(event, task_id=TASK, state_ref=state_ref_for(CHECKPOINT_SHA),
                          request_id="r3", as_of=AS_OF)
    request["task_id"] = "news_relevance_eurusd.v2"
    result = run(request, registry)
    assert result["status"] == Status.INVALID_INPUT and "compatible" in result["why"]
    assert backend.seen == [], "an unknown task never reaches the model"


def test_an_undeclared_combination_is_refused_before_the_model_loads(stub_registry):
    registry, provider, backend = stub_registry
    event = corpus_events()["00_relevant"]
    request = request_for(event, task_id=TASK, state_ref=state_ref_for(CHECKPOINT_SHA),
                          request_id="r4", as_of=AS_OF)
    request["output_kind"] = "marginal_quantiles"
    assert run(request, registry)["status"] == Status.UNSUPPORTED_TASK
    assert backend.seen == []


def test_the_token_budget_refuses_instead_of_truncating(manifest_file):
    class Counting(StubLaya):
        def predict(self, state, questions=None):
            raise Refusal("TOKEN_BUDGET_EXCEEDED: select a documented short news field, do not silently truncate")

    provider = LayaNewsProvider(environ={"NEWS_SIGNAL_MANIFEST": str(manifest_file)},
                                backend_factory=lambda config, manifest: Counting())
    registry = Registry()
    registry.register(provider)
    event = corpus_events("refusals")["03_over_token_budget"]
    result = run(request_for(event, task_id=TASK, state_ref=state_ref_for(CHECKPOINT_SHA),
                             request_id="r5", as_of=AS_OF), registry)
    assert result["status"] == Status.INVALID_INPUT
    assert "TOKEN_BUDGET_EXCEEDED" in result["outputs"]["relevance"]["why"]


# --- CL04: identity across restart, batch order and permutation ------------------------------------------------------------

def test_a_result_is_bound_to_the_news_item_it_is_about(stub_registry):
    registry, provider, _backend = stub_registry
    events = corpus_events()
    request = request_for(events["00_relevant"], task_id=TASK, state_ref=state_ref_for(CHECKPOINT_SHA),
                          request_id="r6", as_of=AS_OF)
    request["inputs"]["news_event"] = events["05_unrelated"]          # the declared population no longer matches the input
    result = run(request, registry)
    assert result["status"] == Status.INVALID_INPUT
    assert "population" in (result["why"] or "")


def test_order_and_permutation_do_not_change_a_result(stub_registry):
    registry, provider, _backend = stub_registry
    events = [corpus_events()[k] for k in ("00_relevant", "05_unrelated", "08_ambiguous")]
    forward = [classify_event(e, task_id=TASK, as_of=AS_OF, registry=registry)["receipt"] for e in events]
    backward = [classify_event(e, task_id=TASK, as_of=AS_OF, registry=registry)["receipt"] for e in reversed(events)]
    by_id = {r["event_id"]: r for r in backward}
    for receipt in forward:
        other = by_id[receipt["event_id"]]
        assert receipt["features"] == other["features"], "the same item answered in another order is the same answer"
        assert receipt["input_sha256"] == other["input_sha256"]


# --- CL05: one broken provider does not take the others with it --------------------------------------------------------------

def test_a_broken_provider_is_named_and_the_others_stay_usable(monkeypatch, stub_registry):
    registry, provider, _backend = stub_registry

    class Broken:
        name = "broken_provider"

        def capabilities(self):
            raise RuntimeError("this provider's backend is not installed")

    with pytest.raises(Exception):
        Registry().register(Broken())
    # the failure is per provider: the working one still answers
    event = corpus_events()["00_relevant"]
    assert classify_event(event, task_id=TASK, as_of=AS_OF, registry=registry)["status"] == "SHADOW_ONLY"


def test_a_request_naming_an_absent_provider_names_it_and_loads_nothing(stub_registry):
    registry, provider, backend = stub_registry
    event = corpus_events()["00_relevant"]
    request = request_for(event, task_id=TASK, state_ref=state_ref_for(CHECKPOINT_SHA),
                          request_id="r7", as_of=AS_OF)
    request["provider_ref"] = "laya_news_remote"
    result = run(request, registry)
    assert result["status"] == Status.UNSUPPORTED_TASK and "laya_news_remote" in result["why"]
    assert backend.seen == []


# --- CL06: persistence, duplicate and revision, replay after restart ----------------------------------------------------------

def test_a_duplicate_is_one_decision_and_a_revision_is_linked_to_it(stub_registry, tmp_path):
    registry, provider, _backend = stub_registry
    store = ShadowStore(tmp_path / "shadow")
    events = corpus_events()
    first = classify_event(events["00_relevant"], task_id=TASK, as_of=AS_OF, registry=registry, store=store)
    again = classify_event(events["10_duplicate_of_ecb-hold-001"], task_id=TASK, as_of=AS_OF,
                           registry=registry, store=store)
    revised = classify_event(events["11_revision_of_ecb-hold-001"], task_id=TASK, as_of=AS_OF,
                             registry=registry, store=store)
    assert first["stored"]["disposition"] == "STORED"
    assert again["stored"]["disposition"] == "DUPLICATE"
    assert again["stored"]["record_sha256"] == first["stored"]["record_sha256"]
    assert revised["stored"]["disposition"] == "REVISION"
    assert revised["stored"]["supersedes"] == [first["stored"]["record_sha256"]]
    report = replay(tmp_path / "shadow")
    assert report["records"] == 2 and report["revisions"] == 1 and report["actionable_events"] == 1
    assert report["integrity_failures"] == [] and report["broker_calls"] == 0
    assert report["inference_performed"] is False


def test_replay_reports_an_altered_record_instead_of_repairing_it(stub_registry, tmp_path):
    registry, provider, _backend = stub_registry
    store = ShadowStore(tmp_path / "shadow")
    classify_event(corpus_events()["00_relevant"], task_id=TASK, as_of=AS_OF, registry=registry, store=store)
    path = next((tmp_path / "shadow").rglob("*.json"))
    record = json.loads(path.read_text())
    record["features"]["relevance"]["label"] = "unrelated"
    path.write_text(json.dumps(record))
    report = replay(tmp_path / "shadow")
    assert len(report["integrity_failures"]) == 1
    assert json.loads(path.read_text())["features"]["relevance"]["label"] == "unrelated", "the record was not rewritten"


def test_the_persisted_result_survives_a_new_process(stub_registry, tmp_path):
    registry, provider, _backend = stub_registry
    store = ShadowStore(tmp_path / "shadow")
    classify_event(corpus_events()["00_relevant"], task_id=TASK, as_of=AS_OF, registry=registry, store=store)
    run_cli = subprocess.run([sys.executable, "-m", "news_signal", "replay", "--store", str(tmp_path / "shadow")],
                             capture_output=True, text=True)
    assert run_cli.returncode == 0, run_cli.stderr
    report = json.loads(run_cli.stdout)
    assert report["records"] == 1 and report["integrity_failures"] == []


# --- CL07: the corpus is sealed before anything scores it ----------------------------------------------------------------------

def test_the_corpus_seal_matches_the_files_on_disk():
    import hashlib
    seal = json.loads((CORPUS / "SEAL.json").read_text())
    for name, recorded in seal["files"].items():
        assert hashlib.sha256((CORPUS / name).read_bytes()).hexdigest() == recorded, name
    corpus = json.loads((CORPUS / "corpus.json").read_text())
    assert corpus["status"] == "SMOKE_TEST_AUTHOR_WRITTEN_LABELS"
    assert sum(corpus["class_counts"].values()) == len(corpus["items"])


def test_the_scorer_reports_counts_and_never_calls_laya_the_truth():
    sys.path.insert(0, str(ROOT / "tools"))
    import score_corpus

    corpus = json.loads((CORPUS / "corpus.json").read_text())
    predictions = {entry["event_id"] + ":" + entry["category"]: entry["label"] for entry in corpus["items"]}
    report = score_corpus.score(corpus, predictions)
    assert report["macro_f1"] == 1.0 and report["coverage"] == 1.0
    assert report["status"] == "SMOKE_TEST_AUTHOR_WRITTEN_LABELS"
    assert report["class_counts"] == corpus["class_counts"]
    wrong = dict(predictions)
    first = corpus["items"][0]
    wrong[first["event_id"] + ":" + first["category"]] = "unclear"
    degraded = score_corpus.score(corpus, wrong)
    assert degraded["macro_f1"] < 1.0 and degraded["confusion"]["related"]["unclear"] == 1


# --- the existing task keeps its identity ---------------------------------------------------------------------------------------

def test_the_legacy_task_digest_did_not_move():
    assert digest(task_questions("news_triage.v1")) == digest(QUESTIONS)
    assert set(TASKS) == {"news_triage.v1", "news_relevance_eurusd.v1"}
    assert list(task_questions(TASK)) == ["relevance"]


# --- the device check has to be able to pass on real hardware -----------------------------------------------------------------

def test_the_gpu_uuid_comparison_accepts_the_form_torch_actually_prints():
    """`nvidia-smi` and CUDA_VISIBLE_DEVICES use `GPU-<uuid>`; `torch.cuda.get_device_properties().uuid` prints the bare uuid.
    Compared literally the check can never pass on a real device, which makes it a refusal that hides a missing check."""
    from news_signal.backends import same_gpu

    mask = "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"
    torch_form = "a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"
    assert same_gpu(torch_form, mask) and same_gpu(mask, mask) and same_gpu(mask, torch_form)
    assert not same_gpu("b77fc3ad-db77-b648-dc15-ec79b65e2519", mask), "another device in the host is not this one"
    assert not same_gpu(None, mask) and not same_gpu(torch_form, None)


def test_a_refusal_reports_the_providers_reason_and_not_only_the_runtimes(stub_registry):
    """The runtime says coverage was not established; the provider says why. A receipt that carries only the first hides it."""
    registry, provider, _backend = stub_registry
    event = corpus_events("refusals")["00_wrong_language"]
    receipt = classify_event(event, task_id=TASK, as_of=AS_OF, registry=registry)["receipt"]
    assert receipt["status"] == "REFUSED_WITH_INPUT"
    assert "LANGUAGE_NOT_VALIDATED" in receipt["refusal_reasons"]["relevance"]
    assert "population" in (receipt["why"] or ""), "the runtime's own observation is kept, not replaced"
