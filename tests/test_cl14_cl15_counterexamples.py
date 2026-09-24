"""CL14/CL15/F3: the second round of counterexamples, frozen before repair.

Three findings, three different lessons about what a check is worth:

* a dependency pin that imports is not a pin that WORKS -- the feature added after the pin was published is missing;
* a record that hashes to itself is not a record that belongs where it was found -- content integrity says nothing about
  placement, and a valid answer to another question served as this one's is the worst kind of wrong;
* counting a value after the SDK has already sliced it measures the slice, not the value -- `fits` was computed from
  `option[:48]`, so it could never be False for the reason it existed.
"""

import json
from pathlib import Path
import threading

import pytest

from m5phet.runtime import Registry

from news_signal.application import classify_event
from news_signal.backends import FixtureBackend, sequence_budget
from news_signal.core import Refusal, digest
from news_signal.provider import LayaNewsProvider
from news_signal.question import ad_hoc_task
from news_signal.shadow import ShadowStore, record_identity

from test_classification_first import AS_OF, CHECKPOINT_SHA, StubLaya, TASK, corpus_events

ROOT = Path(__file__).resolve().parents[1]

SPEC_A = {"name": "answer", "question": "Which economy is named?",
          "options": [("us", "United States"), ("eu", "Euro area"), ("none", "Neither")]}
SPEC_B = {**SPEC_A, "question": "Which economy is explicitly excluded?"}


class WordTokenizer:
    """Deterministic and named: one token per whitespace-separated word. Not the checkpoint's tokenizer."""

    def encode(self, text, add_special_tokens=False):
        return text.split()


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


# --- F1 / CL14: the declared installation must support the feature it declares -----------------------------------------------

def test_the_declared_pin_carries_the_runtime_feature_the_public_ask_needs():
    """`e752ed1` imports, and refuses every user question: task-kind compatibility landed in a later commit."""
    import tomllib

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    pin = next(d for d in declared if "m5phet" in d)
    commit = pin.split("@")[-1].strip()
    proof = json.loads((ROOT / "docs/INSTALL_VERIFICATION.json").read_text())
    assert proof["pinned_commit"] == commit, "the retained proof must be of the pin that is declared now"
    assert proof["editable_or_pythonpath_substitution"] is False
    # the evidence must show a SUCCESSFUL user question through the installed entry point, not only a refusal
    asked = proof.get("installed_entry_point_ask")
    assert asked, "absence of weights is not a demonstration that the feature works"
    assert asked["status"] == "SHADOW_ONLY"
    assert asked["task_id"].startswith("adhoc.question.v1:")
    assert asked["backend_kind"] == "NON_MODEL_FIXTURE", "a controlled backend, declared as not a model"
    assert asked["entry_point_discovered"] == "laya_news"


# --- F2 / CL15: a record must belong where it was found -----------------------------------------------------------------------

def _store_one(registry, tmp_path, spec, event=None):
    store = ShadowStore(tmp_path)
    outcome = classify_event(event or corpus_events()["00_relevant"], task_id=ad_hoc_task(**spec)["task_id"],
                             question_spec=spec, as_of=AS_OF, registry=registry, store=store)
    return store, outcome


def test_an_intact_record_of_another_question_is_not_this_questions_answer(stub, tmp_path):
    """Copying B's whole, valid record onto A's key used to return B's answer as A's DUPLICATE, with no integrity failure."""
    registry, _backend = stub
    store_b, outcome_b = _store_one(registry, tmp_path / "b", SPEC_B)
    store_a = ShadowStore(tmp_path / "a")
    event = corpus_events()["00_relevant"]
    first = classify_event(event, task_id=ad_hoc_task(**SPEC_A)["task_id"], question_spec=SPEC_A,
                           as_of=AS_OF, registry=registry, store=store_a)
    key_a = next((tmp_path / "a").rglob("*.json"))
    intact_b = next((tmp_path / "b").rglob("*.json")).read_text()
    key_a.write_text(intact_b)                                   # a whole, self-consistent record in the wrong place
    again = classify_event(event, task_id=ad_hoc_task(**SPEC_A)["task_id"], question_spec=SPEC_A,
                           as_of=AS_OF, registry=registry, store=store_a)
    assert again["stored"]["disposition"] != "DUPLICATE", "a record for another question is not this one's result"
    returned = store_a.find_record(again["stored"]["record_sha256"])
    assert returned["task_id"] == ad_hoc_task(**SPEC_A)["task_id"]
    report = store_a.replay()
    assert report["integrity_failures"], "a misplaced record must be reported, not passed over in silence"


def test_a_record_whose_identity_does_not_match_its_key_is_quarantined_not_served(stub, tmp_path):
    registry, _backend = stub
    store, outcome = _store_one(registry, tmp_path / "s", SPEC_A)
    path = next((tmp_path / "s").rglob("*.json"))
    record = json.loads(path.read_text())
    record["task_id"] = ad_hoc_task(**SPEC_B)["task_id"]          # still self-consistent after rehashing
    record["record_sha256"] = digest({k: v for k, v in record.items() if k != "record_sha256"})
    path.write_text(json.dumps(record))
    report = store.replay()
    assert report["integrity_failures"], "the record no longer hashes to the key it sits under"
    assert store.find_record(record["record_sha256"]) is None or \
        store.find_record(record["record_sha256"])["integrity"] != "OK"


def test_every_quarantined_version_is_preserved(stub, tmp_path):
    registry, _backend = stub
    store, first = _store_one(registry, tmp_path / "s", SPEC_A)
    path = next((tmp_path / "s").rglob("*.json"))
    for tamper in ("first", "second"):
        record = json.loads(path.read_text())
        record["features"] = {"answer": {"label": tamper}}
        path.write_text(json.dumps(record))
        classify_event(corpus_events()["00_relevant"], task_id=ad_hoc_task(**SPEC_A)["task_id"], question_spec=SPEC_A,
                       as_of=AS_OF, registry=registry, store=store)
    quarantined = sorted((tmp_path / "s" / "quarantine").glob("*.json"))
    assert len(quarantined) == 2, "a second quarantine must not overwrite the first"


def test_malformed_json_in_the_store_is_refused_not_reused(stub, tmp_path):
    registry, _backend = stub
    store, _first = _store_one(registry, tmp_path / "s", SPEC_A)
    path = next((tmp_path / "s").rglob("*.json"))
    path.write_text('{"schema": "news_shadow.v2", "truncat')
    again = classify_event(corpus_events()["00_relevant"], task_id=ad_hoc_task(**SPEC_A)["task_id"], question_spec=SPEC_A,
                           as_of=AS_OF, registry=registry, store=store)
    assert again["stored"]["disposition"] != "DUPLICATE"
    assert store.replay()["integrity_failures"]


def test_concurrent_writers_on_the_same_key_agree_on_one_record(stub, tmp_path):
    """The earlier test used distinct event ids, which is the easy case: nothing contended."""
    registry, _backend = stub
    store = ShadowStore(tmp_path / "s")
    event = corpus_events()["00_relevant"]
    task = ad_hoc_task(**SPEC_A)["task_id"]
    results, errors = [], []

    def write():
        try:
            results.append(classify_event(event, task_id=task, question_spec=SPEC_A, as_of=AS_OF,
                                          registry=registry, store=store))
        except Exception as exc:                                 # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    report = store.replay()
    assert report["records"] == 1, "one evaluation is one record however many writers raced for it"
    assert report["integrity_failures"] == [] and report["actionable_events"] == 1
    assert len({r["stored"]["record_sha256"] for r in results}) == 1


# --- F3: the budget must count what was ASKED, not what the SDK kept ------------------------------------------------------------

def test_an_option_longer_than_the_sdk_keeps_does_not_fit():
    """`opt_ids` slices each option to 48 tokens upstream. Counting after the slice made `fits` blind to exactly that cut."""
    tokenizer = WordTokenizer()
    long_option = " ".join(f"word{i}" for i in range(60))
    questions = {"answer": {"type": "choice", "instructions": "Which economy is named?",
                            "criteria": {"us": long_option, "eu": "Euro area", "none": "Neither"}}}
    report = sequence_budget(tokenizer, "short state", questions)
    assert report["answer"]["fits"] is False
    assert report["answer"].get("options_truncated"), "the option that would be cut must be named"


def test_an_option_at_the_limit_still_fits():
    tokenizer = WordTokenizer()
    exact = " ".join(f"word{i}" for i in range(46))               # 46 words + the mask token stays inside 48
    questions = {"answer": {"type": "choice", "instructions": "Which economy is named?",
                            "criteria": {"us": exact, "eu": "Euro area", "none": "Neither"}}}
    assert sequence_budget(tokenizer, "short state", questions)["answer"]["fits"] is True


def test_the_public_path_refuses_a_question_whose_option_would_be_cut(stub):
    registry, backend = stub
    # short words on purpose: under the character limit, over the SDK's 48-token option cut, so only the token
    # accounting can catch it
    spec = {**SPEC_A, "options": [("us", " ".join(f"w{i}" for i in range(60))),
                                  ("eu", "Euro area"), ("none", "Neither")]}
    outcome = classify_event(corpus_events()["00_relevant"], task_id=ad_hoc_task(**spec)["task_id"],
                             question_spec=spec, as_of=AS_OF, registry=registry)
    assert outcome["status"] == "REFUSED_WITH_INPUT"
    assert "TOKEN_BUDGET_EXCEEDED" in json.dumps(outcome["receipt"]["refusal_reasons"])


@pytest.mark.parametrize("field", ["head", "state"])
def test_each_limit_is_measured_on_the_untruncated_text(field):
    tokenizer = WordTokenizer()
    long = " ".join(f"word{i}" for i in range(700))
    questions = {"answer": {"type": "choice",
                            "instructions": long if field == "head" else "Which economy is named?",
                            "criteria": {"us": "United States", "eu": "Euro area"}}}
    report = sequence_budget(tokenizer, long if field == "state" else "short state", questions)
    assert report["answer"]["fits"] is False


def test_the_accounting_matches_the_pinned_upstream_builder():
    """Differential against the real thing, where it is installed. Skipped elsewhere rather than approximated."""
    laya = pytest.importorskip("laya.common")
    tokenizer = pytest.importorskip("transformers")              # only to make the skip reason explicit
    from news_signal.backends import sequence_budget as budget

    class Tok:
        """The upstream builder needs a tokenizer object; this one is deterministic and declares itself."""
        mask_token, mask_token_id, cls_token_id, sep_token_id = "[MASK]", 4, 1, 2

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [10 + i for i, _w in enumerate(text.split())]}

        def encode(self, text, add_special_tokens=False):
            return self(text)["input_ids"]

    tok = Tok()
    questions = {"answer": {"type": "choice", "instructions": "Which economy is named?",
                            "criteria": {"us": "United States of America", "eu": "Euro area"}}}
    state = " ".join(f"word{i}" for i in range(50))
    ids, _markers = laya.build_sequence(tok, state, {"t": "choice", "ins": questions["answer"]["instructions"],
                                                     "criteria": questions["answer"]["criteria"]},
                                        max_len=512, head_max_len=192)
    mine = budget(tok, state, questions)["answer"]
    assert mine["fits"] is True
    assert len(ids) <= 512
