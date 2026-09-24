"""CL08: the four store and installation counterexamples, frozen before their repair.

Each test states the behaviour the application must have, not the behaviour it had when the finding was written. Against
`bb281cc` every one of them fails; that failure is the point. They are kept afterwards as the regression that stops the same
mistake returning under another name.

The shape shared by findings 1 and 2 is worth naming once: the store keyed a result by text that came from outside (the
`event_id`) and treated "same news bytes" as "same result". Neither holds. An identifier from a feed is not a filename, and
the answer to a question depends on the question, the model, the clock and the policy as much as on the news.
"""

import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from m5phet.runtime import Registry

from news_signal.application import classify_event
from news_signal.core import Refusal, digest, load_json
from news_signal.provider import LayaNewsProvider
from news_signal.shadow import ShadowStore

from test_classification_first import AS_OF, CHECKPOINT_SHA, StubLaya, TASK, corpus_events

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples/eurusd"


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


# --- finding 1: an identifier from a feed is not a path ------------------------------------------------------------------

@pytest.mark.parametrize("event_id", ["../escaped", "../../escaped", "/absolute/escaped", "a/../../escaped",
                                      "nested/child", ".", "..", "", " ", "con:trol\nchar"])
def test_an_external_identifier_can_never_place_a_file_outside_the_store(stub, tmp_path, event_id):
    """`../escaped` used to be joined straight onto the store root and written there, with the application reporting success."""
    registry, _backend = stub
    store_root = tmp_path / "store"
    event = dict(corpus_events()["00_relevant"], event_id=event_id)
    outcome = classify_event(event, task_id=TASK, as_of=AS_OF, registry=registry, store=ShadowStore(store_root))
    written = [p for p in tmp_path.rglob("*.json") if p.name != "manifest.json"]
    outside = [p for p in written if store_root not in p.parents and p.parent != store_root]
    assert not outside, f"the store wrote outside its own root: {outside}"
    # an empty or blank identifier is REFUSED upstream by the record validator, which is correct; what this case fixes is
    # that no identifier, accepted or refused, may decide where a file lands
    assert outcome["status"] in ("SHADOW_ONLY", "REFUSED_WITH_INPUT")
    if outcome["stored"] is not None:
        assert written, "a stored result must actually be on disk inside the root"
        inside = [p for p in written if store_root in p.parents]
        assert inside, "the record must be under the store root"


def test_a_symlinked_store_entry_cannot_redirect_a_write(stub, tmp_path):
    """A directory inside the store that points elsewhere is an escape with extra steps."""
    registry, _backend = stub
    store_root = tmp_path / "store"
    store_root.mkdir()
    target = tmp_path / "elsewhere"
    target.mkdir()
    event = corpus_events()["00_relevant"]
    store = ShadowStore(store_root)
    # whatever internal layout the store chooses, a symlink planted in its root must not carry a write out of it
    for name in ("00", digest(event)[:2], event["event_id"]):
        link = store_root / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    classify_event(event, task_id=TASK, as_of=AS_OF, registry=registry, store=store)
    assert not list(target.rglob("*.json")), "a symlink inside the store redirected a write out of it"


# --- finding 2: identity, integrity and what may be reused ----------------------------------------------------------------

def test_a_refusal_does_not_answer_for_a_later_successful_evaluation(stub, tmp_path):
    """Same news, refused before it was available, then classified at receipt time. Those are two evaluations, not one."""
    registry, _backend = stub
    store = ShadowStore(tmp_path / "store")
    future = load_json(CORPUS / "refusals/01_future_receipt.json")
    refused = classify_event(future, task_id=TASK, as_of=AS_OF, registry=registry, store=store)
    accepted = classify_event(future, task_id=TASK, as_of=future["received_at"], registry=registry, store=store)
    assert refused["status"] == "REFUSED_WITH_INPUT"
    assert accepted["status"] == "SHADOW_ONLY"
    assert accepted["stored"]["disposition"] != "DUPLICATE", "a later, different evaluation is not a duplicate of a refusal"
    kept = store.records_for(future["event_id"])
    assert {r["status"] for r in kept} == {"REFUSED_WITH_INPUT", "SHADOW_ONLY"}, "the retry retains its history"
    assert any(r["status"] == "SHADOW_ONLY" for r in kept), "the successful result must be retrievable, not hidden"


def test_a_different_task_is_a_different_evaluation(stub, tmp_path):
    """Asking another question of the same news used to return the previous task's stored answer."""
    registry, _backend = stub
    store = ShadowStore(tmp_path / "store")
    event = corpus_events()["00_relevant"]
    first = classify_event(event, task_id=TASK, as_of=AS_OF, registry=registry, store=store)
    other = classify_event(event, task_id="news_triage.v1", as_of=AS_OF, registry=registry, store=store)
    assert other["receipt"]["task_id"] == "news_triage.v1"
    assert other["stored"]["disposition"] != "DUPLICATE"
    tasks = {r["task_id"] for r in store.records_for(event["event_id"])}
    assert tasks == {TASK, "news_triage.v1"}, f"both evaluations must be retained, found {tasks}"
    stored = store.find_record(other["stored"]["record_sha256"])
    assert stored["task_id"] == "news_triage.v1", "a receipt must never point at another task's persisted result"


def test_only_an_identical_evaluation_may_be_idempotent(stub, tmp_path):
    registry, _backend = stub
    store = ShadowStore(tmp_path / "store")
    event = corpus_events()["00_relevant"]
    first = classify_event(event, task_id=TASK, as_of=AS_OF, registry=registry, store=store)
    same = classify_event(event, task_id=TASK, as_of=AS_OF, registry=registry, store=store)
    assert same["stored"]["disposition"] == "DUPLICATE"
    assert same["stored"]["record_sha256"] == first["stored"]["record_sha256"]
    later = classify_event(event, task_id=TASK, as_of="2026-09-24T12:05:00Z", registry=registry, store=store)
    assert later["stored"]["disposition"] != "DUPLICATE", "another decision clock is another evaluation"


def test_a_tampered_record_is_never_reused_and_never_counted(stub, tmp_path):
    """The store used to hand back a corrupted record as a duplicate, and the replay counted it as an actionable event."""
    registry, _backend = stub
    store = ShadowStore(tmp_path / "store")
    event = corpus_events()["00_relevant"]
    first = classify_event(event, task_id=TASK, as_of=AS_OF, registry=registry, store=store)
    path = next(p for p in (tmp_path / "store").rglob("*.json"))
    damaged = json.loads(path.read_text())
    damaged["features"]["relevance"]["label"] = "TAMPERED"
    path.write_text(json.dumps(damaged))
    again = classify_event(event, task_id=TASK, as_of=AS_OF, registry=registry, store=store)
    assert again["stored"]["disposition"] != "DUPLICATE", "a record that fails its own digest is not a result to reuse"
    report = store.replay()
    assert report["integrity_failures"], "the damaged record is still reported"
    assert report["actionable_events"] == 1, "a corrupt record must not add to the count of decided events"
    labels = {r["features"]["relevance"]["label"] for r in store.records_for(event["event_id"])
              if r["status"] == "SHADOW_ONLY" and r.get("integrity") != "FAILED"}
    assert "TAMPERED" not in labels


def test_an_interrupted_publication_leaves_no_half_record(stub, tmp_path):
    registry, _backend = stub
    store_root = tmp_path / "store"
    store = ShadowStore(store_root)
    event = corpus_events()["00_relevant"]
    classify_event(event, task_id=TASK, as_of=AS_OF, registry=registry, store=store)
    (store_root / "interrupted.json.tmp").write_text('{"schema": "news_shadow.v2", "truncat')
    report = store.replay()
    assert report["records"] == 1, "an unfinished temporary file is not a record"
    assert report["actionable_events"] == 1


def test_concurrent_writers_do_not_lose_or_mix_results(stub, tmp_path):
    registry, _backend = stub
    store = ShadowStore(tmp_path / "store")
    events = [dict(corpus_events()["00_relevant"], event_id=f"concurrent-{i}") for i in range(8)]
    errors = []

    def write(item):
        try:
            classify_event(item, task_id=TASK, as_of=AS_OF, registry=registry, store=store)
        except Exception as exc:                                    # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(e,)) for e in events]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    report = store.replay()
    assert report["records"] == len(events) and report["integrity_failures"] == []
    for item in events:
        kept = store.records_for(item["event_id"])
        assert len(kept) == 1 and kept[0]["input_sha256"] == digest(item)


# --- finding 3: the declared installation must be able to import the application --------------------------------------------

def test_the_declared_m5phet_pin_provides_the_runtime_the_application_imports():
    """The published pin resolved to a commit with no `m5phet.runtime`; the suite passed only because another one was installed."""
    import tomllib

    pinned = None
    for line in (ROOT / "pyproject.toml").read_text().splitlines():
        if "m5phet" in line and "git+" in line:
            pinned = line
    assert pinned, "the dependency on M5PHET must be declared"
    commit = pinned.split("@")[-1].strip().strip('"],  ')
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), f"pin an exact commit, found {commit!r}"
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert any("m5phet" in d for d in declared)
    # the contract this application actually imports
    from m5phet.runtime import ENTRY_POINT_GROUP, Registry, run          # noqa: F401
    assert ENTRY_POINT_GROUP == "m5phet.providers"
    proof = ROOT / "docs/INSTALL_VERIFICATION.json"
    assert proof.is_file(), "a clean-resolution install must be verified and its evidence retained"
    evidence = json.loads(proof.read_text())
    assert evidence["pinned_commit"] == commit
    assert evidence["imported_m5phet_runtime"] is True
    assert evidence["entry_point_discovered"] == "laya_news"
    assert evidence["editable_or_pythonpath_substitution"] is False
