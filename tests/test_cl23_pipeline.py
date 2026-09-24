"""CL23: recorded source -> queue -> installed provider -> persisted result -> acknowledgement, including crash and retry.

The three parts existed and were tested apart. What was never tested is the join, and the join is where a batch consumer
quietly loses work: an entry closed with a digest that resolves to nothing, a result attached to the entry next to it, an
item classified twice because the process died between the two writes, or an item closed forever because a checkpoint was
missing for a minute. Every test below is one of those, written so that it fails if the guarantee is removed.

The classifier here is the fixture stand-in (`StubLaya`) behind the real Registry and the real runtime, so these establish
the CONTRACT of the pipeline. They do not measure the model: that is the pilot's job and its evidence is a parity report.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from m5phet.runtime import Registry

from news_signal.collector import ACKNOWLEDGED, FAILED, PENDING, DurableQueue, RecordedDirectorySource, collect
from news_signal.core import Refusal, load_json
from news_signal.pipeline import (CLASSIFIED, RECOVERED, REFUSED, acknowledge_result, drain, existing_result,
                                  planned_evaluation)
from news_signal.provider import LayaNewsProvider
from news_signal.shadow import ShadowStore
from news_signal import pipeline as pipeline_module

from test_classification_first import AS_OF, CHECKPOINT_SHA, StubLaya, TASK

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples/eurusd"

#: the corpus was recorded around this moment; the queue stamps it, so the decision clock AS_OF is 15 minutes later
RECEIVED = "2026-09-24T11:50:30Z"

#: 13 recorded files, of which one is a duplicate of another: the queue holds 12 entries
CORPUS_ENTRIES = 12


def registry_with(backend, tmp_path, *, manifest=True):
    """The real Registry and the real runtime, with a fixture where the weights would be -- as CL08 builds it."""
    environ = {"NEWS_SIGNAL_DEVICE": "cpu"}
    if manifest:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"schema": "news_checkpoint.v1", "sha256": CHECKPOINT_SHA, "files": {}}))
        environ["NEWS_SIGNAL_MANIFEST"] = str(path)
    else:
        environ["NEWS_SIGNAL_MANIFEST"] = str(tmp_path / "absent.json")
    registry = Registry()
    registry.register(LayaNewsProvider(environ=environ, backend_factory=lambda config, state: backend))
    return registry


@pytest.fixture
def stub(tmp_path):
    backend = StubLaya()
    return registry_with(backend, tmp_path), backend


def news(name, folder="events", **over):
    event = load_json(CORPUS / folder / name)
    event.pop("received_at")                                     # the queue stamps the receipt clock, never the file
    event.update(over)
    return event


def queue_of(tmp_path, *events):
    """A queue holding exactly the named items, offered under the recorded receipt clock."""
    book = DurableQueue(tmp_path / "queue")
    for event in events:
        book.offer(event, received_at=RECEIVED)
    return book


def store_of(tmp_path):
    return ShadowStore(tmp_path / "shadow")


# --- the join itself: a recorded directory reaches the store, and every entry is closed by its own result ------------------

def test_a_recorded_source_reaches_the_store_and_every_entry_is_acknowledged(stub, tmp_path):
    registry, backend = stub
    book, store = DurableQueue(tmp_path / "queue"), store_of(tmp_path)
    collected = collect(RecordedDirectorySource(CORPUS / "events"), book, received_at=RECEIVED)
    assert collected["refused"] == 0 and book.report()["entries"] == CORPUS_ENTRIES

    report = drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF)

    assert report["pending_before"] == CORPUS_ENTRIES
    assert report["processed"] == report["acknowledged"] == report["classified"] == CORPUS_ENTRIES
    assert (report["failed"], report["refused"], report["skipped"]) == (0, 0, 0)
    assert report["execution_authorized"] is False and report["broker_calls"] == 0
    assert book.report()["by_state"] == {ACKNOWLEDGED: CORPUS_ENTRIES}
    assert book.pending() == []
    assert len(backend.seen) == CORPUS_ENTRIES, "each pending entry was asked exactly once"
    assert len(store.all_records()) == CORPUS_ENTRIES


def test_every_number_in_the_report_is_a_count_of_entries_it_actually_touched(stub, tmp_path):
    """A report is evidence. A count computed anywhere but from the items it lists can drift from what happened."""
    registry, _backend = stub
    book, store = DurableQueue(tmp_path / "queue"), store_of(tmp_path)
    collect(RecordedDirectorySource(CORPUS / "events"), book, received_at=RECEIVED)
    book.offer(news("00_wrong_language.json", "refusals"), received_at=RECEIVED)

    report = drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF)

    outcomes = [item["outcome"] for item in report["items"]]
    assert report["processed"] == len(report["items"]) == CORPUS_ENTRIES + 1
    assert report["classified"] == outcomes.count(CLASSIFIED)
    assert report["refused"] == outcomes.count(REFUSED) == 1
    assert report["failed"] == outcomes.count(FAILED) == 0
    assert report["acknowledged"] == sum(1 for item in report["items"] if item["acknowledged_with"])
    assert report["classified"] + report["refused"] + report["recovered"] + report["failed"] == report["processed"]


def test_a_limit_leaves_the_rest_pending_and_says_so(stub, tmp_path):
    registry, backend = stub
    book, store = DurableQueue(tmp_path / "queue"), store_of(tmp_path)
    collect(RecordedDirectorySource(CORPUS / "events"), book, received_at=RECEIVED)

    report = drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF, limit=3)

    assert report["processed"] == 3 and report["skipped"] == CORPUS_ENTRIES - 3
    assert len(backend.seen) == 3, "an item beyond the limit is not classified, only left alone"
    assert len(book.pending()) == CORPUS_ENTRIES - 3
    assert sorted(report["skipped_entry_ids"]) == sorted(e["entry_id"] for e in book.pending())


def test_a_second_drain_has_nothing_left_to_do(stub, tmp_path):
    registry, backend = stub
    book, store = DurableQueue(tmp_path / "queue"), store_of(tmp_path)
    collect(RecordedDirectorySource(CORPUS / "events"), book, received_at=RECEIVED)
    drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF)
    asked = len(backend.seen)

    again = drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF)

    assert again["processed"] == 0 and again["items"] == []
    assert len(backend.seen) == asked, "an acknowledged item is never handed to the model a second time"


# --- the acknowledgement names the result, and the RIGHT one ------------------------------------------------------------------

def test_an_acknowledgement_resolves_to_the_record_that_was_persisted(stub, tmp_path):
    """An entry acknowledged with a digest no reader can find is worse than an unfinished one: the item is out of the queue
    and there is nothing to show for it."""
    registry, _backend = stub
    book, store = DurableQueue(tmp_path / "queue"), store_of(tmp_path)
    collect(RecordedDirectorySource(CORPUS / "events"), book, received_at=RECEIVED)

    report = drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF)

    for item in report["items"]:
        entry = book.get(item["entry_id"])
        assert entry["state"] == ACKNOWLEDGED
        assert entry["receipt_sha256"] == item["acknowledged_with"]
        record = store.find_record(entry["receipt_sha256"])
        assert record is not None, "the digest an entry was closed with must resolve in the store"
        assert record["integrity"] == "OK"
        assert record["source_identity"]["input_sha256"] == entry["input_sha256"]
        assert record["source_identity"]["event_id"] == entry["event_id"]


def test_a_revision_and_the_text_it_revises_are_closed_by_their_own_results(stub, tmp_path):
    """The two share an `event_id` and differ only in bytes. A join that matched on the identifier alone would close each of
    them with the other's answer and both would look finished."""
    registry, _backend = stub
    book = queue_of(tmp_path, news("00_relevant.json"), news("11_revision_of_ecb-hold-001.json"))
    store = store_of(tmp_path)
    assert len({e["event_id"] for e in book.pending()}) == 1 and len(book.pending()) == 2

    drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF)

    for entry in book.entries():
        record = store.find_record(entry["receipt_sha256"])
        assert record["source_identity"]["input_sha256"] == entry["input_sha256"]
    assert len({e["receipt_sha256"] for e in book.entries()}) == 2, "two items, two results, no shared digest"


def test_a_result_for_another_item_can_never_acknowledge_this_one(stub, tmp_path):
    """The counterexample the binding exists for. This is the only acknowledgement path in the module, so a foreign digest
    is not something a caller has to remember not to pass."""
    registry, _backend = stub
    book = queue_of(tmp_path, news("00_relevant.json"), news("05_unrelated.json"))
    store = store_of(tmp_path)
    first, second = book.pending()
    drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF, limit=1)
    persisted = store.find_record(book.get(first["entry_id"])["receipt_sha256"])

    with pytest.raises(Refusal, match="MISBOUND_ACKNOWLEDGEMENT"):
        acknowledge_result(book, book.get(second["entry_id"]), persisted)

    assert book.get(second["entry_id"])["state"] == PENDING, "the second item was not closed by the first one's answer"
    assert book.get(second["entry_id"]).get("receipt_sha256") is None


def test_an_unreadable_result_is_not_acknowledged(stub, tmp_path):
    """A record that does not validate is evidence of a corruption, not this item's answer."""
    registry, _backend = stub
    book = queue_of(tmp_path, news("00_relevant.json"))
    store = store_of(tmp_path)
    entry = book.pending()[0]
    drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF)
    record = store.find_record(book.get(entry["entry_id"])["receipt_sha256"])
    damaged = dict(record, integrity="FAILED", integrity_problems=["CONTENT_DIGEST_MISMATCH"])

    with pytest.raises(Refusal, match="UNACKNOWLEDGEABLE_RESULT"):
        acknowledge_result(book, book.get(entry["entry_id"]), damaged)


# --- crash and retry ------------------------------------------------------------------------------------------------------------

def test_a_failure_while_classifying_leaves_the_entry_failed_and_a_retry_completes_it(stub, tmp_path, monkeypatch):
    registry, backend = stub
    book = queue_of(tmp_path, news("00_relevant.json"))
    store = store_of(tmp_path)
    entry_id = book.pending()[0]["entry_id"]

    def falls_over(*a, **k):
        raise RuntimeError("the provider process died mid-inference")

    monkeypatch.setattr(pipeline_module, "classify_event", falls_over)
    report = drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF)

    assert (report["failed"], report["acknowledged"]) == (1, 0)
    failed = book.get(entry_id)
    assert failed["state"] == FAILED and failed["attempts"] == 1
    assert "the provider process died mid-inference" in failed["reason"]
    assert failed.get("receipt_sha256") is None, "a failure is never an acknowledgement"
    assert store.all_records() == [], "nothing was persisted for an item that was never answered"
    assert book.pending() == [], "a failed item is not retried by the drain that failed it"

    monkeypatch.undo()
    back = book.retry(entry_id)
    assert back["state"] == PENDING
    assert [h["state"] for h in back["history"]] == [PENDING, FAILED, PENDING], "the failure is retained through the retry"

    second = drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF)
    assert second["classified"] == 1 and second["acknowledged"] == 1
    done = book.get(entry_id)
    assert done["state"] == ACKNOWLEDGED
    assert store.find_record(done["receipt_sha256"])["integrity"] == "OK"
    assert any(h.get("reason") for h in done["history"]), "the earlier failure is still in the record"


def test_a_crash_between_persisting_and_acknowledging_is_recovered_without_asking_again(stub, tmp_path):
    """The two writes cannot be one. This simulates the process dying in the gap: the result is on disk, the entry is still
    PENDING, and nothing recorded that the item was answered.

    A drain that simply reclassified would ask the model a second time and write a second decision for one item. The next
    drain must recognise its own interrupted work -- this item's bytes under this exact evaluation -- and finish it."""
    registry, backend = stub
    book = queue_of(tmp_path, news("00_relevant.json"))
    store = store_of(tmp_path)
    entry_id = book.pending()[0]["entry_id"]

    def killed(*a, **k):
        raise KeyboardInterrupt("SIGINT between the write and the acknowledgement")

    book.acknowledge = killed                                    # the crash lands exactly in the gap, not before it
    with pytest.raises(KeyboardInterrupt):
        drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF)
    del book.acknowledge

    asked_before_the_crash = len(backend.seen)
    assert asked_before_the_crash == 1
    persisted = store.all_records()
    assert len(persisted) == 1, "the result was written before the crash"
    assert book.get(entry_id)["state"] == PENDING, "the entry was never closed, so it is still owed an answer"

    report = drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF)

    assert report["recovered"] == 1 and report["classified"] == 0
    assert report["items"][0]["store_disposition"] == "ALREADY_PERSISTED"
    assert len(backend.seen) == asked_before_the_crash, "the model was not asked a second time about one decision"
    done = book.get(entry_id)
    assert done["state"] == ACKNOWLEDGED
    assert done["receipt_sha256"] == persisted[0]["record_sha256"]
    assert len(store.all_records()) == 1, "one item, one evaluation, one record"


def test_the_recovered_result_must_be_this_items_own_and_this_exact_evaluation(stub, tmp_path):
    """The recovery lookup is what makes a crash cheap; if it matched loosely it would hand an item somebody else's answer,
    or an answer given under another clock, and never call the model at all."""
    registry, _backend = stub
    book = queue_of(tmp_path, news("00_relevant.json"), news("05_unrelated.json"))
    store = store_of(tmp_path)
    mine, other = book.pending()
    drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF, limit=1)
    plan = planned_evaluation(registry, task_id=TASK, as_of=AS_OF, max_age_seconds=900)

    assert existing_result(store, book.get(mine["entry_id"]), plan) is not None
    assert existing_result(store, book.get(other["entry_id"]), plan) is None, "another item's result is not this one's"
    later = dict(plan, as_of="2026-09-24T12:05:00Z")
    assert existing_result(store, book.get(mine["entry_id"]), later) is None, "another decision clock is another evaluation"
    other_task = dict(plan, task_id="news_triage.v1")
    assert existing_result(store, book.get(mine["entry_id"]), other_task) is None, "another question is another evaluation"


# --- a refusal about the item is a completed outcome; a refusal about the environment is not -------------------------------------

def test_a_refused_item_is_persisted_acknowledged_and_reported_apart_from_the_successes(stub, tmp_path):
    """Wrong language, stale, out of scope: the item was read and judged unusable. That is a result. Retrying it would ask
    the same question of the same bytes forever, and counting it as a success would inflate what was decided."""
    registry, _backend = stub
    book = queue_of(tmp_path, news("00_relevant.json"),
                    news("00_wrong_language.json", "refusals"),
                    news("04_asset_out_of_task_scope.json", "refusals"))
    store = store_of(tmp_path)

    report = drain(book, store, task_id=TASK, registry=registry, as_of=AS_OF)

    assert (report["classified"], report["refused"], report["failed"]) == (1, 2, 0)
    assert report["acknowledged"] == 3
    refused = [item for item in report["items"] if item["outcome"] == REFUSED]
    assert {item["status"] for item in refused} == {"REFUSED_WITH_INPUT"}
    assert "LANGUAGE_NOT_VALIDATED" in " ".join(item["why"] for item in refused)
    assert "ASSET_NOT_IN_TASK_SCOPE" in " ".join(item["why"] for item in refused)
    assert book.report()["by_state"] == {ACKNOWLEDGED: 3}
    for item in refused:
        record = store.find_record(item["acknowledged_with"])
        assert record["status"] == "REFUSED_WITH_INPUT" and record["integrity"] == "OK"
    replayed = store.replay()
    assert replayed["records"] == 3
    assert replayed["actionable_events"] == 1 and replayed["refused_events"] == 2, "a retained refusal is not a decision"


def test_a_missing_checkpoint_fails_the_item_instead_of_closing_it(stub, tmp_path):
    """MODEL_NOT_FITTED says nothing about the news. Acknowledging it would close the item forever on the strength of a file
    that was absent for a minute, and a refusal about this environment must not be stored as a judgement about the news."""
    working, backend = stub
    book = queue_of(tmp_path, news("00_relevant.json"))
    store = store_of(tmp_path)
    entry_id = book.pending()[0]["entry_id"]
    unfitted = registry_with(backend, tmp_path / "no-weights", manifest=False)

    report = drain(book, store, task_id=TASK, registry=unfitted, as_of=AS_OF)

    assert (report["failed"], report["acknowledged"], report["refused"]) == (1, 0, 0)
    assert "MODEL_NOT_FITTED" in report["items"][0]["why"]
    assert book.get(entry_id)["state"] == FAILED
    assert store.all_records() == [], "an environment refusal is not a result about the news"
    assert backend.seen == [], "nothing was loaded and nothing was asked"

    book.retry(entry_id)
    repaired = drain(book, store, task_id=TASK, registry=working, as_of=AS_OF)
    assert repaired["classified"] == 1 and book.get(entry_id)["state"] == ACKNOWLEDGED


def test_an_unknown_task_refuses_before_any_item_is_touched(stub, tmp_path):
    """An operator's typo is not the items' failure. Spending every entry's attempt counter on it would bury the reason."""
    registry, backend = stub
    book = queue_of(tmp_path, news("00_relevant.json"))
    store = store_of(tmp_path)

    with pytest.raises(Refusal, match="UNKNOWN_TASK"):
        drain(book, store, task_id="news_relevance_eurusd.v9", registry=registry, as_of=AS_OF)

    assert [e["state"] for e in book.entries()] == [PENDING]
    assert book.get(book.entries()[0]["entry_id"])["attempts"] == 0
    assert backend.seen == [] and store.all_records() == []


# --- the same path in another process ----------------------------------------------------------------------------------------------

def test_the_cli_drains_the_recorded_corpus_in_a_new_process(tmp_path):
    """The declared non-model backend, so no weights are needed; every receipt it writes says NON_MODEL_FIXTURE."""
    done = subprocess.run([sys.executable, "-m", "news_signal", "drain",
                           "--source", str(CORPUS / "events"),
                           "--queue", str(tmp_path / "queue"),
                           "--store", str(tmp_path / "shadow"),
                           "--task", TASK],
                          capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "NEWS_SIGNAL_BACKEND": "fixture",
                               "PYTHONPATH": str(ROOT / "src")})
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["collection"]["live_feed"] is False
    assert result["drain"]["processed"] == result["drain"]["acknowledged"] == CORPUS_ENTRIES
    assert result["drain"]["failed"] == 0
    assert result["drain"]["execution_authorized"] is False
    assert result["clock_mode"] == "WALL_CLOCK"

    replayed = subprocess.run([sys.executable, "-m", "news_signal", "replay", "--store", str(tmp_path / "shadow")],
                              capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "src")})
    assert replayed.returncode == 0, replayed.stderr
    report = json.loads(replayed.stdout)
    assert report["records"] == CORPUS_ENTRIES and report["integrity_failures"] == []

    # a second invocation over the same directories collects nothing new and finds nothing pending
    again = subprocess.run([sys.executable, "-m", "news_signal", "drain",
                            "--source", str(CORPUS / "events"),
                            "--queue", str(tmp_path / "queue"),
                            "--store", str(tmp_path / "shadow"),
                            "--task", TASK],
                           capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin", "NEWS_SIGNAL_BACKEND": "fixture",
                                "PYTHONPATH": str(ROOT / "src")})
    assert again.returncode == 0, again.stderr
    assert json.loads(again.stdout)["drain"]["processed"] == 0
