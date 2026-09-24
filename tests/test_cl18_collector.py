"""CL18: the collector, its durable queue, and what survives a crash.

Built without a feed entitlement on purpose. What a collector owes its consumer does not depend on where the items came
from, and waiting for an entitlement to write it would have been the wrong order of work.
"""

import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from news_signal.collector import (ACKNOWLEDGED, FAILED, PENDING, DurableQueue, RecordedDirectorySource, collect)
from news_signal.core import Refusal, digest, load_json

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples/eurusd"
RECEIVED = "2026-09-24T11:50:30Z"


def item(**over):
    event = load_json(CORPUS / "events/00_relevant.json")
    event.pop("received_at")
    event.update(over)
    return event


def queue(tmp_path):
    return DurableQueue(tmp_path / "queue")


# --- the receipt clock belongs to this system --------------------------------------------------------------------------

def test_an_item_may_not_supply_the_moment_we_received_it():
    """A source that can set our receipt clock can make a late item look timely."""
    with pytest.raises(Refusal, match="RECEIPT_CLOCK_IS_NOT_THE_SOURCES"):
        DurableQueue("/tmp/unused").offer(item(received_at="2026-09-24T11:50:30Z"))


def test_the_collector_stamps_the_receipt_clock(tmp_path):
    entry, disposition = queue(tmp_path).offer(item(), received_at=RECEIVED)
    assert disposition == "ACCEPTED"
    assert entry["event"]["received_at"] == RECEIVED
    assert entry["received_at"] == RECEIVED


def test_an_item_received_before_it_was_published_is_refused(tmp_path):
    with pytest.raises(Refusal, match="RECEIVED_BEFORE_PUBLISHED"):
        queue(tmp_path).offer(item(published_at="2026-09-24T12:00:00Z"), received_at="2026-09-24T11:00:00Z")


def test_an_unreadable_receipt_clock_refuses_at_the_boundary(tmp_path):
    with pytest.raises(Refusal):
        queue(tmp_path).offer(item(), received_at="not-a-time")


def test_a_foreign_field_is_refused(tmp_path):
    with pytest.raises(Refusal, match="NEWS_FIELDS_MISMATCH"):
        queue(tmp_path).offer(dict(item(), sentiment="bullish"), received_at=RECEIVED)


# --- once, and only once -------------------------------------------------------------------------------------------------

def test_the_same_item_offered_twice_is_one_entry(tmp_path):
    book = queue(tmp_path)
    first, _d = book.offer(item(), received_at=RECEIVED)
    again, disposition = book.offer(item(), received_at="2026-09-24T11:59:00Z")
    assert disposition == "DUPLICATE"
    assert again["entry_id"] == first["entry_id"]
    assert again["received_at"] == RECEIVED, "the first receipt is the receipt; a second sighting does not move it"
    assert book.report()["entries"] == 1


def test_concurrent_offers_of_one_item_produce_one_entry(tmp_path):
    book = queue(tmp_path)
    seen, errors = [], []

    def offer():
        try:
            seen.append(book.offer(item(), received_at=RECEIVED)[0]["entry_id"])
        except Exception as exc:                                 # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=offer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(set(seen)) == 1
    assert book.report()["entries"] == 1


# --- a revision is new information, not a correction of the record ----------------------------------------------------------

def test_a_revision_is_its_own_entry_and_links_to_what_it_revises(tmp_path):
    book = queue(tmp_path)
    original, _d = book.offer(item(), received_at=RECEIVED)
    revised, disposition = book.offer(item(headline="ECB keeps deposit rate unchanged at 2.00% (corrected)"),
                                      received_at="2026-09-24T11:55:00Z")
    assert disposition == "REVISION"
    assert revised["entry_id"] != original["entry_id"]
    assert revised["revises"] == [original["entry_id"]]
    assert revised["revision_index"] == 1
    assert book.get(original["entry_id"]) is not None, "the earlier text is not removed by its revision"
    assert book.report()["events"] == 1 and book.report()["revisions"] == 1


def test_two_different_stories_are_two_events(tmp_path):
    book = queue(tmp_path)
    book.offer(item(), received_at=RECEIVED)
    book.offer(item(event_id="other-001", headline="Something else entirely"), received_at=RECEIVED)
    assert book.report()["events"] == 2 and book.report()["revisions"] == 0


# --- the lifecycle, and what a crash leaves behind -----------------------------------------------------------------------------

def test_an_item_stays_pending_until_a_consumer_names_what_it_produced(tmp_path):
    book = queue(tmp_path)
    entry, _d = book.offer(item(), received_at=RECEIVED)
    assert [e["entry_id"] for e in book.pending()] == [entry["entry_id"]]
    with pytest.raises(Refusal, match="ACKNOWLEDGEMENT_NEEDS_ITS_RESULT"):
        book.acknowledge(entry["entry_id"], receipt_sha256=None)
    done = book.acknowledge(entry["entry_id"], receipt_sha256="r" * 64)
    assert done["state"] == ACKNOWLEDGED and done["receipt_sha256"] == "r" * 64
    assert book.pending() == []


def test_a_failure_is_retained_through_a_retry(tmp_path):
    book = queue(tmp_path)
    entry, _d = book.offer(item(), received_at=RECEIVED)
    failed = book.fail(entry["entry_id"], reason="the provider refused")
    assert failed["state"] == FAILED and failed["attempts"] == 1
    back = book.retry(entry["entry_id"])
    assert back["state"] == PENDING
    states = [h["state"] for h in back["history"]]
    assert states == [PENDING, FAILED, PENDING], "a retry does not erase the failure that preceded it"
    assert any(h.get("reason") == "the provider refused" for h in back["history"])


def test_the_queue_survives_a_new_process(tmp_path):
    book = queue(tmp_path)
    entry, _d = book.offer(item(), received_at=RECEIVED)
    book.acknowledge(entry["entry_id"], receipt_sha256="r" * 64)
    book.offer(item(event_id="pending-001"), received_at=RECEIVED)
    code = ("import json,sys;"
            "sys.path.insert(0, %r);"
            "from news_signal.collector import DurableQueue;"
            "print(json.dumps(DurableQueue(%r).report()))" % (str(ROOT / "src"), str(tmp_path / "queue")))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["entries"] == 2
    assert report["by_state"] == {ACKNOWLEDGED: 1, PENDING: 1}
    assert report["integrity_failures"] == []


def test_an_interrupted_write_leaves_the_previous_state_readable(tmp_path):
    book = queue(tmp_path)
    entry, _d = book.offer(item(), received_at=RECEIVED)
    (tmp_path / "queue" / "interrupted.json.tmp").write_text('{"schema": "news_queue_entry.v1", "trunca')
    report = book.report()
    assert report["entries"] == 1 and report["unfinished_writes"] == 1
    assert report["integrity_failures"] == []


def test_a_tampered_entry_is_reported_and_not_dispatched(tmp_path):
    book = queue(tmp_path)
    entry, _d = book.offer(item(), received_at=RECEIVED)
    path = next((tmp_path / "queue").rglob("*.json"))
    damaged = json.loads(path.read_text())
    damaged["event"]["headline"] = "TAMPERED"
    path.write_text(json.dumps(damaged))
    assert book.report()["integrity_failures"] == [entry["entry_id"]]
    assert book.pending() == [], "an entry that fails its own digest is not handed to a consumer"


# --- a recorded directory is a real source with a declared boundary ---------------------------------------------------------------

def test_a_recorded_directory_is_drained_once_and_declares_it_is_not_a_feed(tmp_path):
    source = RecordedDirectorySource(CORPUS / "events")
    book = queue(tmp_path)
    first = collect(source, book, received_at=RECEIVED)
    assert first["live_feed"] is False
    assert first["source_kind"] == "RECORDED_FILES_NOT_A_LIVE_FEED"
    assert first["accepted"] + first["revisions"] + first["duplicates"] == 13
    assert first["refused"] == 0
    again = collect(source, book, received_at="2026-09-24T12:30:00Z")
    assert again["accepted"] == 0 and again["duplicates"] == first["accepted"] + first["revisions"] + first["duplicates"]
    assert book.report()["entries"] == first["accepted"] + first["revisions"]


def test_the_recorded_corpus_revision_is_recognised_as_one(tmp_path):
    book = queue(tmp_path)
    collect(RecordedDirectorySource(CORPUS / "events"), book, received_at=RECEIVED)
    report = book.report()
    assert report["revisions"] == 1, "the corpus contains one revision of ecb-hold-001"
    assert report["events"] == 11
    assert report["inference_performed"] is False and report["execution_authorized"] is False
