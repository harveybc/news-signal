"""CL20: what the durable queue must refuse to do with an entry it can no longer vouch for.

Four counterexamples, all of them about the same thing: a queue is trusted to say WHICH item is pending and WHETHER it is
intact. Every defect below ends with the queue answering one of those two questions confidently and wrongly -- a corruption
re-signed by a state change, an acceptance reported for a write that did not happen, one item served under another item's
key, and two unrelated sources merged into one event because they happened to reuse an identifier.
"""

import json
import os
from pathlib import Path

import pytest

from news_signal.collector import PENDING, DurableQueue
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


def path_of(tmp_path, entry_id):
    return next((tmp_path / "queue").rglob(f"{entry_id}.json"))


def corrupt(tmp_path, entry_id, headline="TAMPERED"):
    """Edit the entry on disk the way a bad sector or a careless operator would: valid JSON, wrong content."""
    path = path_of(tmp_path, entry_id)
    damaged = json.loads(path.read_text())
    damaged["event"]["headline"] = headline
    path.write_text(json.dumps(damaged))
    return path


# --- CL20-a: a state change is not a laundry ----------------------------------------------------------------------------

def test_a_retry_may_not_re_sign_a_corrupted_entry(tmp_path):
    """The corrupted entry was correctly kept out of `pending()`. A retry then recomputed `entry_sha256` over the corrupted
    content and handed the tampered headline to the classifier with integrity OK."""
    book = queue(tmp_path)
    entry, _d = book.offer(item(), received_at=RECEIVED)
    book.fail(entry["entry_id"], reason="the provider refused")
    corrupt(tmp_path, entry["entry_id"])
    assert book.get(entry["entry_id"])["integrity"] == "FAILED"

    with pytest.raises(Refusal, match="CORRUPT_QUEUE_ENTRY"):
        book.retry(entry["entry_id"])

    assert book.get(entry["entry_id"])["integrity"] == "FAILED", "the refusal did not rewrite the entry"
    assert book.pending() == [], "a corruption may not become dispatchable by changing state"
    assert book.report()["integrity_failures"] == [entry["entry_id"]]


def test_no_transition_launders_a_corrupted_entry(tmp_path):
    """Not only the retry: acknowledging or failing a corrupted entry would re-sign it just as thoroughly."""
    book = queue(tmp_path)
    entry, _d = book.offer(item(), received_at=RECEIVED)
    corrupt(tmp_path, entry["entry_id"])
    for call in (lambda: book.acknowledge(entry["entry_id"], receipt_sha256="r" * 64),
                 lambda: book.fail(entry["entry_id"], reason="the provider refused"),
                 lambda: book.retry(entry["entry_id"])):
        with pytest.raises(Refusal, match="CORRUPT_QUEUE_ENTRY"):
            call()
    assert book.get(entry["entry_id"])["state"] == PENDING, "the state on disk is the last one we could vouch for"


# --- CL20-b: an acceptance is a claim about the disk ----------------------------------------------------------------------

def test_an_acceptance_is_never_reported_for_a_write_that_did_not_happen(tmp_path):
    """The exclusive create loses to a file that is already there. When that file is INVALID the caller was told ACCEPTED
    and handed a digest nothing on disk carries, while the queue still held only the corruption."""
    book = queue(tmp_path)
    first, _d = book.offer(item(), received_at=RECEIVED)
    corrupt(tmp_path, first["entry_id"])

    entry, disposition = book.offer(item(), received_at="2026-09-24T11:55:00Z")
    assert disposition != "DUPLICATE", "the entry on disk was not a valid copy of this item"

    on_disk = book.get(entry["entry_id"])
    assert on_disk is not None, "an ACCEPTED item is on disk"
    assert on_disk["integrity"] == "OK"
    assert on_disk["entry_sha256"] == entry["entry_sha256"], "the digest returned is the digest stored"
    assert [e["entry_id"] for e in book.pending()] == [entry["entry_id"]]
    assert book.report()["quarantined"] == [first["entry_id"]], "the corrupted bytes are set aside, not deleted"


def test_a_write_that_keeps_losing_its_key_refuses_instead_of_claiming_an_acceptance(tmp_path, monkeypatch):
    """The pathological case of the same defect: every exclusive create loses. With nothing of ours on disk there is no
    honest answer but a refusal, and the caller must hear it rather than receive a digest for an absent entry."""
    book = queue(tmp_path)
    first, _d = book.offer(item(), received_at=RECEIVED)
    corrupt(tmp_path, first["entry_id"])

    def always_taken(source, target):
        raise FileExistsError(target)

    monkeypatch.setattr(os, "link", always_taken)
    with pytest.raises(Refusal, match="QUEUE_WRITE_NOT_ACCEPTED"):
        book.offer(item(), received_at=RECEIVED)


# --- CL20-c: an entry belongs where it lies --------------------------------------------------------------------------------

def test_an_intact_entry_under_another_items_key_is_not_served_as_that_item(tmp_path):
    """The self-hash of a whole entry for item B answers "is this file intact?" perfectly. It says nothing about whether
    this is item A, and the queue was asked only the first question before serving it as A's DUPLICATE."""
    book = queue(tmp_path)
    mine, _d = book.offer(item(), received_at=RECEIVED)
    other, _d = book.offer(item(event_id="other-001", headline="Something else entirely"), received_at=RECEIVED)
    path_of(tmp_path, mine["entry_id"]).write_text(path_of(tmp_path, other["entry_id"]).read_text())

    served = book.get(mine["entry_id"])
    assert served["integrity"] == "FAILED", "an intact entry for another item is not this item's entry"
    assert any("MISPLACED_QUEUE_ENTRY" in problem for problem in served.get("integrity_problems") or [])
    assert [e["entry_id"] for e in book.pending()] == [other["entry_id"]]
    assert mine["entry_id"] in book.report()["integrity_failures"]

    again, disposition = book.offer(item(), received_at=RECEIVED)
    assert disposition != "DUPLICATE"
    assert again["entry_id"] == mine["entry_id"]
    assert book.get(again["entry_id"])["event"]["headline"] == item()["headline"]


def test_an_entry_carrying_a_key_its_own_fields_do_not_derive_fails(tmp_path):
    """The key is derived from the entry, then compared -- never taken on the entry's word."""
    book = queue(tmp_path)
    entry, _d = book.offer(item(), received_at=RECEIVED)
    path = path_of(tmp_path, entry["entry_id"])
    forged = json.loads(path.read_text())
    forged["content_sha256"] = "0" * 64
    forged["entry_sha256"] = None
    forged["entry_sha256"] = digest({k: v for k, v in forged.items() if k != "entry_sha256"})
    path.write_text(json.dumps(forged))

    served = book.get(entry["entry_id"])
    assert served["integrity"] == "FAILED", "self-consistent bytes are not a self-consistent identity"
    assert book.pending() == []


# --- CL20-d: a revision is linked within a source, never across sources ------------------------------------------------------

def test_two_sources_reusing_one_event_id_are_two_events(tmp_path):
    """`event_id` is namespaced by the source that issued it. Treating a second wire's unrelated item as a revision of the
    first both invents a correction that nobody published and hides one of the two events from the report."""
    book = queue(tmp_path)
    first, _d = book.offer(item(source="regression-bank"), received_at=RECEIVED)
    second, disposition = book.offer(item(source="other-wire"), received_at=RECEIVED)

    assert disposition == "ACCEPTED", "another source's item is not a revision of ours"
    assert second["revises"] == []
    assert second["revision_index"] == 0
    assert second["entry_id"] != first["entry_id"]
    report = book.report()
    assert report["events"] == 2, "an event identity is the source AND its event_id"
    assert report["revisions"] == 0


def test_a_revision_within_one_source_still_links_when_another_source_shares_the_event_id(tmp_path):
    """The repair must not cost the queue the linking it exists to do."""
    book = queue(tmp_path)
    original, _d = book.offer(item(source="regression-bank"), received_at=RECEIVED)
    book.offer(item(source="other-wire"), received_at=RECEIVED)
    revised, disposition = book.offer(item(source="regression-bank", headline="ECB keeps deposit rate unchanged (corrected)"),
                                      received_at="2026-09-24T11:55:00Z")

    assert disposition == "REVISION"
    assert revised["revises"] == [original["entry_id"]], "linked to its own source's earlier text and to nothing else"
    assert revised["revision_index"] == 1
    report = book.report()
    assert report["events"] == 2 and report["revisions"] == 1
