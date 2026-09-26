"""RP148: the collection boundary, written with no feed entitlement and no model weights.

What an entitlement would add to this file is the arrival of real bytes. Everything the arrival has to be judged against --
whose clock says what, when a second sighting is a duplicate and when it is new information, what a decision at T could have
used, what happens to an item too old to admit, and the difference between a quiet source and a broken one -- is decided
here, from recorded fixtures, and does not become truer when the wire is licensed.

Four claims are tested, each named in RP148:

1. **Receipt clocks.** Every item carries when it reached us, stamped here; a source that declares no publication clock is
   recorded as declaring none, and the receipt clock is never promoted into its place.
2. **Dedup and revisions.** The same item twice is one entry. Changed text is a new entry bound to the first, and the first
   is never rewritten -- which is exactly what makes `known_at(T)` answerable.
3. **Persistent queues.** Already built and tested in `test_cl18_collector.py` and `test_cl23_pipeline.py`; what is added
   here is a collection interrupted mid-batch, replayed without double-counting.
4. **Stale input.** A maximum age is declared or it does not exist. An item past it is refused by name, an item whose age
   cannot be measured is refused rather than admitted as fresh, and a source that hands us nothing is `AWAITED`.

Nothing in this file opens a socket, holds a credential, or loads a model.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from news_signal.collector import (AWAITED, DECLARED_DEFAULT_MAX_AGE_SECONDS, NEVER_PRODUCED, NOT_PROVIDED_BY_SOURCE,
                                   PRODUCING, SILENT_SINCE, SOURCE_DECLARED, SOURCE_NOT_DECLARED, DurableQueue,
                                   RecordedDirectorySource, collect, known_at, source_observation)
from news_signal.core import Refusal, load_json
from news_signal.provider import DEFAULT_MAX_AGE_SECONDS

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples/eurusd"
EVENTS = CORPUS / "events"
COLLECTION = CORPUS / "collection"
RECEIVED = "2026-09-24T11:50:30Z"
LATER = "2026-09-24T12:10:00Z"


def item(**over):
    event = load_json(EVENTS / "00_relevant.json")
    event.pop("received_at")
    event.update(over)
    return event


def clockless():
    """The fixture whose source declares no publication clock at all."""
    return load_json(COLLECTION / "00_no_publication_clock.json")


def queue(tmp_path, **kwargs):
    return DurableQueue(tmp_path / "queue", **kwargs)


# --- 1. receipt clocks, and the one substitution that is never made -----------------------------------------------------

def test_a_declared_publication_clock_is_kept_as_the_sources_claim_and_the_lag_is_measured(tmp_path):
    entry, disposition = queue(tmp_path).offer(item(), received_at=RECEIVED)
    assert disposition == "ACCEPTED"
    assert entry["publication_clock"] == SOURCE_DECLARED
    assert entry["published_at"] == "2026-09-24T11:50:00Z" and entry["received_at"] == RECEIVED
    assert entry["receipt_lag_seconds"] == 30.0
    assert "SOURCE_CLAIM" in entry["publication_bound"], "the source's clock is its claim, not our observation"


def test_a_source_that_declares_no_publication_clock_is_recorded_as_declaring_none(tmp_path):
    entry, disposition = queue(tmp_path).offer(clockless(), received_at=RECEIVED)
    assert disposition == "ACCEPTED"
    assert entry["publication_clock"] == NOT_PROVIDED_BY_SOURCE
    assert entry["published_at"] is None and entry["event"]["published_at"] is None
    assert entry["receipt_lag_seconds"] is None, "an age nobody declared is not an age of zero"
    assert "UPPER_BOUND_ONLY" in entry["publication_bound"]


def test_the_receipt_clock_is_never_substituted_for_a_missing_publication_clock(tmp_path):
    """The one error this discipline exists to prevent. A receipt bounds a publication from above and never from below."""
    book = queue(tmp_path)
    entry, _d = book.offer(clockless(), received_at=RECEIVED)
    on_disk = book.get(entry["entry_id"])
    assert on_disk["published_at"] is None
    assert on_disk["event"]["published_at"] is None
    written = json.loads(next((tmp_path / "queue").rglob("*.json")).read_text())
    assert RECEIVED not in json.dumps(written["published_at"])
    assert written["event"]["published_at"] is None
    assert book.report()["without_publication_clock"] == 1
    assert book.report()["receipt_lag_unmeasurable"] == 1


def test_an_unreadable_publication_clock_is_refused_by_name(tmp_path):
    with pytest.raises(Refusal, match="PUBLICATION_CLOCK_UNREADABLE"):
        queue(tmp_path).offer(item(published_at="yesterday afternoon"), received_at=RECEIVED)


def test_a_naive_publication_clock_names_no_instant_and_is_refused(tmp_path):
    with pytest.raises(Refusal, match="PUBLICATION_CLOCK_UNREADABLE"):
        queue(tmp_path).offer(item(published_at="2026-09-24T11:50:00"), received_at=RECEIVED)


def test_a_half_record_is_refused_by_its_missing_field_name(tmp_path):
    """The fields that must be there are named. A record missing its body used to die inside a digest computation."""
    incomplete = item()
    incomplete.pop("body")
    with pytest.raises(Refusal, match="NEWS_FIELDS_MISSING: body"):
        queue(tmp_path).offer(incomplete, received_at=RECEIVED)


# --- 2. dedup, revisions, and what was knowable at T --------------------------------------------------------------------

def test_the_same_item_seen_later_is_a_duplicate_and_not_a_revision(tmp_path):
    """Two sightings of one story are one story. Only changed bytes are new information."""
    book = queue(tmp_path)
    first, _d = book.offer(item(), received_at=RECEIVED)
    again, disposition = book.offer(item(), received_at=LATER)
    assert disposition == "DUPLICATE"
    assert again["entry_id"] == first["entry_id"]
    assert again["revises"] == [] and again["revision_index"] == 0
    assert again["received_at"] == RECEIVED, "the receipt clock is the first receipt; a re-reading does not move it"
    assert book.report()["revisions"] == 0


def test_a_changed_headline_is_a_new_revision_that_rewrites_nothing(tmp_path):
    book = queue(tmp_path)
    original, _d = book.offer(item(), received_at=RECEIVED)
    before = next((tmp_path / "queue").rglob(f"{original['entry_id']}.json")).read_bytes()
    revised, disposition = book.offer(item(headline="ECB keeps deposit rate unchanged at 2.00% (clarified)"),
                                      received_at=LATER)
    assert disposition == "REVISION"
    assert revised["revises"] == [original["entry_id"]] and revised["revision_index"] == 1
    after = next((tmp_path / "queue").rglob(f"{original['entry_id']}.json")).read_bytes()
    assert after == before, "the predecessor's bytes, digest and receipt clock are untouched by its revision"
    assert book.get(original["entry_id"])["integrity"] == "OK"


def test_what_was_knowable_at_T_never_includes_an_item_received_after_T(tmp_path):
    """The claim RP148 asks to be proved, swept over a range of T rather than asserted at one convenient instant."""
    book = queue(tmp_path)
    early, _d = book.offer(item(), received_at=RECEIVED)
    late, _d = book.offer(item(event_id="pmi-002", headline="Euro-area PMI revised higher"), received_at=LATER)
    for as_of, expected in (("2026-09-24T11:50:29Z", set()),
                            ("2026-09-24T11:50:30Z", {early["entry_id"]}),
                            ("2026-09-24T12:09:59Z", {early["entry_id"]}),
                            ("2026-09-24T12:10:00Z", {early["entry_id"], late["entry_id"]}),
                            ("2026-09-25T00:00:00Z", {early["entry_id"], late["entry_id"]})):
        report = known_at(book, as_of)
        assert set(report["known"]) == expected, as_of
        assert late["entry_id"] not in report["known"] or as_of >= LATER
        assert set(report["known"]) & set(report["not_yet_received"]) == set()
        assert report["inference_performed"] is False and report["execution_authorized"] is False


def test_an_item_not_yet_received_is_named_as_absent_rather_than_dropped(tmp_path):
    book = queue(tmp_path)
    book.offer(item(), received_at=RECEIVED)
    late, _d = book.offer(item(event_id="pmi-002", headline="Euro-area PMI revised higher"), received_at=LATER)
    report = known_at(book, "2026-09-24T11:55:00Z")
    assert report["not_yet_received"] == [late["entry_id"]]
    assert report["events_known"] == 1


def test_known_at_shows_the_earlier_text_before_its_revision_arrives(tmp_path):
    """Because no entry is ever rewritten, discarding what arrived after T leaves exactly the text that stood at T."""
    book = queue(tmp_path)
    original, _d = book.offer(item(), received_at=RECEIVED)
    revised, disposition = book.offer(item(headline="ECB keeps deposit rate unchanged at 2.00% (clarified)"),
                                      received_at=LATER)
    assert disposition == "REVISION"
    before = known_at(book, "2026-09-24T11:55:00Z")
    assert [row["entry_id"] for row in before["latest_known_text"]] == [original["entry_id"]]
    assert [row["revision_index"] for row in before["latest_known_text"]] == [0]
    after = known_at(book, "2026-09-24T12:30:00Z")
    assert [row["entry_id"] for row in after["latest_known_text"]] == [revised["entry_id"]]
    assert [row["revision_index"] for row in after["latest_known_text"]] == [1]


def test_known_at_compares_instants_and_not_the_strings_they_were_written_as(tmp_path):
    book = queue(tmp_path)
    entry, _d = book.offer(item(), received_at="2026-09-24T14:00:00+02:00")
    assert known_at(book, "2026-09-24T11:59:59Z")["known"] == []
    assert known_at(book, "2026-09-24T12:00:00Z")["known"] == [entry["entry_id"]]


def test_known_at_refuses_a_clock_that_names_no_instant(tmp_path):
    book = queue(tmp_path)
    book.offer(item(), received_at=RECEIVED)
    with pytest.raises(Refusal, match="TIMEZONE_AWARE_TIMESTAMP_REQUIRED"):
        known_at(book, "2026-09-24T11:55:00")


def test_known_at_ignores_an_entry_it_cannot_vouch_for(tmp_path):
    """A tampered entry is not evidence of what was knowable; it is evidence of tampering, and it is reported as that."""
    book = queue(tmp_path)
    entry, _d = book.offer(item(), received_at=RECEIVED)
    path = next((tmp_path / "queue").rglob("*.json"))
    damaged = json.loads(path.read_text())
    damaged["event"]["headline"] = "TAMPERED"
    path.write_text(json.dumps(damaged))
    assert known_at(book, LATER)["known"] == []
    assert book.report()["integrity_failures"] == [entry["entry_id"]]


def test_known_at_can_be_asked_of_one_source(tmp_path):
    book = queue(tmp_path)
    mine, _d = book.offer(item(), received_at=RECEIVED)
    book.offer(item(source="another-wire", event_id="foreign-001"), received_at=RECEIVED)
    report = known_at(book, LATER, source="regression-bank")
    assert report["known"] == [mine["entry_id"]] and report["source"] == "regression-bank"


# --- 3. the persistent queue, carried to collection ---------------------------------------------------------------------

def test_a_collection_interrupted_mid_batch_replays_without_double_counting(tmp_path):
    """The queue's crash behaviour, exercised at the boundary that fills it rather than at the one that drains it.

    The drain side of this is `test_cl23_pipeline.py::test_a_crash_between_persisting_and_acknowledging_is_recovered_
    without_asking_again`; this is the collection side."""

    class DiesHalfWay:
        kind = "RECORDED_FILES_NOT_A_LIVE_FEED"
        source_name = "regression-bank"

        def items(self):
            yield item(), "00.json"
            yield item(event_id="second-001", headline="A second story"), "01.json"
            raise KeyboardInterrupt("the process was killed between two items")

    book = queue(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        collect(DiesHalfWay(), book, received_at=RECEIVED)
    assert book.report()["entries"] == 2, "what was accepted before the interruption is on disk"

    class Whole(DiesHalfWay):
        def items(self):
            yield item(), "00.json"
            yield item(event_id="second-001", headline="A second story"), "01.json"
            yield item(event_id="third-001", headline="A third story"), "02.json"

    replayed = collect(Whole(), DurableQueue(tmp_path / "queue"), received_at=LATER)
    assert replayed["duplicates"] == 2 and replayed["accepted"] == 1
    assert replayed["refused"] == 0
    assert DurableQueue(tmp_path / "queue").report()["entries"] == 3, "nothing was counted or stored twice"


def test_the_declared_age_policy_is_recorded_in_the_entry_and_read_back_after_a_restart(tmp_path):
    book = queue(tmp_path, max_age_seconds=900)
    entry, _d = book.offer(item(), received_at=RECEIVED)
    reopened = DurableQueue(tmp_path / "queue", max_age_seconds=900).get(entry["entry_id"])
    assert reopened["staleness_policy"] == "DECLARED" and reopened["max_age_seconds"] == 900
    assert reopened["integrity"] == "OK"


# --- 4. stale input, and a source that says nothing ---------------------------------------------------------------------

def test_with_no_maximum_age_declared_nothing_is_refused_for_age_and_the_record_says_so(tmp_path):
    book = queue(tmp_path)
    entry, disposition = book.offer(load_json(COLLECTION / "01_late_arrival.json"), received_at=RECEIVED)
    assert disposition == "ACCEPTED"
    assert entry["staleness_policy"] == "NOT_DECLARED" and entry["max_age_seconds"] is None
    assert entry["receipt_lag_seconds"] == 21030.0, "the lag is measured and published even when no policy judges it"
    assert book.report()["staleness_policy"] == "NOT_DECLARED"


def test_an_item_older_than_the_declared_maximum_age_is_refused_by_name(tmp_path):
    book = queue(tmp_path, max_age_seconds=900)
    with pytest.raises(Refusal, match="STALE_AT_COLLECTION"):
        book.offer(load_json(COLLECTION / "01_late_arrival.json"), received_at=RECEIVED)
    assert book.report()["entries"] == 0, "a refused item is not in the queue"


def test_an_age_that_cannot_be_measured_is_not_admitted_as_a_fresh_one(tmp_path):
    """A declared freshness policy and an item with no publication clock cannot both be honoured. The item is refused."""
    book = queue(tmp_path, max_age_seconds=900)
    with pytest.raises(Refusal, match="AGE_NOT_MEASURABLE_WITHOUT_PUBLICATION_CLOCK"):
        book.offer(clockless(), received_at=RECEIVED)
    assert book.report()["entries"] == 0


def test_the_same_item_is_accepted_when_the_policy_allows_its_age(tmp_path):
    book = queue(tmp_path, max_age_seconds=86400)
    entry, disposition = book.offer(load_json(COLLECTION / "01_late_arrival.json"), received_at=RECEIVED)
    assert disposition == "ACCEPTED" and entry["max_age_seconds"] == 86400


def test_a_policy_that_is_not_a_whole_number_of_seconds_is_refused(tmp_path):
    for policy in (0, 86401, 900.0, True, "900", -1):
        with pytest.raises(Refusal, match="INVALID_AGE_POLICY"):
            queue(tmp_path, max_age_seconds=policy).offer(item(), received_at=RECEIVED)


def test_the_providers_default_age_is_available_but_never_applied_unless_declared(tmp_path):
    """`DEFAULT_MAX_AGE_SECONDS` is the decision boundary's number. It is re-exported for a caller to DECLARE, not
    inherited: a default applied silently would report a freshness guarantee nobody asked for."""
    assert DECLARED_DEFAULT_MAX_AGE_SECONDS == DEFAULT_MAX_AGE_SECONDS
    undeclared, disposition = queue(tmp_path).offer(load_json(COLLECTION / "01_late_arrival.json"), received_at=RECEIVED)
    assert disposition == "ACCEPTED" and undeclared["max_age_seconds"] is None
    with pytest.raises(Refusal, match="STALE_AT_COLLECTION"):
        DurableQueue(tmp_path / "declared", max_age_seconds=DECLARED_DEFAULT_MAX_AGE_SECONDS).offer(
            load_json(COLLECTION / "01_late_arrival.json"), received_at=RECEIVED)


def test_a_stale_re_offer_is_refused_without_disturbing_the_entry_already_held(tmp_path):
    book = queue(tmp_path, max_age_seconds=900)
    entry, _d = book.offer(item(), received_at=RECEIVED)
    with pytest.raises(Refusal, match="STALE_AT_COLLECTION"):
        book.offer(item(), received_at="2026-09-25T11:50:30Z")
    held = book.get(entry["entry_id"])
    assert held["state"] == "PENDING" and held["received_at"] == RECEIVED and held["integrity"] == "OK"


def test_a_source_that_hands_us_nothing_is_AWAITED_and_no_batch_is_invented(tmp_path):
    empty = tmp_path / "silent"
    empty.mkdir()
    book = queue(tmp_path)
    report = collect(RecordedDirectorySource(empty, source_name="regression-bank"), book, received_at=RECEIVED)
    assert report["observation"] == AWAITED
    assert report["source_observation"]["silence"] == NEVER_PRODUCED
    assert report["items"] == [] and report["accepted"] == 0 and report["items_offered_by_source"] == 0
    assert report["empty_batch_invented"] is False and report["neutral_sentiment_substituted"] is False
    assert book.report()["entries"] == 0, "silence writes no item"


def test_a_source_that_stopped_producing_is_not_the_same_fact_as_one_that_never_produced(tmp_path):
    book = queue(tmp_path)
    live = collect(RecordedDirectorySource(EVENTS, source_name="regression-bank"), book, received_at=RECEIVED)
    assert live["observation"] == PRODUCING and live["source_observation"]["silence"] is None

    empty = tmp_path / "silent"
    empty.mkdir()
    stopped = collect(RecordedDirectorySource(empty, source_name="regression-bank"), book, received_at=LATER)
    assert stopped["observation"] == AWAITED
    assert stopped["source_observation"]["silence"] == SILENT_SINCE
    assert stopped["source_observation"]["last_received_at"] == RECEIVED
    assert stopped["source_observation"]["silence_seconds"] == 1170.0, "the silence is measured, not merely asserted"

    never = collect(RecordedDirectorySource(empty, source_name="a-wire-never-connected"), book, received_at=LATER)
    assert never["source_observation"]["silence"] == NEVER_PRODUCED
    assert never["source_observation"]["items_held"] == 0
    assert never["source_observation"]["last_received_at"] is None


def test_a_silence_is_never_attributed_to_a_source_nobody_declared(tmp_path):
    empty = tmp_path / "silent"
    empty.mkdir()
    report = collect(RecordedDirectorySource(empty), queue(tmp_path), received_at=RECEIVED)
    assert report["observation"] == AWAITED
    assert report["source_observation"]["attribution"] == SOURCE_NOT_DECLARED
    assert report["source_observation"]["silence"] is None, "an outage is a claim about a named source"
    assert report["source_name"] is None


def test_an_observation_can_be_asked_of_the_queue_alone(tmp_path):
    book = queue(tmp_path)
    book.offer(item(), received_at=RECEIVED)
    observed = source_observation(book, "regression-bank", as_of=LATER, produced_now=0)
    assert observed["observation"] == AWAITED and observed["silence"] == SILENT_SINCE
    assert observed["items_held"] == 1 and observed["silence_seconds"] == 1170.0
    assert source_observation(book, "regression-bank", as_of=LATER, produced_now=3)["observation"] == PRODUCING


def test_a_source_whose_items_are_all_refused_is_producing_and_not_silent(tmp_path):
    """The refusals are ours. Calling that source silent would blame it for a policy we declared."""
    book = queue(tmp_path, max_age_seconds=900)
    report = collect(RecordedDirectorySource(COLLECTION, source_name="regression-bank"), book, received_at=RECEIVED)
    assert report["observation"] == PRODUCING and report["items_offered_by_source"] == 3
    assert report["accepted"] == 0 and report["refused"] == 3
    assert report["refusals_by_name"] == {"AGE_NOT_MEASURABLE_WITHOUT_PUBLICATION_CLOCK": 1,
                                          "DECLARED_SOURCE_MISMATCH": 1,
                                          "STALE_AT_COLLECTION": 1}
    assert book.report()["entries"] == 0


def test_an_item_from_another_wire_is_refused_at_a_boundary_that_declares_its_source(tmp_path):
    book = queue(tmp_path)
    report = collect(RecordedDirectorySource(COLLECTION, source_name="regression-bank"), book, received_at=RECEIVED)
    refused = [row for row in report["items"] if row["disposition"] == "REFUSED"]
    assert [row["reference"] for row in refused] == ["02_another_wire.json"]
    assert "DECLARED_SOURCE_MISMATCH" in refused[0]["why"]
    assert book.report()["sources"] == ["regression-bank"]


def test_an_undeclared_boundary_accepts_whatever_the_files_say_and_attributes_nothing(tmp_path):
    book = queue(tmp_path)
    report = collect(RecordedDirectorySource(COLLECTION), book, received_at=RECEIVED)
    assert report["refused"] == 0 and report["accepted"] == 3
    assert book.report()["sources"] == ["another-wire", "regression-bank"]
    assert report["source_observation"]["attribution"] == SOURCE_NOT_DECLARED


# --- the same boundary through the CLI, in a new process ------------------------------------------------------------------

def run(*args):
    done = subprocess.run([sys.executable, "-m", "news_signal", *args], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "src")})
    return done, json.loads(done.stdout)


def test_the_cli_collects_a_recorded_source_and_reports_the_boundary(tmp_path):
    done, result = run("collect", "--source", str(EVENTS), "--queue", str(tmp_path / "queue"),
                       "--source-name", "regression-bank", "--max-age-seconds", "provider-default",
                       "--received-at", RECEIVED, "--known-at", "2026-09-24T11:55:00Z")
    assert done.returncode == 0, done.stderr
    assert result["clock_mode"] == "REPLAY"
    assert result["collection"]["staleness_policy"] == "DECLARED"
    assert result["collection"]["max_age_seconds"] == DEFAULT_MAX_AGE_SECONDS
    assert result["collection"]["observation"] == PRODUCING
    assert result["collection"]["accepted"] + result["collection"]["revisions"] == 12
    assert result["queue"]["entries"] == 12 and result["queue"]["revisions"] == 1
    assert len(result["known_at"]["known"]) == 12 and result["known_at"]["not_yet_received"] == []
    assert result["inference_performed"] is False and result["execution_authorized"] is False


def test_the_cli_reads_back_what_was_knowable_at_an_instant_in_a_second_process(tmp_path):
    first, _r = run("collect", "--source", str(EVENTS), "--queue", str(tmp_path / "queue"),
                    "--source-name", "regression-bank", "--received-at", RECEIVED)
    assert first.returncode == 0, first.stderr
    done, result = run("known-at", "--queue", str(tmp_path / "queue"), "--as-of", "2026-09-24T11:50:29Z")
    assert done.returncode == 0, done.stderr
    assert result["known"] == [] and len(result["not_yet_received"]) == 12
    assert result["execution_authorized"] is False


def test_the_cli_refuses_the_late_and_clockless_fixtures_by_name_and_queues_nothing(tmp_path):
    done, result = run("collect", "--source", str(COLLECTION), "--queue", str(tmp_path / "queue"),
                       "--source-name", "regression-bank", "--max-age-seconds", "900", "--received-at", RECEIVED)
    assert done.returncode == 0, done.stderr
    assert result["collection"]["accepted"] == 0 and result["collection"]["refused"] == 3
    assert sorted(result["collection"]["refusals_by_name"]) == ["AGE_NOT_MEASURABLE_WITHOUT_PUBLICATION_CLOCK",
                                                               "DECLARED_SOURCE_MISMATCH", "STALE_AT_COLLECTION"]
    assert result["queue"]["entries"] == 0


def test_the_cli_refuses_an_age_policy_that_is_not_seconds(tmp_path):
    done, result = run("collect", "--source", str(EVENTS), "--queue", str(tmp_path / "queue"),
                       "--max-age-seconds", "fifteen-minutes")
    assert done.returncode == 2
    assert result["status"] == "REFUSED" and "INVALID_AGE_POLICY" in result["reason"]
