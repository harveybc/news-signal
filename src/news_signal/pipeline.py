"""The join: a queue walked by the installed classifier, and an acknowledgement that can only name what was persisted.

Everything this needs already existed and was tested apart. The collector holds an item until someone finishes with it,
`classify_event` drives the installed provider registry, and the shadow store keeps a result under an identity that
separates WHICH NEWS it is about from WHICH EVALUATION produced it. Nothing walked the one with the other, and the gap
between them is exactly where a batch consumer loses items: an entry marked done with a digest no reader can find, a result
attached to the neighbouring entry, an item classified twice because a crash landed between the two writes, or an item
closed forever because the checkpoint happened to be missing for ten minutes.

Four rules, each of them the shape of a failure that would otherwise be silent.

**An acknowledgement names the record that is on disk.** `drain` accepts no digest from anyone. It persists the receipt,
reads the record BACK out of the store under the identity that receipt derives, and acknowledges with the digest the store
itself recorded for it. There is no path through this module that acknowledges an entry with any other string, so "this
item is finished" and "this result exists" cannot come apart. An acknowledgement carrying a digest nobody can resolve is
worse than an unfinished item: the item is gone from the queue and there is nothing to show for it.

**A result belongs to the entry it is about.** Before acknowledging, the record's source identity -- source, event_id and
the digest of the bytes -- is compared with the entry's. A batch loop that reuses a variable, or a revision processed next
to the text it revises, otherwise closes one item with its neighbour's answer, and both look finished afterwards.

**A crash is not a decision.** Persisting and acknowledging are two writes and a process can die between them. The next
drain therefore asks the store first, for THIS entry's bytes under THIS exact evaluation, and acknowledges what it finds
instead of asking the model again. Classifying twice is not merely wasted work: the second answer is a second record with
its own recorded_at, and "how many times was this decided" stops having an answer.

**A refusal about the item is a completed outcome; a refusal about this environment is not.** Wrong language, stale at the
decision clock, outside the task's asset scope -- the item was read and judged unusable, which is a result, and it is
persisted, acknowledged and reported apart from the successes. An absent checkpoint, an uninstalled provider or a task the
runtime does not support says nothing about the news; acknowledging it would close the item forever on the strength of a
missing file, so it leaves the entry FAILED with its reason for the operator to retry once the environment is repaired.
Such a run is also not written to the result store: a result store holds judgements about news, and a later drain must not
find one of these and mistake it for a decision.

A failed entry is NOT returned to PENDING here. `retry()` is the operator's, and a drain that retried on its own would spin
forever on a deterministic failure while reporting activity. Nothing in this module classifies, scores or trades.
"""

from datetime import datetime, timezone

from .application import classify_event, discover
from .core import Refusal
from .provider import PROVIDER_NAME
from .question import question_identity
from .shadow import record_identity
from .tasks import questions as task_questions, task_digest

DRAIN_SCHEMA = "news_drain.v1"

#: what happened to one entry in this drain; every count in the report is these values tallied, never a separate counter
CLASSIFIED, REFUSED, RECOVERED, FAILED = "CLASSIFIED", "REFUSED", "RECOVERED", "FAILED"

#: the runtime statuses that are a judgement ABOUT THE ITEM. Everything else -- MODEL_NOT_FITTED, UNSUPPORTED_TASK,
#: PROVIDER_ERROR, RESOURCE_EXCEEDED -- is a statement about this process, and closing an item on one of those would
#: record an environment outage as a decision about the news.
ITEM_REFUSAL_STATUSES = ("INVALID_INPUT",)


def _now():
    return datetime.now(timezone.utc).isoformat()


def planned_evaluation(registry, *, task_id, as_of, max_age_seconds, question_spec=None):
    """The part of the evaluation identity that is fixed BEFORE anything is loaded or encoded.

    It is what lets a drain recognise its own interrupted work without asking the model again: the task and the exact
    question, the provider, the sealed state the provider declares it holds, the decision clock and the age policy. The
    model digest is deliberately absent -- it is only known after a load, and the sealed manifest digest is the identity a
    checkpoint is admitted under, so that is what is compared here."""
    caps = (registry.capabilities(PROVIDER_NAME) or {}) if registry is not None else {}
    known = caps.get("known_states") or []
    state_digest = known[0].split(":", 1)[-1] if known else None
    return {"task_id": task_id,
            "task_sha256": task_digest(task_id, question_spec),
            # the order-sensitive identity of the exact words, not the name of the task
            "question_sha256": question_identity(task_questions(task_id, question_spec)),
            "provider_ref": PROVIDER_NAME,
            "state_digest": state_digest,
            "as_of": as_of,
            "max_age_seconds": max_age_seconds,
            # a receipt carries no attempt counter, so every attempt at one evaluation is the same evaluation and lands on
            # the same record. A retry after a crash must not invent a second decision.
            "attempt": 0}


def existing_result(store, entry, plan):
    """The result this exact evaluation already produced for this exact item, or None.

    Matching is on the SOURCE identity the store keeps (source, event_id, input digest) plus the planned evaluation: a
    revision of the same event has different bytes and is a different item, and the same bytes under another clock or
    another question are another result. A record that no longer validates is not returned: acknowledging an entry with a
    corrupted record's digest would present the corruption as this item's answer."""
    for record in store.records_for(entry.get("event_id")):
        if record.get("integrity") != "OK":
            continue
        source = record.get("source_identity") or {}
        if (source.get("source"), source.get("event_id"), source.get("input_sha256")) != (
                entry.get("source"), entry.get("event_id"), entry.get("input_sha256")):
            continue
        evaluation = record.get("evaluation_identity") or {}
        if all(evaluation.get(field) == value for field, value in plan.items()):
            return record
    return None


def acknowledge_result(queue, entry, record):
    """Acknowledge one entry with the digest of the record that is on disk for it. The only acknowledgement in this module.

    The digest is never an argument: it is read off the record, and the record is checked to be about THIS entry first.
    Without that check a loop that answered item B and acknowledged item A would leave both looking finished, with A's
    receipt pointing at news A never read."""
    if not isinstance(record, dict) or record.get("integrity") != "OK" or not record.get("record_sha256"):
        raise Refusal("UNACKNOWLEDGEABLE_RESULT: the record for this entry could not be read back and validated")
    source = record.get("source_identity") or {}
    mine = (entry.get("source"), entry.get("event_id"), entry.get("input_sha256"))
    theirs = (source.get("source"), source.get("event_id"), source.get("input_sha256"))
    if theirs != mine:
        raise Refusal(f"MISBOUND_ACKNOWLEDGEMENT: record {record['record_sha256'][:12]} is about {theirs}, and this entry "
                      f"is {mine}; a valid answer to another item is not this item's result")
    return queue.acknowledge(entry["entry_id"], receipt_sha256=record["record_sha256"])


def _persist(store, receipt):
    """Write the result and read it back under the identity the receipt derives.

    The read-back is the point: `put` reports what it believes it wrote, and the acknowledgement must rest on what a reader
    can actually find. A record that comes back unreadable or misplaced fails the entry instead of closing it."""
    record, disposition = store.put(receipt)
    persisted = store.find_record(record_identity(receipt))
    return persisted, disposition, record


def drain(queue, store, *, task_id, registry=None, as_of=None, limit=None, max_age_seconds=900, question_spec=None,
          report=None):
    """Take PENDING entries, classify each through the installed provider, persist the result and acknowledge the entry.

    Returns what happened per entry and the counts of those outcomes. Nothing here is estimated: every number in the report
    is the length of a list of entries this drain actually touched."""
    # the task is validated once, before an item is touched. An unknown task or a question that does not hash to its own
    # identity is the operator's mistake, and spending every item's attempt counter on it would bury the real reason.
    task_questions(task_id, question_spec)
    if registry is None:
        registry, report = discover()
    clock_mode = "REPLAY" if as_of else "WALL_CLOCK"
    as_of = as_of or _now()
    plan = planned_evaluation(registry, task_id=task_id, as_of=as_of, max_age_seconds=max_age_seconds,
                              question_spec=question_spec)

    waiting = queue.pending()
    if limit is not None:
        if not isinstance(limit, int) or limit < 0:
            raise Refusal("INVALID_DRAIN_LIMIT: a limit is a non-negative count of entries")
        taken, left = waiting[:limit], waiting[limit:]
    else:
        taken, left = waiting, []

    items = []
    for entry in taken:
        items.append(_one(queue, store, entry, plan, task_id=task_id, as_of=as_of,
                          max_age_seconds=max_age_seconds, registry=registry, discovery=report,
                          question_spec=question_spec))

    def counted(outcome):
        return sum(1 for item in items if item["outcome"] == outcome)

    acknowledged = [item for item in items if item["acknowledged_with"] is not None]
    return {"schema": DRAIN_SCHEMA,
            "task_id": task_id, "as_of": as_of, "clock_mode": clock_mode,
            "max_age_seconds": max_age_seconds,
            "evaluation": plan,
            "pending_before": len(waiting),
            "processed": len(items),
            "classified": counted(CLASSIFIED),
            "refused": counted(REFUSED),
            "recovered": counted(RECOVERED),
            "failed": counted(FAILED),
            "acknowledged": len(acknowledged),
            # entries still waiting because the caller asked for fewer than were pending: they were not examined, and
            # counting them as anything else would report work that did not happen
            "skipped": len(left),
            "skipped_entry_ids": [e["entry_id"] for e in left],
            "items": items,
            "queue": queue.report(),
            # a drain answers questions about news; it authorizes nothing and calls no broker
            "execution_authorized": False, "broker_calls": 0,
            "inference_performed": any(item["outcome"] in (CLASSIFIED, REFUSED) for item in items)}


def _one(queue, store, entry, plan, *, task_id, as_of, max_age_seconds, registry, discovery, question_spec):
    """One entry, start to finish. Every return path leaves the entry ACKNOWLEDGED with a digest that resolves, or FAILED
    with a reason -- never dropped, and never acknowledged with nothing behind it."""
    row = {"entry_id": entry["entry_id"], "event_id": entry.get("event_id"), "source": entry.get("source"),
           "input_sha256": entry.get("input_sha256"), "attempts": entry.get("attempts", 0),
           "outcome": None, "status": None, "store_disposition": None, "acknowledged_with": None, "why": None}

    already = existing_result(store, entry, plan)
    if already is not None:
        # a crash between the two writes, found on the next pass. The result exists for this exact evaluation, so asking
        # the model again would produce a second record of one decision; the entry is finished with the record we have.
        acknowledge_result(queue, entry, already)
        row.update(outcome=RECOVERED, status=already.get("status"), store_disposition="ALREADY_PERSISTED",
                   acknowledged_with=already["record_sha256"],
                   why="the result for this evaluation was already on disk; it was acknowledged, not recomputed")
        return row

    try:
        outcome = classify_event(entry["event"], task_id=task_id, as_of=as_of, max_age_seconds=max_age_seconds,
                                 registry=registry, report=discovery, question_spec=question_spec)
    except Exception as exc:                                     # noqa: BLE001 - any failure here is the item's, not a crash
        return _fail(queue, row, f"{exc.__class__.__name__}: {exc}")

    receipt = outcome.get("receipt")
    if receipt is None:
        # the provider is not installed in this environment. Nothing was asked and nothing may be closed.
        return _fail(queue, row, f"NO_RECEIPT: {outcome.get('why') or outcome.get('status')}")
    row["status"] = receipt.get("status")
    if receipt["status"] not in ("SHADOW_ONLY", "REFUSED_WITH_INPUT"):
        return _fail(queue, row, f"UNEXPECTED_RECEIPT_STATUS: {receipt['status']}")
    if receipt["status"] == "REFUSED_WITH_INPUT" and receipt.get("m5phet_status") not in ITEM_REFUSAL_STATUSES:
        return _fail(queue, row, f"ENVIRONMENT_REFUSAL: {receipt.get('m5phet_status')}: {receipt.get('why')}")

    persisted, disposition, _record = _persist(store, receipt)
    row["store_disposition"] = disposition
    if persisted is None or persisted.get("integrity") != "OK":
        problems = "; ".join((persisted or {}).get("integrity_problems") or ["the record could not be read back"])
        return _fail(queue, row, f"RESULT_NOT_READABLE_AFTER_WRITE: {problems}")
    try:
        acknowledge_result(queue, entry, persisted)
    except Refusal as exc:
        return _fail(queue, row, str(exc))
    row["acknowledged_with"] = persisted["record_sha256"]
    row["outcome"] = CLASSIFIED if receipt["status"] == "SHADOW_ONLY" else REFUSED
    if row["outcome"] == REFUSED:
        # the reasons the provider gave, per question. A refusal reported only as "refused" cannot be acted on.
        row["why"] = "; ".join(sorted((receipt.get("refusal_reasons") or {}).values())) or receipt.get("why")
    return row


def _fail(queue, row, reason):
    """The entry stays in the queue, FAILED, with its reason and its attempt count. It is not acknowledged, not deleted and
    not retried here: a failure that disappears is indistinguishable from a decision nobody made."""
    failed = queue.fail(row["entry_id"], reason=reason)
    row.update(outcome=FAILED, why=reason, attempts=failed.get("attempts", row["attempts"]))
    return row
