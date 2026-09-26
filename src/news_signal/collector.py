"""The collector: a durable queue between arriving news and the classifier.

A feed entitlement is not needed to build this, and waiting for one would have been the wrong order of work. What a collector
owes its consumer is the same whether the items come from a licensed wire or a directory of recorded files: every item is
accepted once, kept until it is acknowledged, replayable after a crash, and never silently merged with a different item that
happens to share an identifier.

Three clocks, kept apart. `published_at` is when the source says it published -- its claim, which we did not witness.
`received_at` is when THIS system first held the bytes -- the collector stamps it and nothing downstream may move it.
`as_of` is the moment a decision is being made. A collector that let `received_at` be supplied by the item would let a
source backdate its own availability.

A source that declares no publication clock is recorded as declaring none, by name: `publication_clock` is
`NOT_PROVIDED_BY_SOURCE` and `published_at` stays null. Copying the receipt clock into it is the one error this whole
discipline exists to prevent -- our reading bounds the publication instant from ABOVE and never from below, so a
substituted receipt would place a story inside or outside an event window by guesswork.

`known_at(T)` is the reader that makes those clocks worth keeping: the entries whose receipt clock is at or before T, and
nothing else. An item that arrived after T is absent from it, not ranked lower, because the question is what a decision
made at T could have used. It only answers truthfully because no entry is ever rewritten: a revision is a new entry.

Staleness is a DECLARED policy, never an assumed one. With no maximum age declared the collector refuses nothing for age
and says so (`NOT_DECLARED`); with one declared, an item whose publication-to-receipt lag exceeds it is refused by name,
and an item that carries no publication clock at all is refused too rather than admitted as fresh -- an age that cannot
be measured is not an age that passed.

A source that hands us nothing is `AWAITED`. It is not an empty batch, not zero relevant stories and not neutral
sentiment: those are claims about the news, and this boundary has no news to make them from. A source that produced
before and is silent now (`SILENT_SINCE`, with the silence measured) is a different fact from one that never produced
(`NEVER_PRODUCED`), and reporting them as one number would hide an outage behind a quiet market.

A revision is a new item from the SAME source, under the same `event_id`, with different bytes. It is queued as its own
entry, linked to what it revises, and it does not remove the earlier one: whether a decision made on the earlier text
should be revisited is the consumer's judgement, and erasing the earlier text would take that judgement away. An
`event_id` is issued by a source and means nothing outside it, so two wires reusing one identifier are two events.

Integrity is a claim this queue makes about the disk, and three rules keep that claim honest. An entry is validated
against the KEY IT LIES UNDER as well as against its own digest, because a whole entry for another item answers the
self-hash question perfectly. An entry that fails validation may not change state: re-signing it would recompute
`entry_sha256` over the corrupted content and launder the corruption into something dispatchable, so the corrupted bytes
are quarantined and re-offered instead. And an acceptance is only ever reported for a write that happened -- a caller
holding a digest that no reader can find is worse than a refusal.

Nothing here classifies, scores or trades.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .core import Refusal, canonical, digest, load_json, timestamp
from .provider import DEFAULT_MAX_AGE_SECONDS

QUEUE_SCHEMA = "news_queue_entry.v1"

#: what the source said about its own publication clock. These are two different facts, not two values of one field.
SOURCE_DECLARED = "SOURCE_DECLARED"
NOT_PROVIDED_BY_SOURCE = "NOT_PROVIDED_BY_SOURCE"

PUBLICATION_BOUND_READING = (
    "UPPER_BOUND_ONLY: this item was public no later than `received_at`, which is our clock and not the source's. It may "
    "have been public for hours before. A receipt clock bounds a publication instant from above and never from below, and "
    "substituting one for the other would invent a publication time nobody published.")

SOURCE_DECLARED_READING = (
    "SOURCE_CLAIM: `published_at` is what the source says, not something this system witnessed. `received_at` is ours.")

#: the number the decision boundary already uses. It is re-exported, not applied: a caller that wants the same maximum age
#: at collection DECLARES this constant, so the policy appears in the record instead of being inherited silently.
DECLARED_DEFAULT_MAX_AGE_SECONDS = DEFAULT_MAX_AGE_SECONDS

#: whether a source is handing us items right now, and -- when it is not -- which kind of silence this is
PRODUCING, AWAITED = "PRODUCING", "AWAITED"
NEVER_PRODUCED, SILENT_SINCE = "NEVER_PRODUCED", "SILENT_SINCE"
SOURCE_NOT_DECLARED = "SOURCE_NOT_DECLARED"

#: an entry is in exactly one of these
PENDING, ACKNOWLEDGED, FAILED = "PENDING", "ACKNOWLEDGED", "FAILED"

NEWS_FIELDS = {"schema", "event_id", "source", "asset", "language", "published_at", "received_at", "headline", "body"}

#: corrupted entries are moved here rather than deleted, and this directory is not part of the queue
QUARANTINE = "quarantine"

#: computed by a reader about an entry; they are never written back, so a state change cannot fold them into the signature
DERIVED_FIELDS = ("integrity", "integrity_problems", "derived_entry_id", "found_under", "entry_sha256",
                  "unreadable", "why")


def _now():
    return datetime.now(timezone.utc).isoformat()


def entry_identity(entry):
    """The queue key, DERIVED from what the entry is about: the event it reports and the digest of its bytes -- and that
    digest covers the issuing source, so two wires reusing one `event_id` never collide on a key.

    It is recomputed on every read and then compared. An entry's own `entry_id` field is what it CLAIMS to be; a file that
    was copied onto another item's path claims it just as convincingly as the real one."""
    return digest({"schema": QUEUE_SCHEMA, "event_id": entry.get("event_id"), "bytes": entry.get("content_sha256")})


class DurableQueue:
    """A directory of entries. Every state change is a whole-file replace, so a crash leaves a valid previous state."""

    def __init__(self, directory, max_age_seconds=None):
        """`max_age_seconds` is the DECLARED maximum publication-to-receipt age. `None` declares no policy, and then
        nothing is ever refused for age: a default applied silently would report a freshness guarantee nobody asked for."""
        self.root = Path(directory)
        self.max_age_seconds = max_age_seconds

    def _age_policy(self, override):
        policy = self.max_age_seconds if override is None else override
        if policy is None:
            return None
        if type(policy) is not int or not 1 <= policy <= 86400:
            raise Refusal("INVALID_AGE_POLICY: a declared maximum age is a whole number of seconds from 1 to 86400")
        return policy

    def _resolved_root(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root.resolve(strict=True)

    def _path(self, entry_id):
        if not isinstance(entry_id, str) or len(entry_id) != 64 or any(c not in "0123456789abcdef" for c in entry_id):
            raise Refusal("INVALID_QUEUE_KEY: an entry key is a digest this queue computed, never a supplied name")
        root = self._resolved_root()
        path = root / entry_id[:2] / f"{entry_id}.json"
        self._contained(root, path)
        return path

    @staticmethod
    def _contained(root, path):
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        if probe.resolve() != root and root not in probe.resolve().parents:
            raise Refusal("QUEUE_ESCAPE_REFUSED")

    def _write(self, path, entry, *, exclusive):
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".writing-", suffix=".json.tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False, indent=1, sort_keys=True))
                stream.flush()
                os.fsync(stream.fileno())
            if exclusive:
                try:
                    os.link(temporary, path)
                    return True
                except FileExistsError:
                    return False
            os.replace(temporary, path)
            return True
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _read(self, path, *, expected_key=None):
        """Three questions, not one. Does the entry hash to its own digest? Does its own content derive the key it carries?
        And is that key the one it was found under?

        The last question is the one that was missing. A whole, self-consistent entry for another item, copied onto this
        item's path, answered the first question perfectly -- and was handed to the consumer as this item."""
        if expected_key is None and path.parent.name != QUARANTINE:
            expected_key = path.stem
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            return {"entry_id": expected_key or path.stem, "state": FAILED, "unreadable": True,
                    "integrity": "FAILED", "integrity_problems": [f"UNREADABLE_QUEUE_ENTRY: {path.name}"],
                    "why": f"UNREADABLE_QUEUE_ENTRY: {path.name}"}
        if not isinstance(entry, dict):
            return {"entry_id": expected_key or path.stem, "state": FAILED, "unreadable": True,
                    "integrity": "FAILED", "integrity_problems": ["QUEUE_ENTRY_NOT_A_MAPPING"],
                    "why": f"QUEUE_ENTRY_NOT_A_MAPPING: {path.name}"}
        entry = dict(entry)
        problems = []
        recomputed = digest({k: v for k, v in entry.items() if k != "entry_sha256"})
        if entry.get("entry_sha256") != recomputed:
            problems.append("CONTENT_DIGEST_MISMATCH: the entry does not hash to its own recorded digest")
        derived = entry_identity(entry)
        if entry.get("entry_id") != derived:
            problems.append(f"IDENTITY_MISMATCH: the entry's own fields derive key {derived[:12]} and it carries "
                            f"{str(entry.get('entry_id'))[:12]}")
        if expected_key is not None and derived != expected_key:
            problems.append(f"MISPLACED_QUEUE_ENTRY: this entry belongs under key {derived[:12]} and was found under "
                            f"{expected_key[:12]}; an intact entry for another item is not this item")
        entry["integrity"] = "OK" if not problems else "FAILED"
        entry["integrity_problems"] = problems
        entry["derived_entry_id"] = derived
        entry["found_under"] = expected_key
        return entry

    def entries(self, state=None):
        root = self.root
        if not root.is_dir():
            return []
        rows = [self._read(p) for p in sorted(root.rglob("*.json"))
                if not p.is_symlink() and p.parent.name != QUARANTINE]
        rows = [r for r in rows if state is None or r.get("state") == state]
        return sorted(rows, key=lambda r: (r.get("received_at") or "", r.get("entry_id") or ""))

    def get(self, entry_id):
        path = self._path(entry_id)
        return self._read(path) if path.is_file() else None

    def offer(self, event, *, received_at=None, source_reference=None, max_age_seconds=None):
        """Accept one item. Returns (entry, disposition) with disposition ACCEPTED, DUPLICATE or REVISION.

        `received_at` is stamped HERE. An item that carries its own is refused, because a source that can set the moment we
        held it can make a late item look timely.

        `max_age_seconds` overrides the queue's declared policy for this item only; `None` in both places declares none."""
        if not isinstance(event, dict) or set(event) - NEWS_FIELDS:
            raise Refusal("NEWS_FIELDS_MISMATCH: the collector accepts the declared news record and nothing else")
        missing = sorted(NEWS_FIELDS - {"received_at", "published_at"} - set(event))
        if missing:
            # named here rather than raised as a KeyError deep in the digest: a half-record is the source's problem and
            # the operator needs to know which half is absent.
            raise Refusal(f"NEWS_FIELDS_MISSING: {', '.join(missing)}")
        if event.get("received_at"):
            raise Refusal("RECEIPT_CLOCK_IS_NOT_THE_SOURCES: the collector stamps received_at; an item may not supply it")
        policy = self._age_policy(max_age_seconds)
        stamped = dict(event)
        stamped["received_at"] = received_at or _now()
        received = timestamp(stamped["received_at"])            # an unreadable receipt clock refuses here, not downstream
        claimed = stamped.get("published_at")
        if claimed is None or (isinstance(claimed, str) and not claimed.strip()):
            # The absence is the fact, and it is recorded as the absence. `published_at` stays null; the receipt clock is
            # NOT copied into it, here or anywhere downstream.
            stamped["published_at"] = None
            publication_clock, lag, bound = NOT_PROVIDED_BY_SOURCE, None, PUBLICATION_BOUND_READING
            if policy is not None:
                raise Refusal(f"AGE_NOT_MEASURABLE_WITHOUT_PUBLICATION_CLOCK: a maximum age of {policy}s is declared and "
                              "this item carries no publication clock; admitting it would count an age nobody measured "
                              "as an age that passed")
        else:
            try:
                published = timestamp(claimed)
            except Refusal as exc:
                raise Refusal(f"PUBLICATION_CLOCK_UNREADABLE: published_at={claimed!r} names no instant") from exc
            if published > received:
                raise Refusal("RECEIVED_BEFORE_PUBLISHED: an item cannot have arrived before it existed")
            publication_clock, bound = SOURCE_DECLARED, SOURCE_DECLARED_READING
            lag = (received - published).total_seconds()
            if policy is not None and lag > policy:
                raise Refusal(f"STALE_AT_COLLECTION: this item reached us {lag:.0f}s after the publication clock it "
                              f"declares and the maximum age declared here is {policy}s")
        body_digest = digest({k: stamped.get(k) for k in sorted(NEWS_FIELDS - {"received_at"})})
        entry_id = entry_identity({"event_id": stamped["event_id"], "content_sha256": body_digest})
        # an `event_id` is issued BY a source and is only unique within it. Linking across sources would invent a
        # correction nobody published, and would report two unrelated stories as one event.
        siblings = [e for e in self.entries()
                    if e.get("event_id") == stamped["event_id"] and e.get("source") == stamped["source"]]
        existing = self.get(entry_id)
        if existing is not None and existing.get("integrity") == "OK":
            return existing, "DUPLICATE"
        revises = [e["entry_id"] for e in siblings if e.get("content_sha256") != body_digest]
        entry = {"schema": QUEUE_SCHEMA, "entry_id": entry_id, "state": PENDING,
                 "event": stamped, "event_id": stamped["event_id"], "source": stamped["source"],
                 "content_sha256": body_digest, "input_sha256": digest(stamped),
                 "published_at": stamped["published_at"], "received_at": stamped["received_at"],
                 "publication_clock": publication_clock, "publication_bound": bound,
                 "receipt_lag_seconds": lag,
                 "max_age_seconds": policy, "staleness_policy": "DECLARED" if policy is not None else "NOT_DECLARED",
                 "source_reference": source_reference, "revises": revises,
                 "revision_index": len({e.get("content_sha256") for e in siblings}),
                 "attempts": 0, "history": [{"at": _now(), "state": PENDING}]}
        entry["entry_sha256"] = digest({k: v for k, v in entry.items() if k != "entry_sha256"})
        path = self._path(entry_id)
        if not self._write(path, entry, exclusive=True):
            raced = self.get(entry_id)
            if raced is not None and raced.get("integrity") == "OK":
                return raced, "DUPLICATE"
            # the key is taken by something this queue cannot vouch for. It is moved aside -- it is the evidence of a
            # corruption, not something to overwrite -- and the write is tried again, because reporting ACCEPTED for a
            # write that did not happen hands the caller a digest no reader will ever find and leaves nothing pending.
            if path.is_file():
                entry["quarantined_predecessor"] = self._quarantine(path, entry_id)
                entry["entry_sha256"] = digest({k: v for k, v in entry.items() if k != "entry_sha256"})
            if not self._write(path, entry, exclusive=True):
                settled = self.get(entry_id)
                if settled is not None and settled.get("integrity") == "OK":
                    return settled, "DUPLICATE"
                raise Refusal(f"QUEUE_WRITE_NOT_ACCEPTED: {entry_id} could not be written and nothing was queued")
        return entry, ("REVISION" if revises else "ACCEPTED")

    def _quarantine(self, path, entry_id):
        """A corrupted entry is moved aside, not deleted: deleting it would erase what went wrong.

        The quarantine name carries the BYTES that were quarantined, so a second bad version of the same key cannot
        overwrite the first and every one of them is preserved."""
        root = self._resolved_root()
        try:
            content = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        except OSError:
            content = "unreadable"
        target = root / QUARANTINE / f"{entry_id}.{content}.json"
        self._contained(root, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            path.unlink(missing_ok=True)                        # these exact bytes are already preserved
        else:
            os.replace(path, target)
        return target.relative_to(root).as_posix()

    def _transition(self, entry_id, state, **fields):
        entry = self.get(entry_id)
        if entry is None:
            raise Refusal(f"UNKNOWN_QUEUE_ENTRY: {entry_id}")
        if entry.get("integrity") != "OK":
            # signing here would recompute `entry_sha256` over the corrupted content and make it dispatchable with
            # integrity OK. The state on disk stays the last one this queue could vouch for, and the failure stays in
            # the report where a human can see it; a corruption is repaired by re-offering the item, not by a retry.
            why = "; ".join(entry.get("integrity_problems") or ["integrity FAILED"])
            raise Refusal(f"CORRUPT_QUEUE_ENTRY: {entry_id} may not change state ({why})")
        updated = {k: v for k, v in entry.items() if k not in DERIVED_FIELDS}
        updated.update(fields)
        updated["state"] = state
        updated["history"] = list(entry.get("history") or []) + [{"at": _now(), "state": state, **fields}]
        updated["entry_sha256"] = digest({k: v for k, v in updated.items() if k != "entry_sha256"})
        self._write(self._path(entry_id), updated, exclusive=False)
        return updated

    def acknowledge(self, entry_id, *, receipt_sha256):
        """Only a consumer that names the result it produced may acknowledge an item."""
        if not receipt_sha256:
            raise Refusal("ACKNOWLEDGEMENT_NEEDS_ITS_RESULT: an item is done when something was produced from it")
        return self._transition(entry_id, ACKNOWLEDGED, receipt_sha256=receipt_sha256)

    def fail(self, entry_id, *, reason):
        entry = self.get(entry_id)
        attempts = (entry or {}).get("attempts", 0) + 1
        return self._transition(entry_id, FAILED, reason=str(reason), attempts=attempts)

    def retry(self, entry_id):
        """A failed item returns to PENDING with its history intact: the failure is part of the record, not erased by a retry."""
        return self._transition(entry_id, PENDING)

    def pending(self):
        return [e for e in self.entries(PENDING) if e.get("integrity") == "OK"]

    def quarantined(self):
        """The keys whose corrupted bytes were set aside. They are out of every count and in the report: hiding them
        would turn a detected corruption into a silent one."""
        folder = self.root / QUARANTINE
        if not folder.is_dir():
            return []
        return sorted({path.name.split(".")[0] for path in folder.glob("*.json")})

    def report(self):
        rows = self.entries()
        by_state = {}
        for entry in rows:
            by_state[entry.get("state")] = by_state.get(entry.get("state"), 0) + 1
        return {"schema": "news_queue_report.v1", "entries": len(rows), "by_state": by_state,
                "events": len({(e.get("source"), e.get("event_id")) for e in rows}),
                "revisions": sum(1 for e in rows if e.get("revises")),
                "sources": sorted({e.get("source") for e in rows if e.get("source")}),
                "without_publication_clock": sum(1 for e in rows
                                                 if e.get("publication_clock") == NOT_PROVIDED_BY_SOURCE),
                # the worst publication-to-receipt lag we actually hold, and how many items have none that can be measured
                "max_receipt_lag_seconds": max([e["receipt_lag_seconds"] for e in rows
                                                if isinstance(e.get("receipt_lag_seconds"), (int, float))],
                                               default=None),
                "receipt_lag_unmeasurable": sum(1 for e in rows if e.get("receipt_lag_seconds") is None),
                "staleness_policy": "DECLARED" if self.max_age_seconds is not None else "NOT_DECLARED",
                "max_age_seconds": self.max_age_seconds,
                # the key the bad entry OCCUPIES, not the identity it claims: that key is what an operator must act on
                "integrity_failures": [e.get("found_under") or e.get("entry_id") for e in rows
                                       if e.get("integrity") != "OK"],
                "quarantined": self.quarantined(),
                "unfinished_writes": len(list(self.root.rglob("*.tmp"))) if self.root.is_dir() else 0,
                "inference_performed": False, "execution_authorized": False}


def knowable_entries(queue, as_of, *, source=None):
    """The valid entries whose receipt clock is at or before T.

    An entry received after T is ABSENT, not ranked lower: a decision made at T could not have used it, and a reader that
    merely sorted by recency would hand a backtest a story from its own future. Clocks are compared as instants, never as
    strings: `2026-09-24T12:00:00Z` and `2026-09-24T14:00:00+02:00` are one instant written two ways."""
    cutoff = timestamp(as_of)
    known, later, unreadable = [], [], []
    for entry in queue.entries():
        if entry.get("integrity") != "OK":
            continue
        if source is not None and entry.get("source") != source:
            continue
        try:
            received = timestamp(entry.get("received_at"))
        except Refusal:
            unreadable.append(entry.get("found_under") or entry.get("entry_id"))
            continue
        (known if received <= cutoff else later).append(entry)
    return known, later, unreadable


def known_at(queue, as_of, *, source=None):
    """What was knowable at T, as a report. It answers truthfully only because no entry is ever rewritten: a revision is
    a new entry with its own receipt clock, so discarding what arrived after T leaves exactly the earlier text standing."""
    cutoff = timestamp(as_of)
    known, later, unreadable = knowable_entries(queue, cutoff.isoformat(), source=source)
    latest = {}
    for entry in known:
        key = (entry.get("source"), entry.get("event_id"))
        current = latest.get(key)
        rank = (entry.get("received_at") or "", entry.get("entry_id") or "")
        if current is None or rank > (current.get("received_at") or "", current.get("entry_id") or ""):
            latest[key] = entry
    return {"schema": "news_known_at.v1", "as_of": cutoff.isoformat(), "source": source,
            "known": [e["entry_id"] for e in known],
            # named, not silently dropped: an operator asking "why is this item missing" gets an answer
            "not_yet_received": [e["entry_id"] for e in later],
            "unreadable_receipt_clocks": unreadable,
            "latest_known_text": [{"source": key[0], "event_id": key[1], "entry_id": entry["entry_id"],
                                   "revision_index": entry.get("revision_index"),
                                   "received_at": entry.get("received_at"),
                                   "published_at": entry.get("published_at"),
                                   "publication_clock": entry.get("publication_clock")}
                                  for key, entry in sorted(latest.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or ""))],
            "events_known": len(latest),
            "inference_performed": False, "execution_authorized": False}


def source_observation(queue, source_name, *, as_of=None, produced_now=None):
    """Whether a source is producing, and -- when it is not -- which silence this is.

    `AWAITED` is the whole point. A source that hands us nothing has told us nothing, and the three things a consumer
    might infer from a zero -- an empty batch, no relevant news, neutral sentiment -- are all claims about the news that
    this boundary has no news to support. The two silences are separate facts: a source that produced before and stopped
    is an outage to act on; a source that never produced anything here may simply never have been connected."""
    at = timestamp(as_of) if as_of else datetime.now(timezone.utc)
    if source_name is None:
        # an outage is a statement about a NAMED source. Without a declared name there is nothing to attribute a silence
        # to, and saying "a source is down" would name a source nobody declared.
        return {"schema": "news_source_observation.v1", "source": None, "attribution": SOURCE_NOT_DECLARED,
                "observation": PRODUCING if produced_now else AWAITED, "silence": None,
                "items_held": None, "last_received_at": None, "silence_seconds": None,
                "produced_now": produced_now, "as_of": at.isoformat(),
                "empty_batch_invented": False, "neutral_sentiment_substituted": False}
    held = [e for e in queue.entries() if e.get("source") == source_name and e.get("integrity") == "OK"]
    last = max((e.get("received_at") or "" for e in held), default="") or None
    silence_seconds = None
    if last is not None:
        silence_seconds = (at - timestamp(last)).total_seconds()
    observation = PRODUCING if produced_now else AWAITED
    silence = None if observation == PRODUCING else (SILENT_SINCE if last is not None else NEVER_PRODUCED)
    return {"schema": "news_source_observation.v1", "source": source_name, "attribution": "SOURCE_DECLARED",
            "observation": observation, "silence": silence,
            "items_held": len(held), "last_received_at": last, "silence_seconds": silence_seconds,
            "produced_now": produced_now, "as_of": at.isoformat(),
            # said in the record, not left to a reader's assumption
            "empty_batch_invented": False, "neutral_sentiment_substituted": False}


class RecordedDirectorySource:
    """A source of recorded news files. It is a real source with a real boundary; it is not a live feed and says so.

    A live wire replaces this class and nothing else: the queue, the receipt clock and the revision rules are the same."""

    kind = "RECORDED_FILES_NOT_A_LIVE_FEED"

    def __init__(self, directory, source_name=None):
        """`source_name` declares WHICH source this boundary stands for. It is optional, and its absence costs exactly one
        thing: a silence cannot be attributed, because an outage is a statement about a named source."""
        self.directory = Path(directory)
        self.source_name = source_name

    def items(self):
        for path in sorted(self.directory.glob("*.json")):
            event = load_json(path)
            event.pop("received_at", None)                       # the collector stamps it; a file cannot claim it
            yield event, path.name


def collect(source, queue, *, received_at=None, max_age_seconds=None, as_of=None):
    """Drain a source into the queue. Returns what happened per item; nothing is classified here.

    Every count below is a tally of items this collection actually touched. A source that yielded nothing produces an
    empty `items` list and an `AWAITED` observation -- never a fabricated item and never a neutral reading."""
    declared = getattr(source, "source_name", None)
    results = []
    seen = 0
    for event, reference in source.items():
        seen += 1
        try:
            if declared is not None and isinstance(event, dict) and event.get("source") != declared:
                # this boundary stands for one source. Admitting another's item under its name would attribute a story to
                # a wire that never carried it, and would make the outage report about the wrong source.
                raise Refusal(f"DECLARED_SOURCE_MISMATCH: this boundary collects {declared!r} and the item says "
                              f"{event.get('source')!r}")
            entry, disposition = queue.offer(event, received_at=received_at, source_reference=reference,
                                             max_age_seconds=max_age_seconds)
            results.append({"reference": reference, "entry_id": entry["entry_id"], "disposition": disposition,
                            "event_id": entry["event_id"], "revises": entry.get("revises"),
                            "publication_clock": entry.get("publication_clock"),
                            "receipt_lag_seconds": entry.get("receipt_lag_seconds")})
        except Refusal as exc:
            results.append({"reference": reference, "disposition": "REFUSED", "why": str(exc)})
    refusals = {}
    for row in results:
        if row["disposition"] == "REFUSED":
            name = str(row["why"]).split(":", 1)[0]
            refusals[name] = refusals.get(name, 0) + 1
    lags = [r["receipt_lag_seconds"] for r in results if isinstance(r.get("receipt_lag_seconds"), (int, float))]
    observed = source_observation(queue, declared, as_of=as_of or received_at, produced_now=seen)
    return {"schema": "news_collection.v1", "source_kind": source.kind, "items": results,
            "accepted": sum(1 for r in results if r["disposition"] == "ACCEPTED"),
            "revisions": sum(1 for r in results if r["disposition"] == "REVISION"),
            "duplicates": sum(1 for r in results if r["disposition"] == "DUPLICATE"),
            "refused": sum(1 for r in results if r["disposition"] == "REFUSED"),
            "live_feed": False,
            "source_name": declared,
            "items_offered_by_source": seen,
            "observation": observed["observation"],
            "source_observation": observed,
            # one counter per refusal NAME: "12 refused" cannot be acted on, "12 STALE_AT_COLLECTION" can
            "refusals_by_name": dict(sorted(refusals.items())),
            "max_receipt_lag_seconds": max(lags, default=None),
            "receipt_lag_unmeasurable": sum(1 for r in results
                                            if r.get("publication_clock") == NOT_PROVIDED_BY_SOURCE),
            "staleness_policy": "DECLARED" if queue._age_policy(max_age_seconds) is not None else "NOT_DECLARED",
            "max_age_seconds": queue._age_policy(max_age_seconds),
            "empty_batch_invented": False, "neutral_sentiment_substituted": False}
