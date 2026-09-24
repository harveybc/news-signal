"""The collector: a durable queue between arriving news and the classifier.

A feed entitlement is not needed to build this, and waiting for one would have been the wrong order of work. What a collector
owes its consumer is the same whether the items come from a licensed wire or a directory of recorded files: every item is
accepted once, kept until it is acknowledged, replayable after a crash, and never silently merged with a different item that
happens to share an identifier.

Three clocks, kept apart. `published_at` is when the source says it published. `received_at` is when THIS system first held
the bytes -- the collector stamps it and nothing downstream may move it. `as_of` is the moment a decision is being made.
A collector that let `received_at` be supplied by the item would let a source backdate its own availability.

A revision is a new item under the same `event_id` with different bytes. It is queued as its own entry, linked to what it
revises, and it does not remove the earlier one: whether a decision made on the earlier text should be revisited is the
consumer's judgement, and erasing the earlier text would take that judgement away.

Nothing here classifies, scores or trades.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from .core import Refusal, canonical, digest, load_json, timestamp

QUEUE_SCHEMA = "news_queue_entry.v1"

#: an entry is in exactly one of these
PENDING, ACKNOWLEDGED, FAILED = "PENDING", "ACKNOWLEDGED", "FAILED"

NEWS_FIELDS = {"schema", "event_id", "source", "asset", "language", "published_at", "received_at", "headline", "body"}


def _now():
    return datetime.now(timezone.utc).isoformat()


class DurableQueue:
    """A directory of entries. Every state change is a whole-file replace, so a crash leaves a valid previous state."""

    def __init__(self, directory):
        self.root = Path(directory)

    def _resolved_root(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root.resolve(strict=True)

    def _path(self, entry_id):
        if not isinstance(entry_id, str) or len(entry_id) != 64 or any(c not in "0123456789abcdef" for c in entry_id):
            raise Refusal("INVALID_QUEUE_KEY: an entry key is a digest this queue computed, never a supplied name")
        root = self._resolved_root()
        path = root / entry_id[:2] / f"{entry_id}.json"
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        if probe.resolve() != root and root not in probe.resolve().parents:
            raise Refusal("QUEUE_ESCAPE_REFUSED")
        return path

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

    def _read(self, path):
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            return {"entry_id": path.stem, "state": FAILED, "unreadable": True,
                    "why": f"UNREADABLE_QUEUE_ENTRY: {path.name}"}
        recomputed = digest({k: v for k, v in entry.items() if k != "entry_sha256"})
        entry["integrity"] = "OK" if entry.get("entry_sha256") == recomputed else "FAILED"
        return entry

    def entries(self, state=None):
        root = self.root
        if not root.is_dir():
            return []
        rows = [self._read(p) for p in sorted(root.rglob("*.json")) if not p.is_symlink()]
        rows = [r for r in rows if state is None or r.get("state") == state]
        return sorted(rows, key=lambda r: (r.get("received_at") or "", r.get("entry_id") or ""))

    def get(self, entry_id):
        path = self._path(entry_id)
        return self._read(path) if path.is_file() else None

    def offer(self, event, *, received_at=None, source_reference=None):
        """Accept one item. Returns (entry, disposition) with disposition ACCEPTED, DUPLICATE or REVISION.

        `received_at` is stamped HERE. An item that carries its own is refused, because a source that can set the moment we
        held it can make a late item look timely."""
        if not isinstance(event, dict) or set(event) - NEWS_FIELDS:
            raise Refusal("NEWS_FIELDS_MISMATCH: the collector accepts the declared news record and nothing else")
        if event.get("received_at"):
            raise Refusal("RECEIPT_CLOCK_IS_NOT_THE_SOURCES: the collector stamps received_at; an item may not supply it")
        stamped = dict(event)
        stamped["received_at"] = received_at or _now()
        timestamp(stamped["received_at"])                       # an unreadable receipt clock refuses here, not downstream
        published = timestamp(stamped["published_at"])
        if published > timestamp(stamped["received_at"]):
            raise Refusal("RECEIVED_BEFORE_PUBLISHED: an item cannot have arrived before it existed")
        body_digest = digest({k: stamped[k] for k in sorted(NEWS_FIELDS - {"received_at"})})
        entry_id = digest({"schema": QUEUE_SCHEMA, "event_id": stamped["event_id"], "bytes": body_digest})
        siblings = [e for e in self.entries() if e.get("event_id") == stamped["event_id"]]
        existing = self.get(entry_id)
        if existing is not None and existing.get("integrity") == "OK":
            return existing, "DUPLICATE"
        revises = [e["entry_id"] for e in siblings if e.get("content_sha256") != body_digest]
        entry = {"schema": QUEUE_SCHEMA, "entry_id": entry_id, "state": PENDING,
                 "event": stamped, "event_id": stamped["event_id"], "source": stamped["source"],
                 "content_sha256": body_digest, "input_sha256": digest(stamped),
                 "published_at": stamped["published_at"], "received_at": stamped["received_at"],
                 "source_reference": source_reference, "revises": revises,
                 "revision_index": len({e.get("content_sha256") for e in siblings}),
                 "attempts": 0, "history": [{"at": _now(), "state": PENDING}]}
        entry["entry_sha256"] = digest({k: v for k, v in entry.items() if k != "entry_sha256"})
        if not self._write(self._path(entry_id), entry, exclusive=True):
            raced = self.get(entry_id)
            if raced is not None and raced.get("integrity") == "OK":
                return raced, "DUPLICATE"
        return entry, ("REVISION" if revises else "ACCEPTED")

    def _transition(self, entry_id, state, **fields):
        entry = self.get(entry_id)
        if entry is None:
            raise Refusal(f"UNKNOWN_QUEUE_ENTRY: {entry_id}")
        updated = {k: v for k, v in entry.items() if k not in ("integrity", "entry_sha256")}
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

    def report(self):
        rows = self.entries()
        by_state = {}
        for entry in rows:
            by_state[entry.get("state")] = by_state.get(entry.get("state"), 0) + 1
        return {"schema": "news_queue_report.v1", "entries": len(rows), "by_state": by_state,
                "events": len({e.get("event_id") for e in rows}),
                "revisions": sum(1 for e in rows if e.get("revises")),
                "integrity_failures": [e.get("entry_id") for e in rows if e.get("integrity") != "OK"],
                "unfinished_writes": len(list(self.root.rglob("*.tmp"))) if self.root.is_dir() else 0,
                "inference_performed": False, "execution_authorized": False}


class RecordedDirectorySource:
    """A source of recorded news files. It is a real source with a real boundary; it is not a live feed and says so.

    A live wire replaces this class and nothing else: the queue, the receipt clock and the revision rules are the same."""

    kind = "RECORDED_FILES_NOT_A_LIVE_FEED"

    def __init__(self, directory):
        self.directory = Path(directory)

    def items(self):
        for path in sorted(self.directory.glob("*.json")):
            event = load_json(path)
            event.pop("received_at", None)                       # the collector stamps it; a file cannot claim it
            yield event, path.name


def collect(source, queue, *, received_at=None):
    """Drain a source into the queue. Returns what happened per item; nothing is classified here."""
    results = []
    for event, reference in source.items():
        try:
            entry, disposition = queue.offer(event, received_at=received_at, source_reference=reference)
            results.append({"reference": reference, "entry_id": entry["entry_id"], "disposition": disposition,
                            "event_id": entry["event_id"], "revises": entry.get("revises")})
        except Refusal as exc:
            results.append({"reference": reference, "disposition": "REFUSED", "why": str(exc)})
    return {"schema": "news_collection.v1", "source_kind": source.kind, "items": results,
            "accepted": sum(1 for r in results if r["disposition"] == "ACCEPTED"),
            "revisions": sum(1 for r in results if r["disposition"] == "REVISION"),
            "duplicates": sum(1 for r in results if r["disposition"] == "DUPLICATE"),
            "refused": sum(1 for r in results if r["disposition"] == "REFUSED"),
            "live_feed": False}
