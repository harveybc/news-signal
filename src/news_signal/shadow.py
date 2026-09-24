"""The durable shadow store: a classification result that survives the process that produced it.

Two things this module got wrong once, both fixed here by construction rather than by checking for the bad case.

**A name from outside is never a path.** `event_id` arrives from a feed. It was joined onto the store root, so `../escaped`
wrote outside the store while the application reported success. Records are now keyed by an opaque digest the store computes
itself; the external identifier is retained INSIDE the record, where it belongs, and is never part of a filename. Every write
is additionally checked to land under the resolved root, so a symlink planted in the store cannot redirect it either.

**The same news is not the same result.** An answer depends on the question, the model, the decision clock and the staleness
policy as much as on the bytes of the news. Identity is therefore two things kept apart:

* *source identity* -- who reported it, under which `event_id`, and the digest of the record's bytes;
* *evaluation identity* -- the task and its question set, the model and its state, the decision clock and age policy, and
  the attempt.

Only an identical evaluation may be idempotent. A retry after a refusal, another task, another clock: each is its own record
and all of them are retained. A stored record is validated before it is reused, and one that fails its own digest is
quarantined rather than returned or counted.

Nothing here contacts a broker, a venue or a governance service. `execution_authorized` is False in every record it writes.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from .core import Refusal, canonical, digest

#: the fields of a record that identify WHICH NEWS it is about
SOURCE_IDENTITY = ("source", "event_id", "input_sha256")

#: the fields that identify WHICH EVALUATION produced it. Two results are the same result only when all of these agree.
EVALUATION_IDENTITY = ("task_id", "task_sha256", "question_sha256", "provider_ref", "model_sha256", "state_digest",
                       "as_of", "max_age_seconds", "attempt")

RECORD_SCHEMA = "news_shadow_record.v2"


def _evaluation(receipt):
    binding = receipt.get("binding") or {}
    return {"task_id": receipt.get("task_id"),
            "task_sha256": receipt.get("task_sha256"),
            # the order-sensitive identity of the exact question the model was given
            "question_sha256": receipt.get("question_sha256"),
            "provider_ref": receipt.get("provider_ref"),
            "model_sha256": binding.get("model_sha256"),
            "state_digest": binding.get("state_digest"),
            "as_of": receipt.get("as_of"),
            "max_age_seconds": receipt.get("max_age_seconds"),
            "attempt": receipt.get("attempt", 0)}


def _source(receipt):
    return {"source": receipt.get("source"), "event_id": receipt.get("event_id"),
            "input_sha256": receipt.get("input_sha256")}


def record_identity(receipt):
    """The opaque key. It is computed from what the result IS, never from text that arrived from outside, and never from a
    field a copied record brought with it: `record_id` is DERIVED here and then compared, not trusted."""
    return digest({"schema": RECORD_SCHEMA, "source": _source(receipt), "evaluation": _evaluation(receipt),
                   "status": receipt.get("status")})


class ShadowStore:
    def __init__(self, directory):
        self.root = Path(directory)

    # --- containment ------------------------------------------------------------------------------------------------
    def _resolved_root(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root.resolve(strict=True)

    def _path(self, record_id):
        """`<root>/<shard>/<record_id>.json`, where both parts are hex from our own digest and nothing else."""
        if not isinstance(record_id, str) or len(record_id) != 64 or any(c not in "0123456789abcdef" for c in record_id):
            raise Refusal("INVALID_RECORD_KEY: a store key is a digest this store computed, never a supplied name")
        root = self._resolved_root()
        path = root / record_id[:2] / f"{record_id}.json"
        self._contained(root, path)
        return path

    @staticmethod
    def _contained(root, path):
        """The write must land under the root AFTER resolution, so a symlink in the store cannot carry it out."""
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        resolved = probe.resolve()
        if resolved != root and root not in resolved.parents:
            raise Refusal(f"STORE_ESCAPE_REFUSED: {path.name} resolves outside the store root")

    # --- reading ----------------------------------------------------------------------------------------------------
    def _read(self, path, *, expected_key=None):
        """A record that cannot be parsed is reported as a failure of THAT key, not raised past the caller: a torn file in
        the store must not make the whole store unreadable, and must never be mistaken for an absent record."""
        if expected_key is None and path.parent.name != "quarantine":
            expected_key = path.stem
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            return {"record_sha256": None, "record_id": expected_key, "integrity": "FAILED",
                    "integrity_problems": [f"UNREADABLE_SHADOW_RECORD: {path.name}: {exc.__class__.__name__}"],
                    "unreadable": True, "event_id": None, "status": None}
        if not isinstance(record, dict):
            return {"record_sha256": None, "record_id": expected_key, "integrity": "FAILED",
                    "integrity_problems": ["SHADOW_RECORD_NOT_A_MAPPING"], "unreadable": True,
                    "event_id": None, "status": None}
        return self._checked(record, expected_key=expected_key)

    @staticmethod
    def _checked(record, expected_key=None):
        """Three questions, not one. Does the record hash to its own digest? Does it hash to the KEY it was found under?
        And is that key the identity its own fields derive?

        The middle question is the one that was missing. A whole, self-consistent record for another question, copied onto
        this question's path, answered the first question perfectly -- and was served as this question's result."""
        stored = record.get("record_sha256")
        recomputed = digest({k: v for k, v in record.items() if k != "record_sha256"})
        record = dict(record)
        problems = []
        if stored != recomputed:
            problems.append("CONTENT_DIGEST_MISMATCH: the record does not hash to its own recorded digest")
        derived = record_identity(record)
        if record.get("record_id") != derived:
            problems.append(f"IDENTITY_MISMATCH: the record's own fields derive key {derived[:12]} and it carries "
                            f"{str(record.get('record_id'))[:12]}")
        if expected_key is not None and derived != expected_key:
            problems.append(f"MISPLACED_RECORD: this record belongs under key {derived[:12]} and was found under "
                            f"{expected_key[:12]}; a valid answer to another question is not this one's result")
        record["integrity"] = "OK" if not problems else "FAILED"
        record["integrity_problems"] = problems
        record["recomputed_sha256"] = recomputed
        record["derived_record_id"] = derived
        return record

    def all_records(self, *, include_quarantine=False):
        """Every live record. Quarantined ones are kept on disk and stay out of every count unless asked for by name."""
        root = self.root
        if not root.is_dir():
            return []
        out = []
        for path in sorted(root.rglob("*.json")):
            if path.is_symlink():
                continue
            quarantined = path.parent.name == "quarantine"
            if quarantined and not include_quarantine:
                continue
            record = self._read(path)
            record["quarantined"] = quarantined
            out.append(record)
        return sorted(out, key=lambda r: (r.get("recorded_at") or "", r.get("record_sha256") or ""))

    def records_for(self, event_id):
        """Every retained evaluation of one news identity, whatever its task, clock or outcome."""
        return [r for r in self.all_records() if r.get("event_id") == event_id]

    def find_record(self, reference):
        """Look a record up by its store key or by its content digest. Both are ours; neither came from outside."""
        path = self._path(reference)
        if path.is_file():
            return self._read(path)
        for record in self.all_records(include_quarantine=True):
            if record.get("record_sha256") == reference or record.get("record_id") == reference:
                return record
        return None

    def find(self, event_id, input_sha256):
        """Compatibility reader: the successful evaluations of these exact news bytes, newest last."""
        matches = [r for r in self.records_for(event_id)
                   if r.get("input_sha256") == input_sha256 and r.get("integrity") == "OK"]
        return matches[-1] if matches else None

    # --- writing ----------------------------------------------------------------------------------------------------
    def put(self, receipt):
        """Store one result. Returns (record, disposition): STORED, REVISION, REEVALUATION, DUPLICATE or REPLACED_INVALID.

        DUPLICATE is returned only when the identical evaluation is already on disk AND that stored record still validates."""
        record_id = record_identity(receipt)
        path = self._path(record_id)
        prior = self.records_for(receipt.get("event_id"))
        existing = self._read(path, expected_key=record_id) if path.is_file() else None
        if existing is not None and existing.get("integrity") == "OK":
            return existing, "DUPLICATE"
        invalid_existing = existing is not None
        record = dict(receipt)
        record["schema_record"] = RECORD_SCHEMA
        record["record_id"] = record_id
        record["execution_authorized"] = False
        record["source_identity"] = _source(receipt)
        record["evaluation_identity"] = _evaluation(receipt)
        record["receipt_sha256"] = receipt.get("receipt_sha256")
        same_bytes = [p for p in prior if p.get("input_sha256") == receipt.get("input_sha256")]
        # a REVISION supersedes the earlier text of the same news identity. A RE-EVALUATION of the same text supersedes
        # nothing: the earlier answer to a different question, or under a different clock, remains its own true record.
        record["supersedes"] = [p["record_sha256"] for p in prior
                                if p.get("input_sha256") != receipt.get("input_sha256")]
        record["revision_index"] = len({p.get("input_sha256") for p in prior} - {receipt.get("input_sha256")})
        if invalid_existing:
            record["replaces_invalid_record"] = record_id
            record["quarantined_predecessor"] = self._quarantine(path, record_id)
        record["record_sha256"] = digest({k: v for k, v in record.items() if k != "record_sha256"})
        created = self._write(path, record)
        if not created:
            # another writer won the race for this key between our read and our write. Their record is the record: two
            # callers must not walk away holding different digests for one evaluation.
            raced = self._read(path, expected_key=record_id)
            if raced.get("integrity") == "OK":
                return raced, "DUPLICATE"
            self._quarantine(path, record_id)
            self._write(path, record)
        if invalid_existing:
            disposition = "REPLACED_INVALID"
        elif not prior:
            disposition = "STORED"
        elif same_bytes:
            disposition = "REEVALUATION"
        else:
            disposition = "REVISION"
        return record, disposition

    def _quarantine(self, path, record_id):
        """A record that failed a check is moved aside, not deleted: deleting it would erase what went wrong.

        The quarantine name carries the BYTES that were quarantined, so a second bad version of the same key cannot
        overwrite the first and every one of them is preserved."""
        root = self._resolved_root()
        try:
            content = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        except OSError:
            content = "unreadable"
        target = root / "quarantine" / f"{record_id}.{content}.json"
        self._contained(root, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return target.relative_to(root).as_posix()          # the identical bytes are already preserved
        os.replace(path, target)
        return target.relative_to(root).as_posix()

    def _write(self, path, record) -> bool:
        """Atomic within the store: a reader never sees a half-written record, and an interrupted write leaves a `.tmp`.

        Two writers racing on the SAME key write identical bytes -- the key is derived from the evaluation and the content
        from the record -- so `os.replace` makes the last one win with nothing lost. Different keys never share a path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".writing-", suffix=".json.tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, indent=1, sort_keys=True))
                stream.flush()
                os.fsync(stream.fileno())
            # an exclusive create, not a replace: whoever gets there first owns the key, and the losers read it back rather
            # than overwriting a record whose digest another caller is already holding
            try:
                os.link(temporary, path)
                return True
            except FileExistsError:
                return False
        except BaseException:
            raise
        finally:
            Path(temporary).unlink(missing_ok=True)

    # --- replay -----------------------------------------------------------------------------------------------------
    def replay(self):
        """Read every stored record back and re-derive its identity. A record that fails is REPORTED, not rewritten.

        `actionable_events` counts news identities with at least one VALID, successful evaluation. A refusal, a corrupted
        record and a superseded revision are all retained and none of them adds to that count."""
        report = {"records": 0, "events": 0, "revisions": 0, "reevaluations": 0, "integrity_failures": [],
                  "actionable_events": 0, "refused_events": 0, "quarantined": 0, "unfinished_writes": 0}
        if not self.root.is_dir():
            return report
        by_event = {}
        for record in self.all_records():
            by_event.setdefault(record.get("event_id"), []).append(record)
        # a quarantined record stays OUT of every count and IN the failure report: excluding it from the counts is why it
        # was moved, and hiding it would turn a detected corruption into a silent one
        for record in self.all_records(include_quarantine=True):
            if record.get("quarantined"):
                report["quarantined"] += 1
                report["integrity_failures"].append({"event_id": record.get("event_id"),
                                                     "record_sha256": record.get("record_sha256"),
                                                     "recomputed": record.get("recomputed_sha256"),
                                                     "problems": record.get("integrity_problems"),
                                                     "quarantined": True})
        report["unfinished_writes"] = len(list(self.root.rglob("*.tmp")))
        for event_id, live in sorted(by_event.items(), key=lambda kv: str(kv[0])):
            report["events"] += 1
            report["records"] += len(live)
            bytes_seen = {r.get("input_sha256") for r in live}
            report["revisions"] += max(0, len(bytes_seen) - 1)
            report["reevaluations"] += max(0, len(live) - len(bytes_seen))
            for record in live:
                if record.get("integrity") != "OK":
                    report["integrity_failures"].append({"event_id": event_id,
                                                         "record_sha256": record.get("record_sha256"),
                                                         "recomputed": record.get("recomputed_sha256"),
                                                         "problems": record.get("integrity_problems")})
                if record.get("execution_authorized") is not False:
                    report["integrity_failures"].append({"event_id": event_id, "reason": "EXECUTION_AUTHORIZED_IN_STORE"})
            decided = [r for r in live if r.get("integrity") == "OK" and r.get("status") == "SHADOW_ONLY"]
            if decided:
                report["actionable_events"] += 1
            else:
                report["refused_events"] += 1
        report["broker_calls"] = 0
        return report
