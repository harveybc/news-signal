"""The durable shadow store: a classification result that survives the process that produced it.

Three rules the CL06 acceptance turns on. A result is keyed by the bytes of the news item it is about, so the same item
classified twice is ONE stored decision, not two actionable events. A later revision of the same `event_id` is stored beside
the earlier one and linked to it, because a revision is new information, not a correction of the record. And a replay reads
what is on disk and re-derives its digest: a stored file that no longer hashes to its recorded identity is reported, never
repaired.

Nothing here contacts a broker, a venue or a governance service. `execution_authorized` is False in every record it writes.
"""

import json
from pathlib import Path

from .core import Refusal, canonical, digest


class ShadowStore:
    def __init__(self, directory):
        self.root = Path(directory)

    def _path(self, event_id, input_sha256):
        return self.root / event_id / f"{input_sha256}.json"

    def existing(self, event_id):
        """Every stored revision of one news identity, oldest first by the clock it was recorded at."""
        folder = self.root / event_id
        if not folder.is_dir():
            return []
        records = []
        for path in sorted(folder.glob("*.json")):
            try:
                records.append(json.loads(path.read_text()))
            except (OSError, ValueError) as exc:
                raise Refusal(f"UNREADABLE_SHADOW_RECORD: {path.name}") from exc
        return sorted(records, key=lambda r: (r.get("recorded_at") or "", r.get("input_sha256") or ""))

    def find(self, event_id, input_sha256):
        path = self._path(event_id, input_sha256)
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def put(self, receipt):
        """Store one result. Returns (record, disposition) where disposition is STORED, REVISION or DUPLICATE."""
        event_id, input_sha = receipt["event_id"], receipt["input_sha256"]
        already = self.find(event_id, input_sha)
        if already is not None:
            return already, "DUPLICATE"
        prior = self.existing(event_id)
        record = dict(receipt)
        record["execution_authorized"] = False
        record["supersedes"] = [p["record_sha256"] for p in prior]
        record["revision_index"] = len(prior)
        record["record_sha256"] = digest({k: v for k, v in record.items() if k != "record_sha256"})
        path = self._path(event_id, input_sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False, indent=1, sort_keys=True))
        tmp.replace(path)
        return record, ("REVISION" if prior else "STORED")

    def replay(self):
        """Read every stored record back and re-derive its identity. A record that fails is REPORTED, not rewritten."""
        report = {"records": 0, "events": 0, "revisions": 0, "integrity_failures": [], "actionable_events": 0}
        if not self.root.is_dir():
            return report
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            records = self.existing(folder.name)
            if not records:
                continue
            report["events"] += 1
            report["records"] += len(records)
            report["revisions"] += len(records) - 1
            report["actionable_events"] += 1          # one event identity is one decision, whatever its revision count
            for record in records:
                stored = record.get("record_sha256")
                recomputed = digest({k: v for k, v in record.items() if k != "record_sha256"})
                if stored != recomputed:
                    report["integrity_failures"].append({"event_id": folder.name,
                                                         "input_sha256": record.get("input_sha256"),
                                                         "stored": stored, "recomputed": recomputed})
                if record.get("execution_authorized") is not False:
                    report["integrity_failures"].append({"event_id": folder.name, "reason": "EXECUTION_AUTHORIZED_IN_STORE"})
        report["broker_calls"] = 0
        return report
