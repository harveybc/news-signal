"""The application path, end to end: installed entry point -> Registry -> typed request -> provider -> receipt -> store.

This module is the only place that knows all four layers at once, and it knows them in one direction. It never reaches past
the Registry to call a backend, never repairs a refusal into a result, and never turns a classification into an instruction.
"""

from datetime import datetime, timezone
import resource

from m5phet.runtime import ENTRY_POINT_GROUP, Registry, Status, run

from .core import Refusal, digest
from .provider import PROVIDER_NAME, request_for, state_ref_for
from .shadow import ShadowStore
from .tasks import task_digest

ABSENT_STATE = "laya-checkpoint:ABSENT"


def discover(group=ENTRY_POINT_GROUP, registry=None):
    """Installed discovery, exactly as any consumer would get it. A provider that fails to import is named, not hidden."""
    registry = registry or Registry()
    report = registry.load_entry_points(group)
    return registry, report


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def classify_event(event, *, task_id, as_of, max_age_seconds=900, registry=None, report=None,
                   store=None, request_id=None):
    """One news item through the public path. Returns the envelope, the receipt and what was persisted."""
    if registry is None:
        registry, report = discover()
    provider = registry.get(PROVIDER_NAME)
    if provider is None:
        refusal = (report or {}).get("refused", {}).get(PROVIDER_NAME)
        return {"schema": "news_shadow.v2", "status": "REFUSED", "execution_authorized": False,
                "why": f"provider {PROVIDER_NAME!r} is not installed in this environment",
                "provider_ref": PROVIDER_NAME, "provider_refusal": refusal,
                "registry_names": registry.names(), "discovery": report}
    caps = registry.capabilities(PROVIDER_NAME) or {}
    known = caps.get("known_states") or []
    state_ref = known[0] if known else ABSENT_STATE
    request = request_for(event, task_id=task_id, state_ref=state_ref, as_of=as_of,
                          max_age_seconds=max_age_seconds,
                          request_id=request_id or f"news:{event['event_id']}:{digest(event)[:12]}")
    result = run(request, registry)
    receipt = _receipt(event, task_id, as_of, max_age_seconds, request, result, provider, caps, report)
    stored = None
    if store is not None and receipt["status"] in ("SHADOW_ONLY", "REFUSED_WITH_INPUT"):
        record, disposition = store.put(receipt)
        stored = {"disposition": disposition, "record_sha256": record["record_sha256"],
                  "revision_index": record["revision_index"], "supersedes": record["supersedes"]}
    return {"receipt": receipt, "result": result, "request": request, "stored": stored,
            "status": receipt["status"]}


def _receipt(event, task_id, as_of, max_age_seconds, request, result, provider, caps, report):
    ok = result["status"] == Status.OK
    features = {}
    if ok:
        for question, output in result["outputs"].items():
            payload = output["payload"]
            features[question] = {"label": payload["label"],
                                  "uncalibrated_probabilities": payload["uncalibrated_probabilities"],
                                  "probability_decimals": payload["probability_decimals"],
                                  "sdk_answer": payload["sdk_answer"],
                                  "uncertainty": output["uncertainty"]}
    receipt = {
        "schema": "news_shadow.v2",
        "produced_by": "M5PHET_REGISTRY",
        "status": "SHADOW_ONLY" if ok else "REFUSED_WITH_INPUT",
        "m5phet_status": result["status"],
        "why": result.get("why"),
        # the envelope's `why` reports what the RUNTIME observed, which for a refusal is that no rows were covered. The reason
        # the provider gave lives per question, and reporting only the first would hide it.
        "refusal_reasons": ({q: o.get("why") for q, o in (result.get("outputs") or {}).items() if o.get("why")}
                            if not ok else None),
        "execution_authorized": False,
        "calibration": "UNCALIBRATED",
        "governance": "NOT_REGISTERED_BY_THIS_TOOL",
        "task_id": task_id,
        "task_sha256": task_digest(task_id),
        "provider_ref": PROVIDER_NAME,
        "entry_point_group": (report or {}).get("group", ENTRY_POINT_GROUP),
        "registered_providers": sorted((report or {}).get("registered", [])) or None,
        "event_id": event["event_id"], "source": event["source"], "asset": event["asset"],
        "published_at": event["published_at"], "received_at": event["received_at"],
        "as_of": as_of, "max_age_seconds": max_age_seconds,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": digest(event),
        "request_sha256": result["request_sha256"],
        "binding": result.get("binding"),
        "outputs": result.get("outputs"),
        "features": features or None,
        "model": (provider.last_load or {}),
        "model_identity": provider.identity_for(request["fitted_state_ref"]),
        "latency": {"load_seconds": (provider.last_load or {}).get("seconds"),
                    "load_kind": (provider.last_load or {}).get("kind"),
                    "inference_seconds": (provider.last_inference or {}).get("seconds")},
        "peak_rss_bytes": peak_rss_bytes(),
        "limitations": ["NO_RETURN_FORECAST", "NO_CALIBRATED_CONFIDENCE", "NO_BROKER_ORDERS",
                        "AGREEMENT_WITH_THE_SDK_IS_WRAPPER_FIDELITY_NOT_DOMAIN_ACCURACY"],
    }
    receipt["receipt_sha256"] = digest(receipt)
    return receipt


def classify_file(path, **kwargs):
    from .core import load_json
    return classify_event(load_json(path), **kwargs)


def replay(store_directory):
    """What survives a restart, read from disk with no model and no inference."""
    store = ShadowStore(store_directory)
    report = store.replay()
    report["schema"] = "news_shadow_replay.v1"
    report["inference_performed"] = False
    report["execution_authorized"] = False
    return report
