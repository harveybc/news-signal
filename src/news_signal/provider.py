"""The real M5PHET provider for this distribution: `laya_news`.

The path this exposes is the whole point: installed entry point -> Registry -> typed request -> this provider -> the pinned
Laya Agent -> validated answer -> durable receipt. It adds no second encoder, tokenizer, registry or question compiler; it
delegates to the `LayaBackend` this package already ships and to the task registry in `tasks.py`.

Configuration arrives through the environment because an entry point is constructed with no arguments:

    NEWS_SIGNAL_CHECKPOINT   directory of the materialized checkpoint
    NEWS_SIGNAL_MANIFEST     the sealed manifest written by `news-signal seal-model`
    NEWS_SIGNAL_DEVICE       cpu | cuda:0            (default cpu)
    NEWS_SIGNAL_GPU_UUID     the physical GPU- UUID, required for cuda:0

What it refuses, and when. An absent or unreadable manifest leaves `known_states` empty, so the runtime refuses every
inference with MODEL_NOT_FITTED *before* anything is loaded. A task the provider does not ship is refused by the runtime
through the loaded state's declared compatibility. Everything about the news item itself -- language, size, availability at
the decision clock, asset scope -- is checked before the forward pass and comes back as a typed per-question refusal.
"""

import copy
import json
import os
from pathlib import Path
import time

from . import core
from .core import Refusal, canonical, digest, validate_news
from .question import AD_HOC_PREFIX
from .tasks import TASKS, check_scope, questions as task_questions

PROVIDER_NAME = "laya_news"

#: The one combination this provider has been exercised on. Support is declared, never the product of capability lists.
SUPPORTED = ({"operation": "infer", "family": "classification", "output_kind": "typed_questions"},)

UNCERTAINTY = "UNCALIBRATED_CLASS_PROBABILITIES"

#: the identity a fitted state carries: this checkpoint is zero-shot, so it is fitted to no single task and declares the
#: versioned tasks this distribution ships instead of silently accepting any string
CHECKPOINT_TASK_ID = "laya_zero_shot_checkpoint"

DEFAULT_MAX_AGE_SECONDS = 900


def configuration(environ=None):
    """What this process was configured with. Read once per provider instance; never guessed and never defaulted to a device."""
    env = os.environ if environ is None else environ
    device = env.get("NEWS_SIGNAL_DEVICE", "cpu")
    return {"checkpoint": env.get("NEWS_SIGNAL_CHECKPOINT"), "manifest": env.get("NEWS_SIGNAL_MANIFEST"),
            "device": device, "gpu_uuid": env.get("NEWS_SIGNAL_GPU_UUID")}


def state_ref_for(manifest_sha256):
    return f"laya-checkpoint:{manifest_sha256}"


def _read_manifest(path):
    """The sealed manifest, or None. Reading it touches no weights, so `known_states` is honest without loading a model."""
    if not path:
        return None
    try:
        manifest = core.load_json(Path(path))
    except Refusal:
        return None
    if not isinstance(manifest, dict) or manifest.get("schema") != "news_checkpoint.v1" or not manifest.get("sha256"):
        return None
    return manifest


class LayaNewsProvider:
    """One provider, one engine, one declared combination."""

    name = PROVIDER_NAME

    def __init__(self, environ=None, backend_factory=None):
        self.config = configuration(environ)
        self._manifest = _read_manifest(self.config["manifest"])
        self._backend_factory = backend_factory or self._real_backend
        self._backends = {}
        self.calls = 0
        self.last_load = None
        self.last_inference = None
        self.last_budget = None

    # --- what the runtime checks before anything is loaded ---------------------------------------------------------------
    def capabilities(self):
        manifest = self._manifest
        known = [state_ref_for(manifest["sha256"])] if manifest else []
        return {"provider": self.name,
                "operations": ["infer"],
                "families": ["classification"],
                "output_kinds": ["typed_questions"],
                "uncertainty_methods": [UNCERTAINTY],
                "supported": [dict(entry) for entry in SUPPORTED],
                "known_states": known,
                "tasks": sorted(TASKS),
                "device": self.config["device"],
                "weights_present": manifest is not None,
                "reading": ("probabilities are the pinned SDK's own uncalibrated outputs at its own precision; agreeing with "
                            "the SDK is fidelity of this wrapper, not calibration and not domain accuracy")}

    # --- the fitted state ------------------------------------------------------------------------------------------------
    def _real_backend(self, config, manifest):
        from .backends import LayaBackend
        return LayaBackend(config["checkpoint"], manifest, config["device"], config["gpu_uuid"])

    def load(self, state_ref):
        manifest = self._manifest
        if manifest is None:
            raise Refusal("CHECKPOINT_MANIFEST_UNAVAILABLE: this provider holds no sealed checkpoint")
        if state_ref != state_ref_for(manifest["sha256"]):
            raise Refusal(f"UNKNOWN_FITTED_STATE: {state_ref!r} is not this provider's sealed checkpoint")
        backend = self._backends.get(state_ref)
        cold = backend is None
        started = time.perf_counter()
        if cold:
            backend = self._backends[state_ref] = self._backend_factory(self.config, manifest)
        self.last_load = {"state_ref": state_ref, "kind": "COLD" if cold else "WARM",
                          "seconds": time.perf_counter() - started}
        return {"state_ref": state_ref,
                "digest": manifest["sha256"],
                "model_sha256": digest(backend.identity),
                "task_id": CHECKPOINT_TASK_ID,
                "compatible_task_ids": sorted(TASKS),
                # the model's input IS the question, so this state answers any question this adapter has validated. The KIND
                # names that contract; every member of it is built and checked here before anything is encoded.
                "compatible_task_kinds": [AD_HOC_PREFIX],
                "load_seconds": time.perf_counter() - started,
                "load_kind": "COLD" if cold else "WARM",
                "identity": copy.deepcopy(backend.identity)}

    def identity_for(self, state_ref):
        """The loaded backend's declared identity, or None when nothing was loaded under that reference."""
        backend = self._backends.get(state_ref)
        return copy.deepcopy(backend.identity) if backend is not None else None

    # --- inference -------------------------------------------------------------------------------------------------------
    def infer(self, request, state):
        task_id = request["task_id"]
        spec = (request.get("inputs") or {}).get("question_spec")
        try:
            questions = task_questions(task_id, spec)
        except Refusal as exc:
            requested = list((request.get("output_schema") or {}).get("questions") or ["question"])
            return {"outputs": {q: {"status": "INVALID_INPUT", "why": str(exc)} for q in requested}}
        requested = list((request.get("output_schema") or {}).get("questions") or [])
        event = ((request.get("inputs") or {}).get("news_event"))
        max_age = ((request.get("inputs") or {}).get("max_age_seconds"), DEFAULT_MAX_AGE_SECONDS)
        max_age = max_age[0] if max_age[0] is not None else max_age[1]
        refusal = None
        if sorted(requested) != sorted(questions):
            refusal = (f"QUESTION_POPULATION_MISMATCH: task {task_id} answers {sorted(questions)}; the question set is the "
                       f"model's input, so a subset is a different task, not a filter")
        else:
            try:
                validate_news(event, request["as_of"], max_age)
                check_scope(task_id, event["asset"], spec)
            except Refusal as exc:
                refusal = str(exc)
        if refusal is not None:
            # nothing ran: a refused question carries no number and no population is claimed
            return {"outputs": {q: {"status": "INVALID_INPUT", "why": refusal} for q in (requested or sorted(questions))}}
        backend = self._backends[state["state_ref"]]
        serialized = canonical({k: event[k] for k in ("asset", "headline", "body")})
        started = time.perf_counter()
        try:
            response = backend.predict(serialized, questions)
        except Refusal as exc:
            return {"outputs": {q: {"status": "INVALID_INPUT", "why": str(exc)} for q in requested}}
        elapsed = time.perf_counter() - started
        self.calls += 1
        self.last_inference = {"seconds": elapsed, "task_id": task_id, "event_id": event["event_id"]}
        # what the pinned SDK was actually given, counted before it could cut anything
        self.last_budget = getattr(backend, "last_budget", None)
        labels = core.validate_answers(response, questions)
        outputs = {}
        for question in requested:
            answer = labels[question]
            outputs[question] = {
                "status": "OK",
                "uncertainty": UNCERTAINTY,
                "payload": {"label": answer["label"],
                            # the SDK's own numbers at the SDK's own precision: no rounding, rescaling or renormalization
                            "uncalibrated_probabilities": answer["uncalibrated_probabilities"],
                            "probability_decimals": 4,
                            "calibration": "UNCALIBRATED",
                            # the SDK's answer object carried through verbatim, so a parity check can compare every field
                            # it exposes -- type, choice, probabilities and confidence -- and not just the two we read
                            "sdk_answer": copy.deepcopy(response["answers"][question])},
            }
        return {"outputs": outputs,
                "population": population_of(event),
                "provenance": {"task_id": task_id,
                               "questions_sha256": digest(questions),
                               "questions_as_sent": copy.deepcopy(questions),
                               "state_sha256": digest(serialized),
                               "input_sha256": digest(event),
                               "response_sha256": digest(response),
                               "inference_seconds": elapsed,
                               "sdk_response": copy.deepcopy(response)}}


def population_of(event):
    """The rows a result is about, derived from the news item that was actually read -- never from what the caller declared."""
    return {"event_ids": [event["event_id"]], "asset": event["asset"], "input_sha256": digest(event)}


def request_for(event, *, task_id, state_ref, request_id, as_of, max_age_seconds=DEFAULT_MAX_AGE_SECONDS,
                question_spec=None):
    """A typed M5PHET request for one news item. The population is derived from the item, so a mismatch is detectable.

    For a user-authored question the spec travels IN the request, so the request digest covers the exact words the model was
    given: changing a rubric changes the request identity, not just a label on it."""
    return {"schema_version": "m5phet.task.draft2",
            "request_id": request_id,
            "task_id": task_id,
            "operation": "infer",
            "family": "classification",
            "output_kind": "typed_questions",
            "as_of": as_of,
            "provider_ref": PROVIDER_NAME,
            "fitted_state_ref": state_ref,
            "output_schema": {"questions": sorted(task_questions(task_id, question_spec))},
            "population": population_of(event),
            "inputs": {"news_event": event, "max_age_seconds": max_age_seconds,
                       **({"question_spec": question_spec} if question_spec else {})},
            "execution_constraints": {"partial_results": False}}
