"""The real M5PHET provider for this distribution: `laya_news`.

The path this exposes is the whole point: installed entry point -> Registry -> typed request -> this provider -> the pinned
Laya Agent -> validated answer -> durable receipt. It adds no second encoder, tokenizer, registry or question compiler; it
delegates to the `LayaBackend` this package already ships and to the task registry in `tasks.py`.

Configuration arrives through the environment because an entry point is constructed with no arguments:

    NEWS_SIGNAL_CHECKPOINT   directory of the materialized checkpoint
    NEWS_SIGNAL_MANIFEST     the sealed manifest written by `news-signal seal-model`
    NEWS_SIGNAL_DEVICE       cpu | cuda:0            (default cpu)
    NEWS_SIGNAL_GPU_UUID     the physical GPU- UUID, required for cuda:0

Two doors lead to the same engine. The typed request (`infer`) carries one versioned or user-authored task and is the
path every receipt so far was written under. The question envelope (`answer_questions`) is the unified contract of
`m5phet.questions`: a state, named typed questions in, named typed answers out. This area is the native one -- Laya already
takes named `choice` questions and returns named answers -- so the envelope maps onto the SDK's own question set with no
second compiler: every question passes through the same `build_question` gate, and the whole set goes to the SDK in ONE call,
because the set conditions the encoding.

What it refuses, and when. An absent or unreadable manifest leaves `known_states` empty, so the runtime refuses every
inference with MODEL_NOT_FITTED *before* anything is loaded. A task the provider does not ship is refused by the runtime
through the loaded state's declared compatibility. Everything about the news item itself -- language, size, availability at
the decision clock, asset scope -- is checked before the forward pass and comes back as a typed per-question refusal.
"""

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from . import core
from .backends import sequence_budget
from .core import Refusal, canonical, digest, validate_news
from .question import AD_HOC_PREFIX, SUPPORTED_SCHEMAS, build_question, question_identity
from .tasks import TASKS, check_scope, questions as task_questions

PROVIDER_NAME = "laya_news"

#: The one combination this provider has been exercised on. Support is declared, never the product of capability lists.
SUPPORTED = ({"operation": "infer", "family": "classification", "output_kind": "typed_questions"},)

UNCERTAINTY = "UNCALIBRATED_CLASS_PROBABILITIES"

#: the identity a fitted state carries: this checkpoint is zero-shot, so it is fitted to no single task and declares the
#: versioned tasks this distribution ships instead of silently accepting any string
CHECKPOINT_TASK_ID = "laya_zero_shot_checkpoint"

DEFAULT_MAX_AGE_SECONDS = 900

#: the area this provider declares to the question envelope; one area, one provider
AREA = "classification"

#: the question types the envelope may ask this provider, with the fields each needs. `options` is required because a choice
#: without options is not a question; `instructions` is optional in the ENVELOPE (the contract allows it everywhere) and is
#: then required by the same gate that checks a user-authored question, so an empty one is refused by name, not defaulted.
QUESTION_TYPES = {"choice": {"required": ["options"], "optional": ["instructions"]}}

#: the source a plain-text state is filed under when it is wrapped into an event, exactly as the workbench does
MANUAL_SOURCE = "MANUAL_CONTEXT"

READING = ("probabilities are the pinned SDK's own uncalibrated head outputs at its own precision; the label is the SDK's "
           "argmax; nothing here is calibrated, and agreeing with the SDK is fidelity of this wrapper, not domain accuracy")


def configuration(environ=None):
    """What this process was configured with. Read once per provider instance; never guessed and never defaulted to a device."""
    env = os.environ if environ is None else environ
    device = env.get("NEWS_SIGNAL_DEVICE", "cpu")
    backend = env.get("NEWS_SIGNAL_BACKEND", "laya")
    if backend not in ("laya", "fixture"):
        raise Refusal(f"UNKNOWN_BACKEND: {backend!r} is not one of ('laya', 'fixture')")
    return {"checkpoint": env.get("NEWS_SIGNAL_CHECKPOINT"), "manifest": env.get("NEWS_SIGNAL_MANIFEST"),
            "device": device, "gpu_uuid": env.get("NEWS_SIGNAL_GPU_UUID"), "backend": backend}


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
    area = AREA

    def __init__(self, environ=None, backend_factory=None):
        self.config = configuration(environ)
        self._manifest = _read_manifest(self.config["manifest"])
        if self.config["backend"] == "fixture" and self._manifest is None:
            # a declared non-model has no checkpoint, so its state is named after the fixture itself. Nothing about this can
            # be confused with weights: the identity says NON_MODEL_FIXTURE and travels into every receipt.
            from .backends import FixtureBackend
            self._manifest = {"schema": "news_checkpoint.v1", "sha256": digest(FixtureBackend.identity),
                              "files": {}, "kind": "NON_MODEL_FIXTURE"}
        self._backend_factory = backend_factory or self._real_backend
        self._backends = {}
        self.calls = 0
        self.last_load = None
        self.last_inference = None
        self.last_budget = None
        self.last_budget_checked = None

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
                "question_types": copy.deepcopy(QUESTION_TYPES),
                "device": self.config["device"],
                "backend": self.config["backend"],
                "weights_present": manifest is not None and manifest.get("kind") != "NON_MODEL_FIXTURE",
                "reading": ("probabilities are the pinned SDK's own uncalibrated outputs at its own precision; agreeing with "
                            "the SDK is fidelity of this wrapper, not calibration and not domain accuracy")}

    # --- the fitted state ------------------------------------------------------------------------------------------------
    def _real_backend(self, config, manifest):
        from .backends import FixtureBackend, LayaBackend
        if config["backend"] == "fixture":
            return FixtureBackend()
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
        # The budget belongs here, not inside one backend: the guarantee is the provider's, whichever backend answers. A
        # backend that exposes no tokenizer cannot be checked, and the receipt says so rather than implying it was.
        tokenizer = backend.tokenizer() if callable(getattr(backend, "tokenizer", None)) else None
        budget = sequence_budget(tokenizer, serialized, questions) if tokenizer is not None else None
        self.last_budget = budget
        self.last_budget_checked = budget is not None
        if budget is not None:
            over = sorted(name for name, entry in budget.items() if not entry["fits"])
            if over:
                why = (f"TOKEN_BUDGET_EXCEEDED: {over} would be truncated by the pinned SDK "
                       f"({budget[over[0]]}); shorten the field, the question or its options, "
                       f"do not silently truncate")
                return {"outputs": {q: {"status": "INVALID_INPUT", "why": why} for q in requested}}
        started = time.perf_counter()
        try:
            response = backend.predict(serialized, questions)
        except Refusal as exc:
            return {"outputs": {q: {"status": "INVALID_INPUT", "why": str(exc)} for q in requested}}
        elapsed = time.perf_counter() - started
        self.calls += 1
        self.last_inference = {"seconds": elapsed, "task_id": task_id, "event_id": event["event_id"]}
        # what the pinned SDK was actually given, counted before it could cut anything
        self.last_budget = budget if budget is not None else getattr(backend, "last_budget", None)
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
                               "token_budget": copy.deepcopy(budget),
                               "token_budget_checked": budget is not None,
                               "sdk_response": copy.deepcopy(response)}}

    # --- the question envelope ---------------------------------------------------------------------------------------------
    def question_types(self):
        """The types the envelope may ask, and the fields each needs. Unknown types never reach `answer_questions`."""
        return copy.deepcopy(QUESTION_TYPES)

    def answer_questions(self, state, questions, data, as_of):
        """Answer named `choice` questions about one news item, each on its own, through ONE call to the SDK.

        The news is `data` when given, otherwise `state["news"]`: an event dict validated exactly as the typed request
        validates it, or plain text wrapped into a MANUAL_CONTEXT event the way the workbench wraps it. Each question goes
        through the same gate a user-authored question does, so a duplicate option, an empty instruction or an option the
        pinned SDK would cut are refused under that question's name while the others are still asked. The questions that
        survive are sent together, because Laya encodes the question set with the text: the set IS the input.

        Every answer carries the SDK's own numbers untouched and says they are uncalibrated. Nothing here is a confidence in
        the sense of correctness."""
        from m5phet.questions import MALFORMED_QUESTION, PROVIDER_ERROR, STATE_REQUIRED, refusal

        def refuse_all(kind, why, names):
            return {name: refusal(kind, why, questions[name]["type"]) for name in names}

        if not isinstance(state, dict):
            return refuse_all(STATE_REQUIRED, "`state` must be a mapping describing the news item", questions)
        answers = {}
        # 1. each question is built and checked by name; a refusal here names one question, never the envelope
        built = {}
        for name, question in questions.items():
            try:
                if question["type"] not in SUPPORTED_SCHEMAS:
                    raise Refusal(f"UNSUPPORTED_ANSWER_SCHEMA: {question['type']!r} is not one of {list(SUPPORTED_SCHEMAS)}")
                built.update(build_question(name=name, question=question.get("instructions"),
                                            options=question["options"], schema=question["type"]))
            except Refusal as exc:
                answers[name] = refusal(MALFORMED_QUESTION, str(exc), question["type"])
        askable = [name for name in questions if name not in answers]
        if not askable:
            return answers
        # 2. the state: an event with its own clocks, or text wrapped into one at the present clock
        try:
            event, decision = self._envelope_event(state, data, as_of)
            max_age = state.get("max_age_seconds", DEFAULT_MAX_AGE_SECONDS)
            validate_news(event, decision, max_age)
        except Refusal as exc:
            answers.update(refuse_all(STATE_REQUIRED, str(exc), askable))
            return answers
        # 3. the fitted state: the sealed checkpoint, loaded through the same door the runtime uses
        try:
            manifest = self._manifest
            if manifest is None:
                raise Refusal("CHECKPOINT_MANIFEST_UNAVAILABLE: this provider holds no sealed checkpoint")
            state_ref = state.get("state_ref") or state_ref_for(manifest["sha256"])
            self.load(state_ref)
        except Refusal as exc:
            answers.update(refuse_all(PROVIDER_ERROR, str(exc), askable))
            return answers
        backend = self._backends[state_ref]
        serialized = canonical({k: event[k] for k in ("asset", "headline", "body")})
        # 4. the token budget, per question: one that the pinned SDK would cut is refused by name and never sent
        tokenizer = backend.tokenizer() if callable(getattr(backend, "tokenizer", None)) else None
        budget = sequence_budget(tokenizer, serialized, built) if tokenizer is not None else None
        self.last_budget = budget
        self.last_budget_checked = budget is not None
        if budget is not None:
            for name in list(built):
                if not budget[name]["fits"]:
                    answers[name] = refusal(MALFORMED_QUESTION,
                                            f"TOKEN_BUDGET_EXCEEDED: {name!r} would be truncated by the pinned SDK "
                                            f"({budget[name]}); shorten the instructions or the options, do not "
                                            f"silently truncate", questions[name]["type"])
                    del built[name]
        if not built:
            return answers
        # 5. ONE call for the whole surviving set, in the caller's order
        task_id = f"{AD_HOC_PREFIX}:{question_identity(built)}"
        started = time.perf_counter()
        try:
            response = backend.predict(serialized, built)
            labels = core.validate_answers(response, built)
        except Refusal as exc:
            answers.update(refuse_all(PROVIDER_ERROR, str(exc), list(built)))
            return answers
        elapsed = time.perf_counter() - started
        self.calls += 1
        self.last_inference = {"seconds": elapsed, "task_id": task_id, "event_id": event["event_id"]}
        self.last_budget = budget if budget is not None else getattr(backend, "last_budget", None)
        provenance = {"task_id": task_id,
                      "questions_sha256": digest(built),
                      "state_sha256": digest(serialized),
                      "input_sha256": digest(event),
                      "response_sha256": digest(response),
                      "inference_seconds": elapsed,
                      "token_budget_checked": budget is not None}
        for name, answer in labels.items():
            answers[name] = {"type": "choice",
                             "status": "OK",
                             "label": answer["label"],
                             # the SDK's own numbers at the SDK's own precision: no rounding, rescaling or renormalization
                             "uncalibrated_probabilities": answer["uncalibrated_probabilities"],
                             "probability_decimals": 4,
                             "calibration": "UNCALIBRATED",
                             "uncertainty": UNCERTAINTY,
                             # the SDK's answer object verbatim, so a parity check can compare every field it exposes
                             "sdk_answer": copy.deepcopy(response["answers"][name]),
                             "reading": READING,
                             "population": population_of(event),
                             "provenance": copy.deepcopy(provenance),
                             "execution_authorized": False}
        answers["__state_ref__"] = state_ref
        return answers

    @staticmethod
    def _envelope_event(state, data, as_of):
        """The news item the envelope is about, and the decision clock it is judged at.

        `data` wins when given; otherwise the item travels as `state["news"]`. A mapping is an event and keeps its own
        clocks. Plain text becomes a MANUAL_CONTEXT event stamped at the present clock, exactly as the workbench wraps it,
        which is why text cannot be replayed at a historical `as_of`: it has no clocks of its own to replay."""
        news = data if data is not None else state.get("news")
        if isinstance(news, dict):
            return news, as_of or datetime.now(timezone.utc).isoformat()
        if isinstance(news, str) and news.strip():
            if as_of:
                raise Refusal("HISTORICAL_REPLAY_NEEDS_EVENT: text carries no clocks; replay at an `as_of` needs a news "
                              "event with its actual published_at and received_at")
            asset = state.get("asset")
            if not isinstance(asset, str) or not asset.strip():
                raise Refusal("ASSET_REQUIRED: text news needs `state.asset` to say which instrument it is about")
            now = datetime.now(timezone.utc).isoformat()
            return {"schema": "news_event.v1", "event_id": "manual:" + hashlib.sha256(news.encode()).hexdigest(),
                    "source": MANUAL_SOURCE, "asset": asset, "language": state.get("language", "en"),
                    "published_at": now, "received_at": now, "headline": "Manual context", "body": news}, now
        raise Refusal("NEWS_REQUIRED: the envelope carries no news item; pass an event mapping or text as `data` or "
                      "`state.news`")


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
