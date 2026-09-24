"""NL01: a question written by the user, validated into a task before anything is encoded.

Laya conditions its encoder on the question and its options together with the text, so a user-authored question is not a
parameter of an existing task -- it IS a different task, and its identity has to say so. Everything that reaches the model is
therefore part of the identity: the question, the option names, their descriptions, THEIR ORDER, the answer schema and the
call settings.

What this module refuses, and why each refusal exists rather than a silent fix:

* a question or an option that is empty, not a string, or longer than the encoder's budget -- truncating it would answer a
  different question from the one that was asked, and the caller would never know;
* fewer than two options for a choice -- a question with one answer is not a question;
* duplicate option names, or names that differ only by case or surrounding space -- the SDK returns one probability per
  option name, so two that collide cannot be told apart afterwards;
* an unsupported answer schema -- the supported shapes are declared, and an unsupported one fails by name.

Nothing here calls a model. It is the gate that runs before one is loaded.
"""

from .core import Refusal, digest

#: answer shapes this adapter has been exercised on. `choice` is Laya's typed multiple-choice head.
SUPPORTED_SCHEMAS = ("choice",)

#: the encoder budgets of the pinned SDK as this adapter calls it. The question and its options are encoded WITH the state,
#: so they consume the same budget the news text does; a question that fits alone can still overflow with its options.
MAX_QUESTION_CHARACTERS = 600
MAX_OPTION_NAME_CHARACTERS = 64
MAX_OPTION_DESCRIPTION_CHARACTERS = 400
MAX_OPTIONS = 12
MAX_TOTAL_QUESTION_CHARACTERS = 4000

AD_HOC_PREFIX = "adhoc.question.v1"


def _text(value, field, limit):
    if not isinstance(value, str) or not value.strip():
        raise Refusal(f"QUESTION_FIELD_REQUIRED: {field} must be a non-empty string")
    if len(value) > limit:
        raise Refusal(f"QUESTION_FIELD_TOO_LONG: {field} is {len(value)} characters and the limit is {limit}; "
                      f"this adapter refuses rather than silently truncating what the model is asked")
    return value


def build_question(*, name, question, options, schema="choice"):
    """Validate a user-authored question into the exact mapping the SDK will be given. Order is preserved as written."""
    if schema not in SUPPORTED_SCHEMAS:
        raise Refusal(f"UNSUPPORTED_ANSWER_SCHEMA: {schema!r} is not one of {list(SUPPORTED_SCHEMAS)}")
    name = _text(name, "name", MAX_OPTION_NAME_CHARACTERS)
    question = _text(question, "question", MAX_QUESTION_CHARACTERS)
    if isinstance(options, dict):
        pairs = list(options.items())
    elif isinstance(options, (list, tuple)):
        pairs = [tuple(o) if isinstance(o, (list, tuple)) and len(o) == 2 else (o, "") for o in options]
    else:
        raise Refusal("QUESTION_OPTIONS_REQUIRED: options must be a mapping or a sequence of name/description pairs")
    if not 2 <= len(pairs) <= MAX_OPTIONS:
        raise Refusal(f"QUESTION_OPTION_COUNT: a choice needs between 2 and {MAX_OPTIONS} options, got {len(pairs)}")
    criteria, seen = {}, set()
    for option, description in pairs:
        option = _text(option, "option name", MAX_OPTION_NAME_CHARACTERS)
        description = _text(description, f"option {option!r} description", MAX_OPTION_DESCRIPTION_CHARACTERS)
        key = option.strip().casefold()
        if key in seen:
            raise Refusal(f"DUPLICATE_OPTION: {option!r} collides with another option; the SDK returns one probability per "
                          f"option name, so two that differ only by case or spacing cannot be told apart afterwards")
        seen.add(key)
        criteria[option] = description
    built = {name: {"type": schema, "instructions": question, "criteria": criteria}}
    total = len(question) + sum(len(k) + len(v) for k, v in criteria.items())
    if total > MAX_TOTAL_QUESTION_CHARACTERS:
        raise Refusal(f"QUESTION_BUDGET_EXCEEDED: the question and its options are {total} characters and the limit is "
                      f"{MAX_TOTAL_QUESTION_CHARACTERS}; they are encoded together with the news, not separately")
    return built


def question_identity(questions):
    """An identity that can SEE the order of the options.

    `digest` serializes with sorted keys, which is right for a record and wrong for this: the SDK builds its input sequence
    in the mapping's own order, so two orders are two different inputs and must not share an identity. Serializing the
    options as a LIST keeps their order inside a canonical form."""
    ordered = [{"name": name,
                "type": q["type"],
                "instructions": q["instructions"],
                "options": [[option, description] for option, description in q["criteria"].items()]}
               for name, q in questions.items()]
    return digest(ordered)


def ad_hoc_task(*, name, question, options, schema="choice", assets=None):
    """A complete, self-identifying task built from one user-authored question.

    Its `task_id` contains the digest of the question set, so two different questions can never share an identity and the
    same question always recovers the same one."""
    questions = build_question(name=name, question=question, options=options, schema=schema)
    return {"questions": questions,
            "assets": tuple(assets) if assets else None,
            "family": "classification",
            "output_kind": "typed_questions",
            "task_id": f"{AD_HOC_PREFIX}:{question_identity(questions)}",
            "origin": "USER_AUTHORED_QUESTION",
            "reading": ("the question, its options, their descriptions and their order are all encoded with the news, so they "
                        "are part of this task's identity; changing any of them is a different task, not a parameter")}
