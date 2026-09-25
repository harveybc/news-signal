"""WP09: the labelled corpus this package is measured on, and the rules that make its labels independent.

Every number this package has published so far rested on thirteen rows one of its own authors wrote, which is why the
evaluation protocol calls that provenance ``AUTHOR_WRITTEN_SMOKE`` and refuses to read agreement on it as accuracy. This
module builds the first corpus whose labels nobody here wrote.

**Where the labels come from.** An economic calendar distributed as a dataset carries, per release, the country whose
statistics office or central bank published it, beside the release's own description. The country column is the label:
it existed before this system did, it was produced by the calendar's compiler, and no author of this system chose it for
any row. The question asked of the model is therefore *which economy is named in this news*, and the answer is checked
against the calendar's own country field.

**Three declarations, because a label is a definition and not a fact.**

1. The *mapping* (``COUNTRY_TO_CLASS``): ``Euro Zone`` is ``euro_area``, ``United States`` is ``united_states``, and the
   fourteen countries listed below are ``other``. A country the mapping does not declare is refused by name
   (``UNDECLARED_COUNTRY``) rather than silently swept into ``other``, because a class that absorbs the unknown grows
   whenever the source file does.
2. The *exclusion* (``EXCLUDED_COUNTRIES``): a release published by a euro-area member state — Germany, France, Italy,
   Spain — is dropped from the corpus entirely. Calling it ``other`` would mark a model wrong for answering
   ``euro_area`` about a German release, which is defensible; calling it ``euro_area`` would mark it wrong for answering
   ``other`` about a release the news names as Germany's, which is also defensible. A row whose label is contestable
   measures the definition, not the model, so it is counted and left out. This is the one place this module departs from
   the safest mapping, and it departs by removing rows rather than by relabelling any.
3. The *template* (``TEMPLATE_VERSION``): the model must read a release line, not a bare label. The line is built from
   the row's own description, actual, consensus (the calendar's ``forecast`` column), previous value, unit and expected
   volatility, in a fixed order with fixed wording. The country never appears in it except where the description itself
   names it, which is exactly what the keyword baseline measures.

**What is deliberately absent.** No row is rewritten, no label is adjudicated by hand, no text is chosen for being easy
or hard: the sample is seeded, balanced and stratified over the calendar's years, and the manifest carries the seed, the
source digest, the mapping, the template and the per-class counts so the same corpus can be rebuilt byte for byte.
"""

from __future__ import annotations

import csv
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from .core import Refusal, digest

#: the shape of the manifest this module writes; bumped whenever a declaration below changes the rows
CORPUS_SCHEMA = "news_signal.quality_corpus.v1"

#: the release-line wording. It is versioned because changing it changes the model's input and therefore every number
TEMPLATE_VERSION = "wp09.release_line.v1"

#: the classes, in the order every table, option list and tie-break uses
CLASSES = ("euro_area", "united_states", "other")

#: the column names the calendar file carries positionally, as `feature-eng`'s own loader declares them
COLUMNS = ("event_date", "event_time", "country", "volatility", "description",
           "evaluation", "data_format", "actual", "forecast", "previous")

#: the declared mapping from the calendar's own country field to this corpus's classes
COUNTRY_TO_CLASS = {
    "Euro Zone": "euro_area",
    "United States": "united_states",
    "Australia": "other", "Brazil": "other", "Canada": "other", "China": "other", "Hong Kong": "other",
    "India": "other", "Japan": "other", "New Zealand": "other", "Russia": "other", "Singapore": "other",
    "South Africa": "other", "South Korea": "other", "Switzerland": "other", "United Kingdom": "other",
}

#: euro-area member states: declared, excluded, counted. See declaration 2 in the module docstring
EXCLUDED_COUNTRIES = {
    "France": "EURO_AREA_MEMBER_STATE", "Germany": "EURO_AREA_MEMBER_STATE",
    "Italy": "EURO_AREA_MEMBER_STATE", "Spain": "EURO_AREA_MEMBER_STATE",
}

#: what `map_country` returns for a declared exclusion, so an excluded row is never confused with an unmapped one
EXCLUDED = "EXCLUDED_AMBIGUOUS"

#: the asset the envelope declares. The corpus is macroeconomic, so naming a currency pair here would hand the model a
#: hint about the euro on every single row; "MACRO" names the scope without naming an economy
ASSET = "MACRO"

#: the source name every event carries, so a receipt says which file a row came from without carrying a path
SOURCE_NAME = "economic_calendar_2011_2021"

#: the one question this corpus scores, with its exact instruction and options. Laya encodes the question with the
#: text, so this wording IS part of the measurement and is recorded in the manifest
QUESTION_NAME = "economy"
QUESTION_INSTRUCTIONS = "Which economy is named in this news?"
QUESTION_OPTIONS = [["euro_area", "the euro area"], ["united_states", "the United States"], ["other", "another economy"]]

#: the keyword baseline, declared here and computed on the same rows: the class whose country name appears verbatim in
#: the release line, checked in declared class order, `other` when neither appears
KEYWORD_RULES = (("euro_area", ("euro zone",)), ("united_states", ("united states",)))

#: a value is quoted into the line only when it is a plain number; anything else (a revision note, a blank, a stray
#: quote) becomes "not reported", so the line never carries the source file's parsing accidents
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")

_VOLATILITY = re.compile(r"^(Low|Moderate|High)\b", re.IGNORECASE)


class CorpusError(Refusal):
    """A corpus cannot be built as declared. Names the row or the country; never falls back to a partial corpus."""


def map_country(country):
    """The declared class of one calendar country, `EXCLUDED` for a declared exclusion, a refusal for anything else."""
    name = (country or "").strip()
    if name in EXCLUDED_COUNTRIES:
        return EXCLUDED
    label = COUNTRY_TO_CLASS.get(name)
    if label is None:
        raise CorpusError(f"UNDECLARED_COUNTRY: {name!r} is in no declared class and in no declared exclusion; declare "
                          f"it in COUNTRY_TO_CLASS or in EXCLUDED_COUNTRIES before sampling it")
    return label


def _value(raw, unit):
    text = (raw or "").strip().strip('"').strip()
    if not _NUMBER.match(text):
        return "not reported"
    return f"{text}{unit}" if unit else text


def release_line(row):
    """The headline and body one calendar row becomes, under `TEMPLATE_VERSION`. Deterministic, and it names no country.

    `row` is a mapping over `COLUMNS`. The wording is fixed: whoever changes it must change `TEMPLATE_VERSION`, because
    the model's answer belongs to the exact words it was given."""
    description = (row["description"] or "").strip()
    if not description:
        raise CorpusError("EMPTY_DESCRIPTION: a release with no description is no news line")
    unit = (row["data_format"] or "").strip()
    unit = "" if unit in ("", '"') else unit
    volatility = _VOLATILITY.match((row["volatility"] or "").strip())
    volatility = volatility.group(1).capitalize() if volatility else "not reported"
    stamp = release_datetime(row)
    body = (f"Economic calendar release on {stamp.date().isoformat()} at {stamp.strftime('%H:%M')} UTC. "
            f"{description}: actual {_value(row['actual'], unit)}, "
            f"consensus {_value(row['forecast'], unit)}, "
            f"previous {_value(row['previous'], unit)}. "
            f"Expected volatility: {volatility}.")
    return description[:500], body


def release_datetime(row):
    """The row's own clock, in UTC. A row whose date or time does not parse is refused rather than stamped with today."""
    try:
        moment = datetime.strptime(f"{(row['event_date'] or '').strip()} {(row['event_time'] or '').strip()}",
                                   "%Y/%m/%d %H:%M:%S")
    except ValueError as error:
        raise CorpusError(f"UNPARSEABLE_RELEASE_CLOCK: {row.get('event_date')!r} {row.get('event_time')!r}") from error
    return moment.replace(tzinfo=timezone.utc)


def keyword_prediction(text):
    """The declared naive reader: the class whose country name occurs verbatim, in declared order, else `other`."""
    lowered = text.lower()
    for label, needles in KEYWORD_RULES:
        if any(needle in lowered for needle in needles):
            return label
    return "other"


def event_for(row):
    """One `news_event.v1` carrying the calendar's own clocks, so the run is a historical replay and not a fresh stamp."""
    headline, body = release_line(row)
    stamp = release_datetime(row).isoformat()
    identity = digest({"template": TEMPLATE_VERSION, "row": {key: (row[key] or "").strip() for key in COLUMNS}})
    return {"schema": "news_event.v1", "event_id": f"calendar:{identity[:32]}", "source": SOURCE_NAME,
            "asset": ASSET, "language": "en", "published_at": stamp, "received_at": stamp,
            "headline": headline, "body": body}


def read_rows(csv_path):
    """Every row of the headerless calendar file, in file order, with the declared column names attached."""
    with Path(csv_path).open(newline="", encoding="utf-8", errors="strict") as stream:
        for index, fields in enumerate(csv.reader(stream)):
            if len(fields) != len(COLUMNS):
                continue
            yield index, {name: fields[position] for position, name in enumerate(COLUMNS)}


def source_digest(csv_path):
    import hashlib
    sha = hashlib.sha256()
    with Path(csv_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def candidates(csv_path):
    """Every usable row, with its class, its year and its release line, plus the counts of everything left out."""
    kept, counts = [], {"rows_read": 0, "wrong_field_count": 0, "excluded_member_state": 0,
                        "undeclared_country": 0, "unusable_row": 0}
    for index, row in read_rows(csv_path):
        counts["rows_read"] += 1
        try:
            label = map_country(row["country"])
        except CorpusError:
            counts["undeclared_country"] += 1
            continue
        if label == EXCLUDED:
            counts["excluded_member_state"] += 1
            continue
        try:
            event = event_for(row)
        except CorpusError:
            counts["unusable_row"] += 1
            continue
        kept.append({"row_index": index, "label": label, "year": release_datetime(row).year, "event": event})
    return kept, counts


def _collapse_duplicates(rows, counts):
    """One entry per distinct release line. A line that occurs under two different labels is dropped, not adjudicated."""
    by_text, order = {}, []
    for entry in rows:
        text = entry["event"]["body"]
        if text not in by_text:
            by_text[text] = []
            order.append(text)
        by_text[text].append(entry)
    unique, counts["duplicate_texts_collapsed"], counts["ambiguous_texts_dropped"] = [], 0, 0
    for text in order:
        group = by_text[text]
        labels = {entry["label"] for entry in group}
        if len(labels) > 1:
            counts["ambiguous_texts_dropped"] += len(group)
            continue
        counts["duplicate_texts_collapsed"] += len(group) - 1
        unique.append(group[0])
    return unique


def sample(rows, *, seed, rows_per_class):
    """A balanced, seeded, year-stratified draw: the same seed and the same file always produce the same rows.

    Years are visited in a fixed round-robin so no year can dominate a class, and within a year the order is a seeded
    shuffle whose seed is derived from the class and the year, so adding a class never moves another class's rows."""
    drawn = []
    for label in CLASSES:
        by_year = {}
        for entry in rows:
            if entry["label"] == label:
                by_year.setdefault(entry["year"], []).append(entry)
        for year, entries in by_year.items():
            random.Random(f"{seed}:{label}:{year}").shuffle(entries)
        years, picked, exhausted = sorted(by_year), [], set()
        while len(picked) < rows_per_class and len(exhausted) < len(years):
            for year in years:
                if len(picked) >= rows_per_class:
                    break
                if by_year[year]:
                    picked.append(by_year[year].pop())
                else:
                    exhausted.add(year)
        if len(picked) < rows_per_class:
            raise CorpusError(f"CLASS_TOO_SMALL: {label!r} has {len(picked)} usable rows, fewer than the declared "
                              f"{rows_per_class}; lower rows_per_class rather than unbalancing the corpus")
        drawn.extend(sorted(picked, key=lambda entry: entry["event"]["event_id"]))
    return drawn


def build_corpus(csv_path, *, seed=1729, rows_per_class=150):
    """The corpus and its manifest. Nothing here reads a clock, so two builds of the same file agree byte for byte."""
    if not isinstance(rows_per_class, int) or isinstance(rows_per_class, bool) or rows_per_class < 1:
        raise CorpusError("ROWS_PER_CLASS: a positive integer is required")
    rows, counts = candidates(csv_path)
    rows = _collapse_duplicates(rows, counts)
    items = sample(rows, seed=seed, rows_per_class=rows_per_class)
    per_class = {label: sum(1 for entry in items if entry["label"] == label) for label in CLASSES}
    per_year = {}
    for entry in items:
        per_year.setdefault(str(entry["year"]), {label: 0 for label in CLASSES})[entry["label"]] += 1
    corpus_id = digest([[entry["event"]["event_id"], entry["label"]] for entry in items])
    manifest = {
        "schema": CORPUS_SCHEMA,
        "corpus_id": corpus_id,
        "source": {"name": SOURCE_NAME, "file_name": Path(csv_path).name, "sha256": source_digest(csv_path),
                   "columns": list(COLUMNS)},
        "label_provenance": "INDEPENDENT_LABELS",
        "label_source": "economic calendar country field (dataset column 3)",
        "label_producer": "the economic calendar's compiler, distributed with the dataset",
        "label_statement": ("The label of every row is the calendar's own country field. No author of this system wrote, "
                            "reviewed or adjudicated any label, and no row was written for this test: this is "
                            "INDEPENDENT_LABELS, not AUTHOR_WRITTEN_SMOKE."),
        "mapping": dict(COUNTRY_TO_CLASS),
        "excluded_countries": dict(EXCLUDED_COUNTRIES),
        "exclusion_reason": ("A release by a euro-area member state has a contestable class under this mapping, so the "
                             "row would measure the definition rather than the model. Such rows are dropped and counted, "
                             "never relabelled."),
        "template": {"version": TEMPLATE_VERSION,
                     "headline": "the calendar description, stripped, truncated to 500 characters",
                     "body": ("Economic calendar release on {date} at {HH:MM} UTC. {description}: actual {actual}, "
                              "consensus {consensus}, previous {previous}. Expected volatility: {volatility}."),
                     "value_rule": "a value is quoted with its unit only when it matches ^-?\\d+(\\.\\d+)?$, else 'not reported'",
                     "consensus_column": "forecast",
                     "asset": ASSET},
        "question": {"name": QUESTION_NAME, "type": "choice", "instructions": QUESTION_INSTRUCTIONS,
                     "options": [list(option) for option in QUESTION_OPTIONS]},
        "keyword_baseline": {"rules": [[label, list(needles)] for label, needles in KEYWORD_RULES],
                             "order": list(CLASSES), "default": "other",
                             "statement": "the class whose country name occurs verbatim in the release line, else other"},
        "seed": seed,
        "rows_per_class": rows_per_class,
        "classes": list(CLASSES),
        "counts": dict(counts, sampled=len(items), per_class=per_class, per_year=per_year,
                       unique_candidate_texts=len(rows)),
    }
    return manifest, items
