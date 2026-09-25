"""WP09: the corpus is what it declares, and a number that did not come from the model never reaches a report.

Four things are worth failing over here. The corpus must be reproducible from its declarations alone, or the manifest is
decoration. It must be balanced and stratified, or macro-F1 is a statement about the sample. A country nobody declared
must be refused instead of swept into `other`, or the corpus grows silently whenever the source file does. And a run
answered by the declared fixture must abort by name, because fixture output is indistinguishable from a measurement once
it is in a table. The metric helpers are checked against numbers computed by hand, so a rewrite that changes a definition
fails here rather than in a report.
"""

import json

import pytest

from news_signal.quality_corpus import (CLASSES, EXCLUDED, TEMPLATE_VERSION, CorpusError, build_corpus, event_for,
                                        keyword_prediction, map_country, release_line)
from news_signal.quality_eval import FixtureRun, brier, check_model_answer, closure_table, reliability, skill_from_scores


def row(country, description, *, date="2015/03/04", time="12:30:00", volatility="High Volatility Expected",
        unit="% ", actual="1.5", forecast="1.2", previous="1.1"):
    return {"event_date": date, "event_time": time, "country": country, "volatility": volatility,
            "description": description, "evaluation": "", "data_format": unit,
            "actual": actual, "forecast": forecast, "previous": previous}


def write_calendar(path, rows):
    """A headerless calendar in the source file's own column order, quoted and padded the way the real one is."""
    lines = []
    for entry in rows:
        fields = [entry["event_date"], entry["event_time"], f'"{entry["country"]:<32}"', f'"{entry["volatility"]:<32}"',
                  f'"{entry["description"]:<112}"', f'"{entry["evaluation"]}"', f'"{entry["data_format"]}"',
                  entry["actual"], entry["forecast"], entry["previous"]]
        lines.append(",".join(fields))
    path.write_text("\n".join(lines) + "\n")
    return path


def calendar_of(tmp_path, per_class=40, years=(2011, 2012, 2013)):
    """Enough rows per declared class and year that a balanced, stratified draw is possible without exhausting a year."""
    tags = {"Euro Zone": "EZ", "United States": "US", "United Kingdom": "UK"}
    rows = []
    for index in range(per_class):
        for year in years:
            for country, tag in tags.items():
                rows.append(row(country, f"Indicator {tag} {year} {index}",
                                date=f"{year}/03/04", actual=f"{index}.5", forecast=f"{index}.2",
                                previous=f"{index}.1"))
    return write_calendar(tmp_path / "calendar.csv", rows)


# --- the declarations ---------------------------------------------------------------------------------------------

def test_mapping_refuses_a_country_it_never_declared():
    with pytest.raises(CorpusError) as caught:
        map_country("Narnia")
    assert "UNDECLARED_COUNTRY" in str(caught.value)
    assert "Narnia" in str(caught.value)


def test_member_states_are_excluded_by_name_and_never_relabelled():
    for member in ("Germany", "France", "Italy", "Spain"):
        assert map_country(member) == EXCLUDED
    assert map_country("Euro Zone") == "euro_area"
    assert map_country("United States") == "united_states"
    assert map_country(" United Kingdom  ") == "other"


def test_the_release_line_is_the_declared_template_and_names_no_country():
    headline, body = release_line(row("Euro Zone", "Core CPI"))
    assert headline == "Core CPI"
    assert body == ("Economic calendar release on 2015-03-04 at 12:30 UTC. Core CPI: actual 1.5%, consensus 1.2%, "
                    "previous 1.1%. Expected volatility: High.")
    assert "Euro Zone" not in body


def test_an_unparseable_value_becomes_not_reported_rather_than_a_parsing_accident():
    _, body = release_line(row("Euro Zone", "Current Account", previous='vised From 12.0B>12.4 "'))
    assert "previous not reported" in body


def test_the_event_carries_the_calendar_clock_so_the_run_is_a_replay():
    event = event_for(row("United States", "Nonfarm Payrolls"))
    assert event["published_at"] == event["received_at"] == "2015-03-04T12:30:00+00:00"
    assert event["schema"] == "news_event.v1" and event["language"] == "en"
    assert event["event_id"].startswith("calendar:")


def test_two_rows_differing_only_in_country_get_different_identities():
    assert event_for(row("Euro Zone", "CPI"))["event_id"] != event_for(row("United States", "CPI"))["event_id"]


# --- the sample ---------------------------------------------------------------------------------------------------

def test_the_corpus_is_deterministic_under_its_seed(tmp_path):
    path = calendar_of(tmp_path)
    first, first_items = build_corpus(path, seed=1729, rows_per_class=12)
    second, second_items = build_corpus(path, seed=1729, rows_per_class=12)
    assert first == second
    assert [entry["event"]["event_id"] for entry in first_items] == [entry["event"]["event_id"] for entry in second_items]
    other, _ = build_corpus(path, seed=4104, rows_per_class=12)
    assert other["corpus_id"] != first["corpus_id"]
    assert other["counts"]["per_class"] == first["counts"]["per_class"]


def test_the_corpus_is_balanced_and_stratified_over_years(tmp_path):
    manifest, items = build_corpus(calendar_of(tmp_path), seed=1729, rows_per_class=12)
    assert manifest["counts"]["per_class"] == {label: 12 for label in CLASSES}
    assert len(items) == 36
    per_year = manifest["counts"]["per_year"]
    assert sorted(per_year) == ["2011", "2012", "2013"]
    for counts in per_year.values():
        assert counts == {label: 4 for label in CLASSES}


def test_the_manifest_declares_everything_needed_to_rebuild_it(tmp_path):
    manifest, _ = build_corpus(calendar_of(tmp_path), seed=1729, rows_per_class=12)
    assert manifest["seed"] == 1729 and manifest["rows_per_class"] == 12
    assert manifest["source"]["sha256"] and manifest["template"]["version"] == TEMPLATE_VERSION
    assert manifest["label_provenance"] == "INDEPENDENT_LABELS"
    assert "AUTHOR_WRITTEN_SMOKE" in manifest["label_statement"]
    assert manifest["mapping"]["Euro Zone"] == "euro_area"
    assert set(manifest["excluded_countries"]) == {"France", "Germany", "Italy", "Spain"}
    assert manifest["question"]["instructions"] == "Which economy is named in this news?"
    json.dumps(manifest)                                     # the manifest is written as JSON; it must be serializable


def test_a_class_smaller_than_the_request_is_refused_rather_than_unbalanced(tmp_path):
    with pytest.raises(CorpusError) as caught:
        build_corpus(calendar_of(tmp_path, per_class=2), seed=1729, rows_per_class=50)
    assert "CLASS_TOO_SMALL" in str(caught.value)


def test_a_line_carrying_two_labels_is_dropped_and_one_repeated_under_one_label_is_kept_once(tmp_path):
    shared = dict(description="Shared Indicator", date="2011/03/04")
    rows = [row("Euro Zone", **shared), row("United States", **shared),
            row("United Kingdom", description="UK Only", date="2011/03/04"),
            row("United Kingdom", description="UK Only", date="2011/03/04"),
            row("Euro Zone", description="EZ Only", date="2011/03/04"),
            row("United States", description="US Only", date="2011/03/04")]
    path = write_calendar(tmp_path / "dup.csv", rows)
    manifest, items = build_corpus(path, seed=1729, rows_per_class=1)
    assert manifest["counts"]["ambiguous_texts_dropped"] == 2      # the line under two labels, both occurrences
    assert manifest["counts"]["duplicate_texts_collapsed"] == 1    # the line repeated under one label, kept once
    assert sorted(entry["label"] for entry in items) == ["euro_area", "other", "united_states"]
    assert all("Shared Indicator" not in entry["event"]["body"] for entry in items)


# --- the gate -----------------------------------------------------------------------------------------------------

def test_a_fixture_backend_aborts_the_run_by_name():
    with pytest.raises(FixtureRun) as caught:
        check_model_answer({"status": "OK", "backend": "fixture", "label": "other"}, row="calendar:1")
    assert "FIXTURE_BACKEND_REFUSED" in str(caught.value) and "calendar:1" in str(caught.value)


def test_an_answer_the_provider_marked_as_fixture_aborts_even_when_it_claims_the_real_backend():
    with pytest.raises(FixtureRun) as caught:
        check_model_answer({"status": "OK", "backend": "laya", "non_model_fixture": True}, row="calendar:2")
    assert "NON_MODEL_FIXTURE" in str(caught.value)


def test_a_refused_answer_is_not_silently_counted_as_a_prediction():
    with pytest.raises(FixtureRun) as caught:
        check_model_answer({"status": "REFUSED", "why": "STALE_NEWS"}, row="calendar:3")
    assert "ANSWER_NOT_OK" in str(caught.value)
    with pytest.raises(FixtureRun):
        check_model_answer(None, row="calendar:4")


def test_a_real_answer_passes_through_unchanged():
    answer = {"status": "OK", "backend": "laya", "label": "euro_area"}
    assert check_model_answer(answer, row="calendar:5") is answer


# --- the metrics, against numbers computed by hand -----------------------------------------------------------------

def test_reliability_bins_and_expected_calibration_error_match_a_hand_computation():
    # four rows: two at 0.95 (one correct), two at 0.45 (both correct)
    result = reliability([(0.95, True), (0.95, False), (0.45, True), (0.45, True)])
    top = result["bins"][9]
    assert top["count"] == 2 and top["accuracy"] == 0.5 and top["mean_confidence"] == pytest.approx(0.95)
    middle = result["bins"][4]
    assert middle["count"] == 2 and middle["accuracy"] == 1.0 and middle["mean_confidence"] == pytest.approx(0.45)
    # 0.5 * |0.5 - 0.95| + 0.5 * |1.0 - 0.45| = 0.225 + 0.275 = 0.5
    assert result["expected_calibration_error"] == pytest.approx(0.5)
    assert sum(entry["count"] for entry in result["bins"]) == 4


def test_a_probability_of_exactly_one_lands_in_the_last_bin():
    result = reliability([(1.0, True)])
    assert result["bins"][9]["count"] == 1


def test_the_multiclass_brier_score_matches_a_hand_computation():
    classes = ("a", "b", "c")
    # (0.7-1)^2 + (0.2)^2 + (0.1)^2 = 0.09 + 0.04 + 0.01 = 0.14
    # (0.5)^2 + (0.5-1)^2 + 0 = 0.25 + 0.25 = 0.50 ; mean = 0.32
    score = brier([{"a": 0.7, "b": 0.2, "c": 0.1}, {"a": 0.5, "b": 0.5, "c": 0.0}], ["a", "b"], classes)
    assert score == pytest.approx(0.32)


def test_brier_refuses_a_vector_that_does_not_cover_the_declared_classes():
    with pytest.raises(Exception) as caught:
        brier([{"a": 1.0}], ["a"], ("a", "b"))
    assert "BRIER_LABELS" in str(caught.value)


def test_skill_is_the_declared_ratio_of_errors_and_undefined_against_a_perfect_naive():
    computed = skill_from_scores(0.75, 0.5)
    assert computed["model_error"] == pytest.approx(0.25) and computed["naive_error"] == pytest.approx(0.5)
    assert computed["skill"] == pytest.approx(0.5) and computed["status"] == "OK"
    assert skill_from_scores(0.9, 1.0)["status"].startswith("UNDEFINED")
    assert skill_from_scores(0.9, 1.0)["skill"] is None
    negative = skill_from_scores(0.2, 0.5)
    assert negative["skill"] == pytest.approx(-0.6)          # worse than the naive is a negative skill, never clipped


def test_the_keyword_baseline_reads_only_the_country_name_it_declares():
    assert keyword_prediction("Euro Zone CPI rose") == "euro_area"
    assert keyword_prediction("United States Nonfarm Payrolls") == "united_states"
    assert keyword_prediction("Core CPI: actual 1.5%") == "other"
    # declared order decides a line naming both, and the decision is reproducible rather than dict-order luck
    assert keyword_prediction("United States and Euro Zone") == "euro_area"


def test_the_closure_table_carries_every_column_the_owner_requires():
    table = closure_table([{"arm": "laya_zero_shot", "metric": "macro_f1", "scale": "three classes", "n": 450,
                            "model_error": 0.4, "naive_name": "majority_class, same rows", "naive_error": 0.5,
                            "skill": 0.2, "comparability": "COMPARABLE"}], title="t")
    for column in ("model error", "naive reference", "naive error", "skill", "literature value", "comparability"):
        assert column in table
    assert "0.400000" in table and "NOT_CARRIED" in table
    assert "error = 1 - macro_f1" in table
