"""
Tests for the benchmark report renderer.

A benchmark report is read once, believed, and quoted for months. These
pin the parts that would make it quietly misleading: bars that do not
reflect the numbers beside them, a headline that drifts from its own
table, a query that never finished being averaged in as a large number,
or a report that reads as if it were appended to rather than regenerated.

No database and no stopwatch -- render() is deterministic given its
inputs, which is why the report is built from data rather than printed
as it goes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from report import (  # noqa: E402
    UNKNOWN, bar, chart, empty_queries, findings, grouped_chart, headline,
    log_bar, machine_profile, overhead, render,
)

RESULTS = [
    {"id": "Q1", "query": "Forward 1-hop", "feature": "direction",
     "cold_ms": 100.0, "warm_ms": 50.0, "warm_min_ms": 48.0, "warm_max_ms": 52.0,
     "samples": 5, "raw_sql_ms": 10.0, "nodes": 10, "edges": 9},
    {"id": "Q2", "query": "Backward bounded", "feature": "multi-hop",
     "cold_ms": 400.0, "warm_ms": 200.0, "warm_min_ms": 195.0, "warm_max_ms": 205.0,
     "samples": 5, "raw_sql_ms": 25.0, "nodes": 2000, "edges": 1999},
    {"id": "Q3", "query": "GT range", "feature": "range",
     "cold_ms": 20.0, "warm_ms": 10.0, "warm_min_ms": 9.0, "warm_max_ms": 11.0,
     "samples": 5, "raw_sql_ms": 5.0, "nodes": 5, "edges": 0},
]
DNF = {"id": "Q4", "query": "Deep backward", "feature": "deep multi-hop",
       "dnf": True, "budget_s": 150.0, "cold_ms": 150000.0, "warm_ms": 150000.0,
       "nodes": None, "edges": None}
PROFILE = {"cpu": "Apple M1", "cores": 8, "memory": "16.0 GiB", "postgres": "16.6"}


class TestBars:
    def test_linear_bar_is_proportional(self):
        assert bar(50, 100, width=10).count("█") == 5

    def test_every_bar_is_the_same_width(self):
        assert {len(bar(v, 100, width=20)) for v in (0, 1, 50, 100)} == {20}

    def test_log_bar_keeps_small_values_visible(self):
        """The reason the chart is log scale: on a linear one, 1ms next
        to 60,000ms is an empty line, and the chart shows one query."""
        assert log_bar(1, 1, 60000, width=30).count("█") >= 1
        assert log_bar(60000, 1, 60000, width=30).count("█") == 30

    def test_log_bar_orders_by_magnitude(self):
        widths = [log_bar(v, 1, 10000, width=30).count("█")
                  for v in (1, 10, 100, 1000, 10000)]
        assert widths == sorted(widths)
        assert len(set(widths)) == 5      # each order of magnitude is distinguishable

    def test_log_bar_survives_degenerate_ranges(self):
        assert len(log_bar(5, 5, 5, width=8)) == 8
        assert len(log_bar(0, 1, 100, width=8)) == 8


class TestGroupedChart:
    SERIES = [("warm_ms", "hopai"), ("raw_sql_ms", "raw SQL")]

    def test_one_block_per_query_with_a_bar_per_system(self):
        body = grouped_chart(RESULTS, self.SERIES)
        for row in RESULTS:
            assert row["id"] in body and row["query"] in body
        assert body.count("hopai") == len(RESULTS)
        assert body.count("raw SQL") == len(RESULTS)

    def test_shows_the_feature_each_query_exercises(self):
        """Breadth is the point of the suite; a chart that only shows
        timings hides what was actually covered."""
        assert "[direction]" in grouped_chart(RESULTS, self.SERIES)
        assert "[range]" in grouped_chart(RESULTS, self.SERIES)

    def test_a_dnf_is_labelled_not_drawn_as_a_number(self):
        """Reporting non-completion as its slowest-observed time would
        let it average in with real measurements."""
        body = grouped_chart([*RESULTS, DNF], self.SERIES)
        assert "DNF (>150s)" in body
        assert "150,000.0 ms" not in body

    def test_no_measurements_says_so(self):
        assert "no measurements" in grouped_chart([], self.SERIES)


class TestHeadline:
    def test_names_the_slowest_and_fastest_query(self):
        body = "\n".join(headline(RESULTS))
        assert "Q2" in body and "slowest" in body
        assert "Q3" in body and "fastest" in body

    def test_counts_sub_second_queries(self):
        assert "3 / 3" in "\n".join(headline(RESULTS))

    def test_counts_and_names_queries_that_did_not_finish(self):
        body = "\n".join(headline([*RESULTS, DNF]))
        assert "**1**" in body and "`Q4`" in body

    def test_a_dnf_is_never_the_slowest_measurement(self):
        """It has no measurement. Letting it win 'slowest' would report a
        budget as a timing."""
        body = "\n".join(headline([*RESULTS, DNF]))
        assert "Q2" in body.split("slowest")[1][:40]

    def test_survives_a_run_where_everything_failed(self):
        assert headline([DNF])


class TestFindings:
    def test_names_where_the_library_layer_costs_most(self):
        body = "\n".join(findings(RESULTS))
        assert "Q2" in body and "8.0x" in body      # 200 / 25
        assert "next to the row count" in body

    def test_calls_out_queries_that_returned_nothing(self):
        empty = [{**RESULTS[0], "id": "Q9", "query": "dud", "nodes": 0}]
        body = "\n".join(findings(empty))
        assert "measured nothing" in body.lower() and "`dud`" in body

    def test_calls_out_non_completion_separately(self):
        body = "\n".join(findings([*RESULTS, DNF]))
        assert "Did not finish" in body and "150s budget" in body

    def test_calls_out_noisy_measurements(self):
        noisy = [{**RESULTS[0], "warm_min_ms": 10.0, "warm_max_ms": 100.0}]
        body = "\n".join(findings(noisy))
        assert "Noisy" in body and "not comparable across commits" in body

    def test_says_nothing_notable_rather_than_inventing_a_finding(self):
        body = "\n".join(findings(RESULTS))
        assert "Noisy" not in body and "Did not finish" not in body


class TestOverhead:
    def test_is_warm_over_raw(self):
        assert overhead(RESULTS[1]) == pytest.approx(8.0)

    def test_is_none_when_the_floor_was_not_measured(self):
        assert overhead({"warm_ms": 10.0}) is None
        assert overhead({"raw_sql_ms": 10.0}) is None


class TestRender:
    @pytest.fixture()
    def body(self) -> str:
        return render([*RESULTS, DNF], PROFILE,
                      {"nodes": 1_000_000, "edges": 818_695},
                      generated_at="2026-01-01 00:00 UTC")

    def test_has_every_section_in_order(self, body):
        order = ["# hopai benchmark", "## 01 — Headline", "## 02 — Every query",
                 "## 03 — Full results", "## 04 — Findings", "## 05 — Environment"]
        positions = [body.index(section) for section in order]
        assert positions == sorted(positions)

    def test_says_it_is_regenerated_not_appended(self, body):
        assert "Rewritten on every run" in body and "Do not edit by hand" in body

    def test_carries_the_machine_profile(self, body):
        for value in ("Apple M1", "16.0 GiB", "16.6"):
            assert value in body

    def test_carries_the_dataset_size_with_separators(self, body):
        assert "1,000,000" in body and "818,695" in body

    def test_carries_every_query_and_its_feature(self, body):
        for row in RESULTS:
            assert row["id"] in body and row["feature"] in body

    def test_states_how_warm_was_measured(self, body):
        assert "median of 5 runs" in body

    def test_is_deterministic(self):
        first = render(RESULTS, PROFILE, generated_at="fixed")
        assert first == render(RESULTS, PROFILE, generated_at="fixed")

    def test_an_empty_run_still_renders(self):
        assert "no measurements" in render([], PROFILE, generated_at="fixed")

    def test_the_raw_sql_column_appears_only_when_measured(self):
        without = [{k: v for k, v in r.items() if k != "raw_sql_ms"} for r in RESULTS]
        assert "Raw SQL" not in render(without, PROFILE, generated_at="fixed")
        assert "Raw SQL" in render(RESULTS, PROFILE, generated_at="fixed")


class TestMachineProfile:
    def test_reports_the_real_machine(self):
        profile = machine_profile()
        assert profile["cores"] and profile["cores"] != UNKNOWN
        assert profile["python"].count(".") == 2
        assert profile["architecture"] != UNKNOWN

    def test_degrades_instead_of_raising_without_a_connection(self):
        assert machine_profile(None)["postgres"] == UNKNOWN

    def test_memory_is_human_readable_or_unknown(self):
        memory = machine_profile()["memory"]
        assert memory == UNKNOWN or memory.endswith("GiB")


class TestLegacyHelpers:
    """chart() and empty_queries() still back the single-series views."""

    def test_chart_is_slowest_first(self):
        assert chart(RESULTS, "warm_ms").splitlines()[0].startswith("Backward")

    def test_empty_queries_finds_zero_row_results(self):
        assert empty_queries([{**RESULTS[0], "query": "dud", "nodes": 0}]) == ["dud"]
