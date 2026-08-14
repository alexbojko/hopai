"""
Tests for the benchmark report renderer.

A benchmark report is read once, believed, and quoted for months. These
pin the parts that would make it quietly misleading: a chart whose bars
do not reflect the numbers beside them, a missing machine profile, or a
report that reads as if it were appended to rather than regenerated.

No database, no stopwatch -- render() is deterministic given its inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from report import (  # noqa: E402
    UNKNOWN, bar, chart, empty_queries, machine_profile, render,
)

RESULTS = [
    {"query": "forward_1hop", "cold_ms": 100.0, "warm_ms": 50.0, "nodes": 10, "edges": 9},
    {"query": "backward_bounded_3hop", "cold_ms": 400.0, "warm_ms": 200.0,
     "nodes": 2000, "edges": 1999},
    {"query": "range_gt", "cold_ms": 20.0, "warm_ms": 10.0, "nodes": 5, "edges": 0},
]
PROFILE = {"cpu": "Apple M1", "cores": 8, "memory": "16.0 GiB", "postgres": "16.6"}


class TestBar:
    def test_the_peak_fills_the_width(self):
        assert bar(100, 100, width=10) == "█" * 10

    def test_length_is_proportional_to_the_value(self):
        assert bar(50, 100, width=10).count("█") == 5
        assert bar(25, 100, width=10).count("█") == 2   # 2.5 rounds to 2

    def test_every_bar_is_the_same_total_width(self):
        """Ragged right edges make a chart unreadable at a glance."""
        assert {len(bar(v, 100, width=20)) for v in (0, 1, 50, 99, 100)} == {20}

    def test_a_tiny_nonzero_value_still_shows(self):
        """Rounding 0.4 to an empty bar would render a real measurement
        as if nothing had been measured."""
        assert bar(0.4, 1000, width=40).startswith("█")

    def test_zero_and_degenerate_peaks_do_not_divide_by_zero(self):
        assert bar(0, 100, width=5) == "░" * 5
        assert bar(5, 0, width=5) == "░" * 5


class TestChart:
    def test_slowest_first(self):
        lines = chart(RESULTS, "warm_ms").splitlines()
        assert lines[0].startswith("backward_bounded_3hop")
        assert lines[-1].startswith("range_gt")

    def test_the_number_is_printed_beside_the_bar(self):
        """The bar is for shape; the number is the measurement. A chart
        without both invites eyeballing a ratio off a terminal."""
        assert "200.0 ms" in chart(RESULTS, "warm_ms")
        assert "50.0 ms" in chart(RESULTS, "warm_ms")

    def test_labels_are_aligned(self):
        starts = {line.index("█") if "█" in line else line.index("░")
                  for line in chart(RESULTS, "warm_ms").splitlines()}
        assert len(starts) == 1

    def test_the_longest_bar_belongs_to_the_largest_number(self):
        lines = chart(RESULTS, "cold_ms").splitlines()
        counts = [line.count("█") for line in lines]
        assert counts == sorted(counts, reverse=True)

    def test_no_measurements_says_so(self):
        assert "no measurements" in chart([], "warm_ms")


class TestEmptyQueries:
    """A traversal that returns nothing did no work. Its timing is real
    and meaningless, and it will always look like the fastest query in
    the run -- so the report must not let it pass as one."""

    def test_finds_queries_that_matched_nothing(self):
        results = RESULTS + [{"query": "forward_1hop_broken", "warm_ms": 1.0,
                              "cold_ms": 2.0, "nodes": 0, "edges": 0}]
        assert empty_queries(results) == ["forward_1hop_broken"]

    def test_a_healthy_run_flags_nothing(self):
        assert empty_queries(RESULTS) == []

    def test_the_chart_marks_them_inline(self):
        results = [{"query": "empty", "warm_ms": 1.0, "nodes": 0, "edges": 0}]
        assert "NO ROWS" in chart(results, "warm_ms")

    def test_the_report_warns_at_the_top(self):
        results = RESULTS + [{"query": "dud", "warm_ms": 1.0, "cold_ms": 1.0,
                              "nodes": 0, "edges": 0}]
        body = render(results, PROFILE, generated_at="fixed")
        assert "Measured nothing" in body and "`dud`" in body
        assert "not how fast the query is" in body


class TestRender:
    @pytest.fixture()
    def body(self) -> str:
        return render(RESULTS, PROFILE, {"nodes": 1_000_000, "edges": 1_799_431},
                      generated_at="2026-01-01 00:00 UTC")

    def test_says_it_is_regenerated_not_appended(self, body):
        """Someone will otherwise hand-edit it, and the next run will
        silently throw their edit away."""
        assert "rewritten on every run" in body
        assert "Do not edit it by hand" in body

    def test_carries_the_machine_profile(self, body):
        for value in ("Apple M1", "16.0 GiB", "16.6"):
            assert value in body

    def test_carries_the_dataset_size_with_separators(self, body):
        """`1000000` and `1,000,000` are the same number and not the same
        at a glance."""
        assert "1,000,000" in body and "1,799,431" in body

    def test_has_both_cold_and_warm_charts(self, body):
        assert "## Warm latency" in body and "## Cold latency" in body
        assert body.count("```") == 4        # two fenced charts

    def test_has_a_table_of_every_measurement(self, body):
        for row in RESULTS:
            assert f"`{row['query']}`" in body

    def test_is_deterministic(self):
        first = render(RESULTS, PROFILE, generated_at="fixed")
        assert first == render(RESULTS, PROFILE, generated_at="fixed")

    def test_an_empty_run_still_renders(self):
        """A benchmark that measured nothing must produce a report saying
        so, not a traceback."""
        body = render([], PROFILE, generated_at="fixed")
        assert "no measurements" in body

    def test_the_timestamp_is_shown(self, body):
        assert "2026-01-01 00:00 UTC" in body


class TestMachineProfile:
    def test_reports_the_real_machine(self):
        profile = machine_profile()
        assert profile["cores"] and profile["cores"] != UNKNOWN
        assert profile["python"].count(".") == 2
        assert profile["architecture"] != UNKNOWN

    def test_degrades_instead_of_raising_without_a_connection(self):
        """The report is worth having even where a probe is unavailable;
        a missing CPU model must not cost you the measurements."""
        assert machine_profile(None)["postgres"] == UNKNOWN

    def test_memory_is_human_readable_or_unknown(self):
        memory = machine_profile()["memory"]
        assert memory == UNKNOWN or memory.endswith("GiB")
