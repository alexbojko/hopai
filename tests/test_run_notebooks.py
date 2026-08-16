"""
Tests for the notebook staleness checker (scripts/run_notebooks.py --check).

The regex that decides what counts as "just a timing" and the logic that
merges split stream output are the riskiest part of that checker -- get
either wrong and --check either misses a real regression or flakes on a
correct notebook. Pinned here rather than trusted by inspection alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_notebooks import (  # noqa: E402
    _coalesce_streams, _normalize_output, diff_notebooks, mask_timings,
)


class TestMaskTimings:
    def test_a_kwarg_style_timing_is_masked_keeping_the_key(self):
        assert mask_timings("Subgraph(nodes=4, edges=4, elapsed_ms=11.4)") \
            == "Subgraph(nodes=4, edges=4, elapsed_ms=<T>)"

    def test_an_inline_timing_with_a_space_is_masked(self):
        assert mask_timings("9.9 ms for all three queries") == "<T> for all three queries"

    def test_an_inline_timing_with_no_space_is_masked(self):
        assert mask_timings("done in 29.6ms") == "done in<T>"

    def test_explain_analyzes_execution_time_is_masked(self):
        assert mask_timings("Execution Time: 4.665 ms") == "Execution Time:<T>"

    def test_microseconds_are_masked(self):
        assert mask_timings("13.1 µs per call") == "<T> per call"

    def test_padding_in_a_formatted_table_is_consumed_with_the_number(self):
        """notebook 08's >8.1f-formatted column pads the same value to a
        different width depending on its digit count -- if the padding
        were left behind, two runs of an unchanged notebook would still
        diff on whitespace alone."""
        assert mask_timings("traverse : 9.3 ms") == "traverse :<T>"
        assert mask_timings("traverse :  12.3 ms") == "traverse :<T>"

    def test_the_sql_alias_near_0_dot_s_is_not_mistaken_for_a_timing(self):
        """The false positive this regex was fitted against: the digit in
        `near_0`, zero digits after a phantom decimal, then the "s" alias
        two tokens later reads like a bare-second timing to a naive
        \\d+\\.?\\d*\\s*(ms|s|...) pattern. A real query from this
        codebase, not a hypothetical."""
        assert mask_timings("near_0.s AS sim_0") == "near_0.s AS sim_0"

    def test_a_row_count_with_no_unit_is_untouched(self):
        assert mask_timings("Subgraph(nodes=4, edges=4)") == "Subgraph(nodes=4, edges=4)"

    def test_text_with_no_timing_is_returned_unchanged(self):
        assert mask_timings("no numbers here at all") == "no numbers here at all"


class TestCoalesceStreams:
    def _stream(self, text: str, name: str = "stdout") -> dict:
        return {"output_type": "stream", "name": name, "text": text}

    def test_adjacent_same_name_streams_merge_into_one(self):
        outputs = [self._stream("hello "), self._stream("world")]
        coalesced = _coalesce_streams(outputs)
        assert len(coalesced) == 1
        assert coalesced[0]["text"] == "hello world"

    def test_a_single_stream_output_is_unaffected(self):
        outputs = [self._stream("hello world")]
        assert _coalesce_streams(outputs) == outputs

    def test_streams_with_different_names_do_not_merge(self):
        outputs = [self._stream("out", name="stdout"), self._stream("err", name="stderr")]
        assert _coalesce_streams(outputs) == outputs

    def test_a_non_stream_output_between_two_streams_breaks_the_merge(self):
        result_output = {"output_type": "execute_result", "data": {"text/plain": "1"}}
        outputs = [self._stream("a"), result_output, self._stream("b")]
        coalesced = _coalesce_streams(outputs)
        assert coalesced == [self._stream("a"), result_output, self._stream("b")]

    def test_genuinely_different_adjacent_prints_still_differ_after_merging(self):
        """Coalescing must not hide a real content difference -- only the
        message-boundary split is noise, never the text itself."""
        before = _coalesce_streams([self._stream("4 nodes")])
        after = _coalesce_streams([self._stream("4 "), self._stream("nodes")])
        differently = _coalesce_streams([self._stream("5 nodes")])
        assert before == after
        assert before != differently


class TestNormalizeOutput:
    def test_execution_count_is_dropped(self):
        output = {"output_type": "execute_result", "execution_count": 7,
                  "data": {"text/plain": "1"}}
        assert "execution_count" not in _normalize_output(output)

    def test_stream_text_as_a_list_is_joined_then_masked(self):
        output = {"output_type": "stream", "name": "stdout", "text": ["9.9 ms fo", "r it"]}
        assert _normalize_output(output)["text"] == "<T> for it"


class TestDiffNotebooks:
    def _cell(self, source: str, outputs: list, tags: list | None = None) -> dict:
        cell = {"cell_type": "code", "source": source, "outputs": outputs}
        if tags:
            cell["metadata"] = {"tags": tags}
        return cell

    def _stream(self, text: str) -> dict:
        return {"output_type": "stream", "name": "stdout", "text": text}

    def test_identical_notebooks_produce_no_mismatches(self):
        cells = [self._cell("print(1)", [self._stream("1\n")])]
        nb = {"cells": cells}
        assert diff_notebooks(nb, nb, "nb.ipynb") == []

    def test_only_a_timing_difference_is_not_reported_stale(self):
        committed = {"cells": [self._cell("bench()", [self._stream("9.9 ms\n")])]}
        fresh = {"cells": [self._cell("bench()", [self._stream("11.2 ms\n")])]}
        assert diff_notebooks(committed, fresh, "nb.ipynb") == []

    def test_a_real_content_change_is_named_by_notebook_and_cell(self):
        committed = {"cells": [self._cell("Subgraph(...)", [self._stream("nodes=4\n")])]}
        fresh = {"cells": [self._cell("Subgraph(...)", [self._stream("nodes=99\n")])]}
        mismatches = diff_notebooks(committed, fresh, "01_quickstart.ipynb")
        assert len(mismatches) == 1
        assert "01_quickstart.ipynb cell 0" in mismatches[0]
        assert "nodes=4" in mismatches[0] and "nodes=99" in mismatches[0]

    def test_a_tagged_cell_is_skipped_even_when_its_output_changed(self):
        committed = {"cells": [self._cell("random_id()", [self._stream("id=1\n")],
                                          tags=["nbval-ignore-output"])]}
        fresh = {"cells": [self._cell("random_id()", [self._stream("id=2\n")],
                                      tags=["nbval-ignore-output"])]}
        assert diff_notebooks(committed, fresh, "nb.ipynb") == []

    def test_a_non_code_cell_is_never_compared(self):
        committed = {"cells": [{"cell_type": "markdown", "source": "# old title"}]}
        fresh = {"cells": [{"cell_type": "markdown", "source": "# new title"}]}
        assert diff_notebooks(committed, fresh, "nb.ipynb") == []

    def test_a_cell_count_mismatch_refuses_to_guess_a_pairing(self):
        committed = {"cells": [self._cell("a", [])]}
        fresh = {"cells": [self._cell("a", []), self._cell("b", [])]}
        mismatches = diff_notebooks(committed, fresh, "nb.ipynb")
        assert len(mismatches) == 1
        assert "committed has 1 cells" in mismatches[0] and "freshly executed has 2" in mismatches[0]

    def test_stream_message_splitting_alone_is_not_reported_stale(self):
        """The exact case #44 was filed about, at the diff_notebooks
        level rather than the coalescing helper alone: nbclient's
        message-boundary nondeterminism must not make --check flaky on
        an unchanged notebook."""
        committed = {"cells": [self._cell("print('4 nodes')", [self._stream("4 nodes\n")])]}
        fresh = {"cells": [self._cell("print('4 nodes')",
                                      [self._stream("4 "), self._stream("nodes\n")])]}
        assert diff_notebooks(committed, fresh, "nb.ipynb") == []
