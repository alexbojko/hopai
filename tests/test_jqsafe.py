"""
The jq subset hopai will accept from a model, and everything it refuses.

DATABASE-FREE on purpose: jqsafe.py builds no SQL and takes no
connection, so this file runs with nothing running -- the same property
that makes the query-shape tests the fastest check in the suite.

Two tests here are worth more than the rest put together, because they
are the ones that hold the module's two claims:

  * TestTheSubsetIsRealJq compiles every accepted program with libjq.
    hopai parses only to REFUSE -- what it accepts is handed to libjq
    verbatim -- so accepting something libjq then rejects would move a
    refusal from validation time to ranking time, one row at a time.
  * TestAcceptedProgramsTerminate spot-checks totality by RUNNING the
    corpus. The claim is structural (no recursion, no generator, no
    loop, no fold), but a claim nothing exercises is a comment.

`jq` is an optional, test-only import -- hopai itself has no new
dependency and never will -- so those two classes skip without it.

The corpus is a module-level TUPLE. mutmut runs the whole suite twice in
one process and shared MUTABLE module state has broken its baseline in
this project before; a tuple cannot be one of those.
"""

from __future__ import annotations

import contextlib
import json
import random
import sys
import time

import pytest

from hopai.jqsafe import (
    MAX_ADDED, MAX_DEPTH, MAX_GROWTH, MAX_LENGTH, UnsafeFilter, is_total, paths_read,
    validate,
)
from hopai.jqsafe import _Parser, _size, _tokenize


def static_bound(program: str):
    """What jqsafe PROMISES about a program's size: `_size()`'s
    `out <= factor * in + extra`, in characters of compact JSON.

    Reached through the parser rather than through validate() because
    the interesting cases are the ones validate() refuses -- the number
    in the refusal is the claim being checked."""
    return _size(_Parser(_tokenize(program, "test"), "test").program())

#: Every construct the subset admits, in the spellings a document
#: projection actually uses. Shared by the allowed-constructs tests, the
#: differential test and the termination test, so a construct added to
#: the grammar without being added here is a construct nothing checks
#: against real jq.
ACCEPTED = (
    ".",
    ".properties",
    ".properties.title",
    ".properties.title?",
    '."quoted key"',
    '.properties."odd key".title',
    ".properties.tags[]",
    ".properties.tags[0]",
    ".properties.tags[-1]",
    ".properties.tags[1:3]",
    ".properties.tags[:2]",
    ".properties.tags[1:]",
    ".properties.tags[]?",
    ".properties.tags[0].name",
    ".[]",
    ".[0]",
    ".properties | .title",
    ".properties.title, .properties.summary",
    "(.properties.title)",
    '.properties.title + ": " + (.properties.summary // "")',
    '[.properties.title, .properties.summary] | join(" -- ")',
    '.properties.tags | map(ascii_downcase) | sort | unique | join(", ")',
    '"title: \\(.properties.title)"',
    '"\\(.properties.title) (\\(.properties.n))"',
    '"escapes: \\" \\\\ \\/ \\n \\t \\u00e9"',
    '.properties | select(.type == "paper") | .title',
    '.properties | select(.type != "draft" and .n >= 1) | .title',
    ".properties.n - 1",
    ".properties.n + 1",
    ".properties.n > 1 and .properties.n < 9",
    '.properties.n == 3 or .properties.type == "paper"',
    ".properties.title | length | tostring",
    ".properties.title | ascii_upcase",
    ".properties.title | not",
    '.properties | has("title")',
    ".properties.tags | first",
    ".properties.tags | last",
    ".properties.tags | reverse",
    ".properties.tags | flatten(1)",
    ".properties.tags | add",
    '.properties.title | split(" ") | join("-")',
    '.properties.title | startswith("Ra")',
    '.properties.title | endswith("ft")',
    '.properties.title | ltrimstr("Ra")',
    '.properties.title | rtrimstr("ft")',
    ".properties | values",
    ".properties.tags[] | strings",
    ".properties.tags[] | numbers",
    ".properties.tags | arrays",
    ".properties | objects",
    ".properties | tojson",
    '.properties.n | tostring | tonumber',
    ".properties.title | type",
    "empty",
    "true",
    "false",
    "null",
    "1 + 2",
    "1.5e3",
    "[]",
    '[.properties.title] | length',
    ".properties.title // .properties.name // \"untitled\"",
    "# what this filter is for\n.properties.title",
    ".properties.tags | first(.[])",
    # A string INSIDE an interpolation: the scanner that finds the
    # closing `)` has to skip it the way jq does, or the `)` inside the
    # nested string would end the interpolation early.
    '"title: \\(.properties.title + " (draft)")"',
    "[.properties.title, .properties.summary] | .[0]",
    # A comment INSIDE an interpolation: where the extent of `\(...)`
    # is decided, and where a comment-blind scanner let a `)` in a
    # comment end the expression early. See
    # TestACommentCannotSmuggleCodeIntoAnInterpolation.
    '"title: \\(.properties.title # which field\n)"',
    '"\\(.properties.title) # not a comment, string text"',
    # A surrogate PAIR, which libjq combines into one character. The
    # halves are refused on their own (TestCharactersThatNeverReachLibjq);
    # the pair has to keep working, and has to mean the same character
    # here as it does in libjq or the field allowlist matches nothing.
    '."\\ud83d\\ude00"',
    '.properties.summary | split(" ") | .[0:30] | join(" ")',
)


class TestTheAllowedSubset:
    @pytest.mark.parametrize("program", ACCEPTED)
    def test_every_allowed_construct_parses(self, program):
        """The corpus is what the differential and termination tests run
        on; a construct that stops parsing here silently stops being
        checked against libjq there."""
        validate(program)
        assert is_total(program) is True

    def test_a_quoted_key_with_a_dot_in_it_stays_one_segment(self):
        """`."a.b"` is ONE property whose name contains a dot, not two.
        Splitting the source on "." -- the tempting shortcut -- would
        report two paths and let an allowlist match the wrong one."""
        assert paths_read('."a.b"') == frozenset({"a.b"})

    def test_a_comment_is_not_a_construct(self):
        """jq comments run to end of line. Treating `#` as an unknown
        character would refuse a filter an operator documented."""
        assert is_total("# doc\n.properties.title") is True

    @pytest.mark.parametrize("program", ["1.a", '1."k"', "1[0]", "1?", "2.5.a"])
    def test_a_number_takes_no_suffix(self, program):
        """libjq's LEXER takes the dot: it reads `1.a` as the number `1.`
        followed by a name and reports a syntax error. Reading it as
        "field a of 1" was the only shape in 37,000 fuzz-generated
        accepted programs that libjq refused -- and every instance of
        that fuzz run's divergence."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    def test_a_backslash_in_a_comment_is_refused(self):
        """jq 1.7 continues a comment onto the NEXT line when it ends
        with `\\`; jq 1.6 does not. `# c\\<newline>.properties.title` is
        therefore an empty program on 1.7 -- libjq answers "Top-level
        program not given" -- while a to-end-of-line reader sees
        `.properties.title`. Refusing the backslash is what keeps this
        module's comment rule identical to every jq's."""
        with pytest.raises(UnsafeFilter) as refused:
            validate("# c \\\n.properties.title")
        assert "backslash inside a comment" in str(refused.value)


class TestForbiddenConstructs:
    """Everything here is refused because the grammar has no production
    for it -- not because its name appears on a list. `_EXPLAIN` only
    decides WHICH sentence comes back."""

    @pytest.mark.parametrize("program", [
        "env",
        "env.HOME",
        "env|keys",
        "$ENV",
        "$ENV.HOME",
    ])
    def test_the_process_environment_is_unreachable(self, program):
        """Measured: `env|keys` returns 133 entries including every
        credential the process holds, and this filter's OUTPUT is posted
        to a third-party reranker. This is the exfiltration the module
        exists to stop."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    @pytest.mark.parametrize("program", [
        "def f: f; f",
        "def f(x): x; f(.)",
        "while(true; .)",
        "until(. > 1; . + 1)",
        "repeat(.)",
        "recurse",
        "recurse(.a)",
        "..",
        "..|.a",
        "range(10)",
        "[range(100000000)]",
        "limit(3; .[])",
    ])
    def test_nothing_that_can_fail_to_terminate_parses(self, program):
        """`def f: f; f` holds the GIL forever -- a watchdog thread
        cannot print and SIGINT is ignored, so only SIGKILL ends it --
        and `[range(100000000)]` makes libjq abort() the process, which
        no Python `except` catches. Neither is survivable at runtime, so
        neither may reach libjq."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    @pytest.mark.parametrize("program", [
        "reduce .[] as $x (0; . + $x)",
        "foreach .[] as $x (0; . + $x)",
        ". as $x | $x",
        "label $out | break $out",
        "$__loc__",
    ])
    def test_folds_bindings_and_non_local_exits_are_out(self, program):
        """No variables means `$ENV` and `$__loc__` have nowhere to
        appear -- they are excluded by the absence of a construct, which
        is stronger than excluding them by name."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    @pytest.mark.parametrize("program", [
        'include "x";',
        'import "x" as y;',
        "modulemeta",
    ])
    def test_module_loading_is_out(self, program):
        """Module loading is the ONE thing that would let a filter call a
        name it does not spell -- which is the assumption the whole
        static analysis rests on."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    @pytest.mark.parametrize("program", [
        'getpath(["properties", "ssn"])',
        'setpath(["a"]; 1)',
        "paths",
        "leaf_paths",
        "to_entries",
        "with_entries(.value)",
        "tostream",
        ".[.key]",
        '.[$k]',
    ])
    def test_computed_paths_are_out(self, program):
        """A computed path is DATA, so no static check can see which
        property it reads. Accepting one would make `fields=` decorative
        -- `getpath(["properties","ssn"])` reads a field the allowlist
        never names."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    @pytest.mark.parametrize("program", [
        'test("re")',
        'match("re")',
        'capture("(?<x>re)")',
        'sub("a"; "b")',
        'gsub("a"; "b")',
        'scan("a")',
        'splits("a")',
        'split("a"; "g")',
    ])
    def test_the_regex_family_is_out(self, program):
        """A regex from a model is a regex hopai would have to run, and
        catastrophic backtracking is the same unkillable hang as
        `def f: f; f`. `split/2` is refused separately from `split/1`
        because the second argument is what makes it a regex."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    @pytest.mark.parametrize("program", [
        "input",
        "inputs",
        "debug",
        "input_line_number",
        "builtins",
        "@base64d",
        '@base64 "\\(.a)"',
        "implode",
        "explode",
    ])
    def test_everything_that_is_not_this_row_is_out(self, program):
        """`input` would put ANOTHER row's data in this row's document,
        and `@base64d` turns opaque bytes into text nothing validated --
        one layer below every check here."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    @pytest.mark.parametrize("program", [
        "if .a then .b else .c end",
        "if .a then .b end",
        "try .a catch .b",
        ".a?//.b",
        "{a: .b}",
        "{(.k): .v}",
        ".a = 1",
        ".a |= 1",
        ".a += 1",
        ".a //= 1",
    ])
    def test_the_remaining_jq_syntax_is_out(self, program):
        """`catch` binds jq's error MESSAGE -- which quotes the offending
        input value -- straight into a document posted to a vendor, and
        an assignment writes the row a projection is only meant to
        read."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    @pytest.mark.parametrize("program", [
        '"x" * 1000000000',
        ".a * 2",
        ".a / .b",
        ".a % 2",
        ".title / \", \"",
    ])
    def test_multiplication_and_division_are_out(self, program):
        """`"x" * 1000000000` is an unbounded memory amplifier written in
        eleven characters, and `/` on strings is a second spelling of
        `split`. Neither has a role in building a document."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    @pytest.mark.parametrize("program", [
        "sort_by(.a)",
        "group_by(.a)",
        "min_by(.a)",
        "any",
        "all",
        "keys",
        "keys_unsorted",
        "del(.a)",
        "walk(.)",
        "getpath",
        "ascii",
        "splits",
        "now",
    ])
    def test_a_function_outside_the_allowlist_is_refused(self, program):
        """The allowlist is the enforcement; `_EXPLAIN` only chooses the
        sentence. A name with no entry there must still be refused --
        otherwise the table would be the security boundary, and a table
        of names is exactly the arms race this module avoids."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    @pytest.mark.parametrize("program", [
        "",
        "   ",
        ".a.b[",
        '"unclosed',
        '"\\(.a"',
        '"\\(.a',
        "}",
        "and .a",
        ".a | or .b",
        ".a +",
        "(.a",
        ".a == .b == .c",
        '.["key"]',
        ".[:]",
        "join",
        "ltrimstr",
        "not(.a)",
        "tostring(.a)",
        "&",
        '"\\q"',
        '"\\u00zz"',
        '."\\(.k)"',
    ])
    def test_a_filter_that_does_not_parse_is_refused(self, program):
        """A parse hopai cannot complete is a filter hopai cannot reason
        about, so it is refused rather than passed through and hoped
        for. `.[:]`, `not(.a)` and `.a == .b == .c` are here because
        LIBJQ rejects them: accepting them would move the failure from
        validation to ranking, one row at a time."""
        with pytest.raises(UnsafeFilter):
            validate(program)


class TestNamesAreTokensNotSubstrings:
    """The reason this module has a tokenizer instead of a substring
    scan. Every case here is a false positive a blacklist produces."""

    @pytest.mark.parametrize("program, expected", [
        (".environment", {"environment"}),
        (".env_var", {"env_var"}),
        (".env", {"env"}),
        (".properties.environment", {"properties.environment"}),
        ('."env"', {"env"}),
        (".definition", {"definition"}),
        (".range", {"range"}),
        (".input.recurse", {"input.recurse"}),
    ])
    def test_a_property_that_spells_a_forbidden_name_is_fine(self, program, expected):
        """`.env` is a property of the row; `env` is jq's builtin. A
        scan for the string "env" refuses both, which would make
        perfectly ordinary graphs unrankable."""
        validate(program)
        assert paths_read(program) == frozenset(expected)

    def test_a_forbidden_name_inside_a_string_is_text(self):
        """`"env"` is data being concatenated into a document, not a
        call. It reads nothing at all."""
        validate('.properties.title + " env "')
        assert paths_read('"env" + "getpath" + "def f: f; f"') == frozenset()

    def test_a_string_cannot_smuggle_a_call(self):
        """jq has NO dynamic dispatch -- `"e" + "nv"` is a string, and
        there is no way to call the builtin it spells. That fact is what
        makes an allowlist sound rather than optimistic, so it is worth
        one test that the string form is accepted and reads nothing."""
        assert paths_read('"e" + "nv"') == frozenset()


class TestErrorsNameTheFix:
    def test_env_says_what_to_write_instead(self):
        """CLAUDE.md rule 3: an error names the fix. A model that gets
        back "invalid filter" retries with another invalid filter."""
        with pytest.raises(UnsafeFilter) as refused:
            validate("env.SECRET")
        message = str(refused.value)
        assert message.startswith("document_from: `env` is not available (offset 0)")
        assert ".properties.title" in message

    def test_the_owner_names_the_option_the_filter_arrived_on(self):
        """The caller wrote `document_from=` or an MCP argument, not
        `jqsafe`. Naming the module in the message would point at the
        wrong file."""
        with pytest.raises(UnsafeFilter) as refused:
            validate("env", owner="rerank.document_from")
        assert str(refused.value).startswith("rerank.document_from:")

    def test_an_offset_points_at_the_construct(self):
        """"Somewhere in this filter" is not a location. The offset is
        what lets a caller -- human or model -- see which part of a long
        projection was the problem."""
        with pytest.raises(UnsafeFilter) as refused:
            validate('.properties.title + " " + env.HOME')
        assert "(offset 26)" in str(refused.value)

    def test_an_offset_inside_an_interpolation_points_into_the_program(self):
        """An interpolation is parsed by a second parser over a slice of
        the source. Reporting its LOCAL offset would name a position the
        caller cannot count to."""
        with pytest.raises(UnsafeFilter) as refused:
            validate('"who: \\(env.USER)"')
        assert "(offset 8)" in str(refused.value)

    def test_an_unknown_function_lists_the_ones_that_work(self):
        """A model cannot guess an allowlist. Naming it turns one
        refusal into a working retry."""
        with pytest.raises(UnsafeFilter) as refused:
            validate("sort_by(.a)")
        message = str(refused.value)
        assert "ascii_downcase" in message and "join" in message and "tostring" in message

    def test_the_wrong_arity_says_so(self):
        """`split("a"; "g")` is the regex form. Refusing it as "unknown
        function" would tell the caller to stop using `split`, which is
        allowed -- the second argument is the problem."""
        with pytest.raises(UnsafeFilter) as refused:
            validate('split("a"; "g")')
        assert "at most one argument" in str(refused.value)

    @pytest.mark.parametrize("program, offset, reason", [
        (".a * 2", 3, "unbounded memory amplifier"),
        (".a / 2", 3, "jq's `/` is splitting"),
        (".a % 2", 3, "arithmetic beyond `+` and `-`"),
        (".a; .b", 2, "separates the arguments of a multi-argument function"),
        (":", 0, "appears only inside a slice"),
        ("-1", 0, "a leading `-` is not in the subset"),
    ])
    def test_a_token_with_a_known_reason_gets_that_reason(self, program, offset, reason):
        """The operators `_UNEXPECTED` names each have a DIFFERENT fix --
        `*` says concatenate, `/` says `split`, `:` says slice -- and
        that lookup is the whole point of the table. Without this the
        suite only asserted THAT these are refused, so dropping the
        lookup (every token falling through to the generic "cannot
        appear here" text), looking the reason up under the wrong key,
        or inverting the `why is None` test so the two messages swap
        places all passed in silence. The `startswith` pins the rest of
        the refusal's shape at the same time: the owner the caller
        passed, the offset of the offending token, and the "this is not
        in the subset" wording -- each of which has vanished under a
        mutant while every test still went green."""
        with pytest.raises(UnsafeFilter) as refused:
            validate(program, owner="rerank.document_from")
        message = str(refused.value)
        assert message.startswith(
            f"rerank.document_from: this is not in the subset (offset {offset}) -- "
        )
        assert reason in message
        assert "cannot appear here" not in message

    def test_a_token_with_no_known_reason_gets_the_generic_one(self):
        """`]` is not in `_UNEXPECTED`, so it takes the other branch, and
        a model that sent one needs to be told what a projection is built
        FROM rather than nothing at all. Without this the generic branch
        was unpinned: it could name no owner, no offset, no token, or --
        with the `why is None` test inverted -- report a `None` reason,
        and the suite still passed because it only checked that `]` was
        refused. The reason is three adjacent literals, so the anchors
        deliberately straddle their joins -- a substring sitting wholly
        inside one piece leaves the other two free to be rewritten."""
        with pytest.raises(UnsafeFilter) as refused:
            validate(".a ]", owner="rerank.document_from")
        message = str(refused.value)
        assert message.startswith(
            "rerank.document_from: `]` cannot appear here (offset 3) -- "
            "the filter does not parse in the jq subset hopai accepts"
        )
        assert "accepts -- a document is built from properties" in message
        assert message.endswith("string literals, `+`, `//` and the allowed functions")
        assert "this is not in the subset" not in message


class TestPathsRead:
    @pytest.mark.parametrize("program, expected", [
        (".properties.title", {"properties.title"}),
        ('.properties.title + ": " + (.properties.summary // "")',
         {"properties.title", "properties.summary"}),
        (".properties.tags[]", {"properties.tags"}),
        (".properties.tags[0].name", {"properties.tags.name"}),
        (".properties.tags[1:3]", {"properties.tags"}),
        (".properties | .title", {"properties.title"}),
        (".properties.title?", {"properties.title"}),
        ('"\\(.a.b) and \\(.c)"', {"a.b", "c"}),
        ("[.a, .b] | join(\" \")", {"a", "b"}),
        (".properties.tags | map(.name) | join(\",\")",
         {"properties.tags", "properties.tags.name"}),
        ('.properties | select(.type == "x") | .title',
         {"properties.type", "properties.title"}),
        ('"a literal"', set()),
        ("1 + 2", set()),
    ])
    def test_the_paths_a_filter_reads(self, program, expected):
        """`fields=` is enforced on exactly this set, so a path this
        misses is a property that leaves the database without anyone
        having allowed it."""
        assert paths_read(program) == frozenset(expected)

    def test_the_whole_row_is_reported_as_a_dot(self):
        """`.` emits the entire row -- every property, including the ones
        an allowlist withheld. Reporting an EMPTY set for it would make
        the most dangerous filter in the language look like the safest."""
        assert paths_read(".") == frozenset({"."})

    def test_a_pass_through_function_still_reports_what_it_emits(self):
        """`first` reads nothing itself, but `.properties.ssn | first`
        emits the ssn. Attributing reads only at the point a value is
        CONSUMED, and never at the point it is emitted, missed this."""
        assert paths_read(".properties.ssn | first") == frozenset({"properties.ssn"})

    def test_navigation_on_the_left_of_a_pipe_is_not_itself_a_read(self):
        """`.properties | .title` reads properties.title. Reporting a
        bare `properties` too would fail an allowlist that names the leaf
        -- the correct, narrow spelling an operator would write."""
        assert paths_read(".properties | .title") == frozenset({"properties.title"})

    def test_a_path_off_a_derived_value_is_attributed_to_its_source(self):
        """`(.a + .b).c` has no top-level path of its own. Reporting `c`
        -- as if it were a row property -- would let an allowlist for `c`
        pass a filter that actually reads `a` and `b`."""
        assert paths_read("(.a + .b).c") == frozenset({"a", "b"})

    def test_a_path_after_a_derived_value_adds_nothing_new(self):
        """`[.a, .b] | .[0]` walks into an array this filter BUILT, not
        into the row. Attributing `.[0]` to the row -- or crashing on a
        context that no longer names a property -- would both be wrong;
        the reads are the two fields that went into the array."""
        assert paths_read("[.a, .b] | .[0]") == frozenset({"a", "b"})


class TestTheFieldAllowlist:
    def test_an_allowed_field_passes(self):
        """The operator-side allowlist has to admit the filter it was
        written for, or it is just a refusal."""
        validate(".properties.title", fields=["properties.title"])

    @pytest.mark.parametrize("program", [
        ".properties.tags[]",
        ".properties.tags[0]",
        ".properties.tags[0].name",
        '.properties.tags | join(", ")',
        ".properties.tags | map(.name) | first",
    ])
    def test_a_path_beneath_an_allowed_one_is_allowed(self, program):
        """`properties.tags[]` is within `properties.tags` -- it is that
        field's own data. Requiring an exact string match would refuse
        every way of actually using an allowed array."""
        validate(program, fields=["properties.tags"])

    def test_a_sibling_is_refused_and_the_message_names_what_is_allowed(self):
        """A refusal that does not say what IS allowed cannot be acted
        on -- by a model or by the operator reading the log."""
        with pytest.raises(UnsafeFilter) as refused:
            validate(".properties.ssn", fields=["properties.title", "properties.summary"])
        message = str(refused.value)
        assert "properties.ssn" in message
        assert "properties.title, properties.summary" in message

    def test_a_parent_of_an_allowed_field_is_refused(self):
        """`.properties` hands back the siblings -- ssn included -- that
        naming `properties.title` was meant to withhold. Allowing a read
        ABOVE an allowed path would make the allowlist trivially
        bypassable by deleting one segment."""
        with pytest.raises(UnsafeFilter):
            validate(".properties", fields=["properties.title"])

    def test_the_whole_row_is_refused_by_name(self):
        """`.` is the one-character version of the same bypass, and it
        deserves a message that says so rather than "reads `..`"."""
        with pytest.raises(UnsafeFilter) as refused:
            validate(".", fields=["properties.title"])
        assert "`.` reads the whole row" in str(refused.value)
        assert "properties.title" in str(refused.value)

    def test_an_allowed_field_may_be_written_with_a_leading_dot(self):
        """Inside the filter the path is `.properties.title`; an operator
        writing the allowlist should not have to know which side wants
        which spelling."""
        validate(".properties.title", fields=[".properties.title"])

    def test_no_allowlist_means_the_operator_decides(self):
        """`fields=None` is the operator's own filter, which is trusted
        with the row it already owns. Refusing there would leave no way
        to write a projection at all."""
        validate(".", fields=None)

    def test_a_read_inside_an_interpolation_is_checked(self):
        """An interpolation is where a second filter hides. Checking only
        top-level paths would let `"\\(.properties.ssn)"` through."""
        with pytest.raises(UnsafeFilter):
            validate('"x \\(.properties.ssn)"', fields=["properties.title"])

    def test_a_non_string_field_is_a_type_error_not_a_refusal(self):
        """A malformed allowlist is the OPERATOR's bug, not a model's
        unsafe filter, and conflating the two would have an application
        catching UnsafeFilter to find its own typo."""
        with pytest.raises(TypeError):
            validate(".a", fields=[1])

    def test_an_empty_path_names_the_way_to_allow_everything(self):
        """`fields=[""]` reads as "allow nothing" or "allow everything"
        depending on the reader. Refusing and naming `fields=None` is
        the only answer that cannot be silently the wrong one."""
        with pytest.raises(ValueError):
            validate(".a", fields=["   "])


class TestGrowthIsBounded:
    def test_repeated_concatenation_is_refused_with_its_factor(self):
        """Totality bounds how MANY values a filter emits, not how big
        they are: `(.+.)` doubles, and doubling once per pipe stage is
        exponential in the length of the PROGRAM. A short filter could
        otherwise ask for more memory than the process has."""
        program = "(.+.)|" * 7 + "."
        with pytest.raises(UnsafeFilter) as refused:
            validate(program)
        message = str(refused.value)
        assert "128 times its own input" in message
        assert f"{MAX_GROWTH}x" in message

    @pytest.mark.parametrize("program", [
        '.properties.tags | join("\\(.)")',
        ".properties.tags | join(.properties.sep)",
        ".properties | has(.key)",
        ".properties.title | split(.properties.sep)",
        ".properties.title | ltrimstr(.properties.prefix)",
        ".properties.tags | flatten(.properties.depth)",
    ])
    def test_a_separator_or_needle_taken_from_the_row_is_refused(self, program):
        """`join` emits its separator once per ELEMENT, so a separator
        interpolating the row grows with the SQUARE of the row -- 9,899
        characters out of a 50-element array, measured -- which the
        growth factor, counting multiples of the input, cannot see. And
        a computed needle is the same invisible argument that keeps
        `getpath` out."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    def test_a_literal_separator_still_works(self):
        """The refusal above must not take `join(", ")` with it -- that
        is the whole reason `join` is in the allowlist."""
        validate('[.properties.title, .properties.summary] | join(" -- ")')

    def test_a_real_projection_is_nowhere_near_the_cap(self):
        """The cap has to be invisible to the filters this feature is
        for -- a document concatenates a handful of fields."""
        validate('.properties.title + ": " + .properties.summary + " " '
                 '+ (.properties.tags | join(", "))')

    def test_the_split_join_bomb_is_refused_and_counted_honestly(self):
        """The amplifier a separator charged as `1 + growth(argument)`
        could not see. `split("")` makes one element per CHARACTER of a
        published field and `join(sep)` writes `sep` between every pair,
        so one stage multiplies a title by len(sep) -- 311 here -- while
        the old arithmetic charged 2 for it however long the separator
        was. Six stages fit inside MAX_LENGTH: reported 64, exactly
        MAX_GROWTH, and ACCEPTED; the real factor is 311**6, about 9e14.

        Measured live before the fix: 20 documents totalling 1,084,831
        bytes POSTed to a provider out of 7-8 character titles, and with
        three stages one document from a FOUR-character title ran past
        45 seconds inside libjq -- no exception, no abort(), and a
        SIGALRM handler that never ran, because one evaluation hands the
        GIL back to nobody. Nothing downstream can measure a document
        libjq is still building, so this has to be refused HERE."""
        separator = "S" * 311
        program = ".properties.title" + f'|split("")|join("{separator}")' * 6
        assert len(program) <= MAX_LENGTH
        with pytest.raises(UnsafeFilter) as refused:
            validate(program, fields=["properties.title"])
        assert f"{MAX_GROWTH}x" in str(refused.value)
        # Honest means "not below what really happens": an upper bound
        # is the point, so over-reporting is fine and 64 was not.
        assert static_bound(program).factor >= 311 ** 6

    @pytest.mark.parametrize("program", [
        '.properties.summary | split(" ") | .[0:30] | join(" ")',
        '.properties.tags | join(", ")',
        '.properties.tags | join("-")',
        '[.properties.title, .properties.summary] | join(" -- ")',
        '.properties.title | split(" ") | join("-")',
        '.properties.tags | map(ascii_downcase) | unique | join(", ")',
        '.properties.title + ": " + (.properties.summary // "")',
        '.properties.body | split("\\n") | .[0:3] | join(" / ")',
    ])
    def test_the_filters_people_actually_write_still_pass(self, program):
        """Charging the separator as a multiplier must not take the
        truncation idiom with it. `split(" ") | .[0:30] | join(" ")` is
        the ordinary way to cap a summary at thirty words, and a review
        of 25 realistic projections found zero refusals before this
        change -- a regression here costs the feature, not an attacker."""
        validate(program)

    def test_a_separator_costs_its_own_length(self):
        """The arithmetic, stated as a test: `join(sep)` over n elements
        emits about len(sep) * n characters, and n is bounded by the
        input, so len(sep) is the multiplier. A flat 2 for every literal
        separator is what the bomb above was built out of."""
        assert static_bound('.tags | join(" ")').factor == 2
        assert static_bound('.tags | join(", ")').factor == 3
        assert static_bound('.tags | join(" -- ")').factor == 5
        assert static_bound(f'.tags | join("{"x" * 300}")').factor == 301
        # A number is a literal too, and jq writes it out the same way:
        # the charge is the characters it prints as, not a flat 1.
        assert static_bound(".tags | join(37)").factor == 3

    def test_a_slice_credits_the_separator_back(self):
        """`.[0:30]` bounds the ELEMENT COUNT whatever the row holds, so
        the separators cost at most 29 * len(sep) characters -- a
        constant out of the program's own text rather than a multiple of
        the row. Without that credit this pair would be identical and a
        long separator after a slice would be refused for an
        amplification it cannot produce."""
        long_separator = "x" * 300
        with pytest.raises(UnsafeFilter):
            validate(f'.properties.tags | join("{long_separator}")')
        validate(f'.properties.tags[0:30] | join("{long_separator}")')
        assert static_bound(
            f'.properties.tags[0:30] | join("{long_separator}")').factor == 1

    def test_a_literal_multiplied_by_a_slice_is_still_refused(self):
        """The credit is not a hole: a slice long enough turns the same
        program text back into an amplifier, and `extra` is where that
        shows up. `.[0:1000000] | join("<300 characters>")` is 300MB of
        separator from 30 characters of program."""
        with pytest.raises(UnsafeFilter) as refused:
            validate(f'.properties.tags[0:1000000] | join("{"x" * 300}")')
        assert str(MAX_ADDED) in str(refused.value)

    @pytest.mark.parametrize("program", [
        ".properties.tags[] + .properties.tags[]",
        '.properties.title | split("") | [.[] + .[]]',
        ".properties.tags[] - .properties.tags[]",
        '.properties.tags[] + " " + .properties.title',
        ".properties.tags | map(.[] + .[])",
    ])
    def test_a_stream_on_both_sides_of_an_operator_is_refused(self, program):
        """The second amplifier of the same family, found while fixing
        the first. jq's binaries run over the CARTESIAN PRODUCT of their
        two streams -- `(.a,.b) + (.c,.d)` emits four values -- so
        `[.[] + .[]]` squares an array's length, and
        `split("") | [.[]+.[]] | [.[]+.[]]` measured 70,001 characters
        out of a TEN-character field while an arithmetic that counted
        only sizes charged 2 per stage. The row squared is not a
        multiple of the row, so the refusal says that instead of naming
        a number."""
        with pytest.raises(UnsafeFilter) as refused:
            validate(program)
        assert "SQUARE of the row" in str(refused.value)

    def test_a_constant_emitted_once_per_element_is_a_factor(self):
        """`map(f)` runs f once per ELEMENT, so f's constant is emitted
        once per element and the element count is bounded by the input
        -- which makes it a multiplier, exactly as `join`'s separator
        is. Charging it as a constant would leave
        `map(["<", .] | join(""))` looking free."""
        assert static_bound('.tags | map(. + "abc")').factor >= 4
        assert static_bound(f'.tags | map(. + "{"x" * 300}")').factor > MAX_GROWTH

    def test_a_pipe_of_plain_steps_does_not_accumulate_growth(self):
        """A pipe MULTIPLIES its sides' factors, so a step that neither
        grows nor shrinks must count as exactly 1. Charging a function
        for its own ARGUMENTS made `ltrimstr("x") | join(", ")` look
        like 4x, and a chain of six ordinary steps then tripped a cap
        meant for exponential concatenation."""
        validate('.properties.tags[] | ascii_downcase | ltrimstr("a") | rtrimstr("z") '
                 '| split(" ") | reverse | unique | sort | flatten(1) | join(", ")')


class TestLimits:
    def test_a_filter_longer_than_the_limit_is_refused(self):
        """A projection is one expression over one row. The cheapest way
        to bound the parser's work is to bound its input, and the
        message has to say which limit was hit rather than fail to
        parse somewhere in the middle."""
        with pytest.raises(UnsafeFilter) as refused:
            validate('."' + "x" * (MAX_LENGTH + 10) + '"')
        assert str(MAX_LENGTH) in str(refused.value)

    def test_deep_nesting_is_refused_before_python_runs_out_of_stack(self):
        """A recursive-descent parser meeting `((((...))))` raises
        RecursionError -- which is not an UnsafeFilter, so a caller
        catching UnsafeFilter would crash on a filter that is merely
        silly."""
        program = "(" * (MAX_DEPTH + 5) + "." + ")" * (MAX_DEPTH + 5)
        with pytest.raises(UnsafeFilter) as refused:
            validate(program)
        assert str(MAX_DEPTH) in str(refused.value)

    @pytest.mark.parametrize("frames", [0, 400, 800])
    def test_the_depth_refusal_holds_from_a_DEEP_caller_stack(self, frames):
        """MAX_DEPTH is a promise about the FILTER; RecursionError is
        about the PROCESS, and the frames already on the stack when
        validate() is called are not ours. A caller inside a web
        framework, a test runner or a mutation harness starts far closer
        to Python's ceiling, so without reserved headroom the guard is
        reached only AFTER Python gives up -- and RecursionError is not
        an UnsafeFilter, so `except UnsafeFilter` crashes on a filter
        that is merely silly.

        Mutation testing found this: its mutants tree runs the parser
        from a deeper stack than pytest alone, and this exact filter
        raised RecursionError there while passing here."""
        program = "(" * (MAX_DEPTH + 5) + "." + ")" * (MAX_DEPTH + 5)

        def deeper(n):
            if n:
                return deeper(n - 1)
            with pytest.raises(UnsafeFilter):
                validate(program)
            return True

        assert deeper(frames)

    def test_the_recursion_limit_is_put_back(self):
        """Headroom is borrowed for ONE parse of a MAX_LENGTH-bounded
        filter, not lifted for the process: a validator that quietly
        raised the ceiling and left it raised would turn someone else's
        runaway recursion into a segfault instead of an exception."""
        before = sys.getrecursionlimit()
        with pytest.raises(UnsafeFilter):
            validate("(" * (MAX_DEPTH + 5) + "." + ")" * (MAX_DEPTH + 5))
        assert sys.getrecursionlimit() == before
        validate(".properties.title")
        assert sys.getrecursionlimit() == before

    def test_nested_interpolations_are_refused_by_the_scanner_too(self):
        """The parser's depth guard runs too late for these: measuring
        where each `\\(...)` ENDS is done by two scanners that recurse
        into each other, before a single token exists. 399 nested
        interpolations fit inside MAX_LENGTH and raised RecursionError
        out of a function whose whole contract is UnsafeFilter or
        nothing."""
        program = '"' + '\\("' * 399 + ".a" + '")' * 399 + '"'
        assert len(program) <= MAX_LENGTH
        with pytest.raises(UnsafeFilter) as refused:
            validate(program)
        assert str(MAX_DEPTH) in str(refused.value)

    def test_a_non_string_filter_is_a_type_error(self):
        """`document_from=['.a']` is a caller mistake with a different
        fix than an unsafe filter, so it gets a different exception."""
        with pytest.raises(TypeError):
            validate([".a"])


class TestIsTotal:
    def test_it_answers_yes_for_the_subset(self):
        """`is_total` is the parse, so anything validate() accepts it
        must accept -- two answers to one question is a defect."""
        assert is_total(".properties.title") is True

    @pytest.mark.parametrize("program", ["def f: f; f", "recurse", "range(3)", "..", ""])
    def test_it_answers_no_without_raising(self, program):
        """It is a predicate, so a caller can branch on it. Letting the
        UnsafeFilter escape would make every call site a try/except."""
        assert is_total(program) is False

    def test_it_does_not_apply_the_field_allowlist(self):
        """Totality is about termination, not about which properties a
        filter may read -- a filter can be perfectly total and still
        read the wrong field. Conflating them would make `is_total`
        answer a question nobody asked."""
        assert is_total(".properties.ssn") is True


class TestTheSubsetIsRealJq:
    """hopai parses only to refuse: what it accepts goes to libjq
    verbatim. So the subset must be a genuine SUBSET."""

    @pytest.mark.parametrize("program", ACCEPTED)
    def test_libjq_compiles_everything_this_module_accepts(self, program):
        """Accepting something libjq rejects moves the failure from one
        validation call to every candidate row at ranking time -- and
        the caller would see a jq syntax error from a filter hopai had
        already called safe."""
        jq = pytest.importorskip("jq")
        validate(program)
        jq.compile(program)


class TestACommentCannotSmuggleCodeIntoAnInterpolation:
    """The hole this class exists for, and the reason a second scanner
    is never allowed to grow back.

    `_scan_paren` decides how far a `\\(...)` interpolation extends --
    which half of the filter is an EXPRESSION and which half is inert
    text. It used to count parentheses while knowing nothing about `#`
    comments, so a `)` written inside a comment closed the interpolation
    here and not in libjq. Everything after that point was "string text"
    this module never tokenized, while libjq read on to the real `)` and
    ran it. Confirmed live: `env.FAKE_SECRET` came back as
    `sk-leak-me`, and the same shape reached `def f: f; f` (only SIGKILL
    ended it) and `[range(100000000)]` (SIGABRT). SOUNDNESS and TOTALITY
    were both false for these inputs -- the analysis was being applied
    to a program that was not the one that ran.

    The payloads here are never executed. They are asserted refused."""

    @pytest.mark.parametrize("program", [
        '"\\(.properties.a # )\n|env.FAKE_SECRET)"',
        '"\\(.properties.a # )\n|$ENV.FAKE_SECRET)"',
        '"a\\(.properties.a)b\\(null # )\n// $ENV.HOME)c"',
        '"\\(.a # )\n|env)"',
    ])
    def test_the_environment_payloads_are_refused(self, program):
        """The exfiltration this module exists to stop, wearing a
        comment. `env` is refused when it is INSIDE the interpolation --
        which it is, once the extent is measured the way libjq measures
        it."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    @pytest.mark.parametrize("program", [
        '"\\(. # )\ndef f: f; f)"',
        '"\\(. # )\n|[range(100000000)])"',
        '"\\(. # )\n|recurse)"',
        '"\\(. # )\n|reduce .[] as $x (0; .+$x))"',
    ])
    def test_the_non_terminating_payloads_are_refused_never_run(self, program):
        """`def f: f; f` holds the GIL with SIGINT ignored and
        `[range(100000000)]` aborts the process -- neither is survivable,
        so this test asserts the refusal and never evaluates them."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    def test_the_allowlist_bypass_is_refused(self):
        """`"\\(null # )\\n// .properties.ssn)"` returned the ssn while
        paths_read() reported NOTHING -- the allowlist was not bypassed,
        it was never consulted, because the read was hidden in what this
        module thought was string text."""
        with pytest.raises(UnsafeFilter) as refused:
            validate('"\\(null # )\n// .properties.ssn)"',
                     fields=["properties.a"])
        assert "properties.ssn" in str(refused.value)

    def test_the_smuggled_tail_is_part_of_the_program(self):
        """The direct regression: what came after the comment's `)` is
        now tokenized and reported. This assertion returned an EMPTY set
        while the bug was live."""
        assert paths_read('"\\(null # )\n// .properties.ssn)"') == \
            frozenset({"properties.ssn"})

    def test_a_comment_that_swallows_the_closing_paren_is_refused(self):
        """The benign direction of the same confusion: in `"\\(1 # x)"`
        the comment eats `)"` and libjq reports an unterminated
        interpolation. Accepting it fails safe but still breaks the
        genuine-subset claim the whole design rests on."""
        with pytest.raises(UnsafeFilter) as refused:
            validate('"\\(1 # x)"')
        assert "never closed" in str(refused.value)

    def test_a_comment_inside_an_interpolation_still_works(self):
        """The fix must not refuse the ordinary case -- an interpolation
        spanning lines with a note in it is legal jq, and libjq and this
        module must agree on what it means."""
        jq = pytest.importorskip("jq")
        program = '"title: \\(.properties.title # which one\n)"'
        validate(program, fields=["properties.title"])
        assert paths_read(program) == frozenset({"properties.title"})
        assert jq.compile(program).input(
            {"properties": {"title": "Raft", "ssn": "123"}}).all() == ["title: Raft"]


#: One value per field, so that what a filter EMITS can be traced back to
#: the fields it read. Array elements carry the array's own path, which
#: is the convention paths_read() uses: an index is not a segment.
SENTINEL_ROW = {
    "properties": {
        "title": "S:properties.title",
        "summary": "S:properties.summary",
        "type": "S:properties.type",
        "name": "S:properties.name",
        "n": 3,
        "tags": ["S:properties.tags:0", "S:properties.tags:1"],
        "odd key": {"title": "S:properties.odd key.title"},
        "ssn": "S:properties.ssn",
    },
    "quoted key": "S:quoted key",
    "a": {"b": "S:a.b"},
    "c": "S:c",
}


def _sentinels(value, out: dict):
    """Every sentinel in SENTINEL_ROW, mapped to the path it sits at."""
    if isinstance(value, dict):
        for item in value.values():
            _sentinels(item, out)
    elif isinstance(value, list):
        for item in value:
            _sentinels(item, out)
    elif isinstance(value, str) and value.startswith("S:"):
        out[value] = value[2:].split(":")[0]
    return out


class TestPathsReadAgreesWithWhatLibjqRuns:
    """The property the security posture actually needs, and the one
    nothing tested while the interpolation bug was live: not "libjq
    compiles this too", but "libjq MEANS the same thing".

    Compilability is too weak a check -- every payload in the class above
    was valid jq that compiled and ran. So this evaluates each accepted
    program against a row whose every field carries its own sentinel, and
    asserts that every field the OUTPUT proves was touched is covered by
    what paths_read() reported. A validator that misreads where an
    expression ends fails here, because libjq's answer contains a
    sentinel the validator never named."""

    @staticmethod
    def _covered(path: str, reported) -> bool:
        return any(named == "." or path == named or path.startswith(named + ".")
                   for named in reported)

    def test_every_field_that_reaches_the_output_was_reported(self):
        """`fields=` is enforced on paths_read()'s answer, so a field
        that reaches a vendor without appearing there is a field nobody
        allowed. Run over the whole corpus in one test, with a floor on
        how many sentinels were actually observed, so the check cannot
        quietly become vacuous."""
        jq = pytest.importorskip("jq")
        known = _sentinels(SENTINEL_ROW, {})
        observed = 0
        for program in ACCEPTED:
            reported = paths_read(program)
            try:
                answer = jq.compile(program).input(SENTINEL_ROW).all()
            except ValueError:
                continue                    # a runtime type error emits nothing
            printed = json.dumps(answer)
            for sentinel, path in known.items():
                if sentinel not in printed:
                    continue
                observed += 1
                assert self._covered(path, reported), (
                    f"{program!r} emitted {path} but paths_read() reported {set(reported)}")
        assert observed > 20, "the corpus stopped exercising the sentinels"

    def test_the_check_can_fail(self):
        """A test that cannot fail proves nothing. `.properties.ssn`
        emits a sentinel that a report of `properties.title` does not
        cover, which is exactly the shape of the bug this guards."""
        assert not self._covered("properties.ssn", frozenset({"properties.title"}))
        assert self._covered("properties.ssn", frozenset({"."}))
        assert self._covered("properties.tags", frozenset({"properties"}))


#: Stages a document projection is built out of, weighted toward the
#: two that manufacture size: `split` makes one element per character,
#: `join` writes its separator between every pair of them. The long
#: separators are the whole point -- a checker over 6,000 generated
#: programs found 50 places where the old arithmetic under-reported,
#: and every one of them was a `join` with a long literal separator.
GROWTH_STAGES = (
    'split("")', 'split(" ")', 'split("a")',
    'join("")', 'join(" ")', 'join(", ")', 'join("-")', 'join(" -- ")',
    f'join("{"Z" * 40}")', f'join("{"Q" * 120}")',
    ".[0:3]", ".[0:30]", ".[1:2]", ".[2:]", ".[:4]", ".[-2:]", ".[]",
    "map(ascii_downcase)", 'map(. + "xy")', "map(tostring)", 'map(split(""))',
    'map(["<", .] | join(""))',
    '. + "tail"', "ascii_downcase", "sort", "unique", "reverse", "tostring",
    "tojson", "add", "flatten", "length", "first", "last", ".[0]",
    '"pre \\(.) post"', '[., .] | join("~")', "select(. != null)",
    # Refused today (a stream on both sides of `+` is a cartesian
    # product). Generated anyway, so that loosening that refusal without
    # doing the arithmetic makes this property test fail rather than
    # quietly stop covering the shape.
    "[.[] + .[]]", '.[] + "x"', "[.[], .[]]",
)

GROWTH_STARTS = (".t", ".tags", ".n", ".", ".tags[]", "[.t, .n]")

#: Deliberately SMALL, because the bound is stated against the input:
#: a big row would make every ratio comfortable and the property
#: vacuous. Compact JSON, because that is the tightest reading of "how
#: many characters is this value" -- the measurement must not be
#: flattered by whitespace it did not have to emit.
GROWTH_ROW = {"t": "alpha beta gamma delta", "n": 7, "tags": ["x", "yy", "zzz", ""]}


def _measure(value) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False))


class TestGrowthIsNeverUnderReported:
    """The property the growth cap needs and that no example test can
    establish: for every program hopai ACCEPTS, what libjq really emits
    fits inside the bound hopai computed.

    `_size()` promises `out <= factor * in + extra` in characters of
    compact JSON. A single number below what actually happens is the
    whole bug class -- the 6-stage `split("")`/`join()` bomb reported 64
    while multiplying the row by 9e14 -- so this generates the shapes
    that amplify, RUNS them, and compares."""

    def test_the_bound_holds_against_what_libjq_emits(self):
        """A statically computed factor nothing measures is a comment.
        Run against the old arithmetic this finds 16 violations in this
        corpus -- `split("") | map(split(""))`, and every `join` with a
        separator longer than one character."""
        jq = pytest.importorskip("jq")
        row_size = _measure(GROWTH_ROW)
        generator = random.Random(20260817)
        checked, seen = 0, set()
        for _ in range(1500):
            program = generator.choice(GROWTH_STARTS) + "".join(
                " | " + generator.choice(GROWTH_STAGES)
                for _ in range(generator.randint(1, 4)))
            if program in seen:
                continue
            seen.add(program)
            try:
                validate(program)
            except UnsafeFilter:
                continue            # refused: the bound is not a promise about it
            bound = static_bound(program)
            try:
                outputs = jq.compile(program).input_value(GROWTH_ROW).all()
            except Exception:
                continue            # a runtime type error emits nothing to measure
            emitted = sum(_measure(output) for output in outputs)
            checked += 1
            assert emitted <= bound.factor * row_size + bound.extra, (
                f"{program!r} emitted {emitted} characters, over the "
                f"{bound.factor} * {row_size} + {bound.extra} this module promised")
        assert checked > 200, "the generator stopped producing runnable programs"

    def test_the_check_can_fail(self):
        """A property test that cannot fail proves nothing: this is the
        comparison the loop makes, against the number the old
        arithmetic produced for a 120-character separator."""
        assert static_bound(f'.tags | join("{"Q" * 120}")').factor > 2


class TestCharactersThatNeverReachLibjq:
    """The differential in the direction the ACCEPTED corpus cannot
    reach: characters that make `jq.compile()` fail on a program this
    module called safe.

    A raw NUL is the plain one -- the binding hands libjq a UTF-8 C
    string, so the NUL ends it and libjq reports `unexpected end of
    file` on the half it saw, while hopai had analysed the whole thing.
    The direction is safe (libjq refuses, nothing runs) but the caller
    gets a raw parse error instead of a refusal that names the fix,
    which breaks the promise this module's whole design rests on: what
    it accepts, libjq accepts.

    The corpus never contained a NUL byte or a surrogate, which is why
    the differential test above passed for as long as it did."""

    HOSTILE = (
        '.properties.title + "a\x00b"',       # a raw NUL inside a string literal
        '.properties.title + "a\\u0000b"',    # the same NUL, spelled as an escape
        '."\ud83d"',                          # a raw lone surrogate in a field name
        '"\\ud83d"',                          # a lone HIGH surrogate escape
        '"\\ude00"',                          # a lone LOW surrogate escape
        '"\\ud83dx"',                         # a pair that is not one
        ".a # \x00",                          # a NUL where nothing even tokenizes it
    )

    @pytest.mark.parametrize("program", HOSTILE)
    def test_hopai_refuses_them(self, program):
        """Fail closed and name the fix. `\\u0000` and the lone LOW
        surrogate are the two libjq accepts -- the first puts a NUL in a
        document POSTed to a vendor, where it truncates whatever C
        string reads it, and the second silently yields U+FFFD, content
        the row never contained. Both are refused for the same reason
        `implode` and `@base64d` are."""
        with pytest.raises(UnsafeFilter):
            validate(program)

    #: The ones libjq compiles happily, with the reason hopai refuses
    #: them anyway. Over-refusing is always allowed -- the subset is a
    #: SUBSET -- but each one still has to have a reason written down,
    #: or the list becomes a list of tastes.
    OVER_REFUSED = {
        '.properties.title + "a\\u0000b"':
            "the NUL survives into the document, where it truncates whatever C string "
            "reads it on the way to the vendor",
        '"\\ude00"':
            "libjq quietly yields U+FFFD, content the row never contained -- the same "
            "reason `implode` and `@base64d` are out",
        ".a # \x00":
            "one rule for NUL bytes everywhere beats one rule per context, and nothing "
            "legitimate puts one in a comment",
    }

    @pytest.mark.parametrize("program", HOSTILE)
    def test_libjq_agrees_or_the_difference_is_ours_to_own(self, program):
        """What makes the list above a DIFFERENTIAL corpus rather than a
        list of tastes: either libjq refuses it too -- in which case
        accepting it would have moved a parse error from validation to
        ranking time, one row at a time -- or it is one of the three
        hopai over-refuses on purpose, with the reason recorded."""
        jq = pytest.importorskip("jq")
        try:
            jq.compile(program)
        except Exception:
            return                              # libjq refuses it: agreement
        assert program in self.OVER_REFUSED, \
            f"{program!r} compiles in libjq, so the refusal above needs a reason"

    def test_a_surrogate_pair_still_means_one_character(self):
        """The fix must refuse the halves without breaking the whole.
        `."\\ud83d\\ude00"` names ONE character in libjq; a validator
        that kept the two surrogates separate would report a field name
        no allowlist could ever match -- and hand Python a string that
        cannot be encoded to UTF-8 at all."""
        jq = pytest.importorskip("jq")
        program = '."\\ud83d\\ude00"'
        assert paths_read(program) == frozenset({"\U0001f600"})
        assert jq.compile(program).input({"\U0001f600": "yes"}).all() == ["yes"]


class TestAcceptedProgramsTerminate:
    """The totality claim, spot-checked by running it. The proof is
    structural -- no recursion, no generator, no loop, no fold -- but a
    proof nothing exercises is a comment."""

    ROW = SENTINEL_ROW

    def test_every_accepted_program_finishes_in_milliseconds(self):
        """`def f: f; f` holds the GIL forever and only SIGKILL ends it,
        so there is no timeout to fall back on: if a program in the
        corpus could loop, this test would never report -- which is
        itself the signal."""
        jq = pytest.importorskip("jq")
        started = time.monotonic()
        for program in ACCEPTED:
            # A runtime type error (`ascii_downcase` on a number) is jq
            # answering, which is all this test needs: it stopped.
            with contextlib.suppress(ValueError):
                jq.compile(program).input(self.ROW).all()
        assert time.monotonic() - started < 5.0
