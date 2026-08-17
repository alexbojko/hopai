"""
hopai.jqsafe

Deciding whether a jq filter is safe to run, BEFORE libjq ever sees it.

A reranker scores (query, document) pairs, so every candidate row has to
be projected into one string. That projection is written as a jq filter
-- the one small language every model already writes fluently -- and it
may arrive from a model through an MCP tool call:

    document_from='.properties.title + ": " + (.properties.summary // "")'

    from hopai.jqsafe import validate, paths_read, UnsafeFilter

    validate(document_from, fields=["properties.title", "properties.summary"])
    paths_read(document_from)   # {"properties.title", "properties.summary"}

WHY THIS FILE EXISTS. jq is not a sandbox, and three of its properties
are load-bearing against us. Each was measured against the `jq` PyPI
package (CFFI bindings to libjq), not assumed:

  - `env` and `$ENV` return the PROCESS ENVIRONMENT -- `env|keys` gave
    133 entries on the machine this was written on, including every
    credential the application holds. The filter's OUTPUT is posted to a
    third-party reranker API, so this is a one-line exfiltration of a
    database password to a vendor.
  - `def f: f; f` is a non-terminating program, and WITHIN A SINGLE
    EVALUATION the binding hands the GIL back to nobody: a watchdog
    thread cannot even print, SIGINT is ignored, and only SIGKILL ends
    the process. There is NO in-process timeout that works. The scope
    is the whole of the claim and is measured: across MANY short
    evaluations the binding does release the GIL between them (another
    thread got 5,284,436 iterations over 0.90s of small evaluations
    against 184,711 during one 0.90s evaluation, 29x), which is why
    rerankers.py can say a worker thread recovers ~15% of the loop
    while this module says a runaway filter is uninterruptible. One
    evaluation has no boundary to hand it back at -- so the subset must
    be total, and (see the growth section below) size-bounded, because
    nothing downstream can stop an evaluation once it has started.
  - `[range(100000000)]` makes libjq call abort() -- SIGABRT, not a
    Python MemoryError -- so it cannot be caught either.

What jq does NOT have is equally load-bearing: no shell, no network, no
file writes, module loading off by default, and -- the important one --
NO DYNAMIC DISPATCH TO BUILTINS BY NAME. `builtins` yields the STRING
"env/0"; there is no `eval`, no way to call a function whose name was
computed at runtime.

THE TWO CLAIMS THE WHOLE SECURITY POSTURE RESTS ON.

  1. SOUNDNESS: static analysis is decidable here. Because a name can
     never be constructed at runtime, the set of builtins a program can
     invoke is exactly the set of literal names appearing in its source
     -- which a parser enumerates completely. That is what turns this
     from a blacklist arms race into a decision procedure: we ALLOWLIST
     names, and no amount of string arithmetic inside the filter can
     reach a name that is not written in it. (Module loading would break
     this claim, which is why `import`/`include` are absent from the
     grammar rather than merely refused.)
  2. TOTALITY: termination is structural, not a timeout. Every accepted
     program is built from constructs whose output stream is bounded by
     the input: field access, indexing, iteration over an existing
     array, pipes, and a fixed set of elementwise builtins. There is no
     recursion (`def`), no generator (`range`), no loop (`while`,
     `until`, `repeat`, `recurse`, `..`), no fold (`reduce`, `foreach`)
     and no re-entry (`label`/`break`). By structural induction over the
     parse tree, evaluation of an accepted program halts on any finite
     input -- so the hang above is not "unlikely", it is unreachable.
     `is_total()` is exactly "does this parse in the subset".

BOTH CLAIMS HAVE THE SAME PRECONDITION, and it is the one that failed
in review: this module and libjq must agree about WHICH CHARACTERS ARE
CODE. An allowlist over the names in a program is worth nothing if the
program being analyzed is not the program being run. The way that broke
was lexical, not grammatical -- `_scan_paren` measured how far a
`\\(...)` interpolation extends by counting parentheses without knowing
that jq treats `#` as a comment, so

    "\\(.properties.a # )\\n|env.FAKE_SECRET)"

closed the interpolation HERE at the `)` inside the comment, while libjq
read on to the real one. Everything after it was "string text" that was
never tokenized: `env` returned the process environment, and the same
shape reached `def f: f; f` and `[range(100000000)]`. Hence the rule
that now governs this file: EVERY reader of the source goes through
_skip_trivia and _scan_string, so each of jq's lexical rules has exactly
one implementation and no second scanner can drift. A differential test
on COMPILABILITY could not catch that -- the payload is valid jq -- so
there is a second one on MEANING: each accepted program is run against a
row of per-field sentinels, and every field that reaches the output must
be covered by what paths_read() reported.

Totality bounds the number of outputs but not their SIZE: `(. + .)` is
one node and doubles a string, and `(.+.)|(.+.)|(.+.)|...` doubles once
per stage, so a 60-character program could ask for 2**10 times the row.
So every accepted program also carries a statically computed growth
factor -- how many times its own input it may emit -- and anything above
MAX_GROWTH is refused with that number in the message.

That factor COMPOSES because there are no variable bindings: with `as
$x` absent, no subexpression can name a value from an enclosing context,
so `map(f)` really is f applied elementwise and cannot smuggle the whole
document into each element.

AND CONCATENATION IS NOT THE ONLY AMPLIFIER, which is what the first
version of that arithmetic got wrong and what cost this module its
"unreachable" hang. A CONSTANT WRITTEN ONCE PER ELEMENT IS A FACTOR,
NOT A CONSTANT: `join(sep)` puts its separator between every pair of
elements, and `split("")` manufactures one element per CHARACTER of any
published field, so

    .properties.title | split("") | join("<311 characters>")

emits 311 characters for every character of a title. Six of those
stages fit inside MAX_LENGTH; `join` charged as `1 + growth(argument)`
-- 2, whatever the separator's length -- called that 2**6 = 64 and
ACCEPTED it, while the row was multiplied by 311**6, about 9e14. Live
through the MCP `traverse_graph` tool, a 235-character variant turned
20 titles of 7-8 characters into 1,084,831 bytes POSTed to a provider;
with three stages and a 250-character separator, ONE document built
from a FOUR-character title ran past 45 seconds inside libjq under a
700MB rlimit before it was killed from outside -- no exception, no
abort(), and a SIGALRM handler installed beforehand never ran, because
one evaluation hands the GIL back to nobody. That is the `def f: f; f`
hang this docstring calls unreachable, reached by a program in the
subset. Nothing that MEASURES the finished document can help: the
document does not exist yet, and the process is inside libjq. So the
charge for a separator is now len(sep) as a MULTIPLIER -- an array of n
elements is at least n characters of JSON, so n is bounded by the input
and the separators alone are at most len(sep) times it.

_LITERAL_ARGUMENT is still necessary and was never sufficient. It pins
the separator to program text so that len(sep) is a number this module
can READ (`.tags | join("\\(.)")` measured 9,899 characters out of a
50-element array, and no static analysis can bound a separator the row
supplies) -- but knowing the separator is at most MAX_LENGTH characters
bounds the separator, not len(sep) * element_count, which is the
product that grew.

WHAT WAS REJECTED, and why none of it is enough on its own:

  - A watchdog thread with a timeout. Measured above: it cannot fire
    against the case it is for. libjq yields the GIL between
    evaluations and never inside one, and one evaluation that does not
    return is the whole of the problem.
  - A subprocess per row with rlimits and a scrubbed environment. It
    would work, and it is the wrong shape for this library: a fork+exec
    per candidate row on the ranking path, plus SIGABRT handling, plus a
    second copy of the connection state -- a sidecar in all but name,
    which rule 1 of CLAUDE.md exists to refuse. Static validation costs
    microseconds once per query rather than a process per row.
  - Scanning the source for dangerous substrings. `.environment` and
    `.env_var` are ordinary property names and a substring scan rejects
    both; `"env"` inside a string literal is data. Tokenizing is what
    tells a NAME from a FIELD from TEXT, which is why there is a real
    tokenizer here and not a regex.
  - Writing our own jq evaluator. Then hopai would own jq's semantics
    forever, and every divergence would be a silently different document
    -- the worst thing this library can produce (rule 4). So this module
    PARSES ONLY TO REFUSE: what it accepts is handed to libjq unchanged,
    verbatim, and a differential test asserts that everything this
    parser accepts, `jq.compile()` also accepts. The subset is a genuine
    subset; we never accept something libjq then rejects at runtime.
    That test is not decoration -- it is what caught jq 1.7's comment
    rule, where a comment ending in `\\` swallows the NEXT line and a
    filter this module read as `.a` compiles to nothing at all.

WHAT IS IN THE SUBSET: identity `.`; field access `.foo`, `."quoted"`,
with `?`; indexing `.[0]`, `.[-1]`, slices `.[1:3]`, iteration `.[]`;
pipe; comma; parentheses; array construction `[...]`; string literals
with `\\(...)` interpolation; numbers; `true`/`false`/`null`; `+` and
`-`; the alternative `//`; the comparisons and `and`/`or`; and calls, by
literal name only, to the allowlist in _ARITY.

WHY EACH EXCLUDED FAMILY IS EXCLUDED (the ones a reader will reach for):

  - `test`/`match`/`capture`/`sub`/`gsub`/`scan`/`splits`: a regex from a
    model is a regex we would have to run, and catastrophic backtracking
    on a chosen pattern is the same unkillable hang as `def f: f; f`.
    `startswith`/`endswith`/`split`/`ascii_downcase` cover what a
    document projection actually needs. (This is also why `split` is
    pinned to ONE argument: `split("x"; "g")` is the regex form.)
  - `getpath`/`setpath`/`paths`/`leaf_paths`/`to_entries`: the path is
    DATA, so no static check can see which property is read -- they
    defeat the `fields=` allowlist completely, which is the one thing
    that keeps `.properties.ssn` out of a vendor's logs. Same reason
    `.[expr]` with a non-literal index is refused.
  - `env`/`$ENV`/`$__loc__`/`input`/`inputs`/`debug`/`input_line_number`:
    data that is not the candidate row. The first two are the
    exfiltration vector; the rest leak process state into a document.
  - `implode`/`explode`/`ascii`/`@base64d`: they synthesize content the
    row never contained -- `@base64d` in particular turns opaque bytes
    into text nothing has validated, one layer below every check here.
  - `*`, `/`, `%`: `"x" * 1000000000` is an unbounded memory amplifier
    from a literal, `/` on strings is a second spelling of `split`, and
    none of the three has any role in building a document. `+` (and
    `-` for numbers and arrays) is the whole arithmetic a projection
    needs, and it is bounded by the growth cap above.
  - `if`/`then`/`else`: `//` supplies a default and `select(...)`
    filters, which is the whole of what a projection conditional does;
    a second spelling would buy nothing and cost the `elif` chain.
  - `try`/`catch`: `catch` binds jq's ERROR MESSAGE, and jq quotes the
    offending input value inside it -- straight into the document, which
    is posted to a third party. `?` is the one spelling for "tolerate a
    missing key", and it cannot carry data.
  - `as $x` bindings, and `$` in general: without variables there is
    nowhere for `$ENV` or `$__loc__` to appear, so they are excluded by
    the absence of a construct rather than by name.
  - object construction `{a: .b}`: computed keys (`{(.k): .v}`) are the
    same problem as computed paths, and a document is a string.

Names ARE recognized by spelling in one place only: the error messages.
`_EXPLAIN` exists so that `env` is refused with the sentence that tells
the caller what to write instead, rather than with the generic "not in
the allowlist". Rejection is by allowlist; explanation is by name. Those
two must not be confused -- deleting every entry in `_EXPLAIN` would
make the errors worse and change nothing about what is accepted.
"""

from __future__ import annotations

import math
import re
import sys
from contextlib import contextmanager
from typing import NamedTuple, Optional

__all__ = ["UnsafeFilter", "validate", "paths_read", "is_total"]


class UnsafeFilter(ValueError):
    """A jq filter outside the subset hopai will accept from a model."""


#: A projection is one expression over one row. Anything past this is a
#: program, not a projection, and the cheapest way to bound the parser's
#: work is to bound its input.
MAX_LENGTH = 2000

#: Nesting depth, so a pathological `((((...))))` cannot reach Python's
#: own recursion limit -- which would surface as RecursionError from a
#: validator whose entire job is to answer yes or no.
MAX_DEPTH = 40

#: How many times its own input an accepted filter may emit. Totality
#: bounds the number of outputs, not their size: string `+` doubles, and
#: doubling once per pipe stage is exponential in the PROGRAM's length.
#: 64x a row is far past any real projection (a document concatenates a
#: handful of fields, growth 3 or 4) and far below anything that
#: threatens memory.
MAX_GROWTH = 64

#: Characters an accepted filter may add REGARDLESS of the row -- the
#: `extra` term of _size(). Every one of them is program text, which
#: MAX_LENGTH already bounds, and the multiplier cap already licenses
#: MAX_GROWTH copies of the row; so a filter needing more than the two
#: multiplied together is not writing a literal out, it is multiplying
#: one by an element count (`.[0:1000000] | join("<300 characters>")`),
#: which is the same product MAX_GROWTH exists to bound.
MAX_ADDED = MAX_LENGTH * MAX_GROWTH

#: The functions a filter may call, with the arities jq gives them.
#: Calls are by literal name only -- see the module docstring on why
#: that makes this list exhaustive rather than optimistic.
#:
#: Arity is pinned, not just the name: `split/1` is plain string
#: splitting while `split/2` is the REGEX form, and `first/1`/`last/1`
#: take a filter whose totality is checked like any other subexpression.
_ARITY = {
    "add": frozenset({0}),
    "arrays": frozenset({0}),
    "ascii_downcase": frozenset({0}),
    "ascii_upcase": frozenset({0}),
    "empty": frozenset({0}),
    "endswith": frozenset({1}),
    "first": frozenset({0, 1}),
    "flatten": frozenset({0, 1}),
    "has": frozenset({1}),
    "join": frozenset({1}),
    "last": frozenset({0, 1}),
    "length": frozenset({0}),
    "ltrimstr": frozenset({1}),
    "map": frozenset({1}),
    # jq spells negation as a zero-argument FILTER reading its input from
    # the left -- `.a | not` -- and there is no prefix operator form, so
    # `not(.a)` is a call with one argument too many and refused as one.
    "not": frozenset({0}),
    "numbers": frozenset({0}),
    "objects": frozenset({0}),
    "reverse": frozenset({0}),
    "rtrimstr": frozenset({1}),
    "select": frozenset({1}),
    "sort": frozenset({0}),
    "split": frozenset({1}),
    "startswith": frozenset({1}),
    "strings": frozenset({0}),
    "tojson": frozenset({0}),
    "tonumber": frozenset({0}),
    "tostring": frozenset({0}),
    "type": frozenset({0}),
    "unique": frozenset({0}),
    "values": frozenset({0}),
}

#: Calls whose argument must be a LITERAL -- a string with no
#: interpolation, or a number.
#:
#: Not a style rule: `join` emits its separator ONCE PER ELEMENT, so
#: `.tags | join("\(.)")` is quadratic in the row rather than linear in
#: it -- measured at 9,899 characters out of a 50-element array.
#: Pinning the separator to program text is what lets _size() READ its
#: length and charge len(sep) as a multiplier; a separator the row
#: supplies has no length this module can know.
#:
#: WHAT IT DOES NOT DO, because the comment here used to claim it did:
#: it bounds the SEPARATOR by MAX_LENGTH, never len(sep) * element
#: count. That product is the whole amplifier -- `split("") |
#: join("<311 characters>")` multiplies a field by 311 with a separator
#: well inside every limit on this page -- and it is bounded by the
#: multiplier in _size(), not here. The others are here
#: because a computed needle (`has(.k)`, `split(.sep)`) is the same
#: invisible-to-the-allowlist argument that keeps `getpath` out, and
#: because a separator or a prefix in a document projection is a
#: constant in every real filter anyone writes.
_LITERAL_ARGUMENT = frozenset({
    "endswith", "flatten", "has", "join", "ltrimstr", "rtrimstr", "split", "startswith",
})

#: Calls that hand their input through rather than reading a value out
#: of it -- filters, reorderings and selections. Used by paths_read():
#: `.properties | select(.type == "paper") | .title` reads
#: properties.type and properties.title, and reporting a bare
#: `properties` for the select would make a correct filter fail an
#: allowlist that names the two leaves. Everything NOT listed here
#: consumes its input, and consuming it is a read.
_PASSTHROUGH = frozenset({
    "arrays", "empty", "first", "flatten", "last", "map", "numbers",
    "objects", "reverse", "select", "sort", "strings", "unique", "values",
})

#: The literals jq spells as bare words.
_KEYWORD_LITERALS = frozenset({"true", "false", "null"})

#: Infix words, so they are never mistaken for a call.
_WORD_OPERATORS = frozenset({"and", "or"})

_COMPARISONS = frozenset({"==", "!=", "<", "<=", ">", ">="})

#: Characters that never reach libjq as themselves. The binding hands
#: jq a UTF-8 encoding of the program, and both of these break BEFORE
#: any of the analysis on this page applies: a NUL ends the C string, so
#: libjq compiles the half it saw and reports `unexpected end of file`
#: on a filter this module called safe, and a lone surrogate cannot be
#: encoded at all (UnicodeEncodeError out of the binding, which is not
#: an UnsafeFilter either). Refusing them here keeps the promise that
#: what this module accepts, `jq.compile()` accepts -- and keeps the
#: caller's answer a refusal that names the fix rather than a raw parse
#: error from a layer below.
_UNENCODABLE_RE = re.compile("[\x00\ud800-\udfff]")

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_HEX = frozenset("0123456789abcdefABCDEF")

_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
            "n": "\n", "r": "\r", "t": "\t"}

#: Why a particular spelling is not here, for the caller who reached for
#: it. ERROR QUALITY ONLY: none of these names is what makes a program
#: unsafe -- the allowlist is -- so a name missing from this table is
#: still refused, just with the generic message.
_EXPLAIN = {
    "env": "a document is built from the candidate row only, and this filter's output "
           "is posted to a third-party reranker -- `env` would send it your process "
           "environment. Read a property instead, e.g. `.properties.title`",
    "builtins": "it enumerates jq's own function names and has nothing to do with the row",
    "input": "there is exactly one input, the candidate row -- reading further inputs "
             "would put another row's data in this row's document",
    "inputs": "there is exactly one input, the candidate row -- reading further inputs "
              "would put another row's data in this row's document",
    "debug": "diagnostics belong in hopai's log, not in a document posted to a vendor",
    "stderr": "diagnostics belong in hopai's log, not in a document posted to a vendor",
    "input_line_number": "it reports parser state, which is not part of the row",
    "getpath": "a computed path is DATA, so no static check can see which property it "
               "reads -- it would defeat the field allowlist entirely. Name the "
               "property, e.g. `.properties.title`",
    "setpath": "a document projection reads the row, it never writes it",
    "delpaths": "a document projection reads the row, it never writes it",
    "paths": "it walks every path in the row, so no field allowlist can constrain it. "
             "Name the properties you want, e.g. `.properties.title`",
    "leaf_paths": "it walks every path in the row, so no field allowlist can constrain "
                  "it. Name the properties you want, e.g. `.properties.title`",
    "to_entries": "it turns key names into data, which puts every property in reach of "
                  "a filter that named none of them",
    "with_entries": "it turns key names into data, which puts every property in reach of "
                    "a filter that named none of them",
    "test": "a regex from a model is a regex hopai would have to run, and catastrophic "
            "backtracking on a chosen pattern is an unkillable hang. Use `startswith`, "
            "`endswith`, `split` or `ascii_downcase`",
    "match": "a regex from a model is a regex hopai would have to run, and catastrophic "
             "backtracking on a chosen pattern is an unkillable hang. Use `startswith`, "
             "`endswith`, `split` or `ascii_downcase`",
    "capture": "a regex from a model is a regex hopai would have to run, and catastrophic "
               "backtracking on a chosen pattern is an unkillable hang. Use `startswith`, "
               "`endswith`, `split` or `ascii_downcase`",
    "scan": "a regex from a model is a regex hopai would have to run, and catastrophic "
            "backtracking on a chosen pattern is an unkillable hang. Use `split`",
    "splits": "a regex from a model is a regex hopai would have to run, and catastrophic "
              "backtracking on a chosen pattern is an unkillable hang. Use `split`, which "
              "here splits on a plain string",
    "sub": "a regex from a model is a regex hopai would have to run, and catastrophic "
           "backtracking on a chosen pattern is an unkillable hang. Use `ltrimstr` or "
           "`rtrimstr`",
    "gsub": "a regex from a model is a regex hopai would have to run, and catastrophic "
            "backtracking on a chosen pattern is an unkillable hang. Use `split` and "
            "`join`",
    "def": "this subset has no user-defined functions -- `def f: f; f` is a "
           "non-terminating program that no timeout can interrupt, and having no "
           "recursion at all is what makes every accepted filter terminate",
    "reduce": "a fold over a stream is not in the subset. Use `map`, `add` or `join`",
    "foreach": "a fold over a stream is not in the subset. Use `map`, `add` or `join`",
    "while": "unbounded iteration is not in the subset -- every accepted filter "
             "terminates precisely because no such construct exists. Use `map`",
    "until": "unbounded iteration is not in the subset -- every accepted filter "
             "terminates precisely because no such construct exists. Use `map`",
    "repeat": "unbounded iteration is not in the subset -- every accepted filter "
              "terminates precisely because no such construct exists. Use `map`",
    "recurse": "it walks the row to an unbounded depth, so neither termination nor the "
               "field allowlist can be established. Name the properties you want",
    "range": "it generates an arbitrarily long stream -- `[range(100000000)]` makes "
             "libjq abort() the process, which no Python `except` can catch",
    "limit": "stream control implies a generator to limit, and this subset has none",
    "import": "module loading is off, and it is the one thing that would let a filter "
              "call a name it does not spell",
    "include": "module loading is off, and it is the one thing that would let a filter "
               "call a name it does not spell",
    "modulemeta": "module loading is off, and it is the one thing that would let a "
                  "filter call a name it does not spell",
    "label": "a non-local exit needs a matching `break`, and neither is in the subset",
    "break": "a non-local exit needs a matching `label`, and neither is in the subset",
    "if": "conditionals are not in the subset -- `//` supplies a default and "
          "`select(...)` filters, which is the whole of what a projection needs",
    "then": "conditionals are not in the subset -- `//` supplies a default and "
            "`select(...)` filters",
    "elif": "conditionals are not in the subset -- `//` supplies a default and "
            "`select(...)` filters",
    "else": "conditionals are not in the subset -- `//` supplies a default and "
            "`select(...)` filters",
    "end": "conditionals are not in the subset -- `//` supplies a default and "
           "`select(...)` filters",
    "try": "`catch` binds jq's error MESSAGE, which quotes the offending input value, "
           "into a document that is posted to a vendor. Use `?` to tolerate a missing "
           "key, e.g. `.properties.title?`",
    "catch": "`catch` binds jq's error MESSAGE, which quotes the offending input value, "
             "into a document that is posted to a vendor. Use `?` to tolerate a missing "
             "key, e.g. `.properties.title?`",
    "as": "variable bindings are not in the subset -- which is also why `$ENV` and "
          "`$__loc__` have nowhere to appear",
    "implode": "building text from code points produces content the row never contained",
    "explode": "code-point arithmetic produces content the row never contained",
    "ascii": "code-point arithmetic produces content the row never contained",
    "tostream": "it turns every path in the row into data, so no field allowlist can "
                "constrain it",
}


class _Token(NamedTuple):
    kind: str          # field | dot | name | number | string | op | punct | end
    text: str          # what was written, for error messages
    value: object      # field name, string parts, or None
    pos: int           # offset into the ORIGINAL program, interpolations included


def _refuse(owner: str, pos: int, what: str, why: str):
    """The single shape every rejection takes: what, where, and the fix.

    One helper rather than raise-sites, because an error a model reads
    is part of this module's interface -- an inconsistent one costs a
    retry the caller pays for."""
    raise UnsafeFilter(f"{owner}: {what} (offset {pos}) -- {why}")


# ---------------------------------------------------------------------
# Tokenizer. A tokenizer and not a regex scan, because the whole point
# is to tell a NAME (`env`) from a FIELD (`.environment`) from TEXT
# ("env") -- the three spellings a substring blacklist confuses.
#
# Three functions, and EVERY reader of the source goes through them, so
# each of jq's lexical rules has exactly one implementation:
#   _skip_trivia  whitespace and comments
#   _scan_string  string literals, including their interpolations
#   _tokenize     everything else
# _scan_paren -- which decides where an interpolation ENDS -- drives the
# first two rather than reading characters itself. See its docstring for
# the exfiltration that a second, comment-blind scanner allowed.
# ---------------------------------------------------------------------

def _skip_trivia(src: str, i: int, base: int, owner: str) -> int:
    """Index past the whitespace and comments starting at `i`, or `i`
    itself when there are none."""
    n = len(src)
    while i < n:
        if src[i] in " \t\r\n":
            i += 1
            continue
        if src[i] != "#":
            return i
        i += 1
        while i < n and src[i] != "\n":
            # jq 1.7 lets a comment CONTINUE onto the next line when the
            # line ends with a backslash; jq 1.6 does not. So a comment
            # containing one means two different programs depending on
            # the libjq the application installed -- measured:
            # `# c\<newline>.a` compiles to nothing at all on 1.7, and
            # this module would have accepted `.a`. Refusing the
            # backslash makes our comment rule identical to every jq's.
            if src[i] == "\\":
                _refuse(owner, base + i, "a backslash inside a comment",
                        "jq 1.7 continues a comment onto the next line when it ends "
                        "with `\\` and jq 1.6 does not, so the same filter would mean "
                        "two different things. Drop the backslash")
            i += 1
    return i


def _scan_string(src: str, i: int, base: int, owner: str, depth: int = 0):
    """Scan the string literal starting at src[i] == '"'.

    Returns (parts, index-after-the-closing-quote). `parts` alternates
    plain text (str) and interpolations, each recorded as
    ("interp", source, absolute-offset) so the parser can validate the
    interpolated expression with offsets that still point into the
    program the caller wrote."""
    parts: list = []
    text: list = []
    n = len(src)
    i += 1
    while True:
        if i >= n:
            _refuse(owner, base + i, "the string literal is never closed",
                    'add the missing `"`')
        ch = src[i]
        if ch == '"':
            i += 1
            break
        if ch != "\\":
            text.append(ch)
            i += 1
            continue
        nxt = src[i + 1:i + 2]
        if nxt == "(":
            if text:
                parts.append("".join(text))
                text = []
            end = _scan_paren(src, i + 1, base, owner, depth + 1)
            parts.append(("interp", src[i + 2:end - 1], base + i + 2))
            i = end
            continue
        if nxt == "u":
            digits = src[i + 2:i + 6]
            if len(digits) != 4 or any(c not in _HEX for c in digits):
                _refuse(owner, base + i, "`\\u` needs exactly four hex digits",
                        "write the character itself, or a complete escape like `\\u00e9`")
            code = int(digits, 16)
            start = i
            i += 6
            if 0xD800 <= code <= 0xDBFF:
                # A high surrogate is HALF a character, and jq demands
                # the other half immediately: libjq answers `Invalid
                # \\uXXXX\\uXXXX surrogate pair escape` for both
                # `"\\ud83d"` and `"\\ud83dx"`. Combining the pair the
                # way libjq does is not decoration either -- it is what
                # makes `."\\ud83d\\ude00"` the field named by ONE
                # emoji here and in libjq, rather than two lone
                # surrogates the field allowlist would never match.
                low = src[i + 2:i + 6] if src[i:i + 2] == "\\u" else ""
                if len(low) != 4 or any(c not in _HEX for c in low) \
                        or not 0xDC00 <= int(low, 16) <= 0xDFFF:
                    _refuse(owner, base + start,
                            f"`\\u{digits}` is half of a surrogate pair",
                            "libjq refuses a lone surrogate outright, so hopai must too "
                            "-- write the character itself, or both halves "
                            "(`\\ud83d\\ude00`)")
                code = 0x10000 + ((code - 0xD800) << 10) + (int(low, 16) - 0xDC00)
                i += 6
            elif 0xDC00 <= code <= 0xDFFF:
                # libjq ACCEPTS this one and quietly yields U+FFFD, so
                # the differential test on compilability cannot see it.
                # Refused anyway: a replacement character is content the
                # row never held, which is the same reason `implode` and
                # `@base64d` are out -- and a validator that built the
                # lone surrogate instead would hand back a str that
                # cannot be encoded to UTF-8 at all.
                _refuse(owner, base + start,
                        f"`\\u{digits}` is a lone low surrogate",
                        "libjq turns it into U+FFFD, a character the row never contained "
                        "-- write the character itself")
            elif code == 0:
                # `"\\u0000"` compiles, and then the NUL travels into a
                # document POSTed to a vendor and truncates it at the
                # first C string that touches it. Refused for the same
                # reason as a raw NUL in the source (see _compile).
                _refuse(owner, base + start, "`\\u0000` puts a NUL byte in the document",
                        "a NUL ends a C string, so everything after it is lost wherever "
                        "the document is read -- drop it")
            text.append(chr(code))
            continue
        if nxt in _ESCAPES:
            text.append(_ESCAPES[nxt])
            i += 2
            continue
        _refuse(owner, base + i, f"`\\{nxt}` is not a jq string escape",
                'the escapes are \\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX and \\(...)')
    if text:
        parts.append("".join(text))
    return parts, i


def _scan_paren(src: str, i: int, base: int, owner: str, depth: int = 0) -> int:
    """Index just past the `)` matching src[i] == `(`.

    THE THINGS THAT ARE NOT CODE HAVE TO BE SKIPPED THE WAY LIBJQ SKIPS
    THEM, and this function is where that matters most, because what it
    measures is how far a `\\(...)` interpolation extends -- i.e. which
    part of the filter is an expression and which part is inert text.

    That is not a parsing nicety, it is the security boundary. A version
    of this function that counted parentheses without knowing about
    comments accepted

        "\\(.properties.a # )\\n|env.FAKE_SECRET)"

    because the `)` inside the comment closed the interpolation HERE
    while libjq, honouring the comment, read on to the real `)` and ran
    `env.FAKE_SECRET`. Everything past our fake closing paren was
    "string text" that this module never even tokenized: the allowlist,
    the totality argument and the whole static analysis applied to a
    program that was not the one being run. Strings and comments are
    therefore skipped by the SAME two helpers the tokenizer uses --
    _scan_string and _skip_trivia -- so there is exactly one
    implementation of each of jq's lexical rules, and no second scanner
    to drift out of step with the first."""
    # The scanners recurse into each other for a string inside an
    # interpolation, and they run BEFORE the parser's own nesting guard
    # -- so without this, 399 nested interpolations (which fit inside
    # MAX_LENGTH) raise RecursionError out of a function whose entire
    # contract is UnsafeFilter or nothing.
    if depth > MAX_DEPTH:
        _refuse(owner, base + i, f"the filter nests more than {MAX_DEPTH} deep",
                "a document projection is a few fields joined together; this is a program")
    nesting = 0
    n = len(src)
    while i < n:
        skipped = _skip_trivia(src, i, base, owner)
        if skipped != i:
            i = skipped
            continue
        ch = src[i]
        if ch == '"':
            _, i = _scan_string(src, i, base, owner, depth)
            continue
        if ch == "(":
            nesting += 1
        elif ch == ")":
            nesting -= 1
            if nesting == 0:
                return i + 1
        i += 1
    _refuse(owner, base + i, "an interpolation `\\(` is never closed", "add the missing `)`")


def _tokenize(src: str, owner: str, base: int = 0) -> list:
    """Turn a filter into tokens, refusing characters the subset has no
    place for. `base` is the offset of `src` inside the original program,
    so an error inside `"\\(...)"` still names an offset the caller can
    count to."""
    tokens: list = []
    i, n = 0, len(src)
    while i < n:
        skipped = _skip_trivia(src, i, base, owner)     # whitespace and comments
        if skipped != i:
            i = skipped
            continue
        ch = src[i]
        if ch == ".":
            if src.startswith("..", i):
                _refuse(owner, base + i, "`..` (recursive descent) is not in the subset",
                        "it walks the whole row to an unbounded depth, so neither "
                        "termination nor the field allowlist can be established. Name "
                        "the properties you want, e.g. `.properties.title`")
            match = _NAME_RE.match(src, i + 1)
            if match:
                tokens.append(_Token("field", "." + match.group(), match.group(), base + i))
                i = match.end()
                continue
            if src[i + 1:i + 2] == '"':
                parts, end = _scan_string(src, i + 1, base, owner)
                if any(not isinstance(part, str) for part in parts):
                    _refuse(owner, base + i,
                            "a field name built by interpolation is not in the subset",
                            "the property being read has to be visible to the field "
                            'allowlist, so write it out: `.properties.title`')
                name = "".join(parts)
                tokens.append(_Token("field", "." + src[i + 1:end], name, base + i))
                i = end
                continue
            tokens.append(_Token("dot", ".", None, base + i))
            i += 1
            continue
        if ch == '"':
            parts, end = _scan_string(src, i, base, owner)
            tokens.append(_Token("string", src[i:end], parts, base + i))
            i = end
            continue
        if ch in "0123456789":                          # not .isdigit(): that is true of
            match = _NUMBER_RE.match(src, i)            # unicode digits _NUMBER_RE rejects
            tokens.append(_Token("number", match.group(), match.group(), base + i))
            i = match.end()
            continue
        if ch == "$":
            _refuse(owner, base + i, "variables are not in the subset",
                    "a projection reads the candidate row and nothing else -- which is "
                    "also why `$ENV` and `$__loc__` have nowhere to appear. Read a "
                    "property, e.g. `.properties.title`")
        if ch == "@":
            _refuse(owner, base + i, "`@` format strings are not in the subset",
                    "`@base64d` in particular decodes bytes into text nothing has "
                    "validated. Build the document from properties and string literals")
        if ch == "{":
            _refuse(owner, base + i, "object construction is not in the subset",
                    "a computed key (`{(.k): .v}`) is invisible to the field allowlist, "
                    "and a document is a string -- concatenate with `+` or `join`")
        if ch == "}":
            _refuse(owner, base + i, "`}` has no matching construct in the subset",
                    "object construction is not accepted here")
        if src.startswith("?//", i):
            _refuse(owner, base + i, "the destructuring alternative `?//` is not in the "
                                     "subset",
                    "it belongs to `as` bindings, which are not accepted -- use `//` for "
                    "a default value")
        for spelling in ("//", "==", "!=", "<=", ">="):
            if src.startswith(spelling, i):
                if src.startswith(spelling + "=", i):
                    _refuse(owner, base + i, f"`{spelling}=` assigns to the row",
                            "a projection reads the row, it never updates it")
                tokens.append(_Token("op", spelling, None, base + i))
                i += len(spelling)
                break
        else:
            if ch in "+-*/%<>|":
                if src[i + 1:i + 2] == "=":
                    _refuse(owner, base + i, f"`{ch}=` assigns to the row",
                            "a projection reads the row, it never updates it")
                tokens.append(_Token("op", ch, None, base + i))
                i += 1
                continue
            if ch == "=":
                _refuse(owner, base + i, "`=` assigns to the row",
                        "a projection reads the row, it never updates it. Did you mean "
                        "`==`?")
            if ch in "()[],;:?":
                tokens.append(_Token("punct", ch, None, base + i))
                i += 1
                continue
            match = _NAME_RE.match(src, i)
            if match:
                tokens.append(_Token("name", match.group(), match.group(), base + i))
                i = match.end()
                continue
            _refuse(owner, base + i, f"`{ch}` is not part of any construct in the subset",
                    "a document is built from properties, string literals, `+`, `//` and "
                    "the allowed functions")
    tokens.append(_Token("end", "the end of the filter", None, base + len(src)))
    return tokens


# ---------------------------------------------------------------------
# Parser. Recursive descent over jq's own precedence, producing a tree
# used for exactly two things: refusing, and reporting which properties
# are read. It is NOT an evaluator -- see the module docstring.
#
# Node shapes:
#   ("path", base|None, steps)   steps: ("field", name) | ("index", int)
#                                     | ("slice", lo|None, hi|None)
#                                     | ("iterate",) | ("optional",)
#   ("call", name, [arg, ...])   ("bin", op, left, right)
#   ("pipe", left, right)        ("paren", inner)
#   ("array", inner|None)        ("str", [text | node, ...])
#   ("num", text)                ("lit", "true"|"false"|"null")
# ---------------------------------------------------------------------

#: Tokens that cannot start an expression, with the reason a caller most
#: likely needs. Error quality only; the parser refuses them either way.
_UNEXPECTED = {
    "*": "`*` is not in the subset -- `\"x\" * 1000000000` repeats a string a billion "
         "times, which is an unbounded memory amplifier, and multiplication has no role "
         "in building a document. Concatenate with `+`, or use `join`",
    "/": "`/` is not in the subset -- on strings jq's `/` is splitting, which is spelled "
         "`split(\";\")` here, and division has no role in building a document",
    "%": "`%` is not in the subset -- arithmetic beyond `+` and `-` has no role in "
         "building a document",
    ";": "`;` separates the arguments of a multi-argument function, and none in this "
         "subset takes more than one",
    ":": "`:` appears only inside a slice, e.g. `.properties.tags[0:3]`",
    "-": "a leading `-` is not in the subset -- write a subtraction with both sides, "
         "e.g. `.a - 1`, or use a negative index like `.[-1]`",
}


def _is_literal(node) -> bool:
    """A constant the parser can see whole: a number, or a string with no
    interpolation in it. Anything else is data from the row."""
    if node[0] == "num":
        return True
    return node[0] == "str" and all(isinstance(part, str) for part in node[1])


class _Parser:
    def __init__(self, tokens: list, owner: str, depth: int = 0):
        self.tokens = tokens
        self.owner = owner
        self.i = 0
        self.depth = depth

    # -- token plumbing ------------------------------------------------
    def peek(self) -> _Token:
        return self.tokens[self.i]

    def take(self) -> _Token:
        token = self.tokens[self.i]
        self.i += 1
        return token

    def at(self, kind: str, text: Optional[str] = None) -> bool:
        token = self.peek()
        return token.kind == kind and (text is None or token.text == text)

    def accept(self, kind: str, text: Optional[str] = None) -> bool:
        if self.at(kind, text):
            self.i += 1
            return True
        return False

    def expect(self, kind: str, text: str):
        if not self.accept(kind, text):
            token = self.peek()
            _refuse(self.owner, token.pos, f"expected `{text}`, found `{token.text}`",
                    "the filter does not parse in the jq subset hopai accepts")

    def unexpected(self, token: _Token):
        why = _UNEXPECTED.get(token.text)
        if why is None:
            _refuse(self.owner, token.pos, f"`{token.text}` cannot appear here",
                    "the filter does not parse in the jq subset hopai accepts -- a "
                    "document is built from properties, string literals, `+`, `//` and "
                    "the allowed functions")
        _refuse(self.owner, token.pos, "this is not in the subset", why)

    # -- grammar -------------------------------------------------------
    def program(self):
        node = self.pipe()
        token = self.peek()
        if token.kind != "end":
            self.unexpected(token)
        return node

    def pipe(self):
        # Every nesting level -- parens, call arguments, interpolations,
        # array construction -- passes through here, so one guard bounds
        # them all and the parser can never outrun Python's own stack.
        self.depth += 1
        if self.depth > MAX_DEPTH:
            token = self.peek()
            _refuse(self.owner, token.pos, f"the filter nests more than {MAX_DEPTH} deep",
                    "a document projection is a few fields joined together; this is a "
                    "program")
        node = self.comma()
        while self.accept("op", "|"):
            node = ("pipe", node, self.comma())
        self.depth -= 1
        return node

    def comma(self):
        # Comma IS total: it concatenates two finite streams, so its
        # growth is the sum of its sides and nothing iterates. It stays
        # because `[.a, .b] | join(" ")` is the natural way to build a
        # document from several fields, and forbidding it here would
        # push callers to `+` chains that grow the same amount.
        node = self.alternative()
        while self.accept("punct", ","):
            node = ("bin", ",", node, self.alternative())
        return node

    def alternative(self):
        node = self.disjunction()
        while self.accept("op", "//"):
            node = ("bin", "//", node, self.disjunction())
        return node

    def disjunction(self):
        node = self.conjunction()
        while self.at("name", "or"):
            self.take()
            node = ("bin", "or", node, self.conjunction())
        return node

    def conjunction(self):
        node = self.comparison()
        while self.at("name", "and"):
            self.take()
            node = ("bin", "and", node, self.comparison())
        return node

    def comparison(self):
        node = self.additive()
        if self.peek().kind == "op" and self.peek().text in _COMPARISONS:
            operator = self.take()
            node = ("bin", operator.text, node, self.additive())
            following = self.peek()
            # jq's comparisons are non-associative: `a == b == c` is a
            # syntax error there, so accepting it here would accept
            # something libjq rejects -- the one thing the differential
            # test exists to catch.
            if following.kind == "op" and following.text in _COMPARISONS:
                _refuse(self.owner, following.pos, "comparisons do not chain in jq",
                        "compare once, and join with `and`: `.a > 1 and .a < 9`")
        return node

    def additive(self):
        node = self.postfix()
        while self.peek().kind == "op" and self.peek().text in ("+", "-"):
            operator = self.take()
            node = ("bin", operator.text, node, self.postfix())
        return node

    def postfix(self):
        node = self.primary()
        while True:
            token = self.peek()
            if node[0] == "num" and (token.kind == "field" or token.text in ("[", "?")):
                # jq's LEXER takes the dot: it reads `1.a` as the number
                # `1.` followed by an identifier and calls it a syntax
                # error. Found by fuzzing this parser against libjq --
                # it was the only shape in 37,000 accepted programs that
                # libjq refused, and every instance of it.
                _refuse(self.owner, token.pos, "a number takes no suffix",
                        "libjq lexes `1.a` as the number `1.` followed by a name and "
                        "refuses it. Read the property from the row instead, e.g. "
                        "`.properties.title`")
            if token.kind == "field":
                self.take()
                node = self._extend(node, ("field", token.value))
                continue
            if token.kind == "punct" and token.text == "[":
                self.take()
                node = self._extend(node, self.bracket())
                continue
            if token.kind == "punct" and token.text == "?":
                self.take()
                node = self._extend(node, ("optional",))
                continue
            return node

    @staticmethod
    def _extend(node, step):
        """Grow a path, or start one rooted at a non-path expression.

        `(.a + .b) | tostring` and `(.a).b` are both legal jq; the second
        is a path whose base is an expression, which paths_read() then
        treats as derived data rather than as a named property."""
        if node[0] == "path":
            return ("path", node[1], (*node[2], step))
        return ("path", node, (step,))

    def bracket(self):
        """Everything between `[` and `]` on a path, which is where a
        computed index would otherwise slip past the field allowlist."""
        if self.accept("punct", "]"):
            return ("iterate",)
        token = self.peek()
        if token.kind == "end":
            _refuse(self.owner, token.pos, "`[` is never closed",
                    "add the missing `]` -- `[]` iterates, `[0]` indexes, `[0:3]` slices")
        if token.kind == "string":
            _refuse(self.owner, token.pos, "indexing by a string is not in the subset",
                    "there is one spelling for reading a property, and it is the one the "
                    'field allowlist reads: `.foo` or `."foo bar"`')
        low = self._integer()
        if self.accept("punct", ":"):
            high = self._integer()
            if low is None and high is None:
                # libjq rejects `.[:]` outright, and this module must never
                # accept something libjq then refuses to compile.
                _refuse(self.owner, token.pos, "`[:]` is not a slice",
                        "give at least one end -- `[0:3]`, `[2:]`, `[:3]` -- or `[]` to "
                        "iterate every element")
            self.expect("punct", "]")
            return ("slice", low, high)
        if low is None:
            _refuse(self.owner, token.pos, "a computed index is not in the subset",
                    "the index would be DATA, so no static check could see which "
                    "property is read -- it would defeat the field allowlist. Use a "
                    "literal, e.g. `.properties.tags[0]`, or `[]` for every element")
        self.expect("punct", "]")
        return ("index", low)

    def _integer(self) -> Optional[int]:
        """An integer literal, or None when the next token is not one --
        which is how `.[:3]` and a refused `.[.key]` are told apart."""
        sign = 1
        mark = self.i
        if self.at("op", "-"):
            self.take()
            sign = -1
        token = self.peek()
        if token.kind == "number" and token.text.isdigit():
            self.take()
            return sign * int(token.text)
        self.i = mark
        return None

    def primary(self):
        token = self.take()
        if token.kind == "dot":
            return ("path", None, ())
        if token.kind == "field":
            return ("path", None, (("field", token.value),))
        if token.kind == "number":
            return ("num", token.text)
        if token.kind == "string":
            return ("str", [part if isinstance(part, str) else self._interpolated(part)
                            for part in token.value])
        if token.kind == "punct" and token.text == "(":
            node = self.pipe()
            self.expect("punct", ")")
            return ("paren", node)
        if token.kind == "punct" and token.text == "[":
            if self.accept("punct", "]"):
                return ("array", None)
            node = self.pipe()
            self.expect("punct", "]")
            return ("array", node)
        if token.kind == "name":
            return self.call(token)
        self.unexpected(token)

    def _interpolated(self, part):
        """Parse `\\(...)` with the same parser, at the same depth.

        Interpolation is the one place a filter can hide a second filter,
        so it gets no relaxation: the inner expression is in the subset
        or the whole program is refused."""
        _, source, offset = part
        inner = _Parser(_tokenize(source, self.owner, offset), self.owner, self.depth)
        return inner.program()

    def call(self, token: _Token):
        name = token.value
        if name in _KEYWORD_LITERALS:
            return ("lit", name)
        if name in _WORD_OPERATORS:
            _refuse(self.owner, token.pos, f"`{name}` needs an expression on its left",
                    "it joins two conditions, e.g. `.a > 1 and .b == \"x\"`")
        if name not in _ARITY:
            why = _EXPLAIN.get(name)
            if why is not None:
                _refuse(self.owner, token.pos, f"`{name}` is not available", why)
            _refuse(self.owner, token.pos, f"`{name}` is not a function this filter may call",
                    "the allowed functions are " + ", ".join(sorted(_ARITY)) +
                    " -- everything else is either unbounded, regex-driven, or reads "
                    "something that is not the candidate row")
        args = []
        if self.at("punct", "("):
            self.take()
            args.append(self.pipe())
            while self.at("punct", ";"):
                separator = self.take()
                _refuse(self.owner, separator.pos,
                        f"`{name}` is accepted with at most one argument",
                        "the two-argument forms in jq are the regex ones (`split/2`, "
                        "`sub/3`), and a regex from a model is a hang hopai cannot "
                        "interrupt")
            self.expect("punct", ")")
        if args and name in _LITERAL_ARGUMENT and not _is_literal(args[0]):
            _refuse(self.owner, token.pos,
                    f"`{name}` takes a literal argument here",
                    "a separator taken from the row is emitted once per element, which "
                    "grows with the SQUARE of the row rather than with the row -- and a "
                    'computed needle is invisible to the field allowlist. Write it out, '
                    'e.g. `join(", ")`')
        if len(args) not in _ARITY[name]:
            _refuse(self.owner, token.pos,
                    f"`{name}` takes {' or '.join(str(a) for a in sorted(_ARITY[name]))} "
                    f"argument(s), not {len(args)}",
                    "libjq would reject the call too, so hopai refuses it here rather "
                    "than at ranking time")
        return ("call", name, args)


# ---------------------------------------------------------------------
# Growth. Totality bounds how MANY values a filter emits; this bounds
# how BIG they get.
#
# THE MEASURE, stated once because everything below is arithmetic in it:
# a value's size is the number of characters in its compact JSON text,
# and _size() answers, summed over every value the filter emits,
#
#     out <= factor * in + extra
#
# `factor` counts multiples of THE ROW -- the term that has to stay
# small, because the row is the one thing whose size an attacker
# chooses. `extra` counts characters that come out of the PROGRAM's own
# text, which MAX_LENGTH already bounds, so it gets its own much larger
# ceiling (MAX_ADDED). Two numbers rather than one because a multiplier
# and a constant behave differently under composition, and collapsing
# them is what went wrong.
#
# IT COMPOSES the way the operators do -- a pipe multiplies, `+` and `,`
# add, `//` takes whichever side runs. The pipe is where the two terms
# meet: running `g` after `f` gives
#
#     out <= g.factor * (f.factor * in + f.extra) + g.extra
#
# so a later stage MULTIPLIES an earlier stage's constant.
#
# THE RULE THAT WAS MISSING, and the whole of the bug: A CONSTANT
# EMITTED ONCE PER VALUE IS A MULTIPLIER, NOT A CONSTANT. Every place
# the subset repeats something -- `join`'s separator between elements,
# `map(f)`'s body per element, the right of a pipe per value the left
# emitted, a literal in a string per interpolated value, both sides of
# `+` over the cartesian product of their streams -- multiplies program
# text by a count that comes from the ROW. Two counts bound those:
#
#   ELEMENTS. An array of n elements is at least n characters of JSON,
#   so n <= in. `join(sep)` therefore costs `1 + len(sep)`, a
#   MULTIPLIER, where it used to cost `1 + growth(sep)` = 2 for every
#   literal separator however long -- which is what let
#
#       .properties.title | split("") | join("<311 characters>")
#
#   report 2 while multiplying a title by 311 (`split("")` manufactures
#   one element per character), and six of those stages fit inside
#   MAX_LENGTH: reported 64, real 311**6, about 9e14.
#
#   VALUES. A filter emitting n values emits at least n characters, so
#   the same bound applies to a stream: after `.tags[]`, everything
#   downstream runs once per tag. jq's binary operators run over the
#   CARTESIAN PRODUCT of their two streams -- `(.a,.b) + (.c,.d)` emits
#   four values -- so `[.[] + .[]]` SQUARES an array's length, and
#   `.t | split("") | [.[]+.[]] | [.[]+.[]]` measured 70,001 characters
#   out of a ten-character field while the old arithmetic charged 2 per
#   stage. A product of the row with itself is not a multiple of the row
#   at all: _UNBOUNDED is what the arithmetic answers there, because
#   there is no honest number and picking a large one would be a lie in
#   kind rather than in degree.
#
# A SLICE IS THE CREDIT, and the reason `extra` is tracked at all:
# `.[0:30]` bounds the element count to 30 whatever the row holds, so a
# `join` after it costs at most 29 * len(sep) CHARACTERS -- program text
# that stays program text. That is what keeps the ordinary truncation
# idiom `split(" ") | .[0:30] | join(" ")` at a factor of 4 while the
# same shape without the slice pays for its separator.
#
# EVERY UNKNOWN ROUNDS UP: an element count nothing bounds is None, a
# value count nothing bounds is None, and both are then charged as "as
# many as the input has characters".
# ---------------------------------------------------------------------

#: A filter whose output is a PRODUCT of the row with itself rather than
#: a multiple of it -- see VALUES above. It propagates through every
#: rule here (a bigger factor is still bigger after multiplying, adding
#: or taking a max) and is reported with its own sentence, because "N
#: times its own input" is the wrong SHAPE of answer for it.
_UNBOUNDED = math.inf


class _Size(NamedTuple):
    """What a subexpression can emit.

    `factor` and `extra` bound the total characters of JSON it produces:
    `out <= factor * in + extra`, summed over every value. `count`
    bounds how many ELEMENTS each of those values holds, and `outputs`
    how many values there are -- None for either means "bounded only by
    the input", which is the case the two multipliers above charge for.
    Both are upper bounds and both may be rounded up freely; rounding
    DOWN is the bug."""
    factor: float
    extra: int
    count: Optional[int] = None
    outputs: Optional[int] = 1


_UNBOUNDED_SIZE = _Size(_UNBOUNDED, 0, None, None)

#: Calls that hand back the same value, or the same elements reordered
#: or thinned -- so a slice's bound on the element count survives them.
#: `flatten` and `add` are deliberately absent: both can raise the
#: element count above what the slice allowed.
_COUNT_KEPT = frozenset({
    "arrays", "empty", "map", "numbers", "objects", "reverse", "select",
    "sort", "strings", "unique", "values",
})


def _literal_length(node) -> int:
    """Characters a literal argument contributes when it is EMITTED --
    `join`'s separator. _LITERAL_ARGUMENT is what guarantees there is a
    number here to read at all."""
    if node[0] == "num":
        return len(node[1])
    return sum(len(part) for part in node[1] if isinstance(part, str))


def _sliced(step, count: Optional[int]) -> Optional[int]:
    """The element bound a slice leaves behind. A slice never ADDS
    elements, so an unusable one (open-ended, or counting from the end)
    keeps whatever bound was already there rather than losing it."""
    _, low, high = step
    if high is None or high < 0 or (low is not None and low < 0):
        return count
    bound = max(high - (low or 0), 0)
    return bound if count is None else min(count, bound)


def _paired(size: _Size, other: _Size) -> _Size:
    """`size`, emitted once per value `other` produced.

    This is the one rule the growth analysis was missing, in the one
    place every construct that repeats something goes through. A KNOWN
    number of repetitions leaves a constant a constant; an unknown one
    is bounded by `other`'s own size -- every value is at least one
    character of it -- which turns the constant into a multiplier. And
    a row-proportional COUNT meeting a row-proportional SIZE is the row
    squared, which no factor describes."""
    if size.factor == _UNBOUNDED or other.factor == _UNBOUNDED:
        return _UNBOUNDED_SIZE
    if other.outputs is not None:
        return _Size(size.factor * other.outputs, size.extra * other.outputs)
    if size.factor:
        return _UNBOUNDED_SIZE
    return _Size(size.extra * other.factor, size.extra * other.extra)


def _both(left: Optional[int], right: Optional[int]) -> Optional[int]:
    """A count over a cartesian product: unknown if either side is."""
    return None if left is None or right is None else left * right


def _concat(left: _Size, right: _Size) -> _Size:
    """Two streams concatenated pairwise -- `+`, and equally the pieces
    of an interpolated string, which is the same operation with a
    literal on one side."""
    first, second = _paired(left, right), _paired(right, left)
    return _Size(first.factor + second.factor, first.extra + second.extra,
                 None, _both(left.outputs, right.outputs))


def _size(node, count: Optional[int] = None) -> _Size:
    """The bound on `node`'s output, given that its INPUT holds at most
    `count` elements (None = only the input's own size bounds it)."""
    kind = node[0]
    if kind == "path":
        base, steps = node[1], node[2]
        size = _Size(1, 0, count) if base is None else _size(base, count)
        count, outputs = size.count, size.outputs
        for step in steps:
            if step[0] == "slice":
                count = _sliced(step, count)
            elif step[0] == "iterate":
                # `.[]` turns elements into VALUES: the element bound
                # becomes the output count, and each value's own element
                # count is no longer known.
                outputs, count = _both(outputs, count), None
            elif step[0] != "optional":
                # A field or an index yields a value whose element count
                # nothing here knows -- and navigating never grows what
                # it navigates into.
                count = None
        return _Size(size.factor, size.extra, count, outputs)
    if kind == "pipe":
        left = _size(node[1], count)
        right = _size(node[2], left.count)
        if _UNBOUNDED in (left.factor, right.factor):
            # Explicitly, because `_UNBOUNDED * 0` is not a number and a
            # stage with no factor of its own (`... | "constant"`) would
            # otherwise turn the answer into NaN, which compares False
            # against every limit on this page.
            return _UNBOUNDED_SIZE
        outputs = _both(left.outputs, right.outputs)
        repeated = _paired(_Size(0, right.extra), left)
        return _Size(left.factor * right.factor + repeated.factor,
                     right.factor * left.extra + repeated.extra,
                     right.count, outputs)
    if kind == "paren":
        return _size(node[1], count)
    if kind == "array":
        if node[1] is None:
            return _Size(0, 2, 0)                      # `[]`
        inner = _size(node[1], count)
        commas = _paired(_Size(0, 1), inner)           # one per value collected
        return _Size(inner.factor + commas.factor, inner.extra + commas.extra + 2,
                     inner.outputs)
    if kind == "str":
        size = _Size(0, 2)                             # the quotes
        for part in node[1]:
            if isinstance(part, str):
                size = _concat(size, _Size(0, len(part)))
                continue
            # An interpolated NON-string is rendered as JSON, and
            # escaping can double it.
            inner = _size(part, count)
            size = _concat(size, _Size(2 * inner.factor, 2 * inner.extra,
                                       None, inner.outputs))
        return size
    if kind == "bin":
        operator = node[1]
        left, right = _size(node[2], count), _size(node[3], count)
        if operator == ",":
            # Two streams end to end rather than a product: the sizes
            # add, and so do the value counts.
            outputs = None if left.outputs is None or right.outputs is None \
                else left.outputs + right.outputs
            return _Size(left.factor + right.factor, left.extra + right.extra,
                         None, outputs)
        if operator == "+":
            return _concat(left, right)
        if operator == "-":
            # `a - b` never grows its left side, but it still runs once
            # per pair, so the left is repeated per value on the right.
            return _paired(left, right)._replace(
                count=left.count, outputs=_both(left.outputs, right.outputs))
        if operator == "//":
            # Only one side runs, so this is a choice and not a product.
            return _Size(max(left.factor, right.factor),
                         max(left.extra, right.extra), None,
                         _both(left.outputs, right.outputs))
        boolean = _paired(_Size(0, 5), left)           # `true` / `false`
        return _paired(boolean, right)._replace(outputs=_both(left.outputs,
                                                              right.outputs))
    if kind == "call":
        return _call_size(node[1], node[2], count)
    return _Size(0, len(node[1]))                      # num, lit: their own text


def _call_size(name: str, args: list, count: Optional[int]) -> _Size:
    """A call's bound. An argument is a PARAMETER, not something
    appended per element, so it costs nothing -- except where the
    function EMITS it (`join`) or emits its output (`map`, `first`,
    `last`). Charging every argument made `... | ltrimstr("x") |
    join(", ")` look like 4x growth and put ordinary projections near
    the cap."""
    if name == "join":
        separator = _literal_length(args[0]) if args else 0
        if count is None:
            # One separator per element, and an array of n elements is
            # at least n characters of input -- so the separators alone
            # are at most len(sep) times the input. THIS is the term
            # that used to be a flat +1 and let a 311-character
            # separator multiply a row by 311 for a reported cost of 2.
            return _Size(1 + separator, 2)
        # A slice bounded the elements, so the separators are a constant
        # the program wrote out rather than a multiple of the row.
        return _Size(1, max(count - 1, 0) * separator + 2)
    if name == "map" and args:
        # `map(f)` is `[.[] | f]`: f runs once per element, so f's
        # CONSTANT is emitted once per element and the element count is
        # bounded by the input -- the same arithmetic as join's
        # separator. The +1 is the array's own commas. The count
        # survives: map is elementwise and cannot add elements.
        inner = _size(args[0], None)
        if inner.factor == _UNBOUNDED:
            return _UNBOUNDED_SIZE
        return _Size(inner.factor + inner.extra + 1, inner.extra + 2, count)
    if name in ("first", "last") and args:
        return _size(args[0], None)._replace(count=None, outputs=1)
    if name == "split":
        # `split("")` is the element factory the whole amplifier is
        # built on: "abcd" (6 characters of JSON) becomes
        # ["a","b","c","d"] (16), and the ratio approaches 4 as the
        # string grows -- three characters of framing per element, one
        # element per character. Charging 1 for it would leave the
        # nesting `split("") | map(split(""))` free.
        return _Size(4, 4)
    if name in ("tostring", "tojson"):
        return _Size(2, 2)                             # escaping can double a string
    if name in _COUNT_KEPT:
        return _Size(1, 0, count)
    return _Size(1, 0)                                 # every other allowed call shrinks


def _growth(node) -> float:
    """How many times its own input `node` may emit -- the number the
    refusal names, or _UNBOUNDED when no number is the right answer."""
    return _size(node).factor


def _magnitude(number: int) -> str:
    """A number an error message can carry. Growth is a PRODUCT over
    pipe stages, so a filter inside MAX_LENGTH can reach hundreds of
    digits, and printing those would bury the sentence that names the
    fix."""
    if number < 10 ** 9:
        return f"{number:,}"
    return f"about 10^{len(str(number)) - 1}"


# ---------------------------------------------------------------------
# Which properties a filter reads. Used for the operator-side `fields=`
# allowlist and for error messages, so it errs toward reporting MORE:
# an over-reported path makes validate() stricter, never leakier.
# ---------------------------------------------------------------------

def _dotted(ctx) -> str:
    """A context as the caller writes it. The root -- a bare `.`, which
    reads the entire row -- is reported as "." rather than "", so a
    message can name it and an allowlist can refuse it."""
    return ".".join(ctx) if ctx else "."


def _flow(node, ctx, out: set):
    """Walk `node` with `ctx` as the path its input came from, recording
    reads into `out`, and return the path of its OUTPUT (or None once
    the value is derived rather than named).

    The distinction between this and _value() is what keeps
    `.properties | .title` from reporting a read of `properties`: on the
    left of a pipe a path is navigation, and only the value that is
    finally consumed is a read."""
    kind = node[0]
    if kind == "path":
        base, steps = node[1], node[2]
        if base is not None:
            _value(base, ctx, out)
            return None                                # derived; not a named property
        if ctx is None:
            return None
        for step in steps:
            if step[0] == "field":
                ctx = (*ctx, step[1])
            # index / slice / iterate / optional stay inside the same
            # property, which is exactly what the prefix rule wants:
            # `properties.tags[]` is within `properties.tags`.
        return ctx
    if kind == "pipe":
        return _flow(node[2], _flow(node[1], ctx, out), out)
    if kind == "paren":
        return _flow(node[1], ctx, out)
    if kind == "array":
        if node[1] is not None:
            _value(node[1], ctx, out)
        return None
    if kind == "str":
        for part in node[1]:
            if not isinstance(part, str):
                _value(part, ctx, out)
        return None
    if kind == "bin":
        _value(node[2], ctx, out)
        _value(node[3], ctx, out)
        return None
    if kind == "call":
        name, args = node[1], node[2]
        for arg in args:
            # An argument runs against the same data -- inside `map(f)`
            # against its ELEMENTS, which live at the same path.
            _value(arg, ctx, out)
        if name in _PASSTHROUGH:
            return ctx
        if ctx is not None:
            out.add(_dotted(ctx))                      # consuming the value reads it
        return None
    return None                                        # num, lit: reads nothing


def _value(node, ctx, out: set):
    """_flow(), plus the record that makes the result a read: whatever a
    value position ends up holding is emitted, so its path is read.

    This is what catches `.properties.ssn | first` -- `first` passes its
    input through, so nothing inside records anything, but the row's ssn
    is what comes out."""
    result = _flow(node, ctx, out)
    if result is not None:
        out.add(_dotted(result))
    return result


def _read_paths(node) -> frozenset:
    out: set = set()
    _value(node, (), out)
    return frozenset(out)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

#: Python frames each level of filter nesting can cost. The scanners and
#: the parser recurse into each other, so one `(` is several frames, not
#: one -- and under an INSTRUMENTED interpreter it is several more,
#: because a coverage tracer, a profiler or a mutation harness wraps
#: every call: mutmut routes each one through a trampoline, and that is
#: where this was found (RecursionError out of
#: mutmut/mutation/trampoline.py, on a filter this module refuses
#: cleanly under a bare interpreter).
#:
#: Reserving generously is safe because the parser's recursion is
#: bounded BY CONSTRUCTION -- it refuses past MAX_DEPTH -- so this
#: number cannot licence a runaway; it only decides which of the two
#: limits is reached first, and the answer must always be ours.
_FRAMES_PER_LEVEL = 60

#: Frames left over for _growth()/_read_paths(), which walk the tree the
#: parser just built and so recurse to the same depth.
_FRAME_SLACK = 200


def _current_depth() -> int:
    """How many frames are already on the stack below us."""
    depth, frame = 0, sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


@contextmanager
def _stack_headroom():
    """Guarantee MAX_DEPTH is reachable before Python's own limit is.

    MAX_DEPTH is a promise about the FILTER; RecursionError is about the
    PROCESS. The frames already on the stack when validate() is called
    are not ours -- a caller deep inside a web framework, a test runner,
    or a mutation harness starts us far closer to the ceiling than a
    caller at module scope. So the guard could be reached only AFTER
    Python gave up, and RecursionError is not an UnsafeFilter: a caller
    doing `except UnsafeFilter` crashes on a filter that is merely
    silly, which is the exact failure this module's depth limit exists
    to prevent.

    Found by mutation testing, whose mutants tree runs the parser from a
    deeper stack than pytest alone -- the promise had been true only for
    shallow callers, and nothing said so.

    The limit is raised only when it is already too low, and always put
    back: this borrows stack for one parse of a filter bounded by
    MAX_LENGTH, it does not lift a ceiling for the process."""
    need = _current_depth() + MAX_DEPTH * _FRAMES_PER_LEVEL + _FRAME_SLACK
    before = sys.getrecursionlimit()
    if before >= need:
        yield
        return
    sys.setrecursionlimit(need)
    try:
        yield
    finally:
        sys.setrecursionlimit(before)


def _compile(program: str, owner: str):
    if not isinstance(program, str):
        raise TypeError(
            f"{owner}: expected a jq filter as a string, got {type(program).__name__} -- "
            f'e.g. \'.properties.title + ": " + (.properties.summary // "")\''
        )
    if not program.strip():
        raise UnsafeFilter(
            f"{owner}: the filter is empty -- give the projection that builds a document "
            f'from a row, e.g. `.properties.title`'
        )
    if len(program) > MAX_LENGTH:
        raise UnsafeFilter(
            f"{owner}: the filter is {len(program)} characters, over the {MAX_LENGTH} "
            f"character limit -- a document projection is a few fields joined together, "
            f"so this is a program"
        )
    # Before the tokenizer, because these two never reach libjq AS
    # THEMSELVES and the refusal has to come from here rather than as a
    # parse error out of a layer below. See _UNENCODABLE_RE.
    found = _UNENCODABLE_RE.search(program)
    if found is not None:
        nul = found.group() == "\x00"
        raise UnsafeFilter(
            f"{owner}: the filter contains "
            f"{'a NUL byte' if nul else 'a lone surrogate'} at offset {found.start()} -- "
            + ("libjq reads the program as a C string, so the NUL ends it there and "
               "libjq reports `unexpected end of file` on the half it saw"
               if nul else
               "it cannot be encoded as UTF-8, so the filter cannot reach libjq at all")
            + ". Write the document projection in text, e.g. `.properties.title`"
        )
    with _stack_headroom():
        node = _Parser(_tokenize(program, owner), owner).program()
        size = _size(node)
    if size.factor == _UNBOUNDED:
        # No multiple of the row bounds this one, so it gets a sentence
        # rather than a number: `[.[] + .[]]` squares an array's length
        # because jq's binaries run over the cartesian product of their
        # two streams, and stages of it square the square.
        raise UnsafeFilter(
            f"{owner}: this filter's output grows with the SQUARE of the row, not with "
            f"the row -- an operator with a STREAM on both sides (`.tags[] + .tags[]`, "
            f"or a `+` after `.[]`) runs once per pair, so an n-element array emits n*n "
            f"values. Build the document from one value per candidate, e.g. "
            f"`[.properties.tags[]] | join(\", \")`"
        )
    if size.factor > MAX_GROWTH:
        raise UnsafeFilter(
            f"{owner}: this filter can emit {_magnitude(size.factor)} times its own "
            f"input, over the {MAX_GROWTH}x limit -- repeated concatenation doubles at "
            f"every step, and a `join` separator is written once per element while "
            f"`split(\"\")` makes one element per character, so a short filter can ask "
            f"for more memory than the process has. Build the document from the fields "
            f"you need, joined once"
        )
    if size.extra > MAX_ADDED:
        raise UnsafeFilter(
            f"{owner}: this filter adds {_magnitude(size.extra)} characters to every "
            f"document whatever the row holds, over the {MAX_ADDED} character limit -- a "
            f"separator repeated across a slice this long is program text multiplied by "
            f"the slice, not a document. Shorten the separator, or the slice"
        )
    return node


def _allowed(fields, owner: str):
    """Normalize the operator's allowlist. A leading `.` is accepted
    because that is how the same path is written inside the filter, and
    a caller should not have to know which side wants which spelling."""
    normalized = []
    for field in fields:
        if not isinstance(field, str):
            raise TypeError(
                f"{owner}: fields= takes dotted property paths as strings, got "
                f'{type(field).__name__} -- e.g. ["properties.title", "properties.tags"]'
            )
        cleaned = field.strip()
        if cleaned.startswith("."):
            cleaned = cleaned[1:]
        if not cleaned:
            raise ValueError(
                f"{owner}: fields= contains an empty path. To allow the whole row, pass "
                f"fields=None"
            )
        normalized.append(cleaned)
    return tuple(normalized)


def _within(path: str, allowed) -> bool:
    """A read is allowed when it is AT or BENEATH an allowed path --
    `properties.tags[]` reads `properties.tags`, and `.properties.tags[0].name`
    reads `properties.tags.name`, both of which are that field's own
    data. A read ABOVE one (`properties`) is not: it hands back the
    siblings the allowlist exists to withhold."""
    return any(path == field or path.startswith(field + ".") for field in allowed)


def validate(program: str, *, fields=None, owner: str = "document_from") -> None:
    """Raise UnsafeFilter if `program` is outside the subset, or --
    when `fields` is given -- reads a property path outside it.

        validate('.properties.title', fields=["properties.title"])

    `fields` is an allowlist of dotted paths; a read at or beneath one
    is allowed, a read above one is not. `owner` names the option the
    filter arrived on, so the message points at the caller's own
    spelling rather than at this module."""
    node = _compile(program, owner)
    if fields is None:
        return
    allowed = _allowed(fields, owner)
    named = ", ".join(allowed) if allowed else "(none)"
    for path in sorted(_read_paths(node)):
        if _within(path, allowed):
            continue
        if path == ".":
            raise UnsafeFilter(
                f"{owner}: `.` reads the whole row, which is more than this filter may "
                f"see -- the fields it may read are: {named}. Read them by name, e.g. "
                f"`.{allowed[0] if allowed else 'properties.title'}`"
            )
        raise UnsafeFilter(
            f"{owner}: reads `.{path}`, which is not one of the fields this filter may "
            f"see: {named}. Read one of those, or add `{path}` to the fields this "
            f"reranker is allowed to send"
        )


def paths_read(program: str) -> frozenset[str]:
    """Dotted top-level paths the program reads, e.g. {"properties.title"}.
    Used for the operator-side field allowlist and for error messages.

    Index, slice and iteration steps do not add a segment, because they
    stay inside the same property: `.properties.tags[0].name` reads
    `properties.tags.name`. A filter that reads the entire row -- a bare
    `.` -- reports `"."`. Reads are OVER-reported where the analysis
    cannot be sure (a path off a computed base is attributed to the base
    it came from), never under-reported: the set is what `validate()`
    refuses on."""
    return _read_paths(_compile(program, "paths_read"))


def is_total(program: str) -> bool:
    """True when the program parses in the subset (and so terminates).

    There is no third answer and no timeout: the subset has no
    recursion, no generator, no loop and no fold, so parsing IS the
    termination proof. See the module docstring."""
    try:
        _compile(program, "is_total")
    except UnsafeFilter:
        return False
    return True
