#!/usr/bin/env python3
"""
Work out what a PR's diff actually changed, for mutmut to mutate.

    python scripts/mutation_scope.py <base-sha> <head-ref> <changed-file>...
    python scripts/mutation_scope.py --changed-lines <base> <head> <file>...

Two outputs off the same diff, because mutmut needs both to be told
"only what this PR touched", at two different stages:

`--changed-lines` prints `{"<path>": [line, ...]}` as JSON -- the lines
the diff touched. `scripts/mutation_run.py` feeds this to mutmut's
per-line generation filter, so a mutant is only ever CREATED on a
changed line. This is the one that matters: a full run over `hopai/` is
~10000 mutants, and mutating only the changed lines of a small PR is a
few dozen.

The default output prints mutmut-compatible fnmatch patterns,
space-separated -- one `<module>.<mangled-name>__mutmut_*` per top-level
function or method whose line span overlaps a changed line. These go to
`mutmut run` as positional MUTANT_NAMES, which filters what gets
CHECKED. Function-level, because that is the finest granularity mutmut's
naming exposes: it mangles every mutant as `<name>__mutmut_<N>` with no
per-line identifier. It is the coarser of the two, and it is kept as a
second gate so that a stale `mutants/` tree carried over from an earlier
run cannot smuggle in mutants from a line this PR never touched.

Diagnostics (which file produced what) go to stderr in both modes.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys

#: mutmut's own separator between a class name and a method name in a
#: mangled mutant id -- U+01C1 LATIN LETTER LATERAL CLICK, not the
#: dental-click lookalike U+01C0. See mutmut/mutation/trampoline_templates.py.
CLASS_NAME_SEPARATOR = "ǁ"

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.M)


def changed_line_numbers(base: str, head: str, path: str, cwd: str | None = None) -> set[int]:
    """Line numbers the diff touches in `path` at `head`, from unified-diff
    hunk headers -- added lines and the line a pure deletion sits next to."""
    diff = subprocess.run(
        ["git", "diff", "-U0", "--diff-filter=d", base, head, "--", path],
        capture_output=True, text=True, check=True, cwd=cwd,
    ).stdout
    lines: set[int] = set()
    for match in _HUNK_HEADER_RE.finditer(diff):
        start = int(match.group(1))
        count = int(match.group(2)) if match.group(2) is not None else 1
        if count == 0:
            # A pure deletion: nothing added at `start`, but the line it
            # was removed from beforehand is the nearest surviving context.
            lines.add(max(start, 1))
        else:
            lines.update(range(start, start + count))
    return lines


def _mangle(name: str, class_name: str | None) -> str:
    if class_name:
        return f"x{CLASS_NAME_SEPARATOR}{class_name}{CLASS_NAME_SEPARATOR}{name}"
    return f"x_{name}"


def _module_name(path: str) -> str:
    if not path.endswith(".py"):
        raise ValueError(f"not a Python file: {path}")
    return path[:-3].replace("/", ".")


def touched_function_patterns(
    head_ref: str, path: str, changed: set[int], cwd: str | None = None
) -> list[str]:
    """fnmatch patterns for every top-level function/method in `path` (as of
    `head_ref`) whose line span overlaps `changed`. Module-level statements,
    docstrings and comments outside any function body yield no pattern --
    there is no mutant to attribute them to."""
    src = subprocess.run(
        ["git", "show", f"{head_ref}:{path}"],
        capture_output=True, text=True, check=True, cwd=cwd,
    ).stdout
    tree = ast.parse(src, filename=path)
    module = _module_name(path)
    patterns = []

    def overlaps(node: ast.AST) -> bool:
        return any(node.lineno <= ln <= node.end_lineno for ln in changed)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and overlaps(node):
            patterns.append(f"{module}.{_mangle(node.name, None)}__mutmut_*")
        elif isinstance(node, ast.ClassDef) and overlaps(node):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and overlaps(sub):
                    patterns.append(f"{module}.{_mangle(sub.name, node.name)}__mutmut_*")
    return patterns


def resolve_scope(base: str, head: str, paths: list[str], cwd: str | None = None) -> list[str]:
    """The sorted, de-duplicated set of mutmut patterns for every changed
    file, with one diagnostic line per file on stderr."""
    all_patterns: list[str] = []
    for path in paths:
        changed = changed_line_numbers(base, head, path, cwd=cwd)
        patterns = touched_function_patterns(head, path, changed, cwd=cwd)
        print(f"{path}: {len(changed)} changed line(s) -> {len(patterns)} function pattern(s)",
              file=sys.stderr)
        for pattern in patterns:
            print(f"    {pattern}", file=sys.stderr)
        all_patterns.extend(patterns)
    return sorted(set(all_patterns))


def changed_lines_by_file(
    base: str, head: str, paths: list[str], cwd: str | None = None
) -> dict[str, list[int]]:
    """`{path: sorted changed line numbers}`, skipping files the diff
    touched but left with no line numbers to mutate."""
    result: dict[str, list[int]] = {}
    for path in paths:
        changed = changed_line_numbers(base, head, path, cwd=cwd)
        print(f"{path}: {len(changed)} changed line(s)", file=sys.stderr)
        if changed:
            result[path] = sorted(changed)
    return result


def main(argv: list[str]) -> int:
    args = argv[1:]
    as_json = False
    if args and args[0] == "--changed-lines":
        as_json = True
        args = args[1:]
    if len(args) < 2:
        print("usage: mutation_scope.py [--changed-lines] <base-sha> <head-ref> <changed-file>...",
              file=sys.stderr)
        return 2
    base, head, *paths = args
    if as_json:
        print(json.dumps(changed_lines_by_file(base, head, paths)))
    else:
        print(" ".join(resolve_scope(base, head, paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
