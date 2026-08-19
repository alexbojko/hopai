#!/usr/bin/env python3
"""
Compute which mutants a PR's diff actually needs checked.

    python scripts/mutation_scope.py <base-sha> <head-ref> <changed-file>...

Prints mutmut-compatible fnmatch patterns, space-separated, on stdout --
one `<module>.<mangled-name>__mutmut_*` per top-level function or method
whose line span overlaps a changed line. Diagnostics (which file
produced which patterns) go to stderr.

`setup.cfg`'s `source_paths` already narrows which FILES get mutants
generated at all, but every mutant mutmut generates for a covered line
in a 300-line function still gets CHECKED even when the diff only
touched five of those lines -- checking is the expensive half (it reruns
the test suite once per mutant), so a five-line fix in a large,
well-covered function paid for hundreds of unrelated mutants. This
computes the finer-grained scope: the exact set of functions/methods the
diff touched, passed to `mutmut run` as positional MUTANT_NAMES so only
mutants inside those functions are ever checked.

Function-level, not line-level, because that is the finest granularity
mutmut's own naming exposes -- it mangles every mutant of a function as
`<module>.<mangled-name>__mutmut_<N>` with no per-line identifier, and a
nested/inner function's mutants are attributed to the enclosing
top-level function's name rather than named separately. A one-line
change inside a ten-line function checks that function's mutants, not
the file's -- there is nothing finer to ask mutmut for.
"""

from __future__ import annotations

import ast
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


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: mutation_scope.py <base-sha> <head-ref> <changed-file>...", file=sys.stderr)
        return 2
    base, head, *paths = argv[1:]
    print(" ".join(resolve_scope(base, head, paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
