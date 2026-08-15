"""
Fix links written inside notebooks when building the documentation site.

The notebooks link to their neighbours the way GitHub needs -- `[02 ·
Traversal](02_traversal.ipynb)`, `[demo_graph.py](demo_graph.py)`, plain
sibling files -- and that has to keep working, because reading them on GitHub
is still the primary path.

MkDocs rewrites relative links itself, but only for pages it renders from
markdown: mkdocs-jupyter hands it finished HTML, so these links never reach
that pass. `--strict` does not see them either, and they would ship pointing
at files the site does not serve. Two things move under them:

  * `notebooks/X.ipynb` is published at `notebooks/X/index.html`, one
    directory deeper than the file it came from, so a sibling of the notebook
    is one level up from the page.
  * a sibling notebook is published as a page, so it is `../X/`, not a file.

Only BARE filenames are touched -- a target with no `/` in it is the one
thing that can have been written relative to the notebook. Everything with a
slash is either the theme's own chrome (`../`, `../../assets/...`) or an
absolute URL, and rewriting those would break the site's navigation.
"""

from __future__ import annotations

import re

#: A bare sibling filename: no scheme, no slash, no anchor, no query.
_SIBLING = re.compile(r'(href|src)="(?!\w+:|//)([^"/#?]+)"')

_SUFFIX = ".ipynb"


def on_post_page(output: str, page, config) -> str:
    """Re-point every sibling link in a notebook page at what the site serves."""
    if not page.file.src_uri.endswith(_SUFFIX):
        return output

    own_source = page.file.src_uri.rsplit("/", 1)[-1]
    # Without directory URLs the page sits in the same directory as its
    # source, so siblings resolve unchanged and only notebooks need mapping.
    up = "../" if config.get("use_directory_urls", True) else ""

    def replace(match: re.Match) -> str:
        attribute, target = match.group(1), match.group(2)
        # `include_source` copies the notebook's own file next to its page for
        # download. That link already resolves; rewriting it would send the
        # reader to the page they are standing on.
        if target == own_source:
            return match.group(0)
        if target.endswith(_SUFFIX):
            stem = target[: -len(_SUFFIX)]
            destination = f"{up}{stem}/" if up else f"{stem}.html"
        else:
            destination = f"{up}{target}"
        return f'{attribute}="{destination}"'

    return _SIBLING.sub(replace, output)
