#!/usr/bin/env python
"""
Resolve every internal link in the built documentation site and fail on a dead one.

`mkdocs build --strict` checks the links it can see, which is the links in
markdown pages. It cannot see the ones written inside notebooks, because
mkdocs-jupyter hands MkDocs finished HTML -- and those are exactly the links
that move when a notebook becomes a page one directory deeper
(scripts/mkdocs_hooks.py rewrites them). Without this, that rewrite could
stop working and nothing would say so until a reader hit a 404.

    python scripts/check_docs_links.py           # after `mkdocs build`
    python scripts/check_docs_links.py --site out

Absolute links are resolved against the path in `site_url`, because that is
what the built site is mounted at once deployed: on GitHub Pages the site
root is /hopai/, not /.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import urllib.parse

#: Only href/src. Anything with a scheme, a protocol-relative host or a bare
#: fragment points somewhere this script cannot and should not resolve.
_LINK = re.compile(r'(?:href|src)="([^"]+)"')
_EXTERNAL = re.compile(r"\w+:|//|#")


def site_base(config: pathlib.Path) -> str:
    """The path `site_url` mounts the site at, e.g. `/hopai/`."""
    match = re.search(r"^site_url:\s*(\S+)", config.read_text(), flags=re.M)
    if not match:
        return "/"
    return urllib.parse.urlparse(match.group(1).strip("\"'")).path or "/"


def check(site: pathlib.Path, base: str) -> list[str]:
    broken, pages = [], sorted(site.rglob("*.html"))
    if not pages:
        sys.exit(f"error: no pages under {site}/ -- run `mkdocs build` first")

    for page in pages:
        for raw in _LINK.findall(page.read_text(errors="ignore")):
            link = raw.strip()
            if not link or _EXTERNAL.match(link):
                continue
            path = urllib.parse.unquote(link.partition("#")[0].partition("?")[0])
            if not path:
                continue
            if path.startswith("/"):
                if not path.startswith(base):
                    broken.append(f"{page.relative_to(site)} -> {raw}  (outside {base})")
                    continue
                target = site / path[len(base):]
            else:
                target = page.parent / path
            target = target.resolve()
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                broken.append(f"{page.relative_to(site)} -> {raw}")

    print(f"checked {len(pages)} page(s) under {site}/ (site root = {base})")
    return sorted(set(broken))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default="site", type=pathlib.Path,
                        help="the built site directory (default: site)")
    parser.add_argument("--config", default="mkdocs.yml", type=pathlib.Path,
                        help="mkdocs.yml, read for site_url (default: mkdocs.yml)")
    args = parser.parse_args()

    broken = check(args.site, site_base(args.config))
    if broken:
        print(f"\n{len(broken)} broken internal link(s):", file=sys.stderr)
        for entry in broken:
            print(f"  {entry}", file=sys.stderr)
        return 1
    print("no broken internal links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
