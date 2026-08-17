# Runnable documentation

Ten notebooks that go from an empty database to the SQL underneath. They are
documentation, so they are meant to be read with the outputs already there —
on GitHub, or on the [documentation site](https://hopai.readthedocs.io/),
which renders these same files and is published on every release — and they
run, so nothing in them can quietly stop being true.

| Notebook | What it covers |
| --- | --- |
| [01 · Quick start](01_quickstart.ipynb) | Connect, create the schema, write nodes and edges, run the first traversal, read the result |
| [02 · Traversal](02_traversal.ipynb) | The filter language, hop counts, direction, `OPTIONAL`, and the four invariants the engine is built around |
| [03 · Aggregation](03_aggregation.ipynb) | `count`/`sum`/`avg`/`min`/`max` in the database, what they run over, and the empty and non-numeric cases |
| [04 · JSON and Cypher](04_json_and_cypher.ipynb) | The same question in three notations, the LLM tool schemas, Cypher reads and writes, every refusal with its rewrite, and refusing vocabulary the schema does not declare |
| [05 · Constraints](05_constraints.ipynb) | Unique, composite, partial, existence, type and CHECK constraints on JSONB properties |
| [06 · Graph schema](06_graph_schema.ipynb) | Declaring the shape as dataclasses, reading it back as JSON Schema or a Mermaid diagram, value sets and formats and per-type uniqueness, enforcing it all in Postgres, inferring it (whole or sampled) from data that came first, and persisting it for other processes to load |
| [07 · Many graphs](07_many_graphs.ipynb) | Thousands of isolated graphs in one pair of tables, and what keeps them apart |
| [08 · Under the hood](08_under_the_hood.ipynb) | The emitted SQL read with no database, then `EXPLAIN ANALYZE` with one |
| [09 · Vector search](09_vector_search.ipynb) | Exact cosine similarity on nodes and edges, multivector and hybrid ranking, similarity as a traversal seed and as an edge beam, and the two refusals |
| [10 · Reranking](10_reranking.ipynb) | The third retrieval stage: `Rerank` on a flat search and step-wise inside a walk, `document_from` as a jq rule, `candidates` against `k`/`keep`, every refusal, and the total jq subset a model may write |

Read them in order the first time; after that each one stands alone.

## Running them

```bash
pip install -e ".[dev,notebooks]"    # from the repository root
docker compose up -d                 # PostgreSQL matching the default DSN
jupyter lab notebooks/
```

Point `HOPAI_DSN` at your own PostgreSQL to use that instead.

Every notebook owns one PostgreSQL schema (`nb_01_quickstart`, `nb_02_traversal`,
…) and drops and recreates it in its first cell, so "Run All" means the same thing
the second time and the notebooks do not depend on each other or on the test
suite's schemas. [`demo_graph.py`](demo_graph.py) holds that setup, the seven-node
demo graph they share, and two display helpers.

## Keeping them honest

```bash
python scripts/run_notebooks.py              # execute all of them, fail on any error
python scripts/run_notebooks.py 03           # just the ones whose name matches
python scripts/run_notebooks.py --check      # ...and fail if outputs went stale
python scripts/run_notebooks.py --save       # ...and write the outputs back in
```

CI runs `--check` on every PR, with `HOPAI_REQUIRE_DB=1` so a missing database is
an error rather than a skip. A bare run only proves the notebook still executes;
`--check` also diffs the fresh output against the committed one (ignoring
execution metadata and timing-shaped numbers) and names the exact cell that
went stale. Committed outputs are regenerated with `--save` — do that when an
API change makes an output stale, and read the diff rather than trusting it.

The notebooks lean on assertions where the behaviour is the point (fan-in
preserved, dead ends pruned, `NOT` keeping rows with a missing key), so a
regression fails the run instead of quietly printing a different number.
