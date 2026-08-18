# Changelog

## [0.1.0](https://github.com/alexbojko/hopai/compare/v0.0.1...v0.1.0) (2026-08-17)


### ⚠ BREAKING CHANGES

* drop sqlmodel, declare nodes/edges as plain SQLAlchemy Core tables ([#46](https://github.com/alexbojko/hopai/issues/46))

### Features

* an opt-in MCP server for CRUD on the graph and its schema ([#43](https://github.com/alexbojko/hopai/issues/43)) ([be21b96](https://github.com/alexbojko/hopai/commit/be21b9662fcb17cb19952703d52d2aa1ce01c087))
* AsyncGraph -- async traversal, mutation and ingestion via SQLAlchemy's sync/async bridge ([#49](https://github.com/alexbojko/hopai/issues/49)) ([2cf35d5](https://github.com/alexbojko/hopai/commit/2cf35d574c0d356b5e56670913339cc68c389a46))
* define_vectors(..., migrate=True) collapses the 3-call vector setup to one ([#62](https://github.com/alexbojko/hopai/issues/62)) ([8ef5bf2](https://github.com/alexbojko/hopai/commit/8ef5bf2ad398f6ca0fc570be86e3d41e14c13641))
* delete and update, in all three notations ([#33](https://github.com/alexbojko/hopai/issues/33)) ([e4b42f3](https://github.com/alexbojko/hopai/commit/e4b42f36e83bfd6e05f6da477b92c67dde442c7b))
* exact vector search on nodes and edges -- many fields, multivector, no pgvector ([#17](https://github.com/alexbojko/hopai/issues/17)) ([dba76cf](https://github.com/alexbojko/hopai/commit/dba76cf740cd65295e9287df81ccf9ba374532b0))
* extra columns on custom nodes/edges tables ([#48](https://github.com/alexbojko/hopai/issues/48)) ([99e5601](https://github.com/alexbojko/hopai/commit/99e5601beb8cae00e64de938ddc62073306a3534))
* Graph.load_vectors() recovers vector declarations from the database ([#66](https://github.com/alexbojko/hopai/issues/66)) ([89921ac](https://github.com/alexbojko/hopai/commit/89921ac5e1e885f8c2180bc44156f05754a120a8))
* infer the graph schema from existing nodes and edges ([#16](https://github.com/alexbojko/hopai/issues/16)) ([c8c9069](https://github.com/alexbojko/hopai/commit/c8c9069fd302a583ccdf365b86070be65ba15392)), closes [#14](https://github.com/alexbojko/hopai/issues/14)
* inference sampling, schema persistence, and Mermaid diagrams ([#35](https://github.com/alexbojko/hopai/issues/35)) ([662f8e5](https://github.com/alexbojko/hopai/commit/662f8e5afb36ba2c2f9d70d859ab219401948614)), closes [#29](https://github.com/alexbojko/hopai/issues/29) [#30](https://github.com/alexbojko/hopai/issues/30) [#31](https://github.com/alexbojko/hopai/issues/31)
* multivector search returns per-field similarities and boosts ([#71](https://github.com/alexbojko/hopai/issues/71)) ([c67a4f3](https://github.com/alexbojko/hopai/commit/c67a4f32454d6c289749c01b524c4cb9a7521236))
* per-type uniqueness, strict Cypher vocabulary, and richer annotation mappings ([#32](https://github.com/alexbojko/hopai/issues/32)) ([8c89908](https://github.com/alexbojko/hopai/commit/8c89908590c3d0876dd5b7f28476867287f3efd6)), closes [#26](https://github.com/alexbojko/hopai/issues/26) [#27](https://github.com/alexbojko/hopai/issues/27) [#28](https://github.com/alexbojko/hopai/issues/28)
* police declared endpoint types with a constraint trigger ([#24](https://github.com/alexbojko/hopai/issues/24)) ([900d8e8](https://github.com/alexbojko/hopai/commit/900d8e811315912a82510c6dcaa8b868e24ca508)), closes [#21](https://github.com/alexbojko/hopai/issues/21)
* programmatic graph schema with dataclass/pydantic notations and Postgres enforcement ([#15](https://github.com/alexbojko/hopai/issues/15)) ([9e2e671](https://github.com/alexbojko/hopai/commit/9e2e67101e8f1610fcebec349190c72b7c78f76e)), closes [#13](https://github.com/alexbojko/hopai/issues/13)
* reranking -- a third retrieval stage, step-wise through the graph ([#76](https://github.com/alexbojko/hopai/issues/76)) ([ad05de3](https://github.com/alexbojko/hopai/commit/ad05de37b74c0283c453a22443400d920269f1c6))
* schema_violations() -- a read-only dry-run before enforce_schema() ([#23](https://github.com/alexbojko/hopai/issues/23)) ([bf638cb](https://github.com/alexbojko/hopai/commit/bf638cb9f60d62f2080290d1a7ab7270b4f11787)), closes [#20](https://github.com/alexbojko/hopai/issues/20)
* text in, vectors out -- embedding providers, and a model that can search by meaning ([#40](https://github.com/alexbojko/hopai/issues/40)) ([69405f8](https://github.com/alexbojko/hopai/commit/69405f8f7e105ce943c4a36641563dd8fb5d424d))
* tool_schemas() -- the LLM tool definitions describing THIS graph ([#25](https://github.com/alexbojko/hopai/issues/25)) ([54305f6](https://github.com/alexbojko/hopai/commit/54305f64937e0a2fc50de9cc9de3a5d5f0c53726)), closes [#22](https://github.com/alexbojko/hopai/issues/22)
* traverse_graph refuses an oversized result instead of letting the client truncate it ([#70](https://github.com/alexbojko/hopai/issues/70)) ([ce603c2](https://github.com/alexbojko/hopai/commit/ce603c2ee68872b79c899ccae291195a61f08af1))
* vector fields are visible to infer_schema() and tool_schemas() ([#67](https://github.com/alexbojko/hopai/issues/67)) ([a2633a9](https://github.com/alexbojko/hopai/commit/a2633a945c0ffb6d9a446217e66dfbbf2a07984c))


### Bug Fixes

* add count/sum/avg/min/max aggregation across all three front ends ([#9](https://github.com/alexbojko/hopai/issues/9)) ([da34e42](https://github.com/alexbojko/hopai/commit/da34e4225d469ae303967ddb53bda66515cd5471))
* drop sqlmodel, declare nodes/edges as plain SQLAlchemy Core tables ([#46](https://github.com/alexbojko/hopai/issues/46)) ([32e8103](https://github.com/alexbojko/hopai/commit/32e810349d95228cc7d71308795ce153c3c2ffc1))
* infer_schema(sample_percent=) drops edge types its node sample can't support instead of raising ([#61](https://github.com/alexbojko/hopai/issues/61)) ([d086350](https://github.com/alexbojko/hopai/commit/d086350dca841b43a74bc5055aa3faa0c0624143))
* many separate graphs in one database ([#4](https://github.com/alexbojko/hopai/issues/4)) ([fbcdb3f](https://github.com/alexbojko/hopai/commit/fbcdb3fe475e11675101657fbed4fdd2d96cbcc6))
* normalize Boost onto cosine's scale by default, scale="raw" opts out ([#65](https://github.com/alexbojko/hopai/issues/65)) ([b67687d](https://github.com/alexbojko/hopai/commit/b67687dca58f5dfbed7776d25c303175ef0e69b1))
* refuse a declared vector field's floats in add_nodes/add_edges/merge_nodes/merge_edges ([#60](https://github.com/alexbojko/hopai/issues/60)) ([c78cdf9](https://github.com/alexbojko/hopai/commit/c78cdf98fdecc2493a4a272175a0c640f5a081c3))
* resolve embedding text before the greenlet bridge ([#75](https://github.com/alexbojko/hopai/issues/75)) ([281e8cf](https://github.com/alexbojko/hopai/commit/281e8cffca33f53dfa02fc2d3e6f209492ee43c4))


### Documentation

* add notebook 09 to the site nav, and fail the build on the next omission ([#38](https://github.com/alexbojko/hopai/issues/38)) ([a8b874c](https://github.com/alexbojko/hopai/commit/a8b874cd9ab37420e959dae4ecf6facd33384d37))
* eight runnable Jupyter notebooks as executable documentation ([#34](https://github.com/alexbojko/hopai/issues/34)) ([8a68e74](https://github.com/alexbojko/hopai/commit/8a68e74ac67b8a1c40a71f358470ecaf46981ace))
* host the site on Read the Docs instead of GitHub Pages ([#39](https://github.com/alexbojko/hopai/issues/39)) ([526ec88](https://github.com/alexbojko/hopai/commit/526ec88d536dc109f46e61b74dfd0c04b3526bdc))
* publish the notebooks and README as a site on each release ([#37](https://github.com/alexbojko/hopai/issues/37)) ([bee4f7e](https://github.com/alexbojko/hopai/commit/bee4f7e8becec387af7c231356cbdfb6ba1b4ad9))
* shorten README into a landing page ([#72](https://github.com/alexbojko/hopai/issues/72)) ([e33d913](https://github.com/alexbojko/hopai/commit/e33d9132171747162a6912b365b7709e0f9c453f))
* state the exact-scan ceiling in the README, and that min_similarity does not reduce cost ([#59](https://github.com/alexbojko/hopai/issues/59)) ([b2a1b0b](https://github.com/alexbojko/hopai/commit/b2a1b0b5ecc5b51cd47562110a18d5eb07fed860))

## 0.0.1 (2026-08-14)


### Features

* add release-please and PyPI publishing, starting at 0.0.1 ([5f67d3f](https://github.com/alexbojko/hopai/commit/5f67d3fdc9e68b777067c1fcab7f05fee120fdb9))
* release-please + PyPI publishing, starting at 0.0.1 ([6c82db0](https://github.com/alexbojko/hopai/commit/6c82db09a203cc04baf7502c1dbd694ebda61a30))


### Bug Fixes

* freshen up the README with badges, highlights, and section emoji ([#6](https://github.com/alexbojko/hopai/issues/6)) ([be646bf](https://github.com/alexbojko/hopai/commit/be646bf4caf5011a3c4b8986c93eed813fcbe59c))
* make release 0.0.1 ([#5](https://github.com/alexbojko/hopai/issues/5)) ([dcfe4f9](https://github.com/alexbojko/hopai/commit/dcfe4f9aae16af02267da5d6135ff9d0d1faa795))
* make the release job actually produce a 0.0.1 release PR ([#3](https://github.com/alexbojko/hopai/issues/3)) ([fbb9361](https://github.com/alexbojko/hopai/commit/fbb9361d8120896b3b72ed0bb77b0a0543de1dc9))
* Merge pull request [#2](https://github.com/alexbojko/hopai/issues/2) from alexbojko/release/setup-release-please ([6c82db0](https://github.com/alexbojko/hopai/commit/6c82db09a203cc04baf7502c1dbd694ebda61a30))
* start release-please from 0.0.0 so the first release is 0.0.1 ([dea7713](https://github.com/alexbojko/hopai/commit/dea7713d3035336e76e82b14548d4d28ac79b9a0))
