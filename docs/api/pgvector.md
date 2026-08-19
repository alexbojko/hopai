# hopai.pgvector

The optional `vector_backend="pgvector"` backend: why it is opt-in, why
it requires pgvector 0.8, and why it ranks one field per search. See
[Vector search](../reference/vector-search.md#the-pgvector-backend) for
the guide.

::: hopai.pgvector
    options:
      members: [VECTOR_BACKENDS, MINIMUM_PGVECTOR, Vector, validate_backend]
