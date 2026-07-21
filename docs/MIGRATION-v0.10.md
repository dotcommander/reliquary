# Migrating to Reliquary v0.10

Reliquary v0.10 makes two breaking import cleanups. Retrieval behavior is
unchanged.

## Use the singular embedding package name

The import path remains the same, but its declared package name changes from
`embeddings` to `embedding`:

```go
import "github.com/dotcommander/reliquary/embedding"

var embedder embedding.Embedder
request := embedding.Request{Inputs: []string{"hello"}}
```

If you need to migrate call sites gradually, an explicit import alias preserves
the old qualifier temporarily:

```go
import embeddings "github.com/dotcommander/reliquary/embedding"

var embedder embeddings.Embedder
```

The exported types and `ErrInvalidResult` sentinel are unchanged. Validation
error strings now start with `embedding:` instead of `embeddings:`; use
`errors.Is(err, embedding.ErrInvalidResult)` instead of matching error text.

The separate `embed` package remains the deterministic hashing implementation
used by `Quickstart`, demos, and tests.

## Move the index sink import

Update the import path for the ingestion sink:

```diff
-import "github.com/dotcommander/reliquary/indexsink"
+import "github.com/dotcommander/reliquary/pipeline/indexsink"
```

`indexsink.NewSink`, `indexsink.Config`, and `indexsink.Sink` keep the same
names and behavior at the new path. The old path has been removed and there is
no forwarding package.
