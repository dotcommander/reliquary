# internal/hash

```go
digest := hash.SHA256String("body")
fmt.Println(digest.String())
```

`internal/hash` provides stable SHA-256 digest helpers for source, artifact, and
metadata identity.

Use `HashIdentity` for ordered transform ledgers where source, parser,
chunking, model, schema, and profile changes must invalidate derived artifacts.
Empty fields are omitted; part order is significant.

It intentionally stays small: callers decide which bytes are canonical and when
a digest invalidates downstream artifacts.
