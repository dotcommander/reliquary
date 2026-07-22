# ingestfs

`ingestfs` reads a local directory tree as deterministic batches for
`pipeline/ingest`.

```go
reader, err := ingestfs.New(ingestfs.Config{
	Root:      "./docs",
	Source:    "project-docs",
	BatchSize: 64,
})
```

The first `Read` snapshots regular file paths in lexical order. Hidden files
are included. Symlinks and special files are skipped. Each record contains raw
bounded bytes, a slash-normalized root-relative ID, and `source`, `path`, and
`filename` metadata. Decoders retain responsibility for parsing and format
selection.

The cursor token is the last emitted relative path. Pass the returned cursor to
the next read or pipeline run. A nonempty final batch has an empty token. Files
added after the snapshot are ignored; removing or changing a file that remains
to be read fails that batch without returning partial records.

`Include` receives slash-normalized relative paths. Returning false for a
directory prunes its complete subtree.
