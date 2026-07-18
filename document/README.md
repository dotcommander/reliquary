# document

```go
doc := document.Document{
	ID:     "doc-1",
	Format: document.FormatMarkdown,
	Text:   document.NormalizeText("\ufeff# Title\r\nBody"),
}
span := document.Span{StartByte: 0, EndByte: len(doc.Text)}
text, ok := span.Text(doc.Text)
```

`document` owns provider-neutral document value types: `Document`, `Section`,
`Element`, `Span`, `Format`, and `Metadata`.

Callers own parsing, source schemas, and semantic interpretation. This package
only normalizes text and validates byte/rune offsets so downstream chunking,
ingest, retrieval, provenance, and test fixtures can share one document shape.

`Element` records generic pre-chunk structure such as titles, narrative text,
list items, tables, code, page breaks, and unknown blocks. It stores the parser
strategy label and byte span, but it does not select or implement a parser.
