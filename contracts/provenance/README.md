# provenance

```go
digest := hash.SHA256String("source bytes")
artifact := provenance.GeneratedArtifact("a1", "markdown", "file://out.md", "importer", digest)
```

`provenance` gives apps a shared shape for sources, generated artifacts, claims,
and lineage links.

It does not decide whether evidence is sufficient, how claims are ranked, or
what a domain-specific proof policy requires.
