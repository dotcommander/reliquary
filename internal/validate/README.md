# internal/validate

```go
err := validate.NonEmpty("id", "doc-1")
err = validate.Positive("limit", 10)
```

`internal/validate` contains small validation helpers shared by public
packages.

It does not own product validation policy, user-facing messages, or schema
validation frameworks.
