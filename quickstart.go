package reliquary

import (
	"fmt"
	"github.com/dotcommander/reliquary/embed"
)

// InMemory returns a ready-to-use RAG App backed entirely by in-process,
// dependency-free defaults: the deterministic hashing embedder at the given
// dimension plus New's defaults (in-memory index, smart-boundary chunking, and
// the default scoring weights). It is ideal for demos and tests; the hashing
// embedder is not production retrieval quality.
//
// Every input is known-good, so InMemory does not return an error; it supplies
// both required New options itself.
func InMemory(dim int) *App {
	embedder := embed.NewHashing(dim)
	identity := fmt.Sprintf("demo:hashing:1:%d|smart-boundary:220:0", embedder.Model.Dim)
	app, err := New(WithEmbedder(embedder), WithIndexIdentity(identity))
	if err != nil {
		panic("reliquary: InMemory: " + err.Error())
	}
	return app
}

// Quickstart is InMemory(256) — the one-obvious-default entry point for trying
// reliquary with zero configuration.
func Quickstart() *App {
	return InMemory(256)
}
