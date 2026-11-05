package memory

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/dotcommander/reliquary/embedding"
)

type typedNilEmbedder struct{}

func (e *typedNilEmbedder) Embed(context.Context, embedding.Request) (embedding.Result, error) {
	return embedding.Result{}, nil
}

func TestNewSemanticRejectsTypedNilEmbedder(t *testing.T) {
	t.Parallel()
	_, err := NewSemantic(
		SemanticConfig{Embedder: (*typedNilEmbedder)(nil), Model: embedding.ModelRef{Dim: 2}},
		WithTTL(time.Minute),
	)
	if err == nil || !strings.Contains(err.Error(), "Embedder is required") {
		t.Fatalf("NewSemantic error = %v, want embedder-required error", err)
	}
}
