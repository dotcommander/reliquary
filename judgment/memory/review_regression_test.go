package memory

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/judgment"
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

type countingEmbedder struct {
	calls atomic.Int64
}

func (e *countingEmbedder) Embed(_ context.Context, req embedding.Request) (embedding.Result, error) {
	e.calls.Add(1)
	vecs := make([]embedding.Vector, len(req.Inputs))
	for i := range vecs {
		vecs[i] = embedding.Vector{1, 0}
	}
	return embedding.Result{Vectors: vecs}, nil
}

func TestSemanticStoreMismatchedPrimitiveSkipsCacheAndEmbedder(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	emb := &countingEmbedder{}
	store, err := NewSemantic(
		SemanticConfig{Embedder: emb, Model: embedding.ModelRef{Dim: 2}},
		WithTTL(time.Minute),
	)
	if err != nil {
		t.Fatalf("NewSemantic: %v", err)
	}
	defer func() { _ = store.Close() }()

	scoreKey := Key{Tenant: "t1", Schema: "v1", Primitive: PrimitiveScore, Fingerprint: "fp-score"}
	if err := store.PutScore(ctx, scoreKey, "input text", judgment.Score{Value: 80, Confidence: 0.9}); err != nil {
		t.Fatalf("PutScore: %v", err)
	}
	noulKey := Key{Tenant: "t1", Schema: "v1", Primitive: PrimitiveNoul, Fingerprint: "fp-noul"}
	if err := store.PutNoul(ctx, noulKey, "input text", judgment.Noul{Verdict: judgment.VerdictYes, Confidence: 0.9}); err != nil {
		t.Fatalf("PutNoul: %v", err)
	}
	choiceKey := Key{Tenant: "t1", Schema: "v1", Primitive: PrimitiveChoice, Fingerprint: "fp-choice"}
	if err := store.PutChoice(ctx, choiceKey, "input text", judgment.Choice{
		Selected:     "a",
		Distribution: []judgment.OptionProbability{{Option: "a", Probability: 1}},
	}); err != nil {
		t.Fatalf("PutChoice: %v", err)
	}

	callsBefore := emb.calls.Load()

	if _, hit, err := store.GetChoice(ctx, scoreKey, "input text"); err != nil || hit.Kind != HitNone {
		t.Fatalf("GetChoice(scoreKey) = (%+v, %v), want miss", hit, err)
	}
	if _, hit, err := store.GetScore(ctx, noulKey, "input text"); err != nil || hit.Kind != HitNone {
		t.Fatalf("GetScore(noulKey) = (%+v, %v), want miss", hit, err)
	}
	if _, hit, err := store.GetNoul(ctx, choiceKey, "input text"); err != nil || hit.Kind != HitNone {
		t.Fatalf("GetNoul(choiceKey) = (%+v, %v), want miss", hit, err)
	}

	if got := emb.calls.Load(); got != callsBefore {
		t.Fatalf("embedder calls = %d, want %d (must not embed on mismatched primitive)", got, callsBefore)
	}
	if m := store.Metrics(); m.Misses != 3 {
		t.Fatalf("Metrics().Misses = %d, want 3", m.Misses)
	}
}
