package memory

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/judgment"
)

// fakeEmbedder maps inputs to deterministic 3-dimensional vectors by
// input bucket and counts every Embed call.
type fakeEmbedder struct {
	calls int
	fail  bool
}

func (f *fakeEmbedder) Embed(_ context.Context, request embedding.Request) (embedding.Result, error) {
	if f.fail {
		return embedding.Result{}, errors.New("fake embedder unavailable")
	}
	f.calls++
	vectors := make([]embedding.Vector, len(request.Inputs))
	for i, input := range request.Inputs {
		vectors[i] = fakeVector(input)
	}
	return embedding.Result{Model: request.Model, Vectors: vectors}, nil
}

// fakeVector buckets: "alpha parallel*" and its paraphrases are parallel
// (cosine 1.0); "alpha diagonal*" sits near 0.893 against them; "beta*"
// is near-orthogonal; everything else lands on the low-similarity
// diagonal.
func fakeVector(input string) []float32 {
	switch {
	case strings.HasPrefix(input, "alpha parallel"):
		return normalize3(1, 0.05, 0)
	case strings.HasPrefix(input, "alpha diagonal"):
		return normalize3(1, 0.5, 0)
	case strings.HasPrefix(input, "beta"):
		return normalize3(0, 1, 0.05)
	default:
		return normalize3(0.3, 0.3, 0.3)
	}
}

func normalize3(x, y, z float32) []float32 {
	n := float32(math.Sqrt(float64(x*x + y*y + z*z)))
	return []float32{x / n, y / n, z / n}
}

func newSemanticTestStore(t *testing.T, emb *fakeEmbedder, threshold float64, opts ...Option) *SemanticStore {
	t.Helper()
	store, err := NewSemantic(
		SemanticConfig{
			Embedder:  emb,
			Model:     embedding.ModelRef{Provider: "fake", Name: "fake-3", Dim: 3},
			Threshold: threshold,
		},
		append([]Option{WithTTL(time.Hour)}, opts...)...,
	)
	if err != nil {
		t.Fatalf("NewSemantic unexpected error: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return store
}

func semanticScoreKey(fingerprint string) Key {
	return Key{Tenant: "t1", Schema: "s1", Primitive: PrimitiveScore, Fingerprint: fingerprint}
}

func TestSemanticStoreValidatesConfig(t *testing.T) {
	t.Parallel()

	emb := &fakeEmbedder{}
	model := embedding.ModelRef{Provider: "fake", Name: "fake-3", Dim: 3}
	if _, err := NewSemantic(SemanticConfig{Model: model}, WithTTL(time.Hour)); err == nil {
		t.Fatal("NewSemantic error = nil for missing embedder")
	}
	if _, err := NewSemantic(SemanticConfig{Embedder: emb}, WithTTL(time.Hour)); err == nil {
		t.Fatal("NewSemantic error = nil for zero model dims")
	}
	if _, err := NewSemantic(SemanticConfig{Embedder: emb, Model: model, Threshold: 1.5}, WithTTL(time.Hour)); err == nil {
		t.Fatal("NewSemantic error = nil for out-of-range threshold")
	}
	if _, err := NewSemantic(SemanticConfig{Embedder: emb, Model: model}); err == nil {
		t.Fatal("NewSemantic error = nil for missing TTL")
	}
}

func TestSemanticExactHitSkipsEmbedding(t *testing.T) {
	t.Parallel()

	emb := &fakeEmbedder{}
	store := newSemanticTestStore(t, emb, 0)

	result := judgment.Score{Value: 87.5, Confidence: 0.9}
	if err := store.PutScore(context.Background(), semanticScoreKey("fp-a"), "alpha parallel one", result); err != nil {
		t.Fatalf("PutScore unexpected error: %v", err)
	}
	if emb.calls != 1 {
		t.Fatalf("PutScore embed calls = %d, want 1", emb.calls)
	}

	found, hit, err := store.GetScore(context.Background(), semanticScoreKey("fp-a"), "alpha parallel one")
	if err != nil {
		t.Fatalf("GetScore unexpected error: %v", err)
	}
	if hit.Kind != HitExact {
		t.Fatalf("hit kind = %v, want HitExact", hit.Kind)
	}
	if found != result {
		t.Fatalf("score = %#v, want %#v", found, result)
	}
	if emb.calls != 1 {
		t.Fatalf("exact hit embed calls = %d, want 1", emb.calls)
	}
}

func TestSemanticParaphraseHitsAcrossFingerprints(t *testing.T) {
	t.Parallel()

	emb := &fakeEmbedder{}
	store := newSemanticTestStore(t, emb, 0)

	result := judgment.Score{Value: 72, Confidence: 0.8}
	if err := store.PutScore(context.Background(), semanticScoreKey("fp-a"), "alpha parallel one", result); err != nil {
		t.Fatalf("PutScore unexpected error: %v", err)
	}

	found, hit, err := store.GetScore(context.Background(), semanticScoreKey("fp-b"), "alpha parallel two")
	if err != nil {
		t.Fatalf("GetScore unexpected error: %v", err)
	}
	if hit.Kind != HitSemantic {
		t.Fatalf("hit kind = %v, want HitSemantic", hit.Kind)
	}
	if hit.Similarity < 0.999 {
		t.Fatalf("similarity = %v, want ~1.0", hit.Similarity)
	}
	if found != result {
		t.Fatalf("score = %#v, want %#v", found, result)
	}
	if metrics := store.Metrics(); metrics.SemanticHits != 1 {
		t.Fatalf("semantic hits = %d, want 1", metrics.SemanticHits)
	}
}

func TestSemanticBelowThresholdMisses(t *testing.T) {
	t.Parallel()

	emb := &fakeEmbedder{}
	store := newSemanticTestStore(t, emb, 0)

	if err := store.PutScore(context.Background(), semanticScoreKey("fp-a"), "alpha parallel one", judgment.Score{Value: 1, Confidence: 0.5}); err != nil {
		t.Fatalf("PutScore unexpected error: %v", err)
	}

	for name, input := range map[string]string{
		"near miss": "alpha diagonal one", // cosine ~0.893 < 0.92
		"far miss":  "beta unrelated",     // near-orthogonal
	} {
		found, hit, err := store.GetScore(context.Background(), semanticScoreKey("fp-"+name), input)
		if err != nil {
			t.Fatalf("%s: GetScore unexpected error: %v", name, err)
		}
		if hit.Kind != HitNone {
			t.Fatalf("%s: hit kind = %v, want HitNone", name, hit.Kind)
		}
		if found != (judgment.Score{}) {
			t.Fatalf("%s: score = %#v, want zero value", name, found)
		}
	}
}

func TestSemanticThresholdCalibration(t *testing.T) {
	t.Parallel()

	emb := &fakeEmbedder{}
	store := newSemanticTestStore(t, emb, 0.85)

	result := judgment.Score{Value: 10, Confidence: 0.6}
	if err := store.PutScore(context.Background(), semanticScoreKey("fp-a"), "alpha parallel one", result); err != nil {
		t.Fatalf("PutScore unexpected error: %v", err)
	}

	found, hit, err := store.GetScore(context.Background(), semanticScoreKey("fp-b"), "alpha diagonal one")
	if err != nil {
		t.Fatalf("GetScore unexpected error: %v", err)
	}
	if hit.Kind != HitSemantic {
		t.Fatalf("hit kind = %v, want HitSemantic at threshold 0.85", hit.Kind)
	}
	if hit.Similarity < 0.90 || hit.Similarity > 0.92 {
		t.Fatalf("similarity = %v, want the ~0.916 pair", hit.Similarity)
	}
	if found != result {
		t.Fatalf("score = %#v, want %#v", found, result)
	}
}

func TestSemanticScopeIsolation(t *testing.T) {
	t.Parallel()

	emb := &fakeEmbedder{}
	store := newSemanticTestStore(t, emb, 0)

	if err := store.PutScore(context.Background(), semanticScoreKey("fp-a"), "alpha parallel one", judgment.Score{Value: 5, Confidence: 0.5}); err != nil {
		t.Fatalf("PutScore unexpected error: %v", err)
	}

	otherTenant := semanticScoreKey("fp-b")
	otherTenant.Tenant = "t2"
	otherSchema := semanticScoreKey("fp-c")
	otherSchema.Schema = "s2"
	otherPrimitive := baseChoiceKey()
	otherPrimitive.Fingerprint = "fp-d"

	for name, key := range map[string]Key{
		"tenant":    otherTenant,
		"schema":    otherSchema,
		"primitive": otherPrimitive,
	} {
		var hit Hit
		var err error
		if key.Primitive == PrimitiveScore {
			_, hit, err = store.GetScore(context.Background(), key, "alpha parallel two")
		} else {
			_, hit, err = store.GetChoice(context.Background(), key, "alpha parallel two")
		}
		if err != nil {
			t.Fatalf("%s: unexpected error: %v", name, err)
		}
		if hit.Kind != HitNone {
			t.Fatalf("%s: hit kind = %v, want HitNone", name, hit.Kind)
		}
	}
}

func TestSemanticEmptyInputNeverEmbeds(t *testing.T) {
	t.Parallel()

	emb := &fakeEmbedder{}
	store := newSemanticTestStore(t, emb, 0)

	if err := store.PutScore(context.Background(), semanticScoreKey("fp-a"), "alpha parallel one", judgment.Score{Value: 3, Confidence: 0.5}); err != nil {
		t.Fatalf("PutScore unexpected error: %v", err)
	}
	before := emb.calls

	_, hit, err := store.GetScore(context.Background(), semanticScoreKey("fp-b"), "")
	if err != nil {
		t.Fatalf("GetScore unexpected error: %v", err)
	}
	if hit.Kind != HitNone {
		t.Fatalf("hit kind = %v, want HitNone for empty input", hit.Kind)
	}
	if emb.calls != before {
		t.Fatalf("embed calls = %d, want %d", emb.calls, before)
	}
}

func TestSemanticPutEmbedFailure(t *testing.T) {
	t.Parallel()

	emb := &fakeEmbedder{fail: true}
	store := newSemanticTestStore(t, emb, 0)

	if err := store.PutScore(context.Background(), semanticScoreKey("fp-a"), "alpha parallel one", judgment.Score{Value: 1, Confidence: 0.5}); err == nil {
		t.Fatal("PutScore error = nil for failing embedder")
	}
	if metrics := store.Metrics(); metrics.Insertions != 0 {
		t.Fatalf("insertions = %d, want 0", metrics.Insertions)
	}
	if _, _, err := store.GetScore(context.Background(), semanticScoreKey("fp-b"), "alpha parallel two"); err == nil {
		t.Fatal("GetScore error = nil when semantic lookup must embed")
	}
}

func TestSemanticTypedPathsForChoiceAndNoul(t *testing.T) {
	t.Parallel()

	emb := &fakeEmbedder{}
	store := newSemanticTestStore(t, emb, 0)

	choiceKey := baseChoiceKey()
	choice := choiceFixture()
	if err := store.PutChoice(context.Background(), choiceKey, "alpha parallel one", choice); err != nil {
		t.Fatalf("PutChoice unexpected error: %v", err)
	}
	foundChoice, hit, err := store.GetChoice(context.Background(), Key{Tenant: choiceKey.Tenant, Schema: choiceKey.Schema, Primitive: PrimitiveChoice, Fingerprint: "fp-choice-b"}, "alpha parallel two")
	if err != nil {
		t.Fatalf("GetChoice unexpected error: %v", err)
	}
	if hit.Kind != HitSemantic {
		t.Fatalf("choice hit kind = %v, want HitSemantic", hit.Kind)
	}
	if foundChoice.Selected != choice.Selected {
		t.Fatalf("choice = %#v, want selection %q", foundChoice, choice.Selected)
	}

	noulKey := Key{Tenant: "t1", Schema: "s1", Primitive: PrimitiveNoul, Fingerprint: "fp-noul-a"}
	noul := judgment.Noul{Verdict: judgment.VerdictYes, Confidence: 0.97}
	if err := store.PutNoul(context.Background(), noulKey, "alpha parallel one", noul); err != nil {
		t.Fatalf("PutNoul unexpected error: %v", err)
	}
	foundNoul, noulHit, err := store.GetNoul(context.Background(), Key{Tenant: "t1", Schema: "s1", Primitive: PrimitiveNoul, Fingerprint: "fp-noul-b"}, "alpha parallel rephrased")
	if err != nil {
		t.Fatalf("GetNoul unexpected error: %v", err)
	}
	if noulHit.Kind != HitSemantic {
		t.Fatalf("noul hit kind = %v, want HitSemantic", noulHit.Kind)
	}
	if foundNoul != noul {
		t.Fatalf("noul = %#v, want %#v", foundNoul, noul)
	}
}

func TestSemanticMetricsAccounting(t *testing.T) {
	t.Parallel()

	emb := &fakeEmbedder{}
	store := newSemanticTestStore(t, emb, 0)

	if err := store.PutScore(context.Background(), semanticScoreKey("fp-a"), "alpha parallel one", judgment.Score{Value: 9, Confidence: 0.5}); err != nil {
		t.Fatalf("PutScore unexpected error: %v", err)
	}
	_, exactHit, _ := store.GetScore(context.Background(), semanticScoreKey("fp-a"), "alpha parallel one")
	_, semanticHit, _ := store.GetScore(context.Background(), semanticScoreKey("fp-b"), "alpha parallel two")
	_, miss, _ := store.GetScore(context.Background(), semanticScoreKey("fp-c"), "beta unrelated")
	if exactHit.Kind != HitExact || semanticHit.Kind != HitSemantic || miss.Kind != HitNone {
		t.Fatalf("hits = %v, %v, %v; want exact, semantic, none", exactHit.Kind, semanticHit.Kind, miss.Kind)
	}

	metrics := store.Metrics()
	if metrics.Insertions != 1 || metrics.ExactHits != 1 || metrics.SemanticHits != 1 || metrics.Misses != 1 {
		t.Fatalf("metrics = %#v", metrics)
	}
}

func TestSemanticTTLEviction(t *testing.T) {
	t.Parallel()

	synctest.Test(t, func(t *testing.T) {
		emb := &fakeEmbedder{}
		store := newSemanticTestStore(t, emb, 0, WithTTL(30*time.Minute), WithDisableTouchOnHit())

		key := semanticScoreKey("fp-a")
		if err := store.PutScore(context.Background(), key, "alpha parallel one", judgment.Score{Value: 42, Confidence: 0.8}); err != nil {
			t.Fatalf("PutScore unexpected error: %v", err)
		}
		if _, hit, _ := store.GetScore(context.Background(), key, "alpha parallel one"); hit.Kind != HitExact {
			t.Fatalf("fresh hit kind = %v, want HitExact", hit.Kind)
		}

		time.Sleep(31 * time.Minute) // fake clock: past the 30m lifetime
		synctest.Wait()              // let the ttlcache cleanup loop run

		if _, hit, _ := store.GetScore(context.Background(), key, "alpha parallel one"); hit.Kind != HitNone {
			t.Fatalf("expired exact hit kind = %v, want HitNone", hit.Kind)
		}
		if _, hit, _ := store.GetScore(context.Background(), semanticScoreKey("fp-b"), "alpha parallel two"); hit.Kind != HitNone {
			t.Fatalf("expired semantic hit kind = %v, want HitNone", hit.Kind)
		}
	})
}
