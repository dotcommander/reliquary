package memory

import (
	"context"
	"fmt"
	"runtime"
	"testing"
	"testing/synctest"
	"time"

	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/judgment"
	"github.com/jellydator/ttlcache/v3"
)

type snapshotEmbedder func(context.Context, embedding.Request) (embedding.Result, error)

func (f snapshotEmbedder) Embed(ctx context.Context, request embedding.Request) (embedding.Result, error) {
	return f(ctx, request)
}

func snapshotStore(t *testing.T, embedder embedding.Embedder) *SemanticStore {
	t.Helper()
	store, err := NewSemantic(SemanticConfig{
		Embedder: embedder,
		Model:    embedding.ModelRef{Provider: "fixture", Name: "snapshot", Dim: 3},
	}, WithTTL(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return store
}

func snapshotScore(value float64, vec []float32) semanticEntry {
	return semanticEntry{score: &judgment.Score{Value: value, Confidence: 1}, vec: vec}
}

func TestSemanticSnapshotRetainsEligibleCandidatesAfterMutation(t *testing.T) {
	t.Parallel()
	synctest.Test(t, func(t *testing.T) {
		cache := ttlcache.New[Key, semanticEntry](ttlcache.WithTTL[Key, semanticEntry](time.Hour))
		query := []float32{1, 0, 0}
		winner := semanticScoreKey("winner")
		near := semanticScoreKey("near")
		far := semanticScoreKey("far")
		cache.Set(winner, snapshotScore(91, query), ttlcache.DefaultTTL)
		cache.Set(near, snapshotScore(80, []float32{1, 0.5, 0}), ttlcache.DefaultTTL)
		cache.Set(far, snapshotScore(20, []float32{0, 1, 0}), ttlcache.DefaultTTL)
		for _, key := range []Key{
			{Tenant: "other", Schema: "s1", Primitive: PrimitiveScore, Fingerprint: "tenant"},
			{Tenant: "t1", Schema: "other", Primitive: PrimitiveScore, Fingerprint: "schema"},
			{Tenant: "t1", Schema: "s1", Primitive: PrimitiveNoul, Fingerprint: "primitive"},
		} {
			cache.Set(key, snapshotScore(100, query), ttlcache.DefaultTTL)
		}
		cache.Set(semanticScoreKey("no vector"), snapshotScore(100, nil), ttlcache.DefaultTTL)
		cache.Set(semanticScoreKey("expired"), snapshotScore(100, query), time.Nanosecond)
		<-time.After(time.Second) // The synctest clock advances without wall-clock waiting.
		items := cache.Items()
		if _, ok := items[semanticScoreKey("expired")]; ok {
			t.Fatal("snapshot included expired candidate")
		}

		done := make(chan struct{})
		go func() { // Exits after a bounded mutation batch.
			cache.Delete(winner)
			cache.Delete(near)
			cache.Delete(far)
			cache.Set(semanticScoreKey("later"), snapshotScore(99, query), ttlcache.DefaultTTL)
			close(done)
		}()
		<-done
		best, similarity, found := closestSemanticEntry(items, semanticScoreKey("query"), query)
		if !found || best.score == nil || best.score.Value != 91 || similarity != 1 {
			t.Fatalf("snapshot winner = %+v, similarity = %v, found = %v", best.score, similarity, found)
		}
	})
}

func TestSemanticLookupAllowsMutationWhileEmbedding(t *testing.T) {
	t.Parallel()
	started := make(chan struct{})
	release := make(chan struct{})
	store := snapshotStore(t, snapshotEmbedder(func(ctx context.Context, request embedding.Request) (embedding.Result, error) {
		close(started)
		select {
		case <-release:
		case <-ctx.Done():
			return embedding.Result{}, ctx.Err()
		}
		return embedding.Result{Model: request.Model, Vectors: []embedding.Vector{{1, 0, 0}}}, nil
	}))
	type lookupResult struct {
		score judgment.Score
		hit   Hit
		err   error
	}
	result := make(chan lookupResult, 1)
	go func() { // Exits when the released embedder finishes this one lookup.
		score, hit, err := store.GetScore(context.Background(), semanticScoreKey("query"), "query")
		result <- lookupResult{score, hit, err}
	}()
	<-started
	// This insertion must finish before releasing the embedder: no cache lock
	// may be held across Embed, and the production lookup must snapshot afterward.
	store.cache.Set(semanticScoreKey("inserted"), snapshotScore(77, []float32{1, 0, 0}), ttlcache.DefaultTTL)
	close(release)
	got := <-result
	if got.err != nil || got.hit.Kind != HitSemantic || got.score.Value != 77 {
		t.Fatalf("lookup after insertion = %+v", got)
	}
}

func TestSemanticLookupWithConcurrentCacheMutation(t *testing.T) {
	t.Parallel()
	store := snapshotStore(t, snapshotEmbedder(func(_ context.Context, request embedding.Request) (embedding.Result, error) {
		return embedding.Result{Model: request.Model, Vectors: []embedding.Vector{{1, 0, 0}}}, nil
	}))
	winner := semanticScoreKey("stable winner")
	store.cache.Set(winner, snapshotScore(88, []float32{1, 0, 0}), ttlcache.DefaultTTL)
	start := make(chan struct{})
	done := make(chan struct{})
	go func() { // Exits after the start signal and 512 mutation iterations.
		<-start
		for i := range 512 {
			key := semanticScoreKey(fmt.Sprintf("transient-%d", i))
			store.cache.Set(key, snapshotScore(10, []float32{0, 1, 0}), ttlcache.DefaultTTL)
			store.cache.Get(winner) // Move the stable candidate in the mutable LRU list.
			store.cache.Delete(key)
			runtime.Gosched()
		}
		close(done)
	}()
	close(start)
	var lookupErr error
	for range 256 {
		score, hit, err := store.GetScore(context.Background(), semanticScoreKey("query"), "query")
		if err != nil || hit.Kind != HitSemantic || score.Value != 88 {
			lookupErr = fmt.Errorf("score=%+v hit=%+v error=%v", score, hit, err)
			break
		}
		runtime.Gosched()
	}
	<-done
	if lookupErr != nil {
		t.Fatal(lookupErr)
	}
}
