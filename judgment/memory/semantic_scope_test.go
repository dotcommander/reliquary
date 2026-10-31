package memory

import (
	"testing"

	"github.com/jellydator/ttlcache/v3"
)

func TestScopedSnapshotExcludesEvictionsBeforeCallbackAndKeepsReplacements(t *testing.T) {
	t.Parallel()
	// No eviction subscription: model a callback delayed until after lookup.
	store := &SemanticStore{
		cache:  ttlcache.New(ttlcache.WithCapacity[Key, semanticEntry](1)),
		scoped: make(map[semanticScope]map[Key]*ttlcache.Item[Key, semanticEntry]),
	}
	first := semanticScoreKey("first")
	second := semanticScoreKey("second")
	second.Tenant = "other"
	store.setEntry(first, snapshotScore(1, []float32{1, 0, 0}), ttlcache.NoTTL)
	old := store.scopeSnapshot(first)[first]
	store.setEntry(second, snapshotScore(2, []float32{1, 0, 0}), ttlcache.NoTTL)
	if got := store.scopeSnapshot(first); len(got) != 0 {
		t.Fatalf("evicted item leaked into snapshot: %v", got)
	}
	store.setEntry(first, snapshotScore(3, []float32{1, 0, 0}), ttlcache.NoTTL)
	store.removeScoped(old)
	got := store.scopeSnapshot(first)
	if len(got) != 1 || got[first] == old || got[first].Value().score.Value != 3 {
		t.Fatalf("delayed eviction removed replacement: %v", got)
	}
}
