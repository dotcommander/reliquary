package memory

import (
	"time"

	"github.com/dotcommander/reliquary/vector"
	"github.com/jellydator/ttlcache/v3"
)

type semanticScope struct {
	tenant    string
	schema    string
	primitive Primitive
}

func scopeOf(key Key) semanticScope {
	return semanticScope{key.Tenant, key.Schema, key.Primitive}
}

// setEntry serializes registration with snapshots so a completed cache write
// cannot leave a snapshot referring to an older incarnation of the same key.
func (s *SemanticStore) setEntry(key Key, value semanticEntry, ttl time.Duration) {
	s.setMu.Lock()
	defer s.setMu.Unlock()
	item := s.cache.Set(key, value, ttl)
	s.scopedMu.Lock()
	defer s.scopedMu.Unlock()
	// Expiration cleanup may have run before registration.
	if !s.cache.Has(key) {
		return
	}
	scope := scopeOf(key)
	if s.scoped[scope] == nil {
		s.scoped[scope] = make(map[Key]*ttlcache.Item[Key, semanticEntry])
	}
	s.scoped[scope][key] = item
}

func (s *SemanticStore) removeScoped(item *ttlcache.Item[Key, semanticEntry]) {
	key := item.Key()
	s.scopedMu.Lock()
	defer s.scopedMu.Unlock()
	scope := scopeOf(key)
	// Asynchronous eviction of an old incarnation must not remove its replacement.
	if s.scoped[scope][key] == item {
		delete(s.scoped[scope], key)
		if len(s.scoped[scope]) == 0 {
			delete(s.scoped, scope)
		}
	}
}

// scopeSnapshot copies matching live candidates without touching TTL or LRU.
// Has excludes evicted entries even while their asynchronous cleanup is pending.
// Holding setMu prevents a key from being reinserted between membership checks
// and pointer selection; evictions after selection have ordinary snapshot semantics.
func (s *SemanticStore) scopeSnapshot(key Key) map[Key]*ttlcache.Item[Key, semanticEntry] {
	s.setMu.Lock()
	defer s.setMu.Unlock()
	s.scopedMu.RLock()
	defer s.scopedMu.RUnlock()
	items := make(map[Key]*ttlcache.Item[Key, semanticEntry], len(s.scoped[scopeOf(key)]))
	for candidate, item := range s.scoped[scopeOf(key)] {
		if s.cache.Has(candidate) {
			items[candidate] = item
		}
	}
	return items
}

// closestSemanticEntry ranks a nonexpired scope snapshot without traversing the
// cache's mutable LRU list. Item.Value synchronizes reads with cache updates.
func closestSemanticEntry(items map[Key]*ttlcache.Item[Key, semanticEntry], key Key, query []float32) (semanticEntry, float32, bool) {
	var best semanticEntry
	var bestSim float32
	var found bool
	for storedKey, item := range items {
		if storedKey.Tenant != key.Tenant || storedKey.Schema != key.Schema || storedKey.Primitive != key.Primitive {
			continue
		}
		entry := item.Value()
		if len(entry.vec) == 0 {
			continue
		}
		sim := vectors.Cosine32(query, entry.vec)
		if !found || sim > bestSim {
			found, best, bestSim = true, entry, sim
		}
	}
	return best, bestSim, found
}
