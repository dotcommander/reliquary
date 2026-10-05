package memory

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/judgment"
	"github.com/dotcommander/reliquary/vector"
	"github.com/jellydator/ttlcache/v3"
)

// SemanticStore is a judgment cache that answers both exact fingerprint
// hits and semantic matches. Entries whose inputs embed near a stored
// vector (cosine similarity at or above the configured threshold within
// the same tenant, schema, and primitive scope) reuse the cached
// judgment without calling a provider.
//
// The embedder is injected and stays caller-owned; the store performs no
// network or provider I/O of its own beyond that embedder call.
type SemanticStore struct {
	cache     *ttlcache.Cache[Key, semanticEntry]
	embedder  embedding.Embedder
	model     embedding.ModelRef
	threshold float32
	closeOnce sync.Once

	exactHits    atomic.Uint64
	semanticHits atomic.Uint64
	misses       atomic.Uint64
	insertions   atomic.Uint64
}

// semanticEntry is the cached value: exactly one judgment result plus
// the unit of the semantic comparison, the input's embedding.
type semanticEntry struct {
	choice *judgment.Choice
	score  *judgment.Score
	noul   *judgment.Noul
	vec    []float32
}

// SemanticConfig wires the semantic layer. Embedder and Model.Dim are
// required; Threshold is optional and defaults to 0.92.
type SemanticConfig struct {
	// Embedder embeds judgment inputs for semantic comparison.
	Embedder embedding.Embedder
	// Model identifies the embedding space; Dim must be positive and
	// must match every embedded vector.
	Model embedding.ModelRef
	// Threshold is the minimum cosine similarity for a semantic hit.
	// Zero means the 0.92 default.
	Threshold float64
}

// DefaultSemanticThreshold applies when SemanticConfig.Threshold is unset.
const DefaultSemanticThreshold = 0.92

// HitKind classifies a SemanticStore lookup.
type HitKind int

const (
	// HitNone reports a miss.
	HitNone HitKind = iota
	// HitExact reports an exact fingerprint hit.
	HitExact
	// HitSemantic reports a similarity-threshold hit.
	HitSemantic
)

// Hit reports how a lookup resolved. A miss is the zero Hit.
type Hit struct {
	Kind HitKind
	// Similarity is the winning cosine similarity for HitSemantic.
	Similarity float32
}

// SemanticMetrics counts SemanticStore activity.
type SemanticMetrics struct {
	ExactHits    uint64
	SemanticHits uint64
	Misses       uint64
	Insertions   uint64
}

// NewSemantic creates a SemanticStore and starts its background eviction
// goroutine, which exits when Close is called. Cache behavior (TTL,
// capacity, touch-on-hit) reuses the Store options; TTL must be positive.
func NewSemantic(cfg SemanticConfig, opts ...Option) (*SemanticStore, error) {
	if cfg.Embedder == nil {
		return nil, fmt.Errorf("judgment/memory: SemanticConfig.Embedder is required")
	}
	if cfg.Model.Dim <= 0 {
		return nil, fmt.Errorf("judgment/memory: SemanticConfig.Model.Dim must be positive, got %d", cfg.Model.Dim)
	}
	threshold := float32(DefaultSemanticThreshold)
	switch {
	case cfg.Threshold == 0:
	case cfg.Threshold > 0 && cfg.Threshold <= 1:
		threshold = float32(cfg.Threshold)
	default:
		return nil, fmt.Errorf("judgment/memory: SemanticConfig.Threshold must be in (0, 1], got %v", cfg.Threshold)
	}

	var ccfg config
	for _, opt := range opts {
		opt(&ccfg)
	}
	if ccfg.ttl <= 0 {
		return nil, fmt.Errorf("judgment/memory: WithTTL lifetime must be positive, got %v", ccfg.ttl)
	}
	cacheOpts := []ttlcache.Option[Key, semanticEntry]{
		ttlcache.WithTTL[Key, semanticEntry](ccfg.ttl),
	}
	if ccfg.capacity > 0 {
		cacheOpts = append(cacheOpts, ttlcache.WithCapacity[Key, semanticEntry](ccfg.capacity))
	}
	if ccfg.disableTouchOnHit {
		cacheOpts = append(cacheOpts, ttlcache.WithDisableTouchOnHit[Key, semanticEntry]())
	}
	cache := ttlcache.New[Key, semanticEntry](cacheOpts...)
	store := &SemanticStore{
		cache:     cache,
		embedder:  cfg.Embedder,
		model:     cfg.Model,
		threshold: threshold,
	}
	go cache.Start() // Cleanup goroutine: exits when Close calls Stop.
	return store, nil
}

// Metrics returns cumulative lookup and insertion counters.
func (s *SemanticStore) Metrics() SemanticMetrics {
	return SemanticMetrics{
		ExactHits:    s.exactHits.Load(),
		SemanticHits: s.semanticHits.Load(),
		Misses:       s.misses.Load(),
		Insertions:   s.insertions.Load(),
	}
}

// Close stops the background eviction goroutine. It is idempotent.
func (s *SemanticStore) Close() error {
	s.closeOnce.Do(func() { s.cache.Stop() })
	return nil
}

// GetChoice returns the cached choice for key: an exact fingerprint hit
// first, then a semantic match for rawInput within the same scope.
func (s *SemanticStore) GetChoice(ctx context.Context, key Key, rawInput string) (judgment.Choice, Hit, error) {
	if item := s.cache.Get(key); item != nil {
		if stored := item.Value().choice; stored != nil {
			s.exactHits.Add(1)
			return copyChoice(*stored), Hit{Kind: HitExact}, nil
		}
	}
	found, hit, err := s.scan(ctx, key, rawInput)
	if err != nil {
		return judgment.Choice{}, Hit{}, err
	}
	if hit.Kind == HitSemantic && found.choice != nil {
		return copyChoice(*found.choice), hit, nil
	}
	return judgment.Choice{}, Hit{}, nil
}

// PutChoice caches result under key, embedding rawInput for later
// semantic comparison. Keys must carry PrimitiveChoice.
func (s *SemanticStore) PutChoice(ctx context.Context, key Key, rawInput string, result judgment.Choice, opts ...PutOption) error {
	if err := validateKey(key, PrimitiveChoice); err != nil {
		return err
	}
	vec, err := s.embed(ctx, rawInput)
	if err != nil {
		return err
	}
	stored := copyChoice(result)
	s.cache.Set(key, semanticEntry{choice: &stored, vec: vec}, putTTL(opts))
	s.insertions.Add(1)
	return nil
}

// GetScore returns the cached score with GetChoice's resolution rules.
func (s *SemanticStore) GetScore(ctx context.Context, key Key, rawInput string) (judgment.Score, Hit, error) {
	if item := s.cache.Get(key); item != nil {
		if stored := item.Value().score; stored != nil {
			s.exactHits.Add(1)
			return *stored, Hit{Kind: HitExact}, nil
		}
	}
	found, hit, err := s.scan(ctx, key, rawInput)
	if err != nil {
		return judgment.Score{}, Hit{}, err
	}
	if hit.Kind == HitSemantic && found.score != nil {
		return *found.score, hit, nil
	}
	return judgment.Score{}, Hit{}, nil
}

// PutScore caches result under key, embedding rawInput. Keys must carry
// PrimitiveScore.
func (s *SemanticStore) PutScore(ctx context.Context, key Key, rawInput string, result judgment.Score, opts ...PutOption) error {
	if err := validateKey(key, PrimitiveScore); err != nil {
		return err
	}
	vec, err := s.embed(ctx, rawInput)
	if err != nil {
		return err
	}
	stored := result
	s.cache.Set(key, semanticEntry{score: &stored, vec: vec}, putTTL(opts))
	s.insertions.Add(1)
	return nil
}

// GetNoul returns the cached noul with GetChoice's resolution rules.
func (s *SemanticStore) GetNoul(ctx context.Context, key Key, rawInput string) (judgment.Noul, Hit, error) {
	if item := s.cache.Get(key); item != nil {
		if stored := item.Value().noul; stored != nil {
			s.exactHits.Add(1)
			return *stored, Hit{Kind: HitExact}, nil
		}
	}
	found, hit, err := s.scan(ctx, key, rawInput)
	if err != nil {
		return judgment.Noul{}, Hit{}, err
	}
	if hit.Kind == HitSemantic && found.noul != nil {
		return *found.noul, hit, nil
	}
	return judgment.Noul{}, Hit{}, nil
}

// PutNoul caches result under key, embedding rawInput. Keys must carry
// PrimitiveNoul.
func (s *SemanticStore) PutNoul(ctx context.Context, key Key, rawInput string, result judgment.Noul, opts ...PutOption) error {
	if err := validateKey(key, PrimitiveNoul); err != nil {
		return err
	}
	vec, err := s.embed(ctx, rawInput)
	if err != nil {
		return err
	}
	stored := result
	s.cache.Set(key, semanticEntry{noul: &stored, vec: vec}, putTTL(opts))
	s.insertions.Add(1)
	return nil
}

// scan embeds rawInput and returns the closest entry within the key's
// tenant, schema, and primitive scope when its cosine similarity meets
// the threshold.
func (s *SemanticStore) scan(ctx context.Context, key Key, rawInput string) (semanticEntry, Hit, error) {
	if rawInput == "" {
		s.misses.Add(1)
		return semanticEntry{}, Hit{}, nil
	}
	query, err := s.embed(ctx, rawInput)
	if err != nil {
		return semanticEntry{}, Hit{}, err
	}
	best, bestSim, found := closestSemanticEntry(s.cache.Items(), key, query)
	if found && bestSim >= s.threshold {
		s.semanticHits.Add(1)
		return best, Hit{Kind: HitSemantic, Similarity: bestSim}, nil
	}
	s.misses.Add(1)
	return semanticEntry{}, Hit{}, nil
}

// closestSemanticEntry ranks a nonexpired Items snapshot without traversing the
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

// embed embeds one judgment input as a query and validates the result
// against the configured model.
func (s *SemanticStore) embed(ctx context.Context, text string) ([]float32, error) {
	request := embedding.Request{Model: s.model, Inputs: []string{text}, Kind: embedding.KindQuery}
	result, err := s.embedder.Embed(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("judgment/memory: embed judgment input: %w", err)
	}
	if err := embedding.ValidateResult(request, result); err != nil {
		return nil, fmt.Errorf("judgment/memory: %w", err)
	}
	return result.Vectors[0], nil
}
