package reliquary

import (
	"context"
	"fmt"
	"strings"

	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/retrieval"
)

// Ingest chunks each document with the App's strategy, embeds every chunk with
// the App's embedder, and stores the embedded chunks. It returns the number of
// chunks produced.
func (a *App) Ingest(ctx context.Context, docs ...document.Document) (int, error) {
	if err := a.ensureReady(); err != nil {
		return 0, err
	}
	items, err := retrieval.ResultsFromDocuments(docs, a.strategy, a.size, a.overlap)
	if err != nil {
		return 0, err
	}
	if len(items) == 0 {
		return 0, nil
	}
	if err := retrieval.EmbedResults(ctx, a.embedder, items); err != nil {
		return 0, err
	}
	for _, item := range items {
		if item != nil {
			item.IndexIdentity = a.indexIdentity
		}
	}
	if err := a.index.Upsert(ctx, items); err != nil {
		return 0, err
	}
	return len(items), nil
}

// Search embeds the query, asks the index for candidates, applies final hybrid
// scoring, and then applies TopK and optional MMR diversification.
func (a *App) Search(ctx context.Context, query string, opts ...SearchOption) ([]*retrieval.Result, error) {
	if err := a.ensureReady(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}
	cfg := searchConfig{}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	if cfg.err != nil {
		return nil, cfg.err
	}
	embedded, err := a.embedder.Embed(ctx, embeddings.Request{Inputs: []string{query}})
	if err != nil {
		return nil, err
	}
	if len(embedded.Vectors) == 0 {
		return nil, nil
	}
	items, err := a.index.Search(ctx, IndexQuery{
		Identity: a.indexIdentity,
		Text:     query,
		Vector:   embedded.Vectors[0],
		Limit:    cfg.candidateLimit,
		Filter:   cfg.filter,
	})
	if err != nil {
		return nil, err
	}
	if len(items) == 0 {
		return nil, nil
	}
	// Rerank scores results in place; copy each result so Search never mutates
	// stored state and is safe to call concurrently, regardless of how the Index
	// implementation hands back its items.
	scored := make([]*retrieval.Result, 0, len(items))
	for _, item := range items {
		if item == nil {
			continue
		}
		cp := *item
		scored = append(scored, &cp)
	}
	if len(scored) == 0 {
		return nil, nil
	}
	ranked := retrieval.NewScorer(a.weights).RerankEmbedding(embedded.Vectors[0], query, scored)
	k := len(ranked)
	if cfg.kSet {
		k = cfg.k
	}
	if k < 0 {
		k = 0
	}
	if k > len(ranked) {
		k = len(ranked)
	}
	if cfg.mmr {
		return retrieval.Diversify(ranked, k, cfg.lambda), nil
	}
	return ranked[:k], nil
}

// SearchOption tunes a single Search call. Options never mutate the App.
type SearchOption func(*searchConfig)

type searchConfig struct {
	k              int
	kSet           bool
	mmr            bool
	lambda         float64
	candidateLimit int
	filter         map[string]any
	err            error
}

// CandidateLimit bounds index candidate retrieval independently of final TopK.
// Values less than or equal to zero request implementation-defined or
// unbounded candidate retrieval.
func CandidateLimit(n int) SearchOption {
	return func(c *searchConfig) {
		if n > 0 {
			c.candidateLimit = n
		}
	}
}

// WithFilter restricts index candidates by reserved result fields (id,
// document_id, and filename) or scalar metadata values. The filter is
// snapshotted when the option is created; later caller mutations have no
// effect. Compound values such as slices and maps cause Search to return an
// error before embedding the query.
func WithFilter(filter map[string]any) SearchOption {
	snapshot := make(map[string]any, len(filter))
	for key, value := range filter {
		snapshot[key] = value
	}
	return func(c *searchConfig) {
		if c.err != nil {
			return
		}
		c.filter = make(map[string]any, len(snapshot))
		for key, value := range snapshot {
			if !isScalarFilterValue(value) {
				c.err = fmt.Errorf("reliquary search: filter %q must be scalar", key)
				return
			}
			c.filter[key] = value
		}
	}
}

func isScalarFilterValue(value any) bool {
	switch value.(type) {
	case nil, string, bool,
		int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64,
		float32, float64:
		return true
	default:
		return false
	}
}

// TopK limits Search to the k highest-ranked results. k <= 0 returns no
// results. Without it, Search returns all scored results.
func TopK(k int) SearchOption {
	return func(c *searchConfig) {
		c.k = k
		c.kSet = true
	}
}

// WithMMR enables maximal-marginal-relevance diversification with the given
// lambda (0 = maximize diversity, 1 = maximize relevance).
func WithMMR(lambda float64) SearchOption {
	return func(c *searchConfig) {
		c.mmr = true
		c.lambda = lambda
	}
}
