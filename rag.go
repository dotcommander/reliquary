package reliquary

import (
	"context"
	"fmt"
	"strings"

	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/internal/indexutil"
	"github.com/dotcommander/reliquary/retrieval"
)

// Ingest chunks each document with the App's strategy, embeds every chunk, and
// atomically replaces all prior chunks for every supplied document. A document
// that produces no chunks deletes its prior revision. Document IDs must be
// non-blank and unique within the call. It returns the number of chunks produced.
func (a *App) Ingest(ctx context.Context, docs ...document.Document) (int, error) {
	if err := a.ensureReady(); err != nil {
		return 0, err
	}
	seen := make(map[string]struct{}, len(docs))
	for _, doc := range docs {
		if strings.TrimSpace(doc.ID) == "" {
			return 0, ErrInvalidDocumentID
		}
		if _, exists := seen[doc.ID]; exists {
			return 0, fmt.Errorf("%w: %q", ErrDuplicateDocumentID, doc.ID)
		}
		seen[doc.ID] = struct{}{}
	}
	items, err := retrieval.ResultsFromDocuments(docs, a.strategy, a.size, a.overlap)
	if err != nil {
		return 0, err
	}
	if len(docs) == 0 {
		return 0, nil
	}
	if len(items) > 0 {
		if err := retrieval.EmbedResults(ctx, a.embedder, items); err != nil {
			return 0, err
		}
	}
	replacements := make([]DocumentReplacement, len(docs))
	byDocument := make(map[string]int, len(docs))
	for n, doc := range docs {
		replacements[n].DocumentID = doc.ID
		byDocument[doc.ID] = n
	}
	for _, item := range items {
		if item == nil {
			continue
		}
		item.IndexIdentity = a.indexIdentity
		n := byDocument[item.DocumentID]
		replacements[n].Results = append(replacements[n].Results, item)
	}
	if err := a.index.ReplaceDocuments(ctx, replacements); err != nil {
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
	request := embeddings.Request{Inputs: []string{query}}
	embedded, err := a.embedder.Embed(ctx, request)
	if err != nil {
		return nil, err
	}
	if err := embeddings.ValidateResult(request, embedded); err != nil {
		return nil, err
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
// document_id, and filename) or JSON-scalar metadata values. Reserved fields
// match strings only. Metadata keys must be present; explicit nil matches only
// JSON null, strings and booleans are type-exact, and finite numbers compare by
// exact JSON numeric value across accepted Go numeric types and json.Number.
// The filter is snapshotted when the option is created; later caller mutations
// have no effect. Compound and non-finite values cause Search to return an error
// before embedding the query.
func WithFilter(filter map[string]any) SearchOption {
	snapshot := make(map[string]any, len(filter))
	for key, value := range filter {
		snapshot[key] = value
	}
	return func(c *searchConfig) {
		if c.err != nil {
			return
		}
		if err := indexutil.ValidateFilter(snapshot); err != nil {
			c.err = fmt.Errorf("reliquary search: %w", err)
			return
		}
		c.filter = make(map[string]any, len(snapshot))
		for key, value := range snapshot {
			c.filter[key] = value
		}
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
