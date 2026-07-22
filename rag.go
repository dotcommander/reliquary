package reliquary

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/internal/indexutil"
	"github.com/dotcommander/reliquary/internal/validate"
	pipelinelexical "github.com/dotcommander/reliquary/pipeline/lexical"
	"github.com/dotcommander/reliquary/retrieval"
)

const defaultRRFK = 60

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

// Search embeds the query, asks the index for candidates, applies weighted
// hybrid ordering or optional reciprocal rank fusion, then applies an optional
// external reranker, TopK, and MMR diversification.
func (a *App) Search(ctx context.Context, query string, opts ...SearchOption) ([]*retrieval.Result, error) {
	if err := a.ensureReady(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}
	cfg, err := makeSearchConfig(opts)
	if err != nil {
		return nil, err
	}
	request := embedding.Request{Inputs: []string{query}}
	embedded, err := a.embedder.Embed(ctx, request)
	if err != nil {
		return nil, err
	}
	if err := embedding.ValidateResult(request, embedded); err != nil {
		return nil, err
	}
	return a.searchEmbedded(ctx, query, embedded.Vectors[0], cfg)
}

// SearchBatch searches multiple queries while preserving their input order.
// Blank queries produce nil rows. All nonblank queries are embedded in one
// ordered call, validated as a complete batch, and searched sequentially.
func (a *App) SearchBatch(ctx context.Context, queries []string, opts ...SearchOption) ([][]*retrieval.Result, error) {
	if err := a.ensureReady(); err != nil {
		return nil, err
	}
	rows := make([][]*retrieval.Result, len(queries))
	nonblank := make([]string, 0, len(queries))
	positions := make([]int, 0, len(queries))
	for i, query := range queries {
		if strings.TrimSpace(query) == "" {
			continue
		}
		nonblank = append(nonblank, query)
		positions = append(positions, i)
	}
	if len(nonblank) == 0 {
		return rows, nil
	}
	cfg, err := makeSearchConfig(opts)
	if err != nil {
		return nil, err
	}
	request := embedding.Request{Inputs: nonblank}
	embedded, err := a.embedder.Embed(ctx, request)
	if err != nil {
		return nil, err
	}
	if err := embedding.ValidateResult(request, embedded); err != nil {
		return nil, err
	}
	for i, query := range nonblank {
		row, err := a.searchEmbedded(ctx, query, embedded.Vectors[i], cfg)
		if err != nil {
			return nil, err
		}
		rows[positions[i]] = row
	}
	return rows, nil
}

func makeSearchConfig(opts []SearchOption) (searchConfig, error) {
	cfg := searchConfig{}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	return cfg, cfg.err
}

func (a *App) searchEmbedded(ctx context.Context, query string, vector embedding.Vector, cfg searchConfig) ([]*retrieval.Result, error) {
	if cfg.rrf {
		return a.searchEmbeddedRRF(ctx, query, vector, cfg)
	}
	items, err := a.index.Search(ctx, IndexQuery{
		Identity: a.indexIdentity,
		Text:     query,
		Vector:   slices.Clone(vector),
		Limit:    cfg.candidateLimit,
		Filter:   cloneSearchFilter(cfg.filter),
	})
	if err != nil {
		return nil, err
	}
	if len(items) == 0 {
		return nil, nil
	}
	// Rerank scores results in place; clone each result so searches never mutate
	// stored state and remain safe regardless of how the Index returns items.
	scored := make([]*retrieval.Result, 0, len(items))
	for _, item := range items {
		if item == nil {
			continue
		}
		scored = append(scored, indexutil.Clone(item))
	}
	if len(scored) == 0 {
		return nil, nil
	}
	ranked, explanations := scoreCandidates(a.weights, vector, query, scored, cfg.explain, true)
	return finishSearch(ctx, query, cfg, ranked, explanations)
}

func (a *App) searchEmbeddedRRF(ctx context.Context, query string, vector embedding.Vector, cfg searchConfig) ([]*retrieval.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	vectorItems, err := a.index.Search(ctx, IndexQuery{
		Identity: a.indexIdentity,
		Vector:   slices.Clone(vector),
		Limit:    cfg.candidateLimit,
		Filter:   cloneSearchFilter(cfg.filter),
	})
	if err != nil {
		return nil, fmt.Errorf("reliquary search vector lane: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	vectorRanked, vectorByID, err := cloneRRFLane("vector", vectorItems)
	if err != nil {
		return nil, err
	}
	lexicalItems, err := a.index.Search(ctx, IndexQuery{
		Identity: a.indexIdentity,
		Text:     query,
		Limit:    cfg.candidateLimit,
		Filter:   cloneSearchFilter(cfg.filter),
	})
	if err != nil {
		return nil, fmt.Errorf("reliquary search lexical lane: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	lexicalRanked, lexicalByID, err := cloneRRFLane("lexical", lexicalItems)
	if err != nil {
		return nil, err
	}
	inputs := make([]pipelinelexical.FusionInput, 0, 2)
	nonemptyLaneCount := 0
	if len(vectorRanked) > 0 {
		inputs = append(inputs, pipelinelexical.FusionInput{Source: "vector", Candidates: vectorRanked})
		nonemptyLaneCount++
	}
	if len(lexicalRanked) > 0 {
		inputs = append(inputs, pipelinelexical.FusionInput{Source: "lexical", Candidates: lexicalRanked})
		nonemptyLaneCount++
	}
	if nonemptyLaneCount == 0 {
		return nil, nil
	}

	fused := pipelinelexical.FuseRRFByID(inputs, pipelinelexical.FusionOptions{K: cfg.rrfK})
	ranked := make([]*retrieval.Result, 0, len(fused))
	rrfScores := make(map[string]float64, len(fused))
	normalize := (cfg.rrfK + 1) / float64(nonemptyLaneCount)
	for _, candidate := range fused {
		item := vectorByID[candidate.ID]
		if item == nil {
			item = lexicalByID[candidate.ID]
		}
		if item == nil {
			return nil, fmt.Errorf("reliquary search RRF: missing payload for result %q", candidate.ID)
		}
		rrfScores[candidate.ID] = candidate.Score * normalize
		ranked = append(ranked, item)
	}

	// Populate diagnostic component scores on the fused union, then restore the
	// RRF-owned order and CombinedScore. App weights therefore cannot alter RRF.
	ranked, explanations := scoreCandidates(a.weights, vector, query, ranked, cfg.explain, false)
	byID := make(map[string]*retrieval.Result, len(ranked))
	for _, item := range ranked {
		byID[item.ID] = item
	}
	for i, candidate := range fused {
		ranked[i] = byID[candidate.ID]
		ranked[i].CombinedScore = rrfScores[candidate.ID]
		if explanation := explanations[ranked[i]]; explanation != nil {
			vectorRank := firstRRFRank(vectorRanked, candidate.ID)
			lexicalRank := firstRRFRank(lexicalRanked, candidate.ID)
			explanation.RRF = &retrieval.RRFExplanation{
				K:                   cfg.rrfK,
				VectorRank:          vectorRank,
				LexicalRank:         lexicalRank,
				VectorContribution:  normalizedRRFContribution(cfg.rrfK, vectorRank, normalize),
				LexicalContribution: normalizedRRFContribution(cfg.rrfK, lexicalRank, normalize),
				FusedScore:          rrfScores[candidate.ID],
				FusedRank:           i + 1,
			}
		}
	}
	return finishSearch(ctx, query, cfg, ranked, explanations)
}

type explanationMap map[*retrieval.Result]*retrieval.SearchExplanation

func scoreCandidates(weights retrieval.Weights, vector embedding.Vector, query string, candidates []*retrieval.Result, explain, hybridScoreUsed bool) ([]*retrieval.Result, explanationMap) {
	scorer := retrieval.NewScorer(weights)
	if !explain {
		return scorer.RerankEmbedding(vector, query, candidates), nil
	}
	ranked, traces := scorer.RerankWithTrace(retrieval.EmbeddingVector(vector), query, candidates)
	explanations := make(explanationMap, len(ranked))
	for i, result := range ranked {
		explanations[result] = &retrieval.SearchExplanation{
			Hybrid:          traces[i],
			HybridRank:      i + 1,
			HybridScoreUsed: hybridScoreUsed,
		}
	}
	return ranked, explanations
}

func firstRRFRank(ranked pipelinelexical.RankedList, id string) int {
	for i, candidate := range ranked {
		if candidate.ID == id {
			return i + 1
		}
	}
	return 0
}

func normalizedRRFContribution(k float64, rank int, normalize float64) float64 {
	if rank == 0 {
		return 0
	}
	return normalize / (k + float64(rank))
}

func cloneRRFLane(source string, items []*retrieval.Result) (pipelinelexical.RankedList, map[string]*retrieval.Result, error) {
	ranked := make(pipelinelexical.RankedList, 0, len(items))
	byID := make(map[string]*retrieval.Result, len(items))
	for _, item := range items {
		if item == nil {
			continue
		}
		if strings.TrimSpace(item.ID) == "" {
			return nil, nil, fmt.Errorf("reliquary search RRF: %s lane returned a blank result ID", source)
		}
		ranked = append(ranked, pipelinelexical.Candidate{ID: item.ID, Source: source})
		if _, exists := byID[item.ID]; !exists {
			byID[item.ID] = indexutil.Clone(item)
		}
	}
	return ranked, byID, nil
}

func finishSearch(ctx context.Context, query string, cfg searchConfig, ranked []*retrieval.Result, explanations explanationMap) ([]*retrieval.Result, error) {
	if cfg.reranker != nil {
		if err := applyReranker(ctx, query, cfg.reranker, ranked, explanations); err != nil {
			return nil, err
		}
	}
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
		if !cfg.explain {
			return retrieval.Diversify(ranked, k, cfg.lambda), nil
		}
		final, traces := retrieval.DiversifyWithTrace(ranked, k, cfg.lambda)
		for i, result := range final {
			explanations[result].MMR = &traces[i]
		}
		attachExplanations(final, explanations)
		return final, nil
	}
	final := ranked[:k]
	attachExplanations(final, explanations)
	return final, nil
}

func applyReranker(ctx context.Context, query string, reranker retrieval.Reranker, ranked []*retrieval.Result, explanations explanationMap) error {
	if len(ranked) == 0 {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	candidates := make([]*retrieval.Result, len(ranked))
	for i, result := range ranked {
		candidates[i] = indexutil.Clone(result)
	}
	scores, err := reranker.Rerank(ctx, query, candidates)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := retrieval.ValidateRerankScores(len(ranked), scores); err != nil {
		return err
	}
	for i, score := range scores {
		ranked[i].CombinedScore = score
		if explanation := explanations[ranked[i]]; explanation != nil {
			explanation.Reranker = &retrieval.RerankerExplanation{InputRank: i + 1, Score: score}
		}
	}
	slices.SortStableFunc(ranked, func(a, b *retrieval.Result) int {
		return cmp.Compare(b.CombinedScore, a.CombinedScore)
	})
	for i, result := range ranked {
		if explanation := explanations[result]; explanation != nil {
			explanation.Reranker.Rank = i + 1
		}
	}
	return nil
}

func attachExplanations(results []*retrieval.Result, explanations explanationMap) {
	for i, result := range results {
		if explanation := explanations[result]; explanation != nil {
			explanation.FinalRank = i + 1
			result.Explain = explanation
		}
	}
}

func cloneSearchFilter(filter map[string]any) map[string]any {
	if filter == nil {
		return nil
	}
	cloned := make(map[string]any, len(filter))
	for key, value := range filter {
		cloned[key] = value
	}
	return cloned
}

// SearchOption tunes a Search or SearchBatch call. Options never mutate the App.
type SearchOption func(*searchConfig)

type searchConfig struct {
	k              int
	kSet           bool
	mmr            bool
	lambda         float64
	candidateLimit int
	filter         map[string]any
	reranker       retrieval.Reranker
	rrf            bool
	rrfK           float64
	explain        bool
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

// WithRRF enables reciprocal-rank fusion over independent vector-only and
// text-only candidate searches. Non-finite values and k <= 0 use the standard
// constant 60. The last WithRRF option wins.
func WithRRF(k float64) SearchOption {
	if k <= 0 || math.IsNaN(k) || math.IsInf(k, 0) {
		k = defaultRRFK
	}
	return func(c *searchConfig) {
		c.rrf = true
		c.rrfK = k
	}
}

// WithReranker adds an external reranking stage after hybrid scoring or RRF and
// before TopK and MMR. The last WithReranker option wins; nil and typed-nil
// values disable the stage. Concurrent Search calls may invoke the same
// Reranker concurrently, so implementations must provide any required
// synchronization.
func WithReranker(r retrieval.Reranker) SearchOption {
	return func(c *searchConfig) {
		if validate.IsNil(r) {
			c.reranker = nil
			return
		}
		c.reranker = r
	}
}

// WithExplain attaches an ephemeral, typed ranking explanation to each
// returned result. Explanations are never stored in an Index and cover only
// candidates retained for facade-level ranking.
func WithExplain() SearchOption {
	return func(c *searchConfig) {
		c.explain = true
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
