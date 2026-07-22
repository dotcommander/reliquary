package reliquary_test

import (
	"context"
	"errors"
	"math"
	"reflect"
	"slices"
	"sync"
	"testing"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/retrieval"
)

type rrfIndex struct {
	rows          [][]*retrieval.Result
	errors        map[int]error
	queries       []reliquary.IndexQuery
	mutateInputs  bool
	afterSearch   func(int)
	contextValues []any
}

func (*rrfIndex) Upsert(context.Context, []*retrieval.Result) error { return nil }
func (*rrfIndex) ReplaceDocuments(context.Context, []reliquary.DocumentReplacement) error {
	return nil
}
func (*rrfIndex) DeleteDocument(context.Context, string) error { return nil }
func (i *rrfIndex) Search(ctx context.Context, query reliquary.IndexQuery) ([]*retrieval.Result, error) {
	call := len(i.queries)
	i.queries = append(i.queries, cloneRRFQuery(query))
	i.contextValues = append(i.contextValues, ctx.Value(rrfContextKey{}))
	if i.mutateInputs {
		if len(query.Vector) > 0 {
			query.Vector[0] = 99
		}
		if query.Filter != nil {
			query.Filter["tenant"] = "mutated"
		}
	}
	if i.afterSearch != nil {
		i.afterSearch(call + 1)
	}
	if err := i.errors[call+1]; err != nil {
		return nil, err
	}
	if call >= len(i.rows) {
		return nil, nil
	}
	return i.rows[call], nil
}

func cloneRRFQuery(query reliquary.IndexQuery) reliquary.IndexQuery {
	query.Vector = slices.Clone(query.Vector)
	if query.Filter != nil {
		filter := make(map[string]any, len(query.Filter))
		for key, value := range query.Filter {
			filter[key] = value
		}
		query.Filter = filter
	}
	return query
}

type rrfEmbedder struct {
	vectors [][]float32
}

func (e rrfEmbedder) Embed(ctx context.Context, request embedding.Request) (embedding.Result, error) {
	if err := ctx.Err(); err != nil {
		return embedding.Result{}, err
	}
	vectors := make([]embedding.Vector, len(request.Inputs))
	for n := range vectors {
		if n < len(e.vectors) {
			vectors[n] = slices.Clone(e.vectors[n])
		} else {
			vectors[n] = embedding.Vector{float32(n + 1), 0}
		}
	}
	return embedding.Result{Vectors: vectors}, nil
}

type rrfContextKey struct{}

type reusedPayloadIndex struct {
	result *retrieval.Result
}

func (*reusedPayloadIndex) Upsert(context.Context, []*retrieval.Result) error { return nil }
func (*reusedPayloadIndex) ReplaceDocuments(context.Context, []reliquary.DocumentReplacement) error {
	return nil
}
func (*reusedPayloadIndex) DeleteDocument(context.Context, string) error { return nil }
func (i *reusedPayloadIndex) Search(_ context.Context, query reliquary.IndexQuery) ([]*retrieval.Result, error) {
	if query.Text == "" {
		i.result.Content = "vector payload"
	} else {
		i.result.Content = "lexical payload"
	}
	return []*retrieval.Result{i.result}, nil
}

func newRRFApp(t *testing.T, idx reliquary.Index, opts ...reliquary.Option) *reliquary.App {
	t.Helper()
	options := []reliquary.Option{
		reliquary.WithEmbedder(rrfEmbedder{vectors: [][]float32{{1, 0}, {2, 0}}}),
		reliquary.WithIndex(idx),
		reliquary.WithIndexIdentity("rrf-test"),
	}
	options = append(options, opts...)
	app, err := reliquary.New(options...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return app
}

func TestSearchWithoutRRFUsesOneMixedIndexCall(t *testing.T) {
	stored := []*retrieval.Result{
		{ID: "weaker", Content: "other", Embedding: []float64{0, 1}},
		{ID: "hit", Content: "query", Embedding: []float64{1, 0}},
	}
	idx := &rrfIndex{rows: [][]*retrieval.Result{stored}}
	app := newRRFApp(t, idx)
	hits, err := app.Search(context.Background(), "query", reliquary.CandidateLimit(7))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(idx.queries) != 1 {
		t.Fatalf("index calls = %d, want 1", len(idx.queries))
	}
	query := idx.queries[0]
	if query.Text != "query" || !reflect.DeepEqual(query.Vector, []float32{1, 0}) || query.Limit != 7 {
		t.Fatalf("mixed query = %#v", query)
	}
	expected := retrieval.NewScorer(retrieval.DefaultWeights()).RerankEmbedding(embedding.Vector{1, 0}, "query", []*retrieval.Result{
		{ID: "weaker", Content: "other", Embedding: []float64{0, 1}},
		{ID: "hit", Content: "query", Embedding: []float64{1, 0}},
	})
	if !reflect.DeepEqual(hits, expected) {
		t.Fatalf("default output = %#v, want existing weighted output %#v", hits, expected)
	}
}

func TestWithRRFFusionMathOrderingAndPayloads(t *testing.T) {
	vectorA := &retrieval.Result{ID: "a", Content: "vector payload", Metadata: map[string]any{"lane": "vector"}, Embedding: []float64{1, 0}, CombinedScore: 99}
	vectorADuplicate := &retrieval.Result{ID: "a", Content: "duplicate payload"}
	lexicalA := &retrieval.Result{ID: "a", Content: "lexical payload", Metadata: map[string]any{"lane": "lexical"}}
	idx := &rrfIndex{rows: [][]*retrieval.Result{
		{nil, vectorA, vectorADuplicate, {ID: "b", Content: "vector only", Embedding: []float64{0, 1}}},
		{{ID: "c", Content: "lexical only"}, lexicalA, {ID: "d", Content: "lexical tail"}},
	}}
	app := newRRFApp(t, idx)
	hits, err := app.Search(context.Background(), "query", reliquary.WithRRF(60))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if got, want := resultIDs(hits), []string{"a", "c", "b", "d"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("IDs = %v, want %v", got, want)
	}
	wantA := (1.0/61 + 1.0/62) * 61 / 2
	if math.Abs(hits[0].CombinedScore-wantA) > 1e-12 || hits[1].CombinedScore != .5 {
		t.Fatalf("scores = [%v %v], want [%v .5]", hits[0].CombinedScore, hits[1].CombinedScore, wantA)
	}
	if hits[0].Content != "vector payload" || hits[0].Metadata["lane"] != "vector" {
		t.Fatalf("overlap payload = %#v, want vector lane", hits[0])
	}
	if hits[1].Content != "lexical only" || hits[2].Content != "vector only" {
		t.Fatalf("lane-only payloads = %#v", hits)
	}
	if hits[0].EmbeddingScore == 0 || hits[0].KeywordScore == 0 {
		t.Fatalf("diagnostic component scores were not populated: %#v", hits[0])
	}
	if vectorA.CombinedScore != 99 || vectorA.Metadata["lane"] != "vector" || vectorA.Embedding[0] != 1 {
		t.Fatalf("stored vector payload mutated: %#v", vectorA)
	}
	hits[0].Metadata["lane"] = "changed"
	hits[0].Embedding[0] = 42
	if vectorA.Metadata["lane"] != "vector" || vectorA.Embedding[0] != 1 {
		t.Fatalf("returned payload aliases index state: %#v", vectorA)
	}
}

func TestWithRRFDefaultNegativeAndLastOccurrenceWins(t *testing.T) {
	tests := []struct {
		name string
		opts []reliquary.SearchOption
		k    float64
	}{
		{name: "zero defaults", opts: []reliquary.SearchOption{reliquary.WithRRF(0)}, k: 60},
		{name: "negative defaults", opts: []reliquary.SearchOption{reliquary.WithRRF(-4)}, k: 60},
		{name: "NaN defaults", opts: []reliquary.SearchOption{reliquary.WithRRF(math.NaN())}, k: 60},
		{name: "infinity defaults", opts: []reliquary.SearchOption{reliquary.WithRRF(math.Inf(1))}, k: 60},
		{name: "last wins", opts: []reliquary.SearchOption{reliquary.WithRRF(2), reliquary.WithRRF(10)}, k: 10},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx := &rrfIndex{rows: [][]*retrieval.Result{
				{{ID: "a"}, {ID: "b"}},
				{{ID: "b"}, {ID: "a"}},
			}}
			hits, err := newRRFApp(t, idx).Search(context.Background(), "query", tt.opts...)
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			want := (1/(tt.k+1) + 1/(tt.k+2)) * (tt.k + 1) / 2
			if len(hits) != 2 || math.Abs(hits[0].CombinedScore-want) > 1e-12 || hits[0].ID != "a" || hits[1].ID != "b" {
				t.Fatalf("hits = %#v, want tied a,b at %v", hits, want)
			}
		})
	}
}

func TestWithRRFEmptyLaneNormalizationAndWeightIndependence(t *testing.T) {
	tests := []struct {
		name string
		rows [][]*retrieval.Result
		ids  []string
	}{
		{name: "vector only", rows: [][]*retrieval.Result{{{ID: "v"}}, nil}, ids: []string{"v"}},
		{name: "lexical only", rows: [][]*retrieval.Result{nil, {{ID: "l"}}}, ids: []string{"l"}},
		{name: "both empty", rows: make([][]*retrieval.Result, 2)},
		{name: "rank one overlap", rows: [][]*retrieval.Result{{{ID: "same"}}, {{ID: "same"}}}, ids: []string{"same"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hits, err := newRRFApp(t, &rrfIndex{rows: tt.rows}).Search(context.Background(), "query", reliquary.WithRRF(60))
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			if got := resultIDs(hits); !reflect.DeepEqual(got, tt.ids) {
				t.Fatalf("IDs = %v, want %v", got, tt.ids)
			}
			if len(hits) == 1 && hits[0].CombinedScore != 1 {
				t.Fatalf("top score = %v, want 1", hits[0].CombinedScore)
			}
		})
	}

	rows := func() [][]*retrieval.Result {
		return [][]*retrieval.Result{
			{{ID: "a", Content: "query", Embedding: []float64{1, 0}}, {ID: "b", Content: "other", Embedding: []float64{0, 1}}},
			{{ID: "b", Content: "other"}, {ID: "a", Content: "query"}},
		}
	}
	defaults, err := newRRFApp(t, &rrfIndex{rows: rows()}).Search(context.Background(), "query", reliquary.WithRRF(60))
	if err != nil {
		t.Fatalf("default weights: %v", err)
	}
	custom, err := newRRFApp(t, &rrfIndex{rows: rows()}, reliquary.WithWeights(retrieval.Weights{Filename: 1})).Search(context.Background(), "query", reliquary.WithRRF(60))
	if err != nil {
		t.Fatalf("custom weights: %v", err)
	}
	for i := range defaults {
		if defaults[i].ID != custom[i].ID || defaults[i].CombinedScore != custom[i].CombinedScore {
			t.Fatalf("weights changed RRF output: default %#v, custom %#v", defaults, custom)
		}
	}
}

func TestWithRRFOptionCanBeReusedConcurrently(t *testing.T) {
	app := reliquary.Quickstart()
	option := reliquary.WithRRF(-1)
	var wg sync.WaitGroup
	errorsByCall := make(chan error, 16)
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := app.Search(context.Background(), "query", option)
			errorsByCall <- err
		}()
	}
	wg.Wait()
	close(errorsByCall)
	for err := range errorsByCall {
		if err != nil {
			t.Fatalf("concurrent Search: %v", err)
		}
	}
}

func TestWithRRFLaneQueriesCloneInputsAndPropagateContext(t *testing.T) {
	idx := &rrfIndex{mutateInputs: true, rows: make([][]*retrieval.Result, 2)}
	app := newRRFApp(t, idx)
	ctx := context.WithValue(context.Background(), rrfContextKey{}, "context")
	filter := map[string]any{"tenant": "original"}
	if _, err := app.Search(ctx, "query", reliquary.WithRRF(60), reliquary.CandidateLimit(9), reliquary.WithFilter(filter)); err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(idx.queries) != 2 {
		t.Fatalf("queries = %d, want 2", len(idx.queries))
	}
	vector, lexical := idx.queries[0], idx.queries[1]
	if vector.Text != "" || !reflect.DeepEqual(vector.Vector, []float32{1, 0}) || lexical.Text != "query" || lexical.Vector != nil {
		t.Fatalf("lane shapes = vector %#v, lexical %#v", vector, lexical)
	}
	if vector.Limit != 9 || lexical.Limit != 9 || vector.Filter["tenant"] != "original" || lexical.Filter["tenant"] != "original" {
		t.Fatalf("lane options = vector %#v, lexical %#v", vector, lexical)
	}
	if idx.contextValues[0] != "context" || idx.contextValues[1] != "context" || filter["tenant"] != "original" {
		t.Fatalf("context/filter propagation = %#v/%#v/%#v", idx.contextValues, vector.Filter, lexical.Filter)
	}
}

func TestWithRRFClonesVectorPayloadBeforeLexicalSearch(t *testing.T) {
	idx := &reusedPayloadIndex{result: &retrieval.Result{ID: "shared"}}
	hits, err := newRRFApp(t, idx).Search(context.Background(), "query", reliquary.WithRRF(60))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 || hits[0].Content != "vector payload" {
		t.Fatalf("hits = %#v, want vector payload cloned before lexical search", hits)
	}
}

func TestWithRRFRejectsBlankIDAndLaneErrors(t *testing.T) {
	t.Run("whitespace ID", func(t *testing.T) {
		idx := &rrfIndex{rows: [][]*retrieval.Result{{{ID: " \t"}}, nil}}
		hits, err := newRRFApp(t, idx).Search(context.Background(), "query", reliquary.WithRRF(60))
		if err == nil || hits != nil {
			t.Fatalf("Search = %#v, %v, want blank-ID error", hits, err)
		}
	})
	for _, failAt := range []int{1, 2} {
		t.Run(string(rune('0'+failAt)), func(t *testing.T) {
			want := errors.New("lane failed")
			idx := &rrfIndex{rows: make([][]*retrieval.Result, 2), errors: map[int]error{failAt: want}}
			hits, err := newRRFApp(t, idx).Search(context.Background(), "query", reliquary.WithRRF(60))
			if !errors.Is(err, want) || hits != nil || len(idx.queries) != failAt {
				t.Fatalf("Search = %#v, %v, calls %d", hits, err, len(idx.queries))
			}
		})
	}
}

func TestWithRRFCancellationBetweenLanesIsFatal(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	idx := &rrfIndex{rows: make([][]*retrieval.Result, 2)}
	idx.afterSearch = func(call int) {
		if call == 1 {
			cancel()
		}
	}
	hits, err := newRRFApp(t, idx).Search(ctx, "query", reliquary.WithRRF(60))
	if !errors.Is(err, context.Canceled) || hits != nil || len(idx.queries) != 1 {
		t.Fatalf("Search = %#v, %v, calls %d", hits, err, len(idx.queries))
	}
}

func TestWithRRFRerankerOverridesScoresBeforeTopKAndMMR(t *testing.T) {
	idx := &rrfIndex{rows: [][]*retrieval.Result{
		{{ID: "a", Embedding: []float64{1, 0}}, {ID: "b", Embedding: []float64{0, 1}}},
		{{ID: "a"}, {ID: "b"}},
	}}
	reranker := &scriptedReranker{scores: []float64{0.1, 0.9}}
	hits, err := newRRFApp(t, idx).Search(context.Background(), "query",
		reliquary.WithRRF(60), reliquary.WithReranker(reranker), reliquary.TopK(1), reliquary.WithMMR(1))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(reranker.calls) != 1 || resultIDs(reranker.calls[0].candidates)[0] != "a" {
		t.Fatalf("reranker input = %#v", reranker.calls)
	}
	if len(hits) != 1 || hits[0].ID != "b" || hits[0].CombinedScore != .9 {
		t.Fatalf("hits = %#v, want reranked b", hits)
	}
}

func TestWithRRFMMRUsesNormalizedRelevance(t *testing.T) {
	idx := &rrfIndex{rows: [][]*retrieval.Result{
		{{ID: "a", Embedding: []float64{1, 0}}, {ID: "b", Embedding: []float64{0, 1}}, {ID: "c", Embedding: []float64{-1, 0}}},
		{{ID: "a"}, {ID: "c"}, {ID: "b"}},
	}}
	hits, err := newRRFApp(t, idx).Search(context.Background(), "query", reliquary.WithRRF(60), reliquary.TopK(2), reliquary.WithMMR(1))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 2 || hits[0].ID != "a" || hits[0].CombinedScore != 1 {
		t.Fatalf("MMR hits = %#v, want normalized top a", hits)
	}
}

func TestSearchBatchWithRRFHandlesBlanksDuplicatesAndNoPartialFailure(t *testing.T) {
	t.Run("rows and calls", func(t *testing.T) {
		idx := &rrfIndex{rows: [][]*retrieval.Result{
			{{ID: "one"}}, {{ID: "one"}},
			{{ID: "two"}}, {{ID: "two"}},
		}}
		app := newRRFApp(t, idx)
		rows, err := app.SearchBatch(context.Background(), []string{"same", " ", "same"}, reliquary.WithRRF(60))
		if err != nil {
			t.Fatalf("SearchBatch: %v", err)
		}
		if len(rows) != 3 || rows[1] != nil || rows[0][0].ID != "one" || rows[2][0].ID != "two" || len(idx.queries) != 4 {
			t.Fatalf("rows/calls = %#v/%d", rows, len(idx.queries))
		}
		if idx.queries[0].Text != "" || idx.queries[1].Text != "same" || idx.queries[2].Text != "" || idx.queries[3].Text != "same" {
			t.Fatalf("query order = %#v", idx.queries)
		}
	})
	t.Run("no partial matrix", func(t *testing.T) {
		want := errors.New("third lane failed")
		idx := &rrfIndex{
			rows:   [][]*retrieval.Result{{{ID: "one"}}, {{ID: "one"}}, nil},
			errors: map[int]error{3: want},
		}
		rows, err := newRRFApp(t, idx).SearchBatch(context.Background(), []string{"one", "two"}, reliquary.WithRRF(60))
		if !errors.Is(err, want) || rows != nil || len(idx.queries) != 3 {
			t.Fatalf("SearchBatch = %#v, %v, calls %d", rows, err, len(idx.queries))
		}
	})
}

func resultIDs(results []*retrieval.Result) []string {
	if results == nil {
		return nil
	}
	ids := make([]string, len(results))
	for i, result := range results {
		ids[i] = result.ID
	}
	return ids
}
