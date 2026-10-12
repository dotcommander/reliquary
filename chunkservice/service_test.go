package chunkservice

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/dotcommander/reliquary/chunking"
)

func TestRequestValidate(t *testing.T) {
	t.Parallel()
	valid := Request{Text: "hello", ChunkSize: 100, Overlap: 10}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid request rejected: %v", err)
	}

	cases := []struct {
		name string
		req  Request
		want string
	}{
		{"empty text", Request{ChunkSize: 100}, "text is required"},
		{"zero size", Request{Text: "x", ChunkSize: 0}, "chunk_size must be greater than 0"},
		{"negative overlap", Request{Text: "x", ChunkSize: 10, Overlap: -1}, "overlap must be greater than or equal to 0"},
		{"overlap too large", Request{Text: "x", ChunkSize: 10, Overlap: 10}, "overlap must be less than chunk_size"},
		{"unknown strategy", Request{Text: "x", ChunkSize: 10, Strategy: "vibes"}, "unknown strategy"},
		{"unknown encoding", Request{Text: "x", ChunkSize: 10, TokenEncoding: "gpt9"}, "unknown token_encoding"},
		{
			"separator without recursive",
			Request{Text: "x", ChunkSize: 10, Strategy: chunking.WordBoundary, SeparatorProfile: ProfileCJKThai},
			"separator_profile requires the recursive strategy",
		},
		{
			"semantic options without semantic",
			Request{Text: "x", ChunkSize: 10, Strategy: chunking.WordBoundary, Semantic: &SemanticOptions{}},
			"semantic options require the semantic strategy",
		},
		{
			"semantic min/max inverted",
			Request{Text: "x", ChunkSize: 10, Strategy: chunking.Semantic, Semantic: &SemanticOptions{MinChunkChars: 500, MaxChunkChars: 100}},
			"max_chunk_chars must be greater than or equal to",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := tc.req.Validate()
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !IsRequestValidationError(err) {
				t.Fatalf("expected request validation error, got %T", err)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.want)
			}
		})
	}
}

func TestStrategyWithDefault(t *testing.T) {
	t.Parallel()
	if got := withDefaultStrategy(""); got != chunking.SmartBoundary {
		t.Fatalf("empty strategy default = %q, want smart_boundary", got)
	}
	if got := withDefaultStrategy(chunking.HardCut); got != chunking.HardCut {
		t.Fatalf("explicit strategy changed = %q", got)
	}
}

func TestBatchValidateSharedParams(t *testing.T) {
	t.Parallel()
	if err := (BatchRequest{}).Validate(); err == nil {
		t.Fatal("empty texts must fail the batch")
	}
	bad := BatchRequest{Texts: []string{"a"}, ChunkSize: 0}
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "chunk_size") {
		t.Fatalf("shared params must be validated once, got %v", err)
	}
}

type stubEmbedder struct {
	dims int
	err  error
}

func (s *stubEmbedder) EmbedBatch(_ context.Context, _ []string) ([][]float32, error) {
	if s.err != nil {
		return nil, s.err
	}
	return makeVectors(s.dims), nil
}

func makeVectors(dims int) [][]float32 {
	return [][]float32{make([]float32, dims)}
}

func TestProcessRequestDefaultsToSmartBoundary(t *testing.T) {
	t.Parallel()
	resp, err := New().Process(context.Background(), Request{
		Text:      "One sentence. Two sentences, actually.",
		ChunkSize: 100,
	})
	if err != nil {
		t.Fatalf("ProcessRequest: %v", err)
	}
	if resp.Metadata.StrategyUsed != chunking.SmartBoundary {
		t.Fatalf("strategy = %q, want smart_boundary", resp.Metadata.StrategyUsed)
	}
	if resp.Metadata.TotalChunks != len(resp.Chunks) || len(resp.Chunks) == 0 {
		t.Fatalf("metadata total %d vs chunks %d", resp.Metadata.TotalChunks, len(resp.Chunks))
	}
	for i, c := range resp.Chunks {
		if c.ContentHash == "" || len(c.ContentHash) != 16 {
			t.Fatalf("chunk %d missing 16-hex content_hash: %q", i, c.ContentHash)
		}
		if c.Text == "" {
			t.Fatalf("chunk %d empty", i)
		}
	}
}

func TestProcessRequestTokenBasedCountsOriginalText(t *testing.T) {
	t.Parallel()
	text := "The quick brown fox. It jumps over the lazy dog repeatedly."
	resp, err := New().Process(context.Background(), Request{
		Text:          text,
		ChunkSize:     8,
		Overlap:       0,
		Strategy:      chunking.TokenBased,
		TokenEncoding: EncodingCL100K,
	})
	if err != nil {
		t.Fatalf("ProcessRequest: %v", err)
	}
	if resp.Metadata.TotalTokens <= 0 {
		t.Fatalf("token_based must report total tokens, got %d", resp.Metadata.TotalTokens)
	}
	if resp.Metadata.TokenEncoding != EncodingCL100K {
		t.Fatalf("encoding = %q, want cl100k_base", resp.Metadata.TokenEncoding)
	}
}

func TestProcessRequestMaxTokensCeilingSplits(t *testing.T) {
	t.Parallel()
	text := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 40)
	resp, err := New().Process(context.Background(), Request{
		Text:      text,
		ChunkSize: 10000,
		Strategy:  chunking.WordBoundary,
		MaxTokens: 16,
	})
	if err != nil {
		t.Fatalf("ProcessRequest: %v", err)
	}
	if len(resp.Chunks) < 2 {
		t.Fatalf("ceiling must split oversized chunks, got %d", len(resp.Chunks))
	}
	if resp.Metadata.MaxTokens != 16 {
		t.Fatalf("metadata.max_tokens = %d, want 16", resp.Metadata.MaxTokens)
	}
	for i, c := range resp.Chunks {
		if c.TokenCount <= 0 || c.TokenCount > 16 {
			t.Fatalf("chunk %d token_count %d outside (0, 16]", i, c.TokenCount)
		}
	}
}

func TestProcessRequestHeadingAwareCarriesPath(t *testing.T) {
	t.Parallel()
	text := "# Title\n\nIntro paragraph.\n\n## Section\n\nSection body text.\n"
	resp, err := New().Process(context.Background(), Request{
		Text:      text,
		ChunkSize: 100,
		Strategy:  chunking.HeadingAware,
	})
	if err != nil {
		t.Fatalf("ProcessRequest: %v", err)
	}
	sawPath := false
	for _, c := range resp.Chunks {
		if len(c.Path) > 0 {
			sawPath = true
		}
	}
	if !sawPath {
		t.Fatal("heading_aware must carry path breadcrumbs")
	}
}

func TestProcessRequestRecursiveCJKThaiProfile(t *testing.T) {
	t.Parallel()
	text := "第一段落です。続きます。\n\n第二段落です。ここで終わります。"
	resp, err := New().Process(context.Background(), Request{
		Text:             text,
		ChunkSize:        20,
		Overlap:          0,
		Strategy:         chunking.Recursive,
		SeparatorProfile: ProfileCJKThai,
	})
	if err != nil {
		t.Fatalf("ProcessRequest: %v", err)
	}
	if resp.Metadata.SeparatorProfile != ProfileCJKThai {
		t.Fatalf("separator_profile = %q, want cjk_thai", resp.Metadata.SeparatorProfile)
	}
	if len(resp.Chunks) == 0 {
		t.Fatal("expected chunks")
	}
}

func TestProcessRequestSemanticNeedsEmbedder(t *testing.T) {
	t.Parallel()
	_, err := New().Process(context.Background(), Request{
		Text:      "text",
		ChunkSize: 100,
		Strategy:  chunking.Semantic,
	})
	if !IsRequestValidationError(err) {
		t.Fatalf("expected validation error, got %v", err)
	}
}

func TestProcessRequestSemanticStrictOnEmbeddingFailure(t *testing.T) {
	t.Parallel()
	svc := New().WithEmbedder(&stubEmbedder{err: errors.New("provider down")})
	_, err := svc.Process(context.Background(), Request{
		// Three or more analyzable units, or the chunker falls back before
		// ever calling the embedder (see reliquary ChunkSemantic).
		Text:      "The first sentence establishes the topic of this paragraph. The second sentence adds supporting detail and vocabulary. A third sentence provides enough units for semantic boundary detection to engage embedding at all.",
		ChunkSize: 400,
		Strategy:  chunking.Semantic,
	})
	if err == nil {
		t.Fatal("semantic must fail when the embedder fails mid-run")
	}
	if IsRequestValidationError(err) {
		t.Fatalf("mid-run failure is not a client error: %v", err)
	}
	if !strings.Contains(err.Error(), "refusing to emit fallback chunks as semantic") {
		t.Fatalf("error must explain the strict fallback policy: %v", err)
	}
}

func TestProcessRequestIncludeEmbeddings(t *testing.T) {
	t.Parallel()
	if _, err := New().Process(context.Background(), Request{
		Text:              "hello world",
		ChunkSize:         100,
		IncludeEmbeddings: true,
	}); !IsRequestValidationError(err) {
		t.Fatalf("include_embeddings without embedder must be a validation error, got %v", err)
	}

	svc := New().WithEmbedder(&multiStubEmbedder{dims: 3}).WithEmbeddingModel("test-model")
	resp, err := svc.Process(context.Background(), Request{
		Text:              "hello world",
		ChunkSize:         100,
		IncludeEmbeddings: true,
	})
	if err != nil {
		t.Fatalf("ProcessRequest: %v", err)
	}
	if len(resp.Chunks) == 0 || len(resp.Chunks[0].Embedding) != 3 {
		t.Fatalf("expected 3-dim vectors on every chunk")
	}
	if resp.Metadata.EmbeddingModel != "test-model" || resp.Metadata.EmbeddingDimensions != 3 {
		t.Fatalf("embedding metadata = %q/%d", resp.Metadata.EmbeddingModel, resp.Metadata.EmbeddingDimensions)
	}
}

// multiStubEmbedder returns one vector per input so include_embeddings can
// attach in order.
type multiStubEmbedder struct {
	dims int
}

func (m *multiStubEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = make([]float32, m.dims)
		for j := range out[i] {
			out[i][j] = float32(i)
		}
	}
	return out, nil
}

func TestProcessBatchRequestPerItemResults(t *testing.T) {
	t.Parallel()
	svc := New()
	resp, err := svc.ProcessBatch(context.Background(), BatchRequest{
		Texts:     []string{"A valid text.", ""},
		ChunkSize: 100,
	})
	if err != nil {
		t.Fatalf("ProcessBatchRequest: %v", err)
	}
	if len(resp.Results) != 2 {
		t.Fatalf("got %d results, want 2", len(resp.Results))
	}
	if resp.Results[0].Response == nil || resp.Results[0].Error != "" {
		t.Fatalf("item 0 must succeed: %+v", resp.Results[0])
	}
	if resp.Results[1].Response != nil || resp.Results[1].Error == "" {
		t.Fatalf("item 1 (empty text) must be a per-item error: %+v", resp.Results[1])
	}
}

// fixedTokenCounter reports a rune-halved token estimate with a fixed cap,
// keeping preflight tests deterministic without a tiktoken download.
type fixedTokenCounter struct{ limit int }

func (f fixedTokenCounter) CountTokens(text string) int { return len([]rune(text)) / 2 }
func (f fixedTokenCounter) MaxTokens() int              { return f.limit }

func TestPreflightCounterSplitsBoundaryChunks(t *testing.T) {
	t.Parallel()
	// Each sentence is ~6 "tokens" under the test counter; a budget of 8
	// keeps sentences together but splits the pair, and without the counter
	// one chunk holds everything.
	text := "One sentence here. Two sentences now."
	without := New().WithEmbedder(nil)
	req := Request{Text: text, ChunkSize: 200, Overlap: 0, Strategy: chunking.SentenceBoundary}
	base, _, err := without.Chunks(context.Background(), req)
	if err != nil {
		t.Fatalf("no-preflight run failed: %v", err)
	}
	req.PreflightCounter = fixedTokenCounter{limit: 8}
	limited, _, err := without.Chunks(context.Background(), req)
	if err != nil {
		t.Fatalf("preflight run failed: %v", err)
	}
	if len(limited) <= len(base) {
		t.Fatalf("preflight counter did not split further: %d chunks vs %d base", len(limited), len(base))
	}
}
