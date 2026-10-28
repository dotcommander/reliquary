package reliquary_test

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/chunking"
	"github.com/dotcommander/reliquary/chunkservice"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/index"
)

func semanticSentences() []string {
	return []string{
		"Orchard apples grow on carefully tended trees.",
		"Orchard pears ripen through the sunny summer.",
		"Ocean waves carry water toward distant shores.",
		"Ocean tides move with the pull of the moon.",
	}
}

func semanticPolicy() reliquary.ChunkingOptions {
	return reliquary.ChunkingOptions{
		Strategy: chunking.Semantic, ChunkSize: 2000,
		Semantic: &chunkservice.SemanticOptions{MinChunkChars: 1, MaxChunkChars: 2000, BreakSensitivity: 1},
	}
}

type semanticTestEmbedder struct {
	calls     []embedding.Request
	identical bool
	hook      func(context.Context, int) error
}

func (e *semanticTestEmbedder) Embed(ctx context.Context, req embedding.Request) (embedding.Result, error) {
	req.Inputs = slices.Clone(req.Inputs)
	e.calls = append(e.calls, req)
	if e.hook != nil {
		if err := e.hook(ctx, len(e.calls)); err != nil {
			return embedding.Result{}, err
		}
	}
	result := embedding.Result{Vectors: make([]embedding.Vector, len(req.Inputs))}
	for i, text := range req.Inputs {
		result.Vectors[i] = embedding.Vector{1, 0}
		if !e.identical && strings.HasPrefix(text, "Ocean") {
			result.Vectors[i] = embedding.Vector{0, 1}
		}
	}
	return result, nil
}

type semanticTestIndex struct {
	index.Index
	replaceCalls int
}

func (i *semanticTestIndex) ReplaceDocuments(ctx context.Context, docs []index.DocumentReplacement) error {
	i.replaceCalls++
	return i.Index.ReplaceDocuments(ctx, docs)
}

func newSemanticApp(t *testing.T, e *semanticTestEmbedder, idx index.Index, identity string) *reliquary.App {
	t.Helper()
	app, err := reliquary.New(reliquary.WithEmbedder(e), reliquary.WithIndex(idx), reliquary.WithIndexIdentity(identity), reliquary.WithChunking(semanticPolicy()))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return app
}

func TestAppSemanticEmbeddingDrivesBoundaries(t *testing.T) {
	t.Parallel()
	sentences := semanticSentences()
	text := strings.Join(sentences, " ")
	for _, identical := range []bool{false, true} {
		name := "topic vectors"
		if identical {
			name = "identical vectors"
		}
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			e := &semanticTestEmbedder{identical: identical}
			idx := reliquary.NewMemoryIndex()
			app := newSemanticApp(t, e, idx, "semantic-test")
			want := []string{strings.Join(sentences[:2], " "), strings.Join(sentences[2:], " ")}
			if identical {
				want = []string{text}
			}
			n, err := app.Ingest(context.Background(), document.Document{ID: "topics", Text: text})
			if err != nil || n != len(want) {
				t.Fatalf("Ingest = %d, %v, want %d chunks", n, err, len(want))
			}
			if len(e.calls) != 2 {
				t.Fatalf("embedding calls = %d, want boundary and final calls", len(e.calls))
			}
			if !reflect.DeepEqual(e.calls[0].Inputs, sentences) {
				t.Fatalf("boundary inputs = %q, want %q", e.calls[0].Inputs, sentences)
			}
			if !reflect.DeepEqual(e.calls[1].Inputs, want) {
				t.Fatalf("final inputs = %q, want %q", e.calls[1].Inputs, want)
			}
			for _, req := range e.calls {
				if req.Model != (embedding.ModelRef{}) || req.Kind != embedding.KindDocument {
					t.Fatalf("embedding model/kind = %#v/%v", req.Model, req.Kind)
				}
			}
			hits, err := idx.Search(context.Background(), reliquary.IndexQuery{Identity: "semantic-test"})
			if err != nil || len(hits) != len(want) {
				t.Fatalf("indexed chunks = %d, %v", len(hits), err)
			}
			got := make([]string, len(hits))
			for i, hit := range hits {
				got[i] = hit.Content
			}
			slices.Sort(got)
			slices.Sort(want)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("indexed content = %q, want %q", got, want)
			}
		})
	}
}

func TestAppSemanticFailuresPreserveBatchRevisions(t *testing.T) {
	t.Parallel()
	providerErr := errors.New("provider unavailable")
	cases := []struct {
		name          string
		failCall      int
		failure       error
		cancelBefore  bool
		cancelSuccess bool
		docID         string
	}{
		{name: "boundary provider failure", failCall: 1, failure: providerErr, docID: "first"},
		{name: "final embedding failure", failCall: 3, failure: providerErr},
		{name: "later document boundary failure", failCall: 2, failure: providerErr, docID: "second"},
		{name: "boundary cancellation", failCall: 1, failure: context.Canceled, docID: "first"},
		{name: "already canceled", cancelBefore: true, failure: context.Canceled},
		{name: "final embedder cancels but succeeds", failCall: 3, cancelSuccess: true, failure: context.Canceled},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			idx := &semanticTestIndex{Index: reliquary.NewMemoryIndex()}
			e := &semanticTestEmbedder{}
			app := newSemanticApp(t, e, idx, "semantic-test")
			originals := []document.Document{{ID: "first", Text: "original first revision"}, {ID: "second", Text: "original second revision"}}
			if _, err := app.Ingest(context.Background(), originals...); err != nil {
				t.Fatalf("seed: %v", err)
			}
			e.calls = nil
			idx.replaceCalls = 0
			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)
			if tc.cancelBefore {
				cancel()
			}
			e.hook = func(_ context.Context, call int) error {
				if call != tc.failCall {
					return nil
				}
				if tc.cancelSuccess {
					cancel()
					return nil
				}
				return tc.failure
			}
			text := strings.Join(semanticSentences(), " ")
			n, err := app.Ingest(ctx, document.Document{ID: "first", Text: text}, document.Document{ID: "second", Text: text})
			if n != 0 || !errors.Is(err, tc.failure) {
				t.Fatalf("Ingest = %d, %v, want 0 and %v", n, err, tc.failure)
			}
			if tc.docID != "" && (!strings.Contains(err.Error(), tc.docID) || strings.Contains(err.Error(), semanticSentences()[0])) {
				t.Fatalf("chunking error must identify document without source text: %v", err)
			}
			if idx.replaceCalls != 0 {
				t.Fatalf("ReplaceDocuments calls = %d, want 0", idx.replaceCalls)
			}
			if tc.cancelBefore && len(e.calls) != 0 {
				t.Fatalf("canceled ingest embedded %d batches", len(e.calls))
			}
			if tc.failCall > 0 && len(e.calls) != tc.failCall {
				t.Fatalf("embedding calls = %d, want %d", len(e.calls), tc.failCall)
			}
			assertSemanticOriginals(t, idx, "semantic-test", originals)
		})
	}
}

func assertSemanticOriginals(t *testing.T, idx index.Index, identity string, docs []document.Document) {
	t.Helper()
	hits, err := idx.Search(context.Background(), reliquary.IndexQuery{Identity: identity})
	if err != nil {
		t.Fatalf("read retained index: %v", err)
	}
	got := make(map[string]string)
	for _, hit := range hits {
		got[hit.DocumentID] = hit.Content
	}
	if len(hits) != len(docs) {
		t.Fatalf("retained chunks = %d, want %d", len(hits), len(docs))
	}
	for _, doc := range docs {
		if got[doc.ID] != doc.Text {
			t.Fatalf("retained %q = %q, want %q", doc.ID, got[doc.ID], doc.Text)
		}
	}
}

func TestAppSemanticRejectsIdentityMismatch(t *testing.T) {
	t.Parallel()
	idx := reliquary.NewMemoryIndex()
	original := document.Document{ID: "doc", Text: "original indexed revision"}
	first := newSemanticApp(t, &semanticTestEmbedder{}, idx, "semantic-policy-v1")
	if _, err := first.Ingest(context.Background(), original); err != nil {
		t.Fatalf("seed: %v", err)
	}
	second := newSemanticApp(t, &semanticTestEmbedder{}, idx, "semantic-policy-v2")
	if _, err := second.Ingest(context.Background(), document.Document{ID: "doc", Text: strings.Join(semanticSentences(), " ")}); !errors.Is(err, reliquary.ErrIdentityMismatch) {
		t.Fatalf("Ingest error = %v, want identity mismatch", err)
	}
	if _, err := second.Search(context.Background(), "orchard"); !errors.Is(err, reliquary.ErrIdentityMismatch) {
		t.Fatalf("Search error = %v, want identity mismatch", err)
	}
	assertSemanticOriginals(t, idx, "semantic-policy-v1", []document.Document{original})
}
