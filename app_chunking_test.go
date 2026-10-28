package reliquary_test

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/chunking"
	"github.com/dotcommander/reliquary/chunkservice"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/embed"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/retrieval"
)

func newPolicyApp(t *testing.T, policy reliquary.ChunkingOptions) (*reliquary.App, reliquary.Index) {
	t.Helper()
	idx := reliquary.NewMemoryIndex()
	app, err := reliquary.New(reliquary.WithEmbedder(embed.NewHashing(16)), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("policy-v1"), reliquary.WithChunking(policy))
	if err != nil {
		t.Fatal(err)
	}
	return app, idx
}

func policyRows(t *testing.T, idx reliquary.Index) []*retrieval.Result {
	t.Helper()
	rows, err := idx.Search(context.Background(), reliquary.IndexQuery{Identity: "policy-v1"})
	if err != nil {
		t.Fatal(err)
	}
	return rows
}

type unusedPolicyEmbedder struct{}

func (unusedPolicyEmbedder) Embed(context.Context, embedding.Request) (embedding.Result, error) {
	panic("construction called embedder")
}

func TestChunkingConfiguration(t *testing.T) {
	t.Parallel()
	invalid := []reliquary.ChunkingOptions{
		{}, {ChunkSize: -1}, {ChunkSize: 8, Overlap: -1}, {ChunkSize: 8, Overlap: 8},
		{ChunkSize: 8, Strategy: "unknown"}, {ChunkSize: 8, TokenEncoding: "unknown"},
		{ChunkSize: 8, MaxTokens: -1}, {ChunkSize: 8, Semantic: &chunkservice.SemanticOptions{}},
		{ChunkSize: 8, Strategy: chunking.Semantic, Semantic: &chunkservice.SemanticOptions{MinChunkChars: -1}},
		{ChunkSize: 8, SeparatorProfile: chunkservice.ProfileCJKThai},
		{ChunkSize: 8, Strategy: chunking.Recursive, CustomSeparators: []string{}},
		{ChunkSize: 8, Strategy: chunking.Recursive, CustomSeparators: []string{"", "|"}},
		{ChunkSize: 8, Strategy: chunking.Recursive, CustomSeparators: []string{"|"}, SeparatorProfile: chunkservice.ProfileDefaultText},
	}
	idx := &recordingIndex{}
	base := []reliquary.Option{reliquary.WithEmbedder(unusedPolicyEmbedder{}), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("policy-v1")}
	for i, policy := range invalid {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			t.Parallel()
			if _, err := reliquary.New(append(base[:len(base):len(base)], reliquary.WithChunking(policy))...); err == nil {
				t.Fatalf("New accepted %+v", policy)
			}
		})
	}
	legacy := reliquary.WithChunker(chunking.SmartBoundary, 8, 0)
	configured := reliquary.WithChunking(reliquary.ChunkingOptions{ChunkSize: 8})
	for _, opts := range [][]reliquary.Option{{legacy, configured}, {configured, legacy}} {
		if _, err := reliquary.New(append(base[:len(base):len(base)], opts...)...); err == nil {
			t.Fatal("accepted conflicting options")
		}
	}
	if _, err := reliquary.New(append(base[:len(base):len(base)], reliquary.WithChunking(reliquary.ChunkingOptions{}), configured)...); err != nil {
		t.Fatal(err)
	}
}

func TestChunkingCopiesPolicy(t *testing.T) {
	t.Parallel()
	separators := []string{"|", ";"}
	app, _ := newPolicyApp(t, reliquary.ChunkingOptions{Strategy: chunking.Recursive, ChunkSize: 7, CustomSeparators: separators})
	separators[0] = ""
	if n, err := app.Ingest(context.Background(), document.Document{ID: "doc", Text: "aaa|bbb;ccc"}); err != nil || n != 2 {
		t.Fatalf("copied separators: %d, %v", n, err)
	}
	semantic := &chunkservice.SemanticOptions{MinChunkChars: 1, MaxChunkChars: 100}
	app, _ = newPolicyApp(t, reliquary.ChunkingOptions{Strategy: chunking.Semantic, ChunkSize: 100, Semantic: semantic})
	semantic.MinChunkChars = -1
	if _, err := app.Ingest(context.Background(), document.Document{ID: "doc", Text: "First sentence. Second sentence."}); err != nil {
		t.Fatalf("copied semantic options: %v", err)
	}
}

func TestChunkingDefaultMappingAndReplacement(t *testing.T) {
	t.Parallel()
	app, idx := newPolicyApp(t, reliquary.ChunkingOptions{ChunkSize: 20})
	doc := document.Document{ID: "doc", Title: "source.md", Text: "\ufeffAlpha beta gamma.\r\nDelta epsilon zeta.\r\nFinal sentence.", Metadata: document.Metadata{"tag": "kept", retrieval.ContextStartLineKey: "999"}}
	expected, err := retrieval.ResultsFromDocuments([]document.Document{doc}, chunking.SmartBoundary, 20, 0)
	if err != nil {
		t.Fatal(err)
	}
	if n, err := app.Ingest(context.Background(), doc); err != nil || n != len(expected) {
		t.Fatalf("Ingest = %d, %v", n, err)
	}
	rows := policyRows(t, idx)
	byID := make(map[string]*retrieval.Result)
	for _, row := range rows {
		byID[row.ID] = row
	}
	for _, want := range expected {
		got := byID[want.ID]
		if got == nil || got.Content != want.Content || got.DocumentID != want.DocumentID || got.Filename != want.Filename || !reflect.DeepEqual(got.Metadata, want.Metadata) {
			t.Fatalf("mapping mismatch: got %+v want %+v", got, want)
		}
	}
	hits, err := app.Search(context.Background(), "Alpha", reliquary.TopK(1))
	if err != nil || len(hits) != 1 {
		t.Fatalf("search: %v %v", hits, err)
	}
	if n, err := app.Ingest(context.Background(), document.Document{ID: "doc", Text: "short revision"}); err != nil || n != 1 {
		t.Fatalf("replacement: %d %v", n, err)
	}
	if rows := policyRows(t, idx); len(rows) != 1 || rows[0].Content != "short revision" {
		t.Fatalf("obsolete chunks remain: %v", rows)
	}
	if n, err := app.Ingest(context.Background(), document.Document{ID: "doc", Text: "\ufeff \r\n\t"}); err != nil || n != 0 {
		t.Fatalf("empty replacement: %d %v", n, err)
	}
	if rows := policyRows(t, idx); len(rows) != 0 {
		t.Fatalf("empty replacement retained %d rows", len(rows))
	}
}

func TestChunkingIndexedTokenCeilingsAndLines(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, text string
		strategy   chunking.Strategy
		encoding   chunkservice.TokenEncoding
		limit      int
		separators []string
	}{
		{"exact", "hello world", chunking.WordBoundary, chunkservice.EncodingCL100K, 2, nil},
		{"unbroken", strings.Repeat("abcdefghij", 40), chunking.WordBoundary, chunkservice.EncodingO200K, 8, nil},
		{"unicode lines", strings.Repeat("你好世界🌍", 8) + "\n" + strings.Repeat("ภาษาไทย", 8), chunking.Recursive, chunkservice.EncodingCL100K, 8, []string{"\n"}},
		{"semantic", strings.Repeat("First sentence here. Another sentence now. ", 8), chunking.Semantic, chunkservice.EncodingCL100K, 4, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			app, idx := newPolicyApp(t, reliquary.ChunkingOptions{Strategy: tc.strategy, ChunkSize: 10000, TokenEncoding: tc.encoding, MaxTokens: tc.limit, CustomSeparators: tc.separators})
			if _, err := app.Ingest(context.Background(), document.Document{ID: "doc", Text: tc.text}); err != nil {
				t.Fatal(err)
			}
			rows := policyRows(t, idx)
			if len(rows) == 0 {
				t.Fatal("no indexed text")
			}
			if tc.name == "exact" && (len(rows) != 1 || rows[0].Content != tc.text) {
				t.Fatal("exact-limit text changed")
			}
			lines := strings.Split(document.NormalizeText(tc.text), "\n")
			for _, row := range rows {
				count, err := chunking.CountTokens(row.Content, string(tc.encoding))
				if err != nil || count > tc.limit || !utf8.ValidString(row.Content) {
					t.Fatalf("indexed ceiling: %q count=%d err=%v", row.Content, count, err)
				}
				start, okStart := row.Metadata[retrieval.ContextStartLineKey].(int)
				end, okEnd := row.Metadata[retrieval.ContextEndLineKey].(int)
				if !okStart || !okEnd || start < 1 || end < start || end > len(lines) {
					t.Fatalf("invalid lines: %v", row.Metadata)
				}
				if tc.name == "unicode lines" && !strings.Contains(strings.Join(lines[start-1:end], "\n"), row.Content) {
					t.Fatalf("wrong source lines: %v %q", row.Metadata, row.Content)
				}
			}
		})
	}
}

type policyWordCounter struct{}

func (policyWordCounter) CountTokens(s string) int { return len(strings.Fields(s)) }
func (policyWordCounter) MaxTokens() int           { return 3 }

func TestChunkingPreflightAndCeiling(t *testing.T) {
	t.Parallel()
	text := "one two three four five six seven eight nine ten"
	uncapped, _ := newPolicyApp(t, reliquary.ChunkingOptions{Strategy: chunking.WordBoundary, ChunkSize: 1000})
	if n, err := uncapped.Ingest(context.Background(), document.Document{ID: "doc", Text: text}); err != nil || n != 1 {
		t.Fatalf("nil preflight/zero ceiling: %d %v", n, err)
	}
	for _, limit := range []int{0, 2} {
		app, idx := newPolicyApp(t, reliquary.ChunkingOptions{Strategy: chunking.WordBoundary, ChunkSize: 1000, PreflightCounter: policyWordCounter{}, MaxTokens: limit})
		if n, err := app.Ingest(context.Background(), document.Document{ID: "doc", Text: text}); err != nil || n < 2 {
			t.Fatalf("preflight: %d %v", n, err)
		}
		for _, row := range policyRows(t, idx) {
			if len(strings.Fields(row.Content)) > 3 {
				t.Fatal("preflight budget exceeded")
			}
			if limit > 0 {
				n, err := chunking.CountTokens(row.Content, string(chunkservice.EncodingCL100K))
				if err != nil || n > limit {
					t.Fatalf("final ceiling: %d %v", n, err)
				}
			}
		}
	}
}

func TestChunkingEnforcementFailurePreservesBatch(t *testing.T) {
	t.Parallel()
	app, idx := newPolicyApp(t, reliquary.ChunkingOptions{ChunkSize: 100, MaxTokens: 1})
	docs := []document.Document{{ID: "first", Text: "hello"}, {ID: "second", Text: "world"}}
	if _, err := app.Ingest(context.Background(), docs...); err != nil {
		t.Fatal(err)
	}
	n, err := app.Ingest(context.Background(), document.Document{ID: "first", Text: "new"}, document.Document{ID: "second", Text: "🌍"})
	if err == nil || n != 0 || !strings.Contains(err.Error(), "second") {
		t.Fatalf("enforcement failure: %d %v", n, err)
	}
	rows := policyRows(t, idx)
	if len(rows) != 2 {
		t.Fatalf("retained rows = %d", len(rows))
	}
	for _, row := range rows {
		want := "hello"
		if row.DocumentID == "second" {
			want = "world"
		}
		if row.Content != want {
			t.Fatalf("partial replacement: %+v", row)
		}
	}
}
