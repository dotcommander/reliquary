package retrieval

import (
	"errors"
	"testing"

	"github.com/dotcommander/reliquary/chunking"
	"github.com/dotcommander/reliquary/document"
)

func TestBestChunk(t *testing.T) {
	t.Parallel()

	t.Run("preset similarities returns highest", func(t *testing.T) {
		t.Parallel()
		chunks := []ChunkResult{
			{Text: "low", Similarity: 0.3},
			{Text: "high", Similarity: 0.9},
			{Text: "mid", Similarity: 0.5},
		}
		got := BestChunk(nil, chunks)
		if got.Text != "high" {
			t.Errorf("expected text %q, got %q", "high", got.Text)
		}
		if got.Similarity != 0.9 {
			t.Errorf("expected similarity 0.9, got %v", got.Similarity)
		}
	})

	t.Run("zero similarity with embeddings uses Cosine64", func(t *testing.T) {
		t.Parallel()
		// queryEmbedding points toward chunk b.
		query := []float64{0.0, 1.0}
		chunks := []ChunkResult{
			{Text: "a", Embedding: []float64{1.0, 0.0}}, // cosine ~ 0
			{Text: "b", Embedding: []float64{0.0, 1.0}}, // cosine = 1
		}
		got := BestChunk(query, chunks)
		if got.Text != "b" {
			t.Errorf("expected text %q, got %q", "b", got.Text)
		}
		if got.Similarity <= 0.5 {
			t.Errorf("expected similarity > 0.5 for aligned vectors, got %v", got.Similarity)
		}
	})

	t.Run("empty chunks returns zero-value ChunkResult", func(t *testing.T) {
		t.Parallel()
		got := BestChunk(nil, nil)
		// best.Similarity == -2 path → returns ChunkResult{}
		if got.Text != "" || got.Embedding != nil || got.Similarity != 0 {
			t.Errorf("expected zero ChunkResult, got %+v", got)
		}
	})

	t.Run("nil queryEmbedding with zero-similarity chunks no panic", func(t *testing.T) {
		t.Parallel()
		chunks := []ChunkResult{
			{Text: "x", Embedding: []float64{1.0, 0.0}},
			{Text: "y", Embedding: []float64{0.0, 1.0}},
		}
		// score stays 0 for all (no queryEmbedding) — first chunk wins on first > -2
		got := BestChunk(nil, chunks)
		if got.Similarity != 0 {
			t.Errorf("expected similarity 0 when no queryEmbedding, got %v", got.Similarity)
		}
		if got.Text == "" {
			t.Errorf("expected a non-empty text result when chunks exist, got empty")
		}
	})
}

func TestTextChunks(t *testing.T) {
	t.Parallel()

	t.Run("plain text returns non-empty chunks without empty entries", func(t *testing.T) {
		t.Parallel()
		content := "This is a plain text document with enough content to be chunked into pieces. " +
			"It contains multiple sentences to ensure the chunker has material to work with. " +
			"Each sentence adds a bit more content to the overall document."
		chunks := TextChunks(content, 200, 20)
		if len(chunks) == 0 {
			t.Error("expected at least one chunk for non-empty plain text")
		}
		for i, c := range chunks {
			if c == "" {
				t.Errorf("chunk[%d] is empty string", i)
			}
		}
	})

	t.Run("markdown content returns chunks without empty entries", func(t *testing.T) {
		t.Parallel()
		content := "# Introduction\n\nThis is the intro section.\n\n## Details\n\nMore information here.\n"
		chunks := TextChunks(content, 200, 20)
		if len(chunks) == 0 {
			t.Error("expected at least one chunk for markdown content")
		}
		for i, c := range chunks {
			if c == "" {
				t.Errorf("chunk[%d] is empty string", i)
			}
		}
	})

	t.Run("empty content returns empty or nil slice without panic", func(t *testing.T) {
		t.Parallel()
		chunks := TextChunks("", 200, 20)
		// Must not panic; result may be nil or empty slice — both are acceptable.
		for i, c := range chunks {
			if c == "" {
				t.Errorf("unexpected empty-string chunk at index %d", i)
			}
		}
	})
}

func TestResultsFromDocuments(t *testing.T) {
	t.Parallel()

	results, err := ResultsFromDocuments([]document.Document{
		{ID: "doc", Title: "doc.md", Text: "Alpha sentence. Beta sentence."},
	}, chunking.SmartBoundary, 80, 0)
	if err != nil {
		t.Fatalf("ResultsFromDocuments() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("ResultsFromDocuments() len = %d, want 1", len(results))
	}
	if results[0].ID != "doc#0" || results[0].Filename != "doc.md" || results[0].Content == "" {
		t.Fatalf("unexpected result: %+v", results[0])
	}
}

func TestResultsFromDocumentsRejectsInvalidIDs(t *testing.T) {
	t.Parallel()
	if _, err := ResultsFromDocuments([]document.Document{{ID: " ", Text: "content"}}, chunking.SmartBoundary, 80, 0); !errors.Is(err, ErrInvalidDocumentID) {
		t.Fatalf("blank ID error = %v, want ErrInvalidDocumentID", err)
	}
	if _, err := ResultsFromDocuments([]document.Document{{ID: "same", Text: "one"}, {ID: "same", Text: "two"}}, chunking.SmartBoundary, 80, 0); !errors.Is(err, ErrDuplicateDocumentID) {
		t.Fatalf("duplicate ID error = %v, want ErrDuplicateDocumentID", err)
	}
}

func TestFallbackChunker(t *testing.T) {
	t.Parallel()

	fb := fallbackChunker{}
	if fb.Strategy() != chunking.SmartBoundary {
		t.Fatalf("Strategy() = %v, want SmartBoundary", fb.Strategy())
	}

	chunks := fb.Chunk("hello world", 0, -1) // defaults size=2000, overlap=0
	if len(chunks) != 1 || chunks[0].Text != "hello world" {
		t.Fatalf("fb.Chunk default size/overlap = %v", chunks)
	}

	chunks = fb.Chunk("abcdefghij", 4, 2)
	if len(chunks) == 0 {
		t.Fatal("expected chunks")
	}
}
