package retrieval

import (
	"fmt"
	"strings"

	"github.com/dotcommander/reliquary/chunking"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/vector"
)

type ChunkResult struct {
	Text       string
	Embedding  []float64
	Similarity float64
}

// TextChunks splits text into chunks and filters empties.
func TextChunks(content string, size int, overlap int) []string {
	chunker, err := chunking.NewChunker(chunking.SmartBoundary)
	if err != nil {
		chunker = &fallbackChunker{}
	}
	if strings.Contains(content, "\n#") || strings.HasPrefix(strings.TrimSpace(content), "#") {
		if md, err := chunking.NewChunker(chunking.MarkdownAware); err == nil {
			chunker = md
		}
	}
	chunks := chunker.Chunk(content, size, overlap)
	out := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		if text := strings.TrimSpace(chunk.Text); text != "" {
			out = append(out, text)
		}
	}
	return out
}

// ResultsFromDocuments chunks documents into retrieval results using stable
// documentID#chunkID identifiers. It is the small adapter most callers otherwise
// hand-write before embedding and reranking.
func ResultsFromDocuments(docs []document.Document, strategy chunking.Strategy, size, overlap int) ([]*Result, error) {
	chunker, err := chunking.NewChunker(strategy)
	if err != nil {
		return nil, err
	}
	results := make([]*Result, 0, len(docs))
	for _, doc := range docs {
		for _, chunk := range chunker.Chunk(document.NormalizeText(doc.Text), size, overlap) {
			metadata := make(map[string]any, len(doc.Metadata))
			for key, value := range doc.Metadata {
				metadata[key] = value
			}
			results = append(results, &Result{
				ID:         fmt.Sprintf("%s#%d", doc.ID, chunk.ID),
				DocumentID: doc.ID,
				Content:    chunk.Text,
				Filename:   doc.Title,
				Metadata:   metadata,
			})
		}
	}
	return results, nil
}

// BestChunk returns the chunk with highest cosine similarity.
func BestChunk(queryEmbedding []float64, chunks []ChunkResult) ChunkResult {
	best := ChunkResult{Similarity: -2}
	for _, chunk := range chunks {
		score := chunk.Similarity
		if score == 0 && len(queryEmbedding) > 0 && len(chunk.Embedding) > 0 {
			score = vectors.Cosine64(queryEmbedding, chunk.Embedding)
		}
		if score > best.Similarity {
			best = chunk
			best.Similarity = score
		}
	}
	if best.Similarity == -2 {
		return ChunkResult{}
	}
	return best
}

// fallbackChunker matches chunking.chunker behavior for fallback paths.
type fallbackChunker struct{}

func (fallbackChunker) Chunk(text string, size int, overlap int) []chunking.Chunk {
	if size <= 0 {
		size = 2000
	}
	if overlap < 0 {
		overlap = 0
	}
	step := max(size-overlap, 1)
	chunks := make([]chunking.Chunk, 0, (len(text)/step)+1)
	for i := 0; i < len(text); i += step {
		end := min(i+size, len(text))
		if end <= i {
			break
		}
		chunks = append(chunks, chunking.Chunk{ID: len(chunks), Text: text[i:end]})
		if end >= len(text) {
			break
		}
	}
	return chunks
}

func (fallbackChunker) Strategy() chunking.Strategy { return chunking.SmartBoundary }
