package retrieval

import (
	"errors"
	"fmt"
	"strings"

	"github.com/dotcommander/reliquary/chunking"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/textutil"
	"github.com/dotcommander/reliquary/vector"
)

// ErrInvalidDocumentID reports a blank document identifier.
var ErrInvalidDocumentID = errors.New("retrieval: document ID must not be blank")

// ErrDuplicateDocumentID reports duplicate identifiers in one document batch.
var ErrDuplicateDocumentID = errors.New("retrieval: duplicate document ID")

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
	seen := make(map[string]struct{}, len(docs))
	for _, doc := range docs {
		if strings.TrimSpace(doc.ID) == "" {
			return nil, ErrInvalidDocumentID
		}
		if _, exists := seen[doc.ID]; exists {
			return nil, fmt.Errorf("%w: %q", ErrDuplicateDocumentID, doc.ID)
		}
		seen[doc.ID] = struct{}{}
	}
	chunker, err := chunking.NewChunker(strategy)
	if err != nil {
		return nil, err
	}
	results := make([]*Result, 0, len(docs))
	for _, doc := range docs {
		normalized := document.NormalizeText(doc.Text)
		chunks := chunker.Chunk(normalized, size, overlap)
		cursor := 0
		for _, chunk := range chunks {
			metadata := make(map[string]any, len(doc.Metadata))
			for key, value := range doc.Metadata {
				if key == ContextStartLineKey || key == ContextEndLineKey {
					continue
				}
				metadata[key] = value
			}
			span, ok := resolveContextChunkSpan(normalized, chunk, cursor)
			if ok {
				startLine, endLine := chunking.LineRangeForSpan(normalized, span)
				metadata[ContextStartLineKey] = startLine
				metadata[ContextEndLineKey] = endLine
				cursor = chunking.NextChunkCursor(span)
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

func resolveContextChunkSpan(source string, chunk chunking.Chunk, cursor int) (chunking.ChunkSpan, bool) {
	span, ok := chunking.ResolveChunkSpan(source, chunk, cursor)
	if ok && span.Start >= cursor {
		return span, true
	}

	start, end, ok := textutil.FragmentRange(source, chunk.Text, cursor, textutil.NormalizedEarly)
	if !ok || start < cursor {
		return chunking.ChunkSpan{}, false
	}
	return chunking.ChunkSpan{Start: start, End: end}, true
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
