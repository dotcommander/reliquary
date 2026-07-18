package chunking

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

// benchmarkProse generates repeated paragraph prose with many sentence boundaries.
func benchmarkProse(sentences int) string {
	var buf strings.Builder
	for i := 0; i < sentences; i++ {
		fmt.Fprintf(&buf, "This is benchmark sentence number %d with some additional words for length. ", i)
	}
	return buf.String()
}

// benchmarkMarkdown generates Markdown with repeated sections, lists, fenced code, and a table.
func benchmarkMarkdown(sections int) string {
	var buf strings.Builder
	for i := 0; i < sections; i++ {
		fmt.Fprintf(&buf, "## Section %d\n\n", i)
		fmt.Fprintf(&buf, "This is the content for section %d with some explanatory text.\n\n", i)
		fmt.Fprintf(&buf, "```go\nfunc handler%d() {\n\treturn\n}\n```\n\n", i)
		fmt.Fprintf(&buf, "- item one for section %d\n- item two for section %d\n\n", i, i)
		fmt.Fprintf(&buf, "| Col A | Col B |\n|---|---|\n| %d | %d |\n\n", i, i+1)
	}
	return buf.String()
}

// benchmarkNestedHeadings generates H1/H2/H3 nested sections large enough to trigger
// recursive heading splitting.
func benchmarkNestedHeadings(sections int) string {
	var buf strings.Builder
	for i := 0; i < sections; i++ {
		fmt.Fprintf(&buf, "# Chapter %d\n\n", i)
		fmt.Fprintf(&buf, "Introduction to chapter %d with enough content to be meaningful.\n\n", i)
		for j := 0; j < 3; j++ {
			fmt.Fprintf(&buf, "## Section %d.%d\n\n", i, j)
			fmt.Fprintf(&buf, "Content for section %d.%d with additional detail.\n\n", i, j)
			for k := 0; k < 2; k++ {
				fmt.Fprintf(&buf, "### Subsection %d.%d.%d\n\n", i, j, k)
				fmt.Fprintf(&buf, "Detailed content for subsection %d.%d.%d.\n\n", i, j, k)
			}
		}
	}
	return buf.String()
}

// benchmarkSemanticText generates alternating topic groups for semantic chunking.
func benchmarkSemanticText(groups int) string {
	topics := []string{
		"The architecture of neural networks has evolved significantly over the past decade. Deep learning models now power everything from image recognition to natural language processing.",
		"Database optimization requires careful indexing strategies and query planning. Proper normalization reduces redundancy while maintaining data integrity across tables.",
	}
	var buf strings.Builder
	for i := 0; i < groups; i++ {
		buf.WriteString(topics[i%len(topics)])
		buf.WriteString(" ")
	}
	return buf.String()
}

// benchmarkEmbedder is a deterministic mock embedder returning fixed-length vectors.
type benchmarkEmbedder struct {
	dim int
}

func (e benchmarkEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	vecs := make([][]float32, len(texts))
	for i := range texts {
		v := make([]float32, e.dim)
		for j := range v {
			v[j] = float32(i%10+1) / 10.0
		}
		vecs[i] = v
	}
	return vecs, nil
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

func BenchmarkSmartBoundary(b *testing.B) {
	text := benchmarkProse(200)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunker, _ := NewChunker(SmartBoundary)
		chunks := chunker.Chunk(text, 1200, 100)
		if len(chunks) == 0 {
			b.Fatal("expected non-empty chunks")
		}
	}
}

func BenchmarkMarkdownAware(b *testing.B) {
	text := benchmarkMarkdown(20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunker, _ := NewChunker(MarkdownAware)
		chunks := chunker.Chunk(text, 1400, 100)
		if len(chunks) == 0 {
			b.Fatal("expected non-empty chunks")
		}
	}
}

func BenchmarkHeadingAware(b *testing.B) {
	text := benchmarkNestedHeadings(5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunker, _ := NewChunker(HeadingAware)
		chunks := chunker.Chunk(text, 1400, 100)
		if len(chunks) == 0 {
			b.Fatal("expected non-empty chunks")
		}
	}
}

func BenchmarkTokenBased(b *testing.B) {
	text := benchmarkProse(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunker, _ := NewChunker(TokenBased)
		chunks := chunker.Chunk(text, 256, 32)
		if len(chunks) == 0 {
			b.Fatal("expected non-empty chunks")
		}
	}
}

// benchmarkUnicode generates Unicode-heavy text with mixed scripts and emojis.
func benchmarkUnicode(repeats int) string {
	base := "Café naïve résumé über 漢字 ひらがな 한글 🚀🌟🎉 àáâãäå "
	var buf strings.Builder
	buf.Grow(len(base) * repeats)
	for i := 0; i < repeats; i++ {
		buf.WriteString(base)
	}
	return buf.String()
}

// benchmarkOversizedChunks builds a slice of Chunks that exceed a character
// limit, with OriginalText populated for span-propagation cost measurement.
func benchmarkOversizedChunks(n, chunkSize int) ([]Chunk, LimitOptions) {
	para := strings.Repeat("This is a benchmark paragraph with multiple sentences. ", 20)
	chunks := make([]Chunk, n)
	for i := range chunks {
		chunks[i] = buildChunkWithSpan(i, para, i*len(para), (i+1)*len(para))
	}
	opts := LimitOptions{
		MaxChars:     chunkSize,
		Overlap:      0,
		OriginalText: strings.Repeat(para, n),
	}
	return chunks, opts
}

func BenchmarkHardCut_Unicode(b *testing.B) {
	text := benchmarkUnicode(500) // ~15k bytes of mixed Unicode
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunker, _ := NewChunker(HardCut)
		chunks := chunker.Chunk(text, 500, 50)
		if len(chunks) == 0 {
			b.Fatal("expected non-empty chunks")
		}
	}
}

func BenchmarkEnforceHardLimits_OversizedChunk(b *testing.B) {
	chunks, opts := benchmarkOversizedChunks(5, 200)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := EnforceHardLimits(chunks, opts)
		if len(result) == 0 {
			b.Fatal("expected non-empty result")
		}
	}
}

func BenchmarkSemanticMock(b *testing.B) {
	text := benchmarkSemanticText(10)
	embedder := benchmarkEmbedder{dim: 384}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sc, _ := NewSemanticChunker(embedder, SemanticOpts{})
		chunks := sc.ChunkSemantic(context.Background(), text, 1200, 100)
		if len(chunks) == 0 {
			b.Fatal("expected non-empty chunks")
		}
	}
}
