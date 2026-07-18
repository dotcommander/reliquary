package chunking

import (
	"context"
	"errors"
	"strings"
	"unicode/utf8"
)

// BatchEmbedder is satisfied by any embedder that can embed multiple texts.
// Defined here so semantic chunking can accept any embedding implementation.
type BatchEmbedder interface {
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}

// ErrNilEmbedder is returned by NewSemanticChunker when a nil embedder is
// provided.
var ErrNilEmbedder = errors.New("chunking: embedder must not be nil")

// SemanticOpts configures semantic chunking behavior.
type SemanticOpts struct {
	MaxChunkChars    int     // hard ceiling per chunk (default 1600)
	MinChunkChars    int     // merge groups smaller than this (default 200)
	BreakSensitivity float64 // stddev multiplier: higher = fewer breaks (default 1.0)
	SmoothingWindow  int     // centered moving-average window for similarity smoothing; 0 disables (default). Odd values recommended (e.g. 3).
	CoherenceWindow  int     // two-sided coherence gate: require N coherent neighbors on each side of a candidate break; 0 disables (default). Recommended: 2.
}

// SemanticPlanOptions configures semantic chunk planning from precomputed
// embeddings. FallbackSize and FallbackOverlap are applied only to the final
// hard-limit pass; callers should use their own fallback chunker when
// PlanSemanticChunks returns false.
type SemanticPlanOptions struct {
	MaxChunkChars    int
	MinChunkChars    int
	BreakSensitivity float64
	SmoothingWindow  int
	CoherenceWindow  int
	FallbackSize     int
	FallbackOverlap  int
}

// SemanticPlan describes an accepted semantic chunking plan.
type SemanticPlan struct {
	Units        []SemanticUnit
	Similarities []float64
	Breaks       []int
	Chunks       []Chunk
}

func (o SemanticOpts) withDefaults() SemanticOpts {
	if o.MaxChunkChars <= 0 {
		o.MaxChunkChars = defaultMaxChunkChars
	}
	if o.MinChunkChars <= 0 {
		o.MinChunkChars = 200
	}
	if o.BreakSensitivity <= 0 {
		o.BreakSensitivity = 1.0
	}
	return o
}

func semanticPlanOptionsFromSemantic(opts SemanticOpts, fallbackSize, fallbackOverlap int) SemanticPlanOptions {
	opts = opts.withDefaults()
	return SemanticPlanOptions{
		MaxChunkChars:    opts.MaxChunkChars,
		MinChunkChars:    opts.MinChunkChars,
		BreakSensitivity: opts.BreakSensitivity,
		SmoothingWindow:  opts.SmoothingWindow,
		CoherenceWindow:  opts.CoherenceWindow,
		FallbackSize:     fallbackSize,
		FallbackOverlap:  fallbackOverlap,
	}
}

func (o SemanticPlanOptions) withDefaults() SemanticPlanOptions {
	sem := SemanticOpts{
		MaxChunkChars:    o.MaxChunkChars,
		MinChunkChars:    o.MinChunkChars,
		BreakSensitivity: o.BreakSensitivity,
		SmoothingWindow:  o.SmoothingWindow,
		CoherenceWindow:  o.CoherenceWindow,
	}.withDefaults()
	o.MaxChunkChars = sem.MaxChunkChars
	o.MinChunkChars = sem.MinChunkChars
	o.BreakSensitivity = sem.BreakSensitivity
	o.SmoothingWindow = sem.SmoothingWindow
	o.CoherenceWindow = sem.CoherenceWindow
	return o
}

// SemanticChunker splits text at topic boundaries detected by embedding
// similarity between consecutive sentences. Produces variable-length chunks
// where each chunk covers one coherent topic.
type SemanticChunker struct {
	embedder BatchEmbedder
	opts     SemanticOpts
	fallback Chunker
}

// NewSemanticChunker creates a semantic chunker that falls back to smart
// boundary chunking on embedding failure. Returns ErrNilEmbedder if embedder
// is nil.
func NewSemanticChunker(embedder BatchEmbedder, opts SemanticOpts) (*SemanticChunker, error) {
	if embedder == nil {
		return nil, ErrNilEmbedder
	}
	return &SemanticChunker{
		embedder: embedder,
		opts:     opts.withDefaults(),
		fallback: newSmartBoundaryChunker(),
	}, nil
}

// PlanSemanticChunks plans semantic chunks from caller-supplied units and
// embeddings. It returns false when the supplied data is not suitable for
// semantic planning; callers should then use their preferred fallback chunker.
func PlanSemanticChunks(text string, units []SemanticUnit, embeddings [][]float32, opts SemanticPlanOptions) (SemanticPlan, bool) {
	opts = opts.withDefaults()
	spans, unitTexts := semanticUnitsFromPublic(units)

	if len(spans) < 3 {
		return SemanticPlan{}, false
	}
	if len(embeddings) != len(unitTexts) {
		return SemanticPlan{}, false
	}
	if !validEmbeddingBatch(embeddings) {
		return SemanticPlan{}, false
	}

	sims := make([]float64, len(embeddings)-1)
	for i := range len(embeddings) - 1 {
		sims[i] = dotProduct(embeddings[i], embeddings[i+1])
	}
	if opts.SmoothingWindow >= 2 {
		sims = smoothSimilarities(sims, opts.SmoothingWindow)
	}

	breaks := findBreakpoints(sims, opts.BreakSensitivity, opts.CoherenceWindow)
	groups := groupBySplits(unitTexts, breaks)
	groups = enforceSizeConstraints(groups, opts.MinChunkChars, opts.MaxChunkChars)
	groups = mergeAdjacentGroups(spans, groups, embeddings, defaultSemanticMergeThreshold, text)

	chunks := buildSemanticChunks(spans, groups, text)
	chunks = EnforceHardLimits(chunks, LimitOptions{
		MaxChars:     opts.FallbackSize,
		Overlap:      opts.FallbackOverlap,
		OriginalText: text,
	})

	return SemanticPlan{
		Units:        units,
		Similarities: sims,
		Breaks:       breaks,
		Chunks:       chunks,
	}, true
}

func validEmbeddingBatch(embeddings [][]float32) bool {
	if len(embeddings) == 0 {
		return false
	}
	dim := len(embeddings[0])
	if dim == 0 {
		return false
	}
	for _, emb := range embeddings {
		if len(emb) != dim || isZeroVector(emb) {
			return false
		}
	}
	return true
}

// ChunkSemantic splits text at topic boundaries. Falls back to smart boundary
// chunking if embedding fails or the text is too short for semantic analysis.
// When the input contains structural markers (conversation turns, headings,
// horizontal rules, or paragraph blocks), those are used as semantic atoms
// instead of individual sentences, reducing embedding calls and preserving
// source byte spans where possible.
func (sc *SemanticChunker) ChunkSemantic(ctx context.Context, text string, fallbackSize, fallbackOverlap int) []Chunk {
	// Try structural units first (headings, conversation turns, etc.),
	// fall back to sentence-only mode.
	units := semanticUnits(text)

	// Need at least 3 units for meaningful boundary detection.
	if len(units) < 3 {
		return sc.fallback.Chunk(text, fallbackSize, fallbackOverlap)
	}

	// Filter out very short units that produce poor embeddings.
	units, unitTexts := mergeTinyUnits(units, 20)
	if len(unitTexts) < 3 {
		return sc.fallback.Chunk(text, fallbackSize, fallbackOverlap)
	}

	// Identify non-analyzable units (code fences, tables) that should not
	// drive embedding breakpoints but must still appear in output.
	analyzable := markAnalyzableUnits(unitTexts)
	analyzableTexts := filterAnalyzableTexts(unitTexts, analyzable)

	if len(analyzableTexts) < 3 {
		return sc.fallback.Chunk(text, fallbackSize, fallbackOverlap)
	}

	embeddings, err := sc.embedder.EmbedBatch(ctx, analyzableTexts)
	if err != nil {
		return sc.fallback.Chunk(text, fallbackSize, fallbackOverlap)
	}

	analyzableUnits := make([]SemanticUnit, 0, len(units))
	for i, a := range analyzable {
		if !a {
			continue
		}
		u := units[i]
		analyzableUnits = append(analyzableUnits, SemanticUnit{
			Text:      u.text,
			StartChar: u.start,
			EndChar:   u.end,
		})
	}
	plan, ok := PlanSemanticChunks(text, analyzableUnits, embeddings, semanticPlanOptionsFromSemantic(sc.opts, fallbackSize, fallbackOverlap))
	if !ok {
		return sc.fallback.Chunk(text, fallbackSize, fallbackOverlap)
	}

	// Map analyzable-only breaks back to full unit indices.
	breaks := mapBreaksToFullUnits(plan.Breaks, analyzable)

	// Group units between breakpoints into chunks.
	groups := groupBySplits(unitTexts, breaks)

	// Enforce size constraints: merge tiny groups, split oversized ones.
	groups = enforceSizeConstraints(groups, sc.opts.MinChunkChars, sc.opts.MaxChunkChars)

	// Merge adjacent groups with high cosine similarity.
	// Only applies when all units were analyzable (no markdown noise filtering).
	if len(embeddings) == len(unitTexts) {
		groups = mergeAdjacentGroups(units, groups, embeddings, defaultSemanticMergeThreshold, text)
	}

	// Build chunks with span propagation.
	chunks := buildSemanticChunks(units, groups, text)
	return EnforceHardLimits(chunks, LimitOptions{MaxChars: fallbackSize, Overlap: fallbackOverlap, OriginalText: text})
}

// groupBySplits groups sentences into text blocks at the given break indices.
// Break index i means split AFTER sentence i (between sentence i and i+1).
func groupBySplits(sentences []string, breaks []int) []string {
	if len(breaks) == 0 {
		return []string{strings.Join(sentences, " ")}
	}

	groups := make([]string, 0, len(breaks)+1)
	prev := 0
	for _, b := range breaks {
		split := b + 1 // break after sentence b → next group starts at b+1
		if split > prev && split <= len(sentences) {
			groups = append(groups, strings.Join(sentences[prev:split], " "))
			prev = split
		}
	}
	if prev < len(sentences) {
		groups = append(groups, strings.Join(sentences[prev:], " "))
	}
	return groups
}

// enforceSizeConstraints merges undersized groups and splits oversized ones.
func enforceSizeConstraints(groups []string, minChars, maxChars int) []string {
	// Pass 1: merge undersized groups with their neighbor.
	groups = mergeSmallGroups(groups, minChars)

	// Pass 2: split oversized groups at their internal lowest-similarity point.
	var result []string
	for _, g := range groups {
		if utf8.RuneCountInString(g) <= maxChars {
			result = append(result, g)
			continue
		}
		// Split oversized group at sentence boundaries.
		sents := splitIntoSentences(g)
		if len(sents) <= 1 {
			result = append(result, g)
			continue
		}
		result = append(result, splitOversized(sents, maxChars)...)
	}
	return result
}

// mergeSmallGroups merges groups shorter than minChars into their neighbor.
func mergeSmallGroups(groups []string, minChars int) []string {
	if len(groups) <= 1 {
		return groups
	}

	merged := make([]string, 0, len(groups))
	for _, g := range groups {
		if len(merged) > 0 && utf8.RuneCountInString(g) < minChars {
			// Merge into previous group.
			merged[len(merged)-1] += " " + g
		} else {
			merged = append(merged, g)
		}
	}

	// Check if the first group is too small (couldn't merge backward).
	if len(merged) > 1 && utf8.RuneCountInString(merged[0]) < minChars {
		merged[1] = merged[0] + " " + merged[1]
		merged = merged[1:]
	}

	return merged
}

// splitOversized splits sentences into groups that fit within maxChars (rune count),
// greedily accumulating sentences.
func splitOversized(sentences []string, maxChars int) []string {
	var groups []string
	var buf strings.Builder
	runeCount := 0
	for _, s := range sentences {
		sRunes := utf8.RuneCountInString(s)
		if buf.Len() > 0 && runeCount+1+sRunes > maxChars {
			groups = append(groups, strings.TrimSpace(buf.String()))
			buf.Reset()
			runeCount = 0
		}
		if buf.Len() > 0 {
			buf.WriteString(" ")
			runeCount++
		}
		buf.WriteString(s)
		runeCount += sRunes
	}
	if buf.Len() > 0 {
		groups = append(groups, strings.TrimSpace(buf.String()))
	}
	return groups
}
