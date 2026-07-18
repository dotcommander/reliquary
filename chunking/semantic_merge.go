package chunking

import (
	"math"
	"strings"
	"unicode/utf8"

	"github.com/dotcommander/reliquary/vector"
)

// ---------------------------------------------------------------------------
// Semantic adjacent-merge helpers
// ---------------------------------------------------------------------------

// defaultSemanticMergeThreshold is the cosine similarity above which adjacent
// semantic groups are merged. Conservative: collapses near-duplicates only.
const defaultSemanticMergeThreshold = 0.92

// l2Norm returns the L2 (Euclidean) norm of a float32 vector.
func l2Norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

// cosineSimilarity computes the cosine similarity between two float32 vectors.
// Returns 0 for mismatched dimensions, empty vectors, or zero-norm vectors.
func cosineSimilarity(a, b []float32) float64 {
	return float64(vectors.Cosine32(a, b))
}

// normalizeL2 returns a copy of v normalized to unit L2 length.
// Returns nil for zero-norm vectors.
func normalizeL2(v []float32) []float32 {
	norm := l2Norm(v)
	if norm == 0 {
		return nil
	}
	result := make([]float32, len(v))
	for i, x := range v {
		result[i] = float32(float64(x) / norm)
	}
	return result
}

// weightedAverage returns the weighted average of two float32 vectors.
// Returns nil if lengths differ or either input is empty.
func weightedAverage(a []float32, aWeight float64, b []float32, bWeight float64) []float32 {
	if len(a) != len(b) || len(a) == 0 {
		return nil
	}
	total := aWeight + bWeight
	if total == 0 {
		return nil
	}
	result := make([]float32, len(a))
	for i := range a {
		result[i] = float32((float64(a[i])*aWeight + float64(b[i])*bWeight) / total)
	}
	return result
}

// semanticGroup holds a grouped chunk of text with its embedding
// representative for adjacent merge decisions.
type semanticGroup struct {
	text      string
	start     int // byte offset in original text (0 = unknown)
	end       int // byte offset in original text (0 = unknown)
	embedding []float32
	weight    float64 // based on utf8.RuneCountInString(text)
}

// mergeAdjacentSimilarGroups merges adjacent groups whose cosine similarity
// exceeds the threshold. Merged embeddings are L2-normalized.
func mergeAdjacentSimilarGroups(groups []semanticGroup, threshold float64) []semanticGroup {
	if len(groups) <= 1 {
		return groups
	}

	merged := make([]semanticGroup, 0, len(groups))
	merged = append(merged, groups[0])

	for i := 1; i < len(groups); i++ {
		curr := groups[i]
		prev := &merged[len(merged)-1]

		sim := cosineSimilarity(prev.embedding, curr.embedding)
		if sim >= threshold && prev.embedding != nil && curr.embedding != nil {
			// Merge current into previous.
			prev.text = prev.text + " " + curr.text
			oldWeight := prev.weight
			prev.weight += curr.weight

			// Update embedding as weighted average and normalize.
			avg := weightedAverage(prev.embedding, oldWeight, curr.embedding, curr.weight)
			prev.embedding = normalizeL2(avg)

			// Update span only if contiguous.
			if prev.end == curr.start && prev.start != 0 && prev.end != 0 {
				prev.end = curr.end
			} else {
				prev.start = 0
				prev.end = 0
			}
		} else {
			merged = append(merged, curr)
		}
	}

	return merged
}

// mergeAdjacentGroups converts text groups into semanticGroups with their
// embedding representatives, runs adjacent merge, and returns the merged
// group texts.
func mergeAdjacentGroups(units []textSpan, groups []string, embeddings [][]float32, threshold float64, originalText string) []string {
	if len(groups) <= 1 || len(embeddings) == 0 {
		return groups
	}

	// Map each group's text to the average embedding of its constituent units.
	unitIdx := 0
	semGroups := make([]semanticGroup, len(groups))
	for i, g := range groups {
		semGroups[i].text = g
		semGroups[i].weight = float64(utf8.RuneCountInString(g))

		groupText := g
		consumed := 0
		startUnit := unitIdx
		for unitIdx < len(units) {
			u := units[unitIdx]
			if len(groupText) < len(u.text) {
				break
			}
			if strings.HasPrefix(groupText, u.text) {
				groupText = strings.TrimSpace(groupText[len(u.text):])
				unitIdx++
				consumed++
				continue
			}
			if consumed > 0 && strings.HasPrefix(groupText, " "+u.text) {
				groupText = groupText[len(u.text)+1:]
				unitIdx++
				consumed++
				continue
			}
			break
		}

		if consumed > 0 {
			first := units[startUnit]
			last := units[unitIdx-1]
			semGroups[i].start = first.start
			semGroups[i].end = last.end

			if len(embeddings) > startUnit && consumed <= len(embeddings)-startUnit {
				dim := len(embeddings[startUnit])
				avg := make([]float32, dim)
				for j := 0; j < consumed; j++ {
					for k := range avg {
						if len(embeddings[startUnit+j]) == dim {
							avg[k] += embeddings[startUnit+j][k]
						}
					}
				}
				for k := range avg {
					avg[k] /= float32(consumed)
				}
				semGroups[i].embedding = normalizeL2(avg)
			}
		}
	}

	merged := mergeAdjacentSimilarGroups(semGroups, threshold)
	result := make([]string, len(merged))
	for i, g := range merged {
		result[i] = g.text
	}
	return result
}
