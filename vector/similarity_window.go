package vectors

import "math"

// AverageSimilarity calculates the average cosine similarity between multiple vector pairs.
// Returns 1 when fewer than two embeddings are provided or no pairs exist.
func AverageSimilarity(embeddings [][]float32) float32 {
	if len(embeddings) < 2 {
		return 1
	}

	var totalSimilarity float32
	count := 0

	for i := 0; i < len(embeddings)-1; i++ {
		for j := i + 1; j < len(embeddings); j++ {
			totalSimilarity += Cosine32(embeddings[i], embeddings[j])
			count++
		}
	}

	if count == 0 {
		return 1
	}

	return totalSimilarity / float32(count)
}

// SlidingWindowSimilarity calculates similarity scores using a sliding window.
// Returns similarity scores between consecutive windows.
func SlidingWindowSimilarity(embeddings [][]float32, windowSize int) []float32 {
	if len(embeddings) < 2 || windowSize < 1 {
		return []float32{}
	}

	// Adjust window size if needed
	if windowSize > len(embeddings) {
		windowSize = len(embeddings)
	}

	// Pre-allocate result slice with exact capacity
	numResults := len(embeddings) - 1
	similarities := make([]float32, numResults)

	// Pre-allocate reusable buffers for averaging
	dim := len(embeddings[0])
	currentAvg := make([]float32, dim)
	nextAvg := make([]float32, dim)

	for i := 0; i < numResults; i++ {
		// Calculate average embedding for current window
		windowEnd := i + windowSize
		if windowEnd > len(embeddings) {
			windowEnd = len(embeddings)
		}
		averageEmbeddingInto(embeddings[i:windowEnd], currentAvg)

		// Calculate average embedding for next window
		nextStart := i + 1
		nextEnd := nextStart + windowSize
		if nextEnd > len(embeddings) {
			nextEnd = len(embeddings)
		}
		averageEmbeddingInto(embeddings[nextStart:nextEnd], nextAvg)

		// Calculate similarity
		similarities[i] = Cosine32(currentAvg, nextAvg)
	}

	return similarities
}

// averageEmbeddingInto calculates the average of multiple embeddings into a pre-allocated buffer.
// This avoids allocations when called repeatedly in tight loops.
func averageEmbeddingInto(embeddings [][]float32, result []float32) {
	if len(embeddings) == 0 || len(result) == 0 {
		return
	}

	// Zero the result buffer
	for i := range result {
		result[i] = 0
	}

	dim := len(result)
	for _, embedding := range embeddings {
		for i := 0; i < dim && i < len(embedding); i++ {
			result[i] += embedding[i]
		}
	}

	count := float32(len(embeddings))
	for i := 0; i < dim; i++ {
		result[i] /= count
	}
}

// FindSemanticBoundaries identifies positions where semantic similarity drops below threshold.
func FindSemanticBoundaries(similarities []float32, threshold float32) []int {
	boundaries := []int{}

	for i, sim := range similarities {
		if sim < threshold {
			// Boundary is after position i
			boundaries = append(boundaries, i+1)
		}
	}

	return boundaries
}

// AdaptiveThreshold calculates an adaptive threshold based on similarity distribution.
func AdaptiveThreshold(similarities []float32) float32 {
	if len(similarities) == 0 {
		return 0.7 // Default threshold
	}

	// Calculate mean and standard deviation
	var sum, sumSq float64
	for _, sim := range similarities {
		value := float64(sim)
		sum += value
		sumSq += value * value
	}

	mean := sum / float64(len(similarities))
	variance := (sumSq / float64(len(similarities))) - (mean * mean)
	if variance < 0 {
		variance = 0
	}
	stdDev := math.Sqrt(variance)

	// Adaptive threshold: mean - 1 standard deviation
	// This identifies points that are significantly less similar than average
	threshold := mean - stdDev

	// Clamp to reasonable range
	if threshold < 0.3 {
		threshold = 0.3
	} else if threshold > 0.9 {
		threshold = 0.9
	}

	return float32(threshold)
}

// SmoothSimilarities applies smoothing to reduce noise in similarity scores.
func SmoothSimilarities(similarities []float32, windowSize int) []float32 {
	if len(similarities) == 0 || windowSize < 1 {
		return similarities
	}

	smoothed := make([]float32, len(similarities))
	halfWindow := windowSize / 2

	for i := range similarities {
		start := i - halfWindow
		end := i + halfWindow + 1

		if start < 0 {
			start = 0
		}
		if end > len(similarities) {
			end = len(similarities)
		}

		// Calculate average in window
		var sum float32
		for j := start; j < end; j++ {
			sum += similarities[j]
		}
		smoothed[i] = sum / float32(end-start)
	}

	return smoothed
}
