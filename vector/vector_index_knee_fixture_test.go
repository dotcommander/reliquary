package vectors

import (
	"fmt"
	"math"
	rand "math/rand/v2"
)

func newKneeFixture(rows, dimensions int) kneeFixture {
	rng := rand.New(rand.NewPCG(1, 2))
	fixture := kneeFixture{
		vectors:      make([][]float32, rows),
		blobs:        make([][]byte, rows),
		arena:        make([]byte, 0, rows*dimensions*4),
		chunks:       make([]IndexChunk, rows),
		groups:       make([]string, rows),
		chunkIndexes: make([]int, rows),
	}
	for row := range rows {
		vector := make([]float32, dimensions)
		for dimension := range dimensions {
			vector[dimension] = rng.Float32()*2 - 1
		}
		Normalize32(vector)
		blob := EncodeFloat32Vec(vector)
		group := fmt.Sprintf("row-%08d", row)
		fixture.vectors[row] = vector
		fixture.blobs[row] = blob
		fixture.chunks[row] = IndexChunk{
			Group:      group,
			ChunkIndex: 0,
			Offset:     len(fixture.arena),
			Length:     len(blob),
		}
		fixture.groups[row] = group
		fixture.arena = append(fixture.arena, blob...)
	}
	return fixture
}

func newKneeQueries(count, dimensions int) [][]float32 {
	rng := rand.New(rand.NewPCG(3, 4))
	queries := make([][]float32, count)
	for index := range count {
		query := make([]float32, dimensions)
		for dimension := range dimensions {
			query[dimension] = rng.Float32()*2 - 1
		}
		Normalize32(query)
		queries[index] = query
	}
	return queries
}

func newKneeProfile(kind IndexKind, rows, dimensions, candidateLimit int) (ANNProfile, error) {
	bytes, err := kneePayloadEstimate(kind, rows, dimensions)
	if err != nil {
		return ANNProfile{}, fmt.Errorf("profile %s memory estimate: %w", kind, err)
	}
	switch kind {
	case IndexKindExact:
		return ANNProfile{
			ID:             "exact",
			Kind:           kind,
			Quantization:   "float32",
			MemoryEstimate: MemoryEstimate{Bytes: bytes, Label: kneeMemoryLabel},
		}, nil
	case IndexKindBinary:
		if candidateLimit <= 0 {
			return ANNProfile{}, fmt.Errorf("profile binary candidate limit: must be positive")
		}
		return ANNProfile{
			ID:             fmt.Sprintf("binary-candidates-%d", candidateLimit),
			Kind:           kind,
			CandidateLimit: candidateLimit,
			ExactRescore:   true,
			Quantization:   "median_binary",
			MemoryEstimate: MemoryEstimate{Bytes: bytes, Label: kneeMemoryLabel},
		}, nil
	default:
		return ANNProfile{}, fmt.Errorf("profile kind %q: unsupported", kind)
	}
}

func kneePayloadEstimate(kind IndexKind, rows, dimensions int) (int64, error) {
	if rows <= 0 {
		return 0, fmt.Errorf("rows: must be positive")
	}
	if dimensions <= 0 {
		return 0, fmt.Errorf("dimensions: must be positive")
	}
	rowCount := int64(rows)
	dimensionCount := int64(dimensions)

	switch kind {
	case IndexKindExact:
		values, err := kneeCheckedMultiply(rowCount, dimensionCount, "rows * dimensions")
		if err != nil {
			return 0, err
		}
		return kneeCheckedMultiply(values, 4, "exact float32 bytes")
	case IndexKindBinary:
		if dimensionCount > math.MaxInt64-63 {
			return 0, fmt.Errorf("binary words: int64 overflow")
		}
		words := (dimensionCount + 63) / 64
		rowWords, err := kneeCheckedMultiply(rowCount, words, "rows * binary words")
		if err != nil {
			return 0, err
		}
		rowBytes, err := kneeCheckedMultiply(rowWords, 8, "binary row bytes")
		if err != nil {
			return 0, err
		}
		medianBytes, err := kneeCheckedMultiply(dimensionCount, 4, "binary median bytes")
		if err != nil {
			return 0, err
		}
		if rowBytes > math.MaxInt64-medianBytes {
			return 0, fmt.Errorf("binary payload bytes: int64 overflow")
		}
		return rowBytes + medianBytes, nil
	default:
		return 0, fmt.Errorf("index kind %q: unsupported", kind)
	}
}

func kneeCheckedMultiply(left, right int64, field string) (int64, error) {
	if left < 0 || right < 0 || (right != 0 && left > math.MaxInt64/right) {
		return 0, fmt.Errorf("%s: int64 overflow", field)
	}
	return left * right, nil
}

func validateKneeBuildReport(profile string, report IndexBuildReport, expectedRows int) error {
	checks := []struct {
		field string
		got   int
		want  int
	}{
		{field: "InputRows", got: report.InputRows, want: expectedRows},
		{field: "IndexedRows", got: report.IndexedRows, want: expectedRows},
		{field: "SkippedBadSpan", got: report.SkippedBadSpan},
		{field: "SkippedBadBlob", got: report.SkippedBadBlob},
		{field: "SkippedMissingMetadata", got: report.SkippedMissingMetadata},
		{field: "SkippedDuplicateKey", got: report.SkippedDuplicateKey},
		{field: "DimensionMismatch", got: report.DimensionMismatch},
	}
	for _, check := range checks {
		if check.got != check.want {
			return fmt.Errorf("profile %s build report field %s: got %d, want %d", profile, check.field, check.got, check.want)
		}
	}
	if report.MedianError != "" {
		return fmt.Errorf("profile %s build report field MedianError: %s", profile, report.MedianError)
	}
	if report.QuantizeError != "" {
		return fmt.Errorf("profile %s build report field QuantizeError: %s", profile, report.QuantizeError)
	}
	return nil
}
