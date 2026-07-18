package pq

import (
	"testing"
)

func TestKMeansPlusPlus(t *testing.T) {
	t.Parallel()

	vectors := generateClusteredVectors(4, 25, 8)

	centroids, err := KMeansPlusPlus(vectors, 4)
	if err != nil {
		t.Fatalf("KMeansPlusPlus() error = %v", err)
	}

	if len(centroids) != 4 {
		t.Errorf("KMeansPlusPlus() returned %d centroids, want 4", len(centroids))
	}

	for i, c := range centroids {
		if len(c) != 8 {
			t.Errorf("centroid %d has dimension %d, want 8", i, len(c))
		}
	}
}

func TestKMeansPlusPlusErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		vectors [][]float32
		k       int
		wantErr bool
	}{
		{
			name:    "empty vectors",
			vectors: nil,
			k:       4,
			wantErr: true,
		},
		{
			name:    "k is zero",
			vectors: generateRandomVectors(10, 8),
			k:       0,
			wantErr: true,
		},
		{
			name:    "k greater than vectors",
			vectors: generateRandomVectors(5, 8),
			k:       10,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := KMeansPlusPlus(tt.vectors, tt.k)
			if (err != nil) != tt.wantErr {
				t.Errorf("KMeansPlusPlus() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestKMeans(t *testing.T) {
	t.Parallel()

	// Generate clearly separated clusters
	vectors := generateClusteredVectors(4, 50, 8)

	centroids, err := KMeans(vectors, 4, 25)
	if err != nil {
		t.Fatalf("KMeans() error = %v", err)
	}

	if len(centroids) != 4 {
		t.Errorf("KMeans() returned %d centroids, want 4", len(centroids))
	}

	// Verify each vector is reasonably close to at least one centroid
	for _, v := range vectors {
		minDist := squaredL2Distance(v, centroids[0])
		for _, c := range centroids[1:] {
			dist := squaredL2Distance(v, c)
			if dist < minDist {
				minDist = dist
			}
		}
		// For clustered data with noise ~0.25, max expected distance is small
		if minDist > 5.0 {
			t.Errorf("Vector too far from nearest centroid: distance = %f", minDist)
		}
	}
}

func TestKMeansErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		vectors [][]float32
		k       int
		maxIter int
		wantErr bool
	}{
		{
			name:    "empty vectors",
			vectors: nil,
			k:       4,
			maxIter: 10,
			wantErr: true,
		},
		{
			name:    "k is zero",
			vectors: generateRandomVectors(10, 8),
			k:       0,
			maxIter: 10,
			wantErr: true,
		},
		{
			name:    "maxIter is zero",
			vectors: generateRandomVectors(10, 8),
			k:       4,
			maxIter: 0,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := KMeans(tt.vectors, tt.k, tt.maxIter)
			if (err != nil) != tt.wantErr {
				t.Errorf("KMeans() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestKMeansKGreaterThanVectors(t *testing.T) {
	t.Parallel()

	vectors := generateRandomVectors(5, 8)

	// Should not error, but return k centroids (with duplicates if needed)
	centroids, err := KMeans(vectors, 10, 10)
	if err != nil {
		t.Fatalf("KMeans() error = %v", err)
	}

	if len(centroids) != 10 {
		t.Errorf("KMeans() returned %d centroids, want 10", len(centroids))
	}
}

func TestMiniBatchKMeans(t *testing.T) {
	t.Parallel()

	vectors := generateClusteredVectors(4, 100, 8)

	centroids, err := MiniBatchKMeans(vectors, 4, 32, 50)
	if err != nil {
		t.Fatalf("MiniBatchKMeans() error = %v", err)
	}

	if len(centroids) != 4 {
		t.Errorf("MiniBatchKMeans() returned %d centroids, want 4", len(centroids))
	}
}

func TestMiniBatchKMeansErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		vectors   [][]float32
		k         int
		batchSize int
		maxIter   int
		wantErr   bool
	}{
		{
			name:      "empty vectors",
			vectors:   nil,
			k:         4,
			batchSize: 32,
			maxIter:   10,
			wantErr:   true,
		},
		{
			name:      "k is zero",
			vectors:   generateRandomVectors(100, 8),
			k:         0,
			batchSize: 32,
			maxIter:   10,
			wantErr:   true,
		},
		{
			name:      "batchSize is zero",
			vectors:   generateRandomVectors(100, 8),
			k:         4,
			batchSize: 0,
			maxIter:   10,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := MiniBatchKMeans(tt.vectors, tt.k, tt.batchSize, tt.maxIter)
			if (err != nil) != tt.wantErr {
				t.Errorf("MiniBatchKMeans() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCopyVector(t *testing.T) {
	t.Parallel()

	original := []float32{1.0, 2.0, 3.0}
	copied := copyVector(original)

	// Modify original
	original[0] = 999.0

	// Check copy is independent
	if copied[0] != 1.0 {
		t.Error("copyVector() did not create independent copy")
	}
}

func TestFindNearest(t *testing.T) {
	t.Parallel()

	centroids := [][]float32{
		{0, 0},
		{10, 10},
		{-10, -10},
	}

	tests := []struct {
		name     string
		vector   []float32
		expected int
	}{
		{"near origin", []float32{1, 1}, 0},
		{"near positive", []float32{9, 9}, 1},
		{"near negative", []float32{-9, -9}, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := findNearest(tt.vector, centroids)
			if got != tt.expected {
				t.Errorf("findNearest() = %d, want %d", got, tt.expected)
			}
		})
	}
}

func BenchmarkKMeansPlusPlus(b *testing.B) {
	vectors := generateRandomVectors(1000, 16)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = KMeansPlusPlus(vectors, 256)
	}
}

func BenchmarkMiniBatchKMeans(b *testing.B) {
	vectors := generateRandomVectors(1000, 16)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = MiniBatchKMeans(vectors, 256, 64, 25)
	}
}
