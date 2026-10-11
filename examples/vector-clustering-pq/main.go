// Command vector-clustering-pq demonstrates Reliquary's vector clustering
// and Product Quantization (PQ) compression primitives.
//
// Run from the repository root:
//
//	GOWORK=off go run ./examples/vector-clustering-pq
package main

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/dotcommander/reliquary/examples/internal/examplekit"
	"github.com/dotcommander/reliquary/vector/clustering"
	"github.com/dotcommander/reliquary/vector/pq"
)

func main() {
	examplekit.Run(run)
}

func run(_ context.Context) error {
	const numVectors = 64
	const dim = 32

	// Generate synthetic test vectors with 2 distinct clusters
	vectors64 := make([][]float64, numVectors)
	vectors32 := make([][]float32, numVectors)
	for i := range numVectors {
		vec64 := make([]float64, dim)
		vec32 := make([]float32, dim)
		center := 0.2
		if i >= numVectors/2 {
			center = 0.8
		}
		for d := range dim {
			val := center + (rand.Float64()*0.1 - 0.05)
			vec64[d] = val
			vec32[d] = float32(val)
		}
		vectors64[i] = vec64
		vectors32[i] = vec32
	}

	// 1. K-Means Vector Clustering
	fmt.Println("=== 1. K-Means Clustering ===")
	kmResult := clustering.KMeans(vectors64, clustering.KMeansConfig{
		K:             2,
		MaxIterations: 20,
		Tolerance:     1e-4,
	})
	c0, c1 := 0, 0
	for _, l := range kmResult.Assignments {
		if l == 0 {
			c0++
		} else {
			c1++
		}
	}
	fmt.Printf("Clustered %d vectors of dim %d into %d clusters (sizes: cluster0=%d, cluster1=%d, iterations=%d)\n",
		numVectors, dim, kmResult.K, c0, c1, kmResult.Iterations)

	// 2. Product Quantization (PQ) Compression
	fmt.Println("\n=== 2. Product Quantization (PQ) ===")
	pqCfg := pq.Config{
		Dimension:    dim,
		NumSubspaces: 4,  // 32 dims / 4 subspaces = 8 dims per subspace
		NumCentroids: 16, // 16 centroids per subspace
	}
	quantizer, err := pq.NewQuantizer(pqCfg)
	if err != nil {
		return fmt.Errorf("new pq quantizer: %w", err)
	}
	if err := quantizer.Train(vectors32); err != nil {
		return fmt.Errorf("train pq: %w", err)
	}

	// Compress a vector to compact code bytes
	sampleVec := vectors32[0]
	code, err := quantizer.Encode(sampleVec)
	if err != nil {
		return fmt.Errorf("encode pq: %w", err)
	}
	uncompressedBytes := len(sampleVec) * 4 // float32 = 4 bytes
	compressedBytes := len(code)

	fmt.Printf("Original vector: %d dimensions (%d bytes)\n", len(sampleVec), uncompressedBytes)
	fmt.Printf("PQ encoded code: %v (%d bytes)\n", code, compressedBytes)
	fmt.Printf("Compression ratio: %.1fx smaller\n", float64(uncompressedBytes)/float64(compressedBytes))

	// Reconstruct and compute asymmetric distance
	reconstructed, err := quantizer.Decode(code)
	if err != nil {
		return fmt.Errorf("decode pq: %w", err)
	}
	dist, err := quantizer.AsymmetricDistance(sampleVec, code)
	if err != nil {
		return fmt.Errorf("asymmetric distance: %w", err)
	}
	fmt.Printf("Asymmetric distance to decoded vector: %.4f (dims=%d)\n", dist, len(reconstructed))

	return nil
}
