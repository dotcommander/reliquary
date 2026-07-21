// Package pq implements Product Quantization for efficient vector compression.
//
// # Overview
//
// Product Quantization (PQ) is a technique for compressing high-dimensional vectors
// while preserving approximate distance computations. It works by:
//
//  1. Splitting the vector into M subspaces (e.g., 1536-dim -> 96 subspaces of 16-dim)
//  2. Training K centroids per subspace using k-means clustering
//  3. Encoding each subspace as a single-byte index (if K=256)
//
// This achieves significant compression:
//
//   - Original: 1536 dimensions * 4 bytes = 6,144 bytes per vector
//   - Compressed: 96 subspaces * 1 byte = 96 bytes per vector
//   - Compression ratio: 64x
//
// # Accuracy
//
// PQ provides approximately 95% accuracy for similarity search at recall@10.
// The accuracy depends on:
//
//   - Number of subspaces (M): More subspaces = better accuracy
//   - Number of centroids (K): More centroids = better accuracy
//   - Data distribution: Clustered data compresses better
//
// # Usage
//
// Basic usage:
//
//	// Configure for 1536-dimensional vectors
//	config := pq.Config{
//		NumSubspaces: 96,
//		NumCentroids: 256,
//		Dimension:    1536,
//	}
//
//	// Create quantizer
//	q, err := pq.NewQuantizer(config)
//	if err != nil {
//		return err
//	}
//
//	// Train on sample vectors (need at least NumCentroids vectors)
//	err = q.Train(trainingVectors)
//	if err != nil {
//		return err
//	}
//
//	// Encode a vector
//	codes, err := q.Encode(vector)
//
//	// Decode to approximate vector
//	approx, err := q.Decode(codes)
//
// # Asymmetric Distance Computation
//
// For search, PQ uses Asymmetric Distance Computation (ADC):
//
//   - Query vector stays uncompressed
//   - Database vectors are compressed
//   - Distance is computed between uncompressed query and compressed DB vectors
//
// This provides better accuracy than symmetric distance (both compressed).
//
// For batch search, precompute lookup tables:
//
//	// Build lookup tables for fast batch distance computation
//	tables, err := q.BuildLookupTables(queryVector)
//
//	// O(M) distance computation per vector (instead of O(M * SubspaceDim))
//	for _, codes := range databaseCodes {
//		dist := q.DistanceWithTables(tables, codes)
//	}
//
// # Persistence
//
// Save and load trained quantizers:
//
//	// Save
//	var buf bytes.Buffer
//	q.Save(&buf)
//
//	// Load
//	q2 := &pq.Quantizer{}
//	q2.Load(&buf)
//
// Loads are strict and atomic: malformed or truncated state leaves q2
// unchanged. Serialized codebooks larger than 256 MiB or with more than 65,536
// subspaces are rejected before allocation.
//
// # Configuration Guidelines
//
// For 1536-dimensional vectors (OpenAI text-embedding-3-small):
//
//	config := pq.DefaultConfig() // NumSubspaces=96, NumCentroids=256
//
// For 768-dimensional vectors (smaller models):
//
//	config := pq.Config{
//		NumSubspaces: 48,  // 768/48 = 16 dims per subspace
//		NumCentroids: 256,
//		Dimension:    768,
//	}
//
// Training requirements:
//
//   - Minimum: NumCentroids vectors (256 for default config)
//   - Recommended: 1000-10000 vectors for good codebook quality
package pq
