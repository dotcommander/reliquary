// Package vectors provides zero-dependency primitive vector math for embedding
// and retrieval systems.
//
// # Similarity
//
// Cosine similarity and distance are available in float32 and float64 variants
// (Cosine32, Cosine64, CosineDistance32, CosineDistance64). Use float32 for
// embedding pipelines — it halves memory and is faster on modern hardware.
// Use float64 only when numerical precision is the governing constraint.
// Dot product helpers (Dot32, Dot64, DotFromBlob, TopKFromBlob) are provided
// for callers that normalize up front and want raw dot as a proxy for cosine.
// Euclidean64 and Jaccard (JaccardWords) cover additional similarity regimes.
//
// # Normalization
//
// Normalize32/Normalize64 L2-normalize a vector in place and return the
// original magnitude. NormalizeTo32/NormalizeTo64 return a new normalized
// slice. NormSquared32/NormSquared64 and IsUnit32/IsUnit64 are low-level
// helpers for callers that need to pre-check or validate unit vectors.
//
// # Binary quantization
//
// Quantize encodes a float32 embedding as a BinaryVector ([]uint64 packed
// bits) using per-dimension thresholds; QuantizeInto is the allocation-free
// variant. HammingDistance counts differing bits between two BinaryVectors in
// O(dims/64) time. ComputeMedians/ComputeMediansChecked derive thresholds from
// a corpus of embeddings. Together these support compact storage and fast
// approximate pre-filtering: Hamming distance on quantized vectors is ~10x
// faster than float32 cosine, making it a practical first-pass filter before
// exact re-ranking.
//
// # Pooling and aggregation
//
// MeanPool32 averages a slice of vectors into one. WeightedMeanPool32 weights
// the average by a parallel float64 slice. AverageSimilarity summarizes
// pairwise similarity across a set of embeddings.
//
// # In-memory indexes
//
// ExactIndex is a flat brute-force cosine index suitable for corpora up to
// ~100 K chunks. SearchKeys reranks selected (group, chunk) candidates without
// importing caller storage policy. BinaryIndex is a Hamming-distance pre-filter
// index backed by packed bit blobs, intended for larger corpora or
// memory-constrained environments. Checked constructors return IndexBuildReport
// values so callers can log skipped rows without changing compatibility paths.
// IndexManifest records dimensions, embedding identity, median thresholds, and
// chunking identity so persisted indexes can be rejected when stale.
//
// # Auxiliary utilities
//
// K-means clustering (KMeans, KMeans64, FindOptimalK, FindOptimalK64,
// SilhouetteScore, AverageSilhouetteScore64), sliding-window
// and semantic boundary detection (SlidingWindowSimilarity, FindSemanticBoundaries,
// FindElbowCurvature), Reciprocal Rank Fusion (RRF), score normalization
// (MinMaxNormalize, CosineToUnit, Clamp01), statistical helpers (MeanStddev,
// GaussianSmooth, Gradient), and binary blob encoding/decoding
// (EncodeFloat32Vec, DecodeFloat32Vec, EncodeFloat64Vec, DecodeFloat64Vec) are
// also provided.
//
// # Dependencies
//
// This package's production code uses the standard library only. The enclosing
// reliquary module has dependencies for sibling packages such as chunking.
package vectors
