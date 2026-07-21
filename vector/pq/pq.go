// Package pq implements Product Quantization for vector compression.
//
// Product Quantization (PQ) compresses high-dimensional vectors by:
// 1. Splitting the vector into M subspaces (e.g., 1536-dim -> 96 subspaces of 16-dim)
// 2. Training K centroids per subspace using k-means clustering
// 3. Encoding each subspace as a single byte index (if K=256)
//
// This achieves ~16x compression (1536 float32 = 6KB -> 96 bytes) with ~95% accuracy.
//
// Search uses asymmetric distance computation (ADC): the query vector stays
// uncompressed while database vectors are compressed. This provides better
// accuracy than symmetric distance while maintaining fast search.
package pq

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

const (
	maxSerializedCodebookBytes int64 = 256 << 20
	// Each subspace becomes a separate slice/allocation on load. Bound that
	// shape independently of float payload bytes so hostile tiny-subspace
	// headers cannot cause millions of allocations below the byte limit.
	maxSerializedSubspaces = 1 << 16
)

// Config defines Product Quantization parameters.
type Config struct {
	// NumSubspaces is the number of subspaces (M).
	// The vector dimension must be divisible by this.
	// Common values: 96 for 1536-dim, 48 for 768-dim.
	NumSubspaces int

	// NumCentroids is the number of centroids per subspace (K).
	// 256 allows single-byte encoding. Must be <= 256.
	NumCentroids int

	// Dimension is the original vector dimension.
	Dimension int
}

// DefaultConfig returns a configuration for 1536-dimensional vectors.
func DefaultConfig() Config {
	return Config{
		NumSubspaces: 96,
		NumCentroids: 256,
		Dimension:    1536,
	}
}

// Validate checks the configuration for consistency.
func (c Config) Validate() error {
	if c.NumSubspaces <= 0 {
		return errors.New("NumSubspaces must be positive")
	}
	if c.NumCentroids <= 0 || c.NumCentroids > 256 {
		return errors.New("NumCentroids must be between 1 and 256")
	}
	if c.Dimension <= 0 {
		return errors.New("dimension must be positive")
	}
	if c.Dimension%c.NumSubspaces != 0 {
		return fmt.Errorf("dimension (%d) must be divisible by NumSubspaces (%d)",
			c.Dimension, c.NumSubspaces)
	}
	return nil
}

// SubspaceDim returns the dimensionality of each subspace.
func (c Config) SubspaceDim() int {
	return c.Dimension / c.NumSubspaces
}

// Quantizer handles PQ encoding and decoding.
type Quantizer struct {
	config Config

	// codebooks[m] contains K centroids for subspace m, flattened.
	// Shape: [NumSubspaces][NumCentroids * SubspaceDim]
	codebooks [][]float32

	trained bool
}

// NewQuantizer creates a new quantizer with the given configuration.
func NewQuantizer(config Config) (*Quantizer, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	return &Quantizer{config: config}, nil
}

// Config returns the quantizer configuration.
func (q *Quantizer) Config() Config {
	return q.config
}

// IsTrained returns whether the quantizer has been trained.
func (q *Quantizer) IsTrained() bool {
	return q.trained
}

// Train builds codebooks from sample vectors using k-means clustering.
// Requires at least NumCentroids vectors for meaningful clustering.
func (q *Quantizer) Train(vectors [][]float32) error {
	if len(vectors) < q.config.NumCentroids {
		return fmt.Errorf("need at least %d vectors for training, got %d",
			q.config.NumCentroids, len(vectors))
	}

	// Validate vector dimensions
	for i, v := range vectors {
		if len(v) != q.config.Dimension {
			return fmt.Errorf("vector %d has dimension %d, expected %d",
				i, len(v), q.config.Dimension)
		}
		if err := validateFiniteVector(fmt.Sprintf("vector %d", i), v); err != nil {
			return err
		}
	}

	subspaceDim := q.config.SubspaceDim()
	q.codebooks = make([][]float32, q.config.NumSubspaces)

	// Train each subspace independently
	for m := 0; m < q.config.NumSubspaces; m++ {
		// Extract subspace vectors
		subvectors := make([][]float32, len(vectors))
		for i, v := range vectors {
			subvectors[i] = v[m*subspaceDim : (m+1)*subspaceDim]
		}

		// Run k-means clustering
		centroids, err := KMeans(subvectors, q.config.NumCentroids, 25)
		if err != nil {
			return fmt.Errorf("kmeans for subspace %d: %w", m, err)
		}

		// Flatten centroids for storage
		q.codebooks[m] = make([]float32, q.config.NumCentroids*subspaceDim)
		for k, centroid := range centroids {
			copy(q.codebooks[m][k*subspaceDim:], centroid)
		}
	}

	q.trained = true
	return nil
}

// Encode compresses a vector to PQ codes.
// Returns a byte slice of length NumSubspaces.
func (q *Quantizer) Encode(vector []float32) ([]byte, error) {
	if !q.trained {
		return nil, errors.New("quantizer not trained")
	}
	if len(vector) != q.config.Dimension {
		return nil, fmt.Errorf("vector has dimension %d, expected %d",
			len(vector), q.config.Dimension)
	}
	if err := validateFiniteVector("vector", vector); err != nil {
		return nil, err
	}

	codes := make([]byte, q.config.NumSubspaces)
	subspaceDim := q.config.SubspaceDim()

	for m := 0; m < q.config.NumSubspaces; m++ {
		subvec := vector[m*subspaceDim : (m+1)*subspaceDim]
		codes[m] = byte(q.findNearestCentroid(m, subvec))
	}

	return codes, nil
}

// findNearestCentroid returns the index of the nearest centroid for a subvector.
func (q *Quantizer) findNearestCentroid(subspace int, subvec []float32) int {
	subspaceDim := q.config.SubspaceDim()
	codebook := q.codebooks[subspace]

	minDist := float32(math.MaxFloat32)
	minIdx := 0

	for k := 0; k < q.config.NumCentroids; k++ {
		centroid := codebook[k*subspaceDim : (k+1)*subspaceDim]
		dist := squaredL2Distance(subvec, centroid)
		if dist < minDist {
			minDist = dist
			minIdx = k
		}
	}

	return minIdx
}

// Decode reconstructs an approximate vector from PQ codes.
func (q *Quantizer) Decode(codes []byte) ([]float32, error) {
	if !q.trained {
		return nil, errors.New("quantizer not trained")
	}
	if len(codes) != q.config.NumSubspaces {
		return nil, fmt.Errorf("codes has length %d, expected %d",
			len(codes), q.config.NumSubspaces)
	}
	if err := q.validateCodes(codes); err != nil {
		return nil, err
	}

	vector := make([]float32, q.config.Dimension)
	subspaceDim := q.config.SubspaceDim()

	for m := 0; m < q.config.NumSubspaces; m++ {
		k := int(codes[m])
		centroid := q.codebooks[m][k*subspaceDim : (k+1)*subspaceDim]
		copy(vector[m*subspaceDim:], centroid)
	}

	return vector, nil
}

// AsymmetricDistance computes the squared L2 distance between a query vector
// and a compressed vector without fully decoding it.
// This is more accurate than symmetric distance (both compressed).
func (q *Quantizer) AsymmetricDistance(query []float32, codes []byte) (float32, error) {
	if !q.trained {
		return 0, errors.New("quantizer not trained")
	}
	if len(query) != q.config.Dimension {
		return 0, fmt.Errorf("query has dimension %d, expected %d",
			len(query), q.config.Dimension)
	}
	if err := validateFiniteVector("query", query); err != nil {
		return 0, err
	}
	if len(codes) != q.config.NumSubspaces {
		return 0, fmt.Errorf("codes has length %d, expected %d",
			len(codes), q.config.NumSubspaces)
	}
	if err := q.validateCodes(codes); err != nil {
		return 0, err
	}

	var totalDist float32
	subspaceDim := q.config.SubspaceDim()

	for m := 0; m < q.config.NumSubspaces; m++ {
		querySub := query[m*subspaceDim : (m+1)*subspaceDim]
		k := int(codes[m])
		centroid := q.codebooks[m][k*subspaceDim : (k+1)*subspaceDim]
		totalDist += squaredL2Distance(querySub, centroid)
	}

	return totalDist, nil
}

// LookupTables precomputes distances from query subvectors to all centroids.
// Shape: [NumSubspaces][NumCentroids]
type LookupTables [][]float32

// BuildLookupTables precomputes distance tables for fast batch search.
// After building, use DistanceWithTables for O(M) distance computation.
func (q *Quantizer) BuildLookupTables(query []float32) (LookupTables, error) {
	if !q.trained {
		return nil, errors.New("quantizer not trained")
	}
	if len(query) != q.config.Dimension {
		return nil, fmt.Errorf("query has dimension %d, expected %d",
			len(query), q.config.Dimension)
	}
	if err := validateFiniteVector("query", query); err != nil {
		return nil, err
	}

	tables := make(LookupTables, q.config.NumSubspaces)
	subspaceDim := q.config.SubspaceDim()

	for m := 0; m < q.config.NumSubspaces; m++ {
		querySub := query[m*subspaceDim : (m+1)*subspaceDim]
		tables[m] = make([]float32, q.config.NumCentroids)

		for k := 0; k < q.config.NumCentroids; k++ {
			centroid := q.codebooks[m][k*subspaceDim : (k+1)*subspaceDim]
			tables[m][k] = squaredL2Distance(querySub, centroid)
		}
	}

	return tables, nil
}

func validateFiniteVector(name string, vector []float32) error {
	for i, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("%s has non-finite value at index %d: %v", name, i, value)
		}
	}
	return nil
}

// DistanceWithTables computes distance using precomputed lookup tables.
// O(M) operations instead of O(M * SubspaceDim).
//
// codes must come from this quantizer's Encode (each value < NumCentroids);
// an out-of-range code panics.
func (q *Quantizer) DistanceWithTables(tables LookupTables, codes []byte) float32 {
	var dist float32
	for m := 0; m < q.config.NumSubspaces; m++ {
		if int(codes[m]) >= q.config.NumCentroids {
			panic(fmt.Sprintf("pq: DistanceWithTables: codes[%d]=%d >= NumCentroids=%d", m, codes[m], q.config.NumCentroids))
		}
		dist += tables[m][codes[m]]
	}
	return dist
}

// Save serializes the quantizer to a writer.
// Format: [config][trained flag][codebooks...]
func (q *Quantizer) Save(w io.Writer) error {
	if err := q.validateState(); err != nil {
		return fmt.Errorf("invalid quantizer state: %w", err)
	}

	// Write config
	if err := binary.Write(w, binary.LittleEndian, int32(q.config.NumSubspaces)); err != nil {
		return fmt.Errorf("write NumSubspaces: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, int32(q.config.NumCentroids)); err != nil {
		return fmt.Errorf("write NumCentroids: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, int32(q.config.Dimension)); err != nil {
		return fmt.Errorf("write Dimension: %w", err)
	}

	// Write trained flag
	var trained int32
	if q.trained {
		trained = 1
	}
	if err := binary.Write(w, binary.LittleEndian, trained); err != nil {
		return fmt.Errorf("write trained flag: %w", err)
	}

	// Write codebooks if trained
	if q.trained {
		for m := 0; m < q.config.NumSubspaces; m++ {
			if err := binary.Write(w, binary.LittleEndian, q.codebooks[m]); err != nil {
				return fmt.Errorf("write codebook %d: %w", m, err)
			}
		}
	}

	return nil
}

// Load deserializes the quantizer from a reader.
func (q *Quantizer) Load(r io.Reader) error {
	// Read config
	var numSubspaces, numCentroids, dimension, trained int32
	if err := binary.Read(r, binary.LittleEndian, &numSubspaces); err != nil {
		return fmt.Errorf("read NumSubspaces: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &numCentroids); err != nil {
		return fmt.Errorf("read NumCentroids: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &dimension); err != nil {
		return fmt.Errorf("read Dimension: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &trained); err != nil {
		return fmt.Errorf("read trained flag: %w", err)
	}
	if trained != 0 && trained != 1 {
		return fmt.Errorf("invalid trained flag: %d", trained)
	}

	loaded := Quantizer{config: Config{
		NumSubspaces: int(numSubspaces),
		NumCentroids: int(numCentroids),
		Dimension:    int(dimension),
	}, trained: trained == 1}

	if err := loaded.config.Validate(); err != nil {
		return fmt.Errorf("invalid loaded config: %w", err)
	}
	if _, err := serializedCodebookBytes(loaded.config); err != nil {
		return fmt.Errorf("invalid loaded config: %w", err)
	}

	// Read codebooks if trained
	if loaded.trained {
		subspaceDim := loaded.config.SubspaceDim()
		loaded.codebooks = make([][]float32, loaded.config.NumSubspaces)
		for m := 0; m < loaded.config.NumSubspaces; m++ {
			loaded.codebooks[m] = make([]float32, loaded.config.NumCentroids*subspaceDim)
			if err := binary.Read(r, binary.LittleEndian, loaded.codebooks[m]); err != nil {
				return fmt.Errorf("read codebook %d: %w", m, err)
			}
			for i, v := range loaded.codebooks[m] {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					return fmt.Errorf("codebook %d: non-finite value at index %d: %v", m, i, v)
				}
			}
		}
	}

	*q = loaded
	return nil
}

func (q *Quantizer) validateCodes(codes []byte) error {
	for m, code := range codes {
		if int(code) >= q.config.NumCentroids {
			return fmt.Errorf("codes[%d]=%d must be less than NumCentroids %d", m, code, q.config.NumCentroids)
		}
	}
	return nil
}

func (q *Quantizer) validateState() error {
	if err := q.config.Validate(); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	if _, err := serializedCodebookBytes(q.config); err != nil {
		return err
	}
	if !q.trained {
		if len(q.codebooks) != 0 {
			return errors.New("untrained quantizer has codebooks")
		}
		return nil
	}
	if len(q.codebooks) != q.config.NumSubspaces {
		return fmt.Errorf("codebook count %d, expected %d", len(q.codebooks), q.config.NumSubspaces)
	}
	expectedLen := q.config.NumCentroids * q.config.SubspaceDim()
	for m, codebook := range q.codebooks {
		if len(codebook) != expectedLen {
			return fmt.Errorf("codebook %d has length %d, expected %d", m, len(codebook), expectedLen)
		}
		for i, value := range codebook {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("codebook %d has non-finite value at index %d: %v", m, i, value)
			}
		}
	}
	return nil
}

func serializedCodebookBytes(config Config) (int64, error) {
	if config.NumSubspaces > maxSerializedSubspaces {
		return 0, fmt.Errorf("serialized codebooks exceed %d-subspace limit", maxSerializedSubspaces)
	}
	bytesPerDimension := int64(config.NumCentroids) * 4
	if int64(config.Dimension) > maxSerializedCodebookBytes/bytesPerDimension {
		return 0, fmt.Errorf("serialized codebooks exceed %d-byte limit", maxSerializedCodebookBytes)
	}
	bytes := int64(config.Dimension) * bytesPerDimension
	return bytes, nil
}

// squaredL2Distance computes the squared Euclidean distance between two vectors.
// Returns math.MaxFloat32 if the vectors have different lengths.
func squaredL2Distance(a, b []float32) float32 {
	if len(a) != len(b) {
		return math.MaxFloat32
	}
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// CompressionRatio returns the compression ratio achieved by PQ.
// For 1536-dim vectors with 96 subspaces: 6144 bytes -> 96 bytes = 64x.
func (c Config) CompressionRatio() float64 {
	originalBytes := c.Dimension * 4 // float32 = 4 bytes
	compressedBytes := c.NumSubspaces
	if compressedBytes == 0 {
		// Unvalidated/zero-value Config (NumSubspaces==0) has no compression configured.
		return 0
	}
	return float64(originalBytes) / float64(compressedBytes)
}

// EstimatedAccuracy returns the estimated accuracy based on configuration.
// Higher NumCentroids and NumSubspaces generally improve accuracy.
func (c Config) EstimatedAccuracy() float64 {
	// Empirical estimate: accuracy improves with more centroids and finer subspaces
	// 256 centroids * 96 subspaces = ~95% recall@10
	// This is a rough estimate; actual accuracy depends on data distribution.
	centroidFactor := float64(c.NumCentroids) / 256.0
	subspaceFactor := float64(c.NumSubspaces) / 96.0
	return math.Min(0.99, 0.85+0.10*centroidFactor+0.04*subspaceFactor)
}
