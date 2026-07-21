package pq

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"
)

func TestConfigValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		config  Config
		wantErr bool
	}{
		{
			name:    "valid default config",
			config:  DefaultConfig(),
			wantErr: false,
		},
		{
			name: "valid 768-dim config",
			config: Config{
				NumSubspaces: 48,
				NumCentroids: 256,
				Dimension:    768,
			},
			wantErr: false,
		},
		{
			name: "zero subspaces",
			config: Config{
				NumSubspaces: 0,
				NumCentroids: 256,
				Dimension:    1536,
			},
			wantErr: true,
		},
		{
			name: "zero centroids",
			config: Config{
				NumSubspaces: 96,
				NumCentroids: 0,
				Dimension:    1536,
			},
			wantErr: true,
		},
		{
			name: "too many centroids",
			config: Config{
				NumSubspaces: 96,
				NumCentroids: 257,
				Dimension:    1536,
			},
			wantErr: true,
		},
		{
			name: "dimension not divisible",
			config: Config{
				NumSubspaces: 96,
				NumCentroids: 256,
				Dimension:    1000,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestSubspaceDim(t *testing.T) {
	t.Parallel()

	config := Config{
		NumSubspaces: 96,
		NumCentroids: 256,
		Dimension:    1536,
	}
	if got := config.SubspaceDim(); got != 16 {
		t.Errorf("SubspaceDim() = %d, want 16", got)
	}
}

func TestNewQuantizer(t *testing.T) {
	t.Parallel()

	q, err := NewQuantizer(DefaultConfig())
	if err != nil {
		t.Fatalf("NewQuantizer() error = %v", err)
	}
	if q.IsTrained() {
		t.Error("New quantizer should not be trained")
	}
}

func TestNewQuantizerInvalidConfig(t *testing.T) {
	t.Parallel()

	_, err := NewQuantizer(Config{NumSubspaces: 0})
	if err == nil {
		t.Error("NewQuantizer() should fail with invalid config")
	}
}

// generateRandomVectors creates random vectors for testing.
func generateRandomVectors(n, dim int) [][]float32 {
	vectors := make([][]float32, n)
	for i := range vectors {
		vectors[i] = make([]float32, dim)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()*2 - 1 // [-1, 1]
		}
	}
	return vectors
}

// generateClusteredVectors creates vectors clustered around centers.
func generateClusteredVectors(nClusters, pointsPerCluster, dim int) [][]float32 {
	// Generate cluster centers
	centers := make([][]float32, nClusters)
	for i := range centers {
		centers[i] = make([]float32, dim)
		for j := range centers[i] {
			centers[i][j] = rand.Float32()*10 - 5 // [-5, 5]
		}
	}

	// Generate points around centers
	vectors := make([][]float32, 0, nClusters*pointsPerCluster)
	for _, center := range centers {
		for p := 0; p < pointsPerCluster; p++ {
			vec := make([]float32, dim)
			for j := range vec {
				vec[j] = center[j] + rand.Float32()*0.5 - 0.25 // Small noise
			}
			vectors = append(vectors, vec)
		}
	}
	return vectors
}

func TestTrainAndEncode(t *testing.T) {
	t.Parallel()

	// Use smaller config for faster tests
	config := Config{
		NumSubspaces: 8,
		NumCentroids: 16,
		Dimension:    64,
	}

	q, err := NewQuantizer(config)
	if err != nil {
		t.Fatalf("NewQuantizer() error = %v", err)
	}

	// Generate clustered training data
	vectors := generateClusteredVectors(16, 50, 64)

	// Train
	if err := q.Train(vectors); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	if !q.IsTrained() {
		t.Error("Quantizer should be trained after Train()")
	}

	// Encode a vector
	testVec := vectors[0]
	codes, err := q.Encode(testVec)
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}

	if len(codes) != config.NumSubspaces {
		t.Errorf("Encode() returned %d codes, want %d", len(codes), config.NumSubspaces)
	}
}

func TestEncodeDecodeRoundtrip(t *testing.T) {
	t.Parallel()

	config := Config{
		NumSubspaces: 8,
		NumCentroids: 32,
		Dimension:    64,
	}

	q, err := NewQuantizer(config)
	if err != nil {
		t.Fatalf("NewQuantizer() error = %v", err)
	}

	vectors := generateClusteredVectors(32, 30, 64)
	if err := q.Train(vectors); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	// Encode and decode, check reconstruction error
	for i := 0; i < 10; i++ {
		original := vectors[i]
		codes, err := q.Encode(original)
		if err != nil {
			t.Fatalf("Encode() error = %v", err)
		}

		reconstructed, err := q.Decode(codes)
		if err != nil {
			t.Fatalf("Decode() error = %v", err)
		}

		// Check reconstruction error is reasonable
		mse := meanSquaredError(original, reconstructed)
		if mse > 1.0 { // Reasonable threshold for clustered data
			t.Errorf("Reconstruction MSE = %f, too high", mse)
		}
	}
}

func TestAsymmetricDistance(t *testing.T) {
	t.Parallel()

	config := Config{
		NumSubspaces: 8,
		NumCentroids: 32,
		Dimension:    64,
	}

	q, err := NewQuantizer(config)
	if err != nil {
		t.Fatalf("NewQuantizer() error = %v", err)
	}

	vectors := generateClusteredVectors(32, 30, 64)
	if err := q.Train(vectors); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	query := vectors[0]
	codes, _ := q.Encode(vectors[1])

	dist, err := q.AsymmetricDistance(query, codes)
	if err != nil {
		t.Fatalf("AsymmetricDistance() error = %v", err)
	}

	if dist < 0 {
		t.Error("AsymmetricDistance should be non-negative")
	}

	// Compare with distance using lookup tables
	tables, err := q.BuildLookupTables(query)
	if err != nil {
		t.Fatalf("BuildLookupTables() error = %v", err)
	}

	tableDist := q.DistanceWithTables(tables, codes)
	if math.Abs(float64(dist-tableDist)) > 1e-5 {
		t.Errorf("DistanceWithTables() = %f, AsymmetricDistance() = %f", tableDist, dist)
	}
}

func TestSaveLoad(t *testing.T) {
	t.Parallel()

	config := Config{
		NumSubspaces: 4,
		NumCentroids: 8,
		Dimension:    32,
	}

	q, err := NewQuantizer(config)
	if err != nil {
		t.Fatalf("NewQuantizer() error = %v", err)
	}

	vectors := generateClusteredVectors(8, 20, 32)
	if err := q.Train(vectors); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	// Save
	var buf bytes.Buffer
	if err := q.Save(&buf); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	// Load into new quantizer
	q2 := &Quantizer{}
	if err := q2.Load(&buf); err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	// Verify config matches
	if q2.config != q.config {
		t.Errorf("Loaded config = %+v, want %+v", q2.config, q.config)
	}

	if q2.trained != q.trained {
		t.Errorf("Loaded trained = %v, want %v", q2.trained, q.trained)
	}

	// Verify encoding produces same results
	testVec := vectors[0]
	codes1, _ := q.Encode(testVec)
	codes2, _ := q2.Encode(testVec)

	if !bytes.Equal(codes1, codes2) {
		t.Error("Loaded quantizer produces different codes")
	}
}

func TestUntrainedQuantizer(t *testing.T) {
	t.Parallel()

	q, _ := NewQuantizer(DefaultConfig())

	vec := make([]float32, 1536)
	codes := make([]byte, 96)

	if _, err := q.Encode(vec); err == nil {
		t.Error("Encode() should fail on untrained quantizer")
	}

	if _, err := q.Decode(codes); err == nil {
		t.Error("Decode() should fail on untrained quantizer")
	}

	if _, err := q.AsymmetricDistance(vec, codes); err == nil {
		t.Error("AsymmetricDistance() should fail on untrained quantizer")
	}

	if _, err := q.BuildLookupTables(vec); err == nil {
		t.Error("BuildLookupTables() should fail on untrained quantizer")
	}
}

func TestWrongDimensionErrors(t *testing.T) {
	t.Parallel()

	config := Config{
		NumSubspaces: 8,
		NumCentroids: 16,
		Dimension:    64,
	}

	q, _ := NewQuantizer(config)
	vectors := generateClusteredVectors(16, 50, 64)
	_ = q.Train(vectors)

	// Wrong vector dimension
	wrongVec := make([]float32, 32)
	if _, err := q.Encode(wrongVec); err == nil {
		t.Error("Encode() should fail with wrong dimension")
	}

	// Wrong codes length
	wrongCodes := make([]byte, 4)
	if _, err := q.Decode(wrongCodes); err == nil {
		t.Error("Decode() should fail with wrong codes length")
	}

	// Wrong query dimension for asymmetric distance
	codes, _ := q.Encode(vectors[0])
	if _, err := q.AsymmetricDistance(wrongVec, codes); err == nil {
		t.Error("AsymmetricDistance() should fail with wrong query dimension")
	}
}

func TestRuntimeOperationsRejectNonFiniteVectors(t *testing.T) {
	t.Parallel()

	q := &Quantizer{
		config:    Config{NumSubspaces: 1, NumCentroids: 2, Dimension: 2},
		codebooks: [][]float32{{0, 0, 1, 1}},
		trained:   true,
	}
	codes := []byte{0}

	for name, value := range map[string]float32{
		"nan":               float32(math.NaN()),
		"positive infinity": float32(math.Inf(1)),
		"negative infinity": float32(math.Inf(-1)),
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			vector := []float32{value, 0}
			if _, err := q.Encode(vector); err == nil {
				t.Fatal("Encode() error = nil, want non-finite-vector error")
			}
			if _, err := q.AsymmetricDistance(vector, codes); err == nil {
				t.Fatal("AsymmetricDistance() error = nil, want non-finite-query error")
			}
			if _, err := q.BuildLookupTables(vector); err == nil {
				t.Fatal("BuildLookupTables() error = nil, want non-finite-query error")
			}
		})
	}
}

func TestPQCodesAreValidated(t *testing.T) {
	t.Parallel()

	q := &Quantizer{
		config:    Config{NumSubspaces: 2, NumCentroids: 2, Dimension: 2},
		codebooks: [][]float32{{0, 1}, {0, 1}},
		trained:   true,
	}
	codes := []byte{0, 2}
	if _, err := q.Decode(codes); err == nil {
		t.Fatal("Decode() error = nil, want invalid-code error")
	}
	if _, err := q.AsymmetricDistance([]float32{0, 0}, codes); err == nil {
		t.Fatal("AsymmetricDistance() error = nil, want invalid-code error")
	}

	tables := LookupTables{{0, 1}, {0, 1}}
	defer func() {
		if recover() == nil {
			t.Fatal("DistanceWithTables() did not preserve invalid-code panic contract")
		}
	}()
	q.DistanceWithTables(tables, codes)
}

func TestLoadRejectsMalformedStateAtomically(t *testing.T) {
	t.Parallel()

	q := &Quantizer{
		config:    Config{NumSubspaces: 1, NumCentroids: 2, Dimension: 2},
		codebooks: [][]float32{{0, 0, 1, 1}},
		trained:   true,
	}
	before := savedQuantizer(t, q)

	tests := []struct {
		name string
		data []byte
	}{
		{name: "truncated codebook", data: pqStateBytes(t, 1, 2, 2, 1, []float32{0})},
		{name: "oversized header", data: pqStateBytes(t, 1, 256, 262145, 1, nil)},
		{name: "excessive subspace allocations", data: pqStateBytes(t, 1<<26, 1, 1<<26, 1, nil)},
		{name: "invalid trained flag", data: pqStateBytes(t, 1, 2, 2, 2, nil)},
		{name: "non-finite codebook", data: pqStateBytes(t, 1, 2, 2, 1, []float32{0, 0, 1, float32(math.Inf(1))})},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := q.Load(bytes.NewReader(tt.data)); err == nil {
				t.Fatal("Load() error = nil, want malformed-state error")
			}
			if after := savedQuantizer(t, q); !bytes.Equal(after, before) {
				t.Fatalf("Load() mutated receiver on failure\nbefore: %x\nafter:  %x", before, after)
			}
		})
	}
}

func TestSaveRejectsMalformedOrOversizedState(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		q    *Quantizer
	}{
		{
			name: "missing trained codebooks",
			q:    &Quantizer{config: Config{NumSubspaces: 1, NumCentroids: 2, Dimension: 2}, trained: true},
		},
		{
			name: "untrained with codebooks",
			q: &Quantizer{
				config:    Config{NumSubspaces: 1, NumCentroids: 2, Dimension: 2},
				codebooks: [][]float32{{0, 0, 1, 1}},
			},
		},
		{
			name: "oversized codebooks",
			q:    &Quantizer{config: Config{NumSubspaces: 1, NumCentroids: 256, Dimension: 262145}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.q.Save(&bytes.Buffer{}); err == nil {
				t.Fatal("Save() error = nil, want invalid-state error")
			}
		})
	}
}

func savedQuantizer(t *testing.T, q *Quantizer) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := q.Save(&buf); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	return buf.Bytes()
}

func pqStateBytes(t *testing.T, subspaces, centroids, dimension, trained int32, codebooks []float32) []byte {
	t.Helper()
	var buf bytes.Buffer
	for _, field := range []int32{subspaces, centroids, dimension, trained} {
		if err := binary.Write(&buf, binary.LittleEndian, field); err != nil {
			t.Fatalf("write state header: %v", err)
		}
	}
	if err := binary.Write(&buf, binary.LittleEndian, codebooks); err != nil {
		t.Fatalf("write state codebooks: %v", err)
	}
	return buf.Bytes()
}

func TestInsufficientTrainingVectors(t *testing.T) {
	t.Parallel()

	config := Config{
		NumSubspaces: 8,
		NumCentroids: 256,
		Dimension:    64,
	}

	q, _ := NewQuantizer(config)
	vectors := generateRandomVectors(100, 64) // Less than 256 centroids

	if err := q.Train(vectors); err == nil {
		t.Error("Train() should fail with insufficient vectors")
	}
}

func TestCompressionRatio(t *testing.T) {
	t.Parallel()

	config := DefaultConfig()
	ratio := config.CompressionRatio()

	// 1536 * 4 bytes = 6144 bytes original
	// 96 bytes compressed
	// Ratio = 64
	expected := 64.0
	if math.Abs(ratio-expected) > 0.1 {
		t.Errorf("CompressionRatio() = %f, want %f", ratio, expected)
	}
}

// Benchmark tests

func BenchmarkEncode(b *testing.B) {
	config := Config{
		NumSubspaces: 96,
		NumCentroids: 256,
		Dimension:    1536,
	}

	q, _ := NewQuantizer(config)
	vectors := generateRandomVectors(1000, 1536)
	_ = q.Train(vectors)

	testVec := vectors[0]

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = q.Encode(testVec)
	}
}

func BenchmarkDecode(b *testing.B) {
	config := Config{
		NumSubspaces: 96,
		NumCentroids: 256,
		Dimension:    1536,
	}

	q, _ := NewQuantizer(config)
	vectors := generateRandomVectors(1000, 1536)
	_ = q.Train(vectors)

	codes, _ := q.Encode(vectors[0])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = q.Decode(codes)
	}
}

func BenchmarkAsymmetricDistance(b *testing.B) {
	config := Config{
		NumSubspaces: 96,
		NumCentroids: 256,
		Dimension:    1536,
	}

	q, _ := NewQuantizer(config)
	vectors := generateRandomVectors(1000, 1536)
	_ = q.Train(vectors)

	query := vectors[0]
	codes, _ := q.Encode(vectors[1])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = q.AsymmetricDistance(query, codes)
	}
}

func BenchmarkDistanceWithTables(b *testing.B) {
	config := Config{
		NumSubspaces: 96,
		NumCentroids: 256,
		Dimension:    1536,
	}

	q, _ := NewQuantizer(config)
	vectors := generateRandomVectors(1000, 1536)
	_ = q.Train(vectors)

	query := vectors[0]
	codes, _ := q.Encode(vectors[1])
	tables, _ := q.BuildLookupTables(query)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q.DistanceWithTables(tables, codes)
	}
}

func BenchmarkKMeans(b *testing.B) {
	vectors := generateRandomVectors(1000, 16)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = KMeans(vectors, 256, 25)
	}
}

// Helper functions

func meanSquaredError(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum / float32(len(a))
}

func TestConfigEstimatedAccuracyAndFarthestPoint(t *testing.T) {
	t.Parallel()

	cfg := DefaultConfig()
	q, err := NewQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewQuantizer failed: %v", err)
	}

	if q.Config() != cfg {
		t.Fatalf("q.Config() = %+v, want %+v", q.Config(), cfg)
	}

	acc := cfg.EstimatedAccuracy()
	if acc <= 0 || acc > 1 {
		t.Fatalf("EstimatedAccuracy = %v, want (0, 1]", acc)
	}

	zeroCfg := Config{NumSubspaces: 0}
	if zeroCfg.CompressionRatio() != 0 {
		t.Fatalf("zeroCfg CompressionRatio = %v, want 0", zeroCfg.CompressionRatio())
	}

	// Test findFarthestPoint
	vectors := [][]float32{{0, 0}, {1, 0}, {10, 10}}
	centroids := [][]float32{{0, 0}}
	farthest := findFarthestPoint(vectors, centroids)
	if len(farthest) != 2 || farthest[0] != 10 || farthest[1] != 10 {
		t.Fatalf("findFarthestPoint = %v, want [10, 10]", farthest)
	}
}
