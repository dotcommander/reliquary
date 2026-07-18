package vectors

import (
	"math/rand/v2"
	"testing"
)

// benchVec returns a deterministic float32 vector of the given dimension,
// filled with increasing small values.
func benchVec(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(i+1) * 0.01
	}
	return v
}

// benchNormalized returns a normalized copy of a benchVec.
func benchNormalized(n int) []float32 {
	v := benchVec(n)
	Normalize32(v)
	return v
}

// --- Cosine32 ---

func BenchmarkCosine32_512(b *testing.B) {
	a := benchNormalized(512)
	c := benchNormalized(512)
	b.ResetTimer()
	for b.Loop() {
		Cosine32(a, c)
	}
}

func BenchmarkCosine32_768(b *testing.B) {
	a := benchNormalized(768)
	c := benchNormalized(768)
	b.ResetTimer()
	for b.Loop() {
		Cosine32(a, c)
	}
}

func BenchmarkCosineDistance32_512(b *testing.B) {
	a := benchNormalized(512)
	bVec := benchNormalized(512)
	b.ResetTimer()
	for b.Loop() {
		CosineDistance32(a, bVec)
	}
}

func BenchmarkCosineDistance32_768(b *testing.B) {
	a := benchNormalized(768)
	bVec := benchNormalized(768)
	b.ResetTimer()
	for b.Loop() {
		CosineDistance32(a, bVec)
	}
}

// --- Normalize32 ---

// --- Norm & unit helpers ---

func BenchmarkNormSquared32_768(b *testing.B) {
	v := benchVec(768)
	for b.Loop() {
		NormSquared32(v)
	}
}

func BenchmarkIsUnit32_768(b *testing.B) {
	v := benchVec(768)
	Normalize32(v)
	for b.Loop() {
		IsUnit32(v, 1e-6)
	}
}

func BenchmarkNormalize32_512(b *testing.B) {
	for b.Loop() {
		v := benchVec(512)
		Normalize32(v)
	}
}

func BenchmarkNormalize32_768(b *testing.B) {
	for b.Loop() {
		v := benchVec(768)
		Normalize32(v)
	}
}

// --- DotFromBlob ---

func BenchmarkDotFromBlob_512(b *testing.B) {
	query := benchNormalized(512)
	blob := EncodeFloat32Vec(benchNormalized(512))
	b.ResetTimer()
	for b.Loop() {
		DotFromBlob(query, blob)
	}
}

func BenchmarkDotFromBlob_768(b *testing.B) {
	query := benchNormalized(768)
	blob := EncodeFloat32Vec(benchNormalized(768))
	b.ResetTimer()
	for b.Loop() {
		DotFromBlob(query, blob)
	}
}

// --- Quantize ---

func BenchmarkQuantize_512(b *testing.B) {
	vec := benchVec(512)
	thresholds := make([]float32, 512)
	for i := range thresholds {
		thresholds[i] = 0.5
	}
	b.ResetTimer()
	for b.Loop() {
		Quantize(vec, thresholds)
	}
}

func BenchmarkQuantize_768(b *testing.B) {
	vec := benchVec(768)
	thresholds := make([]float32, 768)
	for i := range thresholds {
		thresholds[i] = 0.5
	}
	b.ResetTimer()
	for b.Loop() {
		Quantize(vec, thresholds)
	}
}

func BenchmarkWeightedMeanPool32_512x8(b *testing.B) {
	vecs := make([][]float32, 8)
	weights := make([]float64, 8)
	for i := range vecs {
		vecs[i] = benchNormalized(512)
		weights[i] = 1
	}
	b.ResetTimer()
	for b.Loop() {
		if _, err := WeightedMeanPool32(vecs, weights); err != nil {
			b.Fatalf("BenchmarkWeightedMeanPool32_512x8: WeightedMeanPool32 returned err=%v", err)
		}
	}
}

func BenchmarkWeightedMeanPool32_768x8(b *testing.B) {
	vecs := make([][]float32, 8)
	weights := make([]float64, 8)
	for i := range vecs {
		vecs[i] = benchNormalized(768)
		weights[i] = 1
	}
	b.ResetTimer()
	for b.Loop() {
		if _, err := WeightedMeanPool32(vecs, weights); err != nil {
			b.Fatalf("BenchmarkWeightedMeanPool32_768x8: WeightedMeanPool32 returned err=%v", err)
		}
	}
}

func BenchmarkQuantizeInto_512(b *testing.B) {
	vec := benchVec(512)
	thresholds := make([]float32, 512)
	for i := range thresholds {
		thresholds[i] = 0.5
	}
	dst := make(BinaryVector, BinaryWords(512))
	b.ResetTimer()
	for b.Loop() {
		if err := QuantizeInto(dst, vec, thresholds); err != nil {
			b.Fatalf("BenchmarkQuantizeInto_512: QuantizeInto returned err=%v", err)
		}
	}
}

func BenchmarkQuantizeInto_768(b *testing.B) {
	vec := benchVec(768)
	thresholds := make([]float32, 768)
	for i := range thresholds {
		thresholds[i] = 0.5
	}
	dst := make(BinaryVector, BinaryWords(768))
	b.ResetTimer()
	for b.Loop() {
		if err := QuantizeInto(dst, vec, thresholds); err != nil {
			b.Fatalf("BenchmarkQuantizeInto_768: QuantizeInto returned err=%v", err)
		}
	}
}

// --- HammingDistance ---

func BenchmarkHammingDistance_768(b *testing.B) {
	// 768 dims = 12 uint64s
	rng := rand.New(rand.NewPCG(42, 0))
	a := make(BinaryVector, 12)
	c := make(BinaryVector, 12)
	for i := range a {
		a[i] = rng.Uint64()
		c[i] = rng.Uint64()
	}
	b.ResetTimer()
	for b.Loop() {
		HammingDistance(a, c)
	}
}

func BenchmarkHammingDistance_1024(b *testing.B) {
	// 1024 dims = 16 uint64s
	rng := rand.New(rand.NewPCG(42, 0))
	a := make(BinaryVector, 16)
	c := make(BinaryVector, 16)
	for i := range a {
		a[i] = rng.Uint64()
		c[i] = rng.Uint64()
	}
	b.ResetTimer()
	for b.Loop() {
		HammingDistance(a, c)
	}
}

// --- ComputeMedians ---

func BenchmarkComputeMedians_768(b *testing.B) {
	const nVecs = 100
	vecs := make([][]float32, nVecs)
	for i := range vecs {
		vecs[i] = benchVec(768)
	}
	b.ResetTimer()
	for b.Loop() {
		ComputeMedians(vecs)
	}
}
