package vectors

import (
	"container/heap"
	"encoding/binary"
	"fmt"
	"math"
	"slices"
)

// EncodeFloat32Vec encodes a []float32 as a raw little-endian byte slice.
// Each float32 occupies 4 bytes; 512 dimensions × 4 bytes = 2048 bytes per vector.
func EncodeFloat32Vec(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// EncodeFloat64Vec encodes a []float64 as a raw little-endian byte slice.
// Each float64 occupies 8 bytes.
func EncodeFloat64Vec(v []float64) []byte {
	if v == nil {
		return nil
	}
	buf := make([]byte, len(v)*8)
	for i, f := range v {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(f))
	}
	return buf
}

// DecodeFloat32Vec decodes a raw little-endian byte slice into []float32.
// Returns nil if len(blob) is not a multiple of 4.
func DecodeFloat32Vec(blob []byte) []float32 {
	out, err := DecodeFloat32Batch([][]byte{blob})
	if err != nil {
		return nil
	}
	return out[0]
}

// DecodeFloat64Vec decodes a raw little-endian byte slice into []float64.
// Returns nil when len(data) is not a multiple of 8.
func DecodeFloat64Vec(data []byte) []float64 {
	if len(data) == 0 {
		return make([]float64, 0)
	}
	if len(data)%8 != 0 {
		return nil
	}
	v := make([]float64, len(data)/8)
	for i := range v {
		v[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]))
	}
	return v
}

// DecodeFloat32Batch decodes a slice of little-endian float32 blobs, preserving
// input order: out[i] corresponds to blobs[i]. It fails fast with an error naming
// the first blob whose length is not a multiple of 4.
func DecodeFloat32Batch(blobs [][]byte) ([][]float32, error) {
	out := make([][]float32, len(blobs))
	for i, blob := range blobs {
		if len(blob)%4 != 0 {
			return nil, fmt.Errorf("vectors: blob at index %d has length %d, not a multiple of 4", i, len(blob))
		}
		v := make([]float32, len(blob)/4)
		for j := range v {
			v[j] = math.Float32frombits(binary.LittleEndian.Uint32(blob[j*4:]))
		}
		out[i] = v
	}
	return out, nil
}

// DotFromBlob computes the dot product between a pre-normalized query vector
// and a raw little-endian float32 BLOB without intermediate allocation.
// Both vectors MUST be L2-normalized (as guaranteed by the embedder); under
// that invariant, dot product == cosine similarity.
// Returns 0 if dimensions do not match.
func DotFromBlob(query []float32, blob []byte) float32 {
	if len(blob) != len(query)*4 {
		return 0
	}
	var dot float32
	for i, q := range query {
		b := math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
		dot += q * b
	}
	return dot
}

// ScoredIndex identifies one scored input row.
type ScoredIndex struct {
	Index int
	Score float32
}

// TopKFromBlob scores raw little-endian float32 blobs against query and returns
// the top limit input indexes. Invalid or dimension-mismatched blobs are
// skipped. Equal scores are ordered by ascending input index.
func TopKFromBlob(query []float32, blobs [][]byte, limit int, minScore float32) []ScoredIndex {
	if len(query) == 0 || limit <= 0 || len(blobs) == 0 {
		return nil
	}

	h := &idxHeap{max: true}
	expectedLen := len(query) * 4
	for i, blob := range blobs {
		if len(blob) != expectedLen {
			continue
		}
		score := DotFromBlob(query, blob)
		if score < minScore {
			continue
		}
		item := idxItem{index: i, score: score}
		if h.Len() < limit {
			heap.Push(h, item)
			continue
		}
		if shouldReplace(score, i, h.Peek(), true) {
			heap.Pop(h)
			heap.Push(h, item)
		}
	}

	out := make([]ScoredIndex, 0, h.Len())
	for h.Len() > 0 {
		item := heap.Pop(h).(idxItem)
		out = append(out, ScoredIndex{Index: item.index, Score: item.score})
	}
	slices.Reverse(out)
	return out
}
