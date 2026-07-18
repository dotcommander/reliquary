package embed

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/dotcommander/reliquary/embedding"
)

func TestNewHashingInvalidDimensionsUseDefault(t *testing.T) {
	t.Parallel()
	for _, dim := range []int{0, -7} {
		h := NewHashing(dim)
		if h.Model.Dim != DefaultHashingDim {
			t.Fatalf("NewHashing(%d) dim = %d, want %d", dim, h.Model.Dim, DefaultHashingDim)
		}
		v := HashVector("hello world", dim)
		if len(v) != DefaultHashingDim {
			t.Fatalf("HashVector dim = %d, want %d", len(v), DefaultHashingDim)
		}
	}
}

func TestHashingEmptyText(t *testing.T) {
	t.Parallel()
	const dim = 16
	v := HashVector("", dim)
	if len(v) != dim {
		t.Fatalf("empty text dim = %d, want %d", len(v), dim)
	}
	for i, x := range v {
		if x != 0 {
			t.Fatalf("empty text vector[%d] = %v, want 0", i, x)
		}
	}
}

func TestHashingDeterministicAndNormalized(t *testing.T) {
	t.Parallel()
	const dim = 64
	a := HashVector("Go garbage collector reclaims memory", dim)
	b := HashVector("Go garbage collector reclaims memory", dim)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("HashVector returned non-deterministic vectors")
	}
	if !almostUnit(a) {
		t.Fatalf("HashVector norm = %.6f, want ~1", norm(a))
	}
}

func TestHashingEmbedCancellation(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := NewHashing(8).Embed(ctx, embeddings.Request{Inputs: []string{"x"}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Embed canceled error = %v, want context.Canceled", err)
	}
}

func TestHashingEmbedResult(t *testing.T) {
	t.Parallel()
	h := NewHashing(8)
	got, err := h.Embed(context.Background(), embeddings.Request{Inputs: []string{"alpha", ""}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if got.Model != h.Model {
		t.Fatalf("Embed model = %#v, want %#v", got.Model, h.Model)
	}
	if len(got.Vectors) != 2 {
		t.Fatalf("Embed vectors = %d, want 2", len(got.Vectors))
	}
	if len(got.Vectors[0]) != 8 || len(got.Vectors[1]) != 8 {
		t.Fatalf("Embed vector dims = %d/%d, want 8/8", len(got.Vectors[0]), len(got.Vectors[1]))
	}
}

func almostUnit(v embeddings.Vector) bool {
	return math.Abs(norm(v)-1) < 1e-5
}

func norm(v embeddings.Vector) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}
