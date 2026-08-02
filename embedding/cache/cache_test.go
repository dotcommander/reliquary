package cache

import (
	"context"
	"errors"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/embedding/embeddingtest"
	"reflect"
	"testing"
)

var testModel = embedding.ModelRef{Provider: "test", Name: "vectors", Version: "v1", Revision: "r1", Dim: 2}

func TestNewValidation(t *testing.T) {
	t.Parallel()
	base := &fakeEmbedder{}
	store := newMemoryStore()

	tests := []struct {
		name  string
		base  embedding.Embedder
		store Store
		cfg   Config
		want  error
	}{
		{name: "nil embedder", store: store, cfg: Config{Identity: "cache"}, want: ErrNilEmbedder},
		{name: "typed nil embedder", base: (*fakeEmbedder)(nil), store: store, cfg: Config{Identity: "cache"}, want: ErrNilEmbedder},
		{name: "nil store", base: base, cfg: Config{Identity: "cache"}, want: ErrNilStore},
		{name: "typed nil store", base: base, store: (*memoryStore)(nil), cfg: Config{Identity: "cache"}, want: ErrNilStore},
		{name: "blank identity", base: base, store: store, cfg: Config{Identity: " \t\n "}, want: ErrInvalidIdentity},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, err := New(test.base, test.store, test.cfg)
			if !errors.Is(err, test.want) {
				t.Fatalf("New() error = %v, want errors.Is(_, %v)", err, test.want)
			}
		})
	}
	if got := base.callCount(); got != 0 {
		t.Fatalf("base calls = %d, want 0", got)
	}
	if got := store.getCount() + store.setCount(); got != 0 {
		t.Fatalf("store calls = %d, want 0", got)
	}

	if _, err := New(nil, nil, Config{}); !errors.Is(err, ErrNilEmbedder) {
		t.Fatalf("New(all invalid) error = %v, want embedder validation first", err)
	}
	if _, err := New(base, nil, Config{}); !errors.Is(err, ErrNilStore) {
		t.Fatalf("New(nil store and blank identity) error = %v, want Store validation first", err)
	}

	const identity = " cache identity "
	cached, err := New(base, store, Config{Identity: identity})
	if err != nil {
		t.Fatalf("New(valid): %v", err)
	}
	if cached.identity != identity {
		t.Fatalf("retained identity = %q, want byte-for-byte %q", cached.identity, identity)
	}
}

func TestCacheKeyGolden(t *testing.T) {
	t.Parallel()
	model := embedding.ModelRef{Provider: "p", Name: "n", Version: "v", Revision: "r", Dim: 2}
	tests := []struct {
		name, identity, input, want string
		model                       embedding.ModelRef
	}{
		{"exact input", "namespace", "input", "1d135e623437b225814ab15fa39689821f49e60daa024f8c4873d1a5709fe563", model},
		{"unicode byte framing", "aé", "é", "d980c9fffa5abfb64eb81caaa10f93ad776329e67b48192dcbc51a32a367b298", model},
		{"zero model", "x", "", "45606cda8b0cde025f459d15c89f810e79842de0f803da2da880f3f876a4a6db", embedding.ModelRef{}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if got := cacheKey(test.identity, test.model, test.input); got != test.want {
				t.Fatalf("cacheKey() = %q, want %q", got, test.want)
			}
		})
	}

	if cacheKey("ab", model, "c") == cacheKey("a", model, "bc") {
		t.Fatal("cacheKey() collided for ambiguous unframed identity/input values")
	}
	if cacheKey("namespace", model, "input") == cacheKey("namespace", model, "input ") {
		t.Fatal("cacheKey() ignored exact input bytes")
	}
	if cacheKey("namespace", model, "input") == cacheKey("other", model, "input") {
		t.Fatal("cacheKey() ignored cache identity")
	}
	if cacheKey("namespace", model, "input") == cacheKey("namespace", embedding.ModelRef{}, "input") {
		t.Fatal("cacheKey() ignored explicit request model")
	}
}

func TestEmbedderContract(t *testing.T) {
	t.Parallel()
	embeddingtest.Run(t, func() embedding.Embedder {
		cached, err := New(&fakeEmbedder{}, newMemoryStore(), Config{Identity: "contract:v1"})
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return cached
	})
}

func TestEmbedEmptyRequest(t *testing.T) {
	t.Parallel()
	t.Run("success delegates without store access", func(t *testing.T) {
		t.Parallel()
		baseVectors := make([]embedding.Vector, 0, 1)
		base := &fakeEmbedder{response: func(embedding.Request) (embedding.Result, error) {
			return embedding.Result{Model: testModel, Vectors: baseVectors}, nil
		}}
		store := newMemoryStore()
		result := mustEmbed(t, mustNew(t, base, store), embedding.Request{})
		if got := base.callCount(); got != 1 {
			t.Fatalf("base calls = %d, want 1", got)
		}
		if got := store.getCount() + store.setCount(); got != 0 {
			t.Fatalf("store calls = %d, want 0", got)
		}
		if result.Model != testModel || len(result.Vectors) != 0 {
			t.Fatalf("empty result = %#v, want resolved model with no vectors", result)
		}
		if cap(result.Vectors) != 0 {
			t.Fatalf("empty result capacity = %d, want cloned zero-capacity slice", cap(result.Vectors))
		}
	})

	t.Run("base error", func(t *testing.T) {
		t.Parallel()
		baseErr := errors.New("empty base failed")
		result, err := mustNew(t, &fakeEmbedder{err: baseErr}, newMemoryStore()).Embed(context.Background(), embedding.Request{})
		if !errors.Is(err, baseErr) || !isZeroResult(result) {
			t.Fatalf("Embed() result=%#v error=%v, want zero result and wrapped base error", result, err)
		}
	})

	t.Run("malformed result", func(t *testing.T) {
		t.Parallel()
		base := &fakeEmbedder{response: func(embedding.Request) (embedding.Result, error) {
			return embedding.Result{Model: testModel, Vectors: []embedding.Vector{{1, 2}}}, nil
		}}
		result, err := mustNew(t, base, newMemoryStore()).Embed(context.Background(), embedding.Request{})
		if !errors.Is(err, embedding.ErrInvalidResult) || !isZeroResult(result) {
			t.Fatalf("Embed() result=%#v error=%v, want zero result and invalid result", result, err)
		}
	})
}

func TestEmbedColdWarmMixedAndDuplicates(t *testing.T) {
	t.Parallel()
	base := &fakeEmbedder{}
	store := newMemoryStore()
	cached := mustNew(t, base, store)

	request := embedding.Request{Inputs: []string{"alpha", "beta", "alpha"}}
	cold := mustEmbed(t, cached, request)
	if got, want := base.requests(), [][]string{{"alpha", "beta"}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("cold base requests = %#v, want %#v", got, want)
	}
	if got := store.getCount(); got != 2 {
		t.Fatalf("cold Get calls = %d, want 2", got)
	}
	if got := store.setCount(); got != 2 {
		t.Fatalf("cold Set calls = %d, want 2", got)
	}
	if cold.Model != testModel || len(cold.Vectors[0]) != testModel.Dim || len(cold.Vectors[1]) != testModel.Dim {
		t.Fatalf("cold model/dimensions = %#v / %d,%d, want %#v / %d", cold.Model, len(cold.Vectors[0]), len(cold.Vectors[1]), testModel, testModel.Dim)
	}
	if !reflect.DeepEqual(cold.Vectors[0], cold.Vectors[2]) {
		t.Fatal("duplicate result values differ")
	}
	cold.Vectors[0][0] = 99
	if cold.Vectors[2][0] == 99 {
		t.Fatal("duplicate result vectors alias")
	}

	warm := mustEmbed(t, cached, request)
	if got := base.callCount(); got != 1 {
		t.Fatalf("warm base calls = %d, want 1", got)
	}
	if got := store.setCount(); got != 2 {
		t.Fatalf("warm Set calls = %d, want 2", got)
	}
	if !reflect.DeepEqual(warm.Vectors[0], vectorFor("alpha")) || !reflect.DeepEqual(warm.Vectors[1], vectorFor("beta")) {
		t.Fatalf("warm vectors = %#v, want original-order cached vectors", warm.Vectors)
	}
	if warm.Model != testModel || len(warm.Vectors[0]) != testModel.Dim || len(warm.Vectors[1]) != testModel.Dim {
		t.Fatalf("warm model/dimensions = %#v / %d,%d, want %#v / %d", warm.Model, len(warm.Vectors[0]), len(warm.Vectors[1]), testModel, testModel.Dim)
	}

	mixed := mustEmbed(t, cached, embedding.Request{Inputs: []string{"beta", "gamma", "beta"}})
	if got, want := base.requests(), [][]string{{"alpha", "beta"}, {"gamma"}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("mixed base requests = %#v, want %#v", got, want)
	}
	if got := store.setCount(); got != 3 {
		t.Fatalf("mixed Set calls = %d, want 3", got)
	}
	if !reflect.DeepEqual(mixed.Vectors, []embedding.Vector{vectorFor("beta"), vectorFor("gamma"), vectorFor("beta")}) {
		t.Fatalf("mixed vectors = %#v, want original order", mixed.Vectors)
	}
	if mixed.Model != testModel || len(mixed.Vectors[0]) != testModel.Dim || len(mixed.Vectors[1]) != testModel.Dim {
		t.Fatalf("mixed model/dimensions = %#v / %d,%d, want %#v / %d", mixed.Model, len(mixed.Vectors[0]), len(mixed.Vectors[1]), testModel, testModel.Dim)
	}
}
