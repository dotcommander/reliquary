package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/internal/validate"
)

var (
	// ErrNilEmbedder indicates that New received a nil Embedder.
	ErrNilEmbedder = errors.New("nil embedder")
	// ErrNilStore indicates that New received a nil Store.
	ErrNilStore = errors.New("nil store")
	// ErrInvalidIdentity indicates that New received a blank cache identity.
	ErrInvalidIdentity = errors.New("invalid cache identity")
	// ErrModelMismatch indicates that entries in one batch resolve to different models.
	ErrModelMismatch = errors.New("embedding model mismatch")
)

// Config configures an Embedder.
type Config struct {
	// Identity namespaces entries by the caller-owned embedding-output contract.
	Identity string
}

// Entry is one resolved embedding cached by a Store.
type Entry struct {
	Model  embedding.ModelRef
	Vector embedding.Vector
}

// Store persists individual embedding entries. Implementations must be safe for
// the concurrency with which callers invoke Embedder.
type Store interface {
	Get(context.Context, string) (Entry, bool, error)
	Set(context.Context, string, Entry) error
}

// Embedder decorates a base embedding.Embedder with a caller-provided Store.
// Its state is immutable after construction.
type Embedder struct {
	base     embedding.Embedder
	store    Store
	identity string
}

// New constructs a Store-backed decorator without performing dependency I/O.
func New(base embedding.Embedder, store Store, cfg Config) (*Embedder, error) {
	if validate.IsNil(base) {
		return nil, fmt.Errorf("embedding cache: %w", ErrNilEmbedder)
	}
	if validate.IsNil(store) {
		return nil, fmt.Errorf("embedding cache: %w", ErrNilStore)
	}
	if strings.TrimSpace(cfg.Identity) == "" {
		return nil, fmt.Errorf("embedding cache: %w", ErrInvalidIdentity)
	}
	return &Embedder{base: base, store: store, identity: cfg.Identity}, nil
}

// Embed embeds request inputs, using cached vectors where available.
func (e *Embedder) Embed(ctx context.Context, request embedding.Request) (embedding.Result, error) {
	if err := contextError(ctx); err != nil {
		return embedding.Result{}, err
	}
	if len(request.Inputs) == 0 {
		result, err := e.base.Embed(ctx, request)
		if err != nil {
			return embedding.Result{}, fmt.Errorf("embedding cache: embed empty request: %w", err)
		}
		if err := embedding.ValidateResult(request, result); err != nil {
			return embedding.Result{}, fmt.Errorf("embedding cache: validate empty result: %w", err)
		}
		return cloneResult(result), nil
	}

	unique := make([]cacheInput, 0, len(request.Inputs))
	byKey := make(map[string]int, len(request.Inputs))
	for position, input := range request.Inputs {
		key := cacheKey(e.identity, request.Model, input)
		if index, ok := byKey[key]; ok {
			unique[index].positions = append(unique[index].positions, position)
			continue
		}
		byKey[key] = len(unique)
		unique = append(unique, cacheInput{key: key, input: input, positions: []int{position}})
	}

	for index := range unique {
		if err := contextError(ctx); err != nil {
			return embedding.Result{}, err
		}
		entry, ok, err := e.store.Get(ctx, unique[index].key)
		if err != nil {
			return embedding.Result{}, fmt.Errorf("embedding cache: lookup key %s: %w", unique[index].key, err)
		}
		if !ok {
			continue
		}
		entry.Vector = cloneVector(entry.Vector)
		entryRequest := embedding.Request{Model: request.Model, Inputs: []string{unique[index].input}}
		entryResult := embedding.Result{Model: entry.Model, Vectors: []embedding.Vector{entry.Vector}}
		if err := embedding.ValidateResult(entryRequest, entryResult); err != nil {
			return embedding.Result{}, fmt.Errorf("embedding cache: validate cached input %d: %w", unique[index].positions[0], err)
		}
		unique[index].entry = entry
		unique[index].hit = true
	}

	if err := contextError(ctx); err != nil {
		return embedding.Result{}, err
	}

	misses := make([]int, 0, len(unique))
	for index := range unique {
		if !unique[index].hit {
			misses = append(misses, index)
		}
	}

	var generated embedding.Result
	if len(misses) > 0 {
		inputs := make([]string, len(misses))
		for resultIndex, uniqueIndex := range misses {
			inputs[resultIndex] = unique[uniqueIndex].input
		}
		result, err := e.base.Embed(ctx, embedding.Request{Model: request.Model, Inputs: inputs})
		if err != nil {
			return embedding.Result{}, fmt.Errorf("embedding cache: embed %d misses: %w", len(misses), err)
		}
		if err := contextError(ctx); err != nil {
			return embedding.Result{}, err
		}
		generated = cloneResult(result)
		if err := embedding.ValidateResult(embedding.Request{Model: request.Model, Inputs: inputs}, generated); err != nil {
			return embedding.Result{}, fmt.Errorf("embedding cache: validate generated result: %w", err)
		}
		for resultIndex, uniqueIndex := range misses {
			unique[uniqueIndex].entry = Entry{Model: generated.Model, Vector: generated.Vectors[resultIndex]}
		}
	}

	var model embedding.ModelRef
	haveModel := false
	for _, item := range unique {
		if !haveModel {
			model = item.entry.Model
			haveModel = true
			continue
		}
		if item.entry.Model != model {
			return embedding.Result{}, fmt.Errorf("embedding cache: input %d: %w", item.positions[0], ErrModelMismatch)
		}
	}

	assembled := embedding.Result{Model: model, Vectors: make([]embedding.Vector, len(request.Inputs))}
	for _, item := range unique {
		for _, position := range item.positions {
			assembled.Vectors[position] = cloneVector(item.entry.Vector)
		}
	}
	if err := embedding.ValidateResult(request, assembled); err != nil {
		return embedding.Result{}, fmt.Errorf("embedding cache: validate assembled result: %w", err)
	}

	for _, uniqueIndex := range misses {
		if err := contextError(ctx); err != nil {
			return embedding.Result{}, err
		}
		item := unique[uniqueIndex]
		entry := Entry{Model: generated.Model, Vector: cloneVector(item.entry.Vector)}
		if err := e.store.Set(ctx, item.key, entry); err != nil {
			return embedding.Result{}, fmt.Errorf("embedding cache: store key %s: %w", item.key, err)
		}
	}
	if err := contextError(ctx); err != nil {
		return embedding.Result{}, err
	}
	return assembled, nil
}

type cacheInput struct {
	key       string
	input     string
	positions []int
	entry     Entry
	hit       bool
}

func cacheKey(identity string, model embedding.ModelRef, input string) string {
	preimage := "reliquary:embedding-cache:v1:" + frame(identity) + frame(embedding.CacheKey(model, input))
	sum := sha256.Sum256([]byte(preimage))
	return hex.EncodeToString(sum[:])
}

func frame(value string) string {
	return strconv.Itoa(len([]byte(value))) + ":" + value
}

func cloneResult(result embedding.Result) embedding.Result {
	cloned := embedding.Result{Model: result.Model, Vectors: make([]embedding.Vector, len(result.Vectors))}
	for index, vector := range result.Vectors {
		cloned.Vectors[index] = cloneVector(vector)
	}
	return cloned
}

func cloneVector(vector embedding.Vector) embedding.Vector {
	return append(embedding.Vector(nil), vector...)
}

func contextError(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("embedding cache: context: %w", err)
	}
	return nil
}
