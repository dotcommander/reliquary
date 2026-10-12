package chunking

import (
	"context"
	"sync"

	"github.com/dotcommander/reliquary/embedding"
)

// LimitBatchSize wraps an embedding.Embedder so each Embed call carries at
// most max inputs: oversized batches are split into sequential requests and
// the vectors are reassembled in input order. max <= 0 returns e unchanged.
// Providers that cap request size (or cost per call) get bounded requests
// without every caller reimplementing the split/reassemble loop.
func LimitBatchSize(e embedding.Embedder, max int) embedding.Embedder {
	if max <= 0 {
		return e
	}
	return &batchLimitedEmbedder{inner: e, max: max}
}

type batchLimitedEmbedder struct {
	inner embedding.Embedder
	max   int
}

func (b *batchLimitedEmbedder) Embed(ctx context.Context, req embedding.Request) (embedding.Result, error) {
	if len(req.Inputs) <= b.max {
		return b.inner.Embed(ctx, req)
	}
	var out []embedding.Vector
	for start := 0; start < len(req.Inputs); start += b.max {
		end := min(start+b.max, len(req.Inputs))
		part := req
		part.Inputs = req.Inputs[start:end]
		res, err := b.inner.Embed(ctx, part)
		if err != nil {
			return embedding.Result{}, err
		}
		if len(res.Vectors) != end-start {
			return embedding.Result{}, embedding.ErrInvalidResult
		}
		out = append(out, res.Vectors...)
	}
	return embedding.Result{Model: req.Model, Vectors: out}, nil
}

// LatchedBatchEmbedder records the first EmbedBatch error of the wrapped
// embedder while letting the call fail through unchanged. SemanticChunker
// deliberately swallows embedding errors and falls back to boundary chunking;
// callers that must distinguish a genuine semantic run from a degraded one
// wrap their embedder with LatchBatchEmbedder and check Err after chunking.
type LatchedBatchEmbedder struct {
	inner BatchEmbedder
	mu    sync.Mutex
	err   error
}

// LatchBatchEmbedder wraps inner so the first embedding error is retained
// for later inspection. A nil inner is returned unchanged: there is nothing
// to observe without an embedder.
func LatchBatchEmbedder(inner BatchEmbedder) *LatchedBatchEmbedder {
	if inner == nil {
		return nil
	}
	return &LatchedBatchEmbedder{inner: inner}
}

func (l *LatchedBatchEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	vectors, err := l.inner.EmbedBatch(ctx, texts)
	if err != nil {
		l.mu.Lock()
		if l.err == nil {
			l.err = err
		}
		l.mu.Unlock()
	}
	return vectors, err
}

// Err returns the first embedding error observed, or nil.
func (l *LatchedBatchEmbedder) Err() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.err
}
