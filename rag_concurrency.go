package reliquary

import (
	"context"
	"sync"

	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/retrieval"
)

// WithSearchConcurrency bounds concurrent complete query pipelines in SearchBatch.
// The last occurrence wins; n < 2 selects sequential execution. Search accepts
// this option but remains a single query. Opting in permits concurrent index and
// reranker calls: dependencies must synchronize their state and honor cancellation.
// Dependencies ignoring cancellation can delay return because all workers are joined.
func WithSearchConcurrency(n int) SearchOption {
	return func(c *searchConfig) { c.concurrency = n }
}

func (a *App) searchConcurrent(ctx context.Context, queries []string, positions []int, vectors []embedding.Vector, cfg searchConfig, rows [][]*retrieval.Result) ([][]*retrieval.Result, error) {
	workCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	jobs := make(chan int)
	var workers sync.WaitGroup
	var first sync.Once
	var queryErr error
	for range min(cfg.concurrency, len(queries)) {
		workers.Go(func() {
			// Workers exit when dispatch closes jobs or cancellation is observed.
			for i := range jobs {
				if workCtx.Err() != nil {
					return
				}
				row, err := a.searchEmbedded(workCtx, queries[i], vectors[i], cfg)
				if err != nil {
					first.Do(func() { queryErr = err; cancel() })
					return
				}
				rows[positions[i]] = row
			}
		})
	}
dispatch:
	for i := range queries {
		if workCtx.Err() != nil {
			break
		}
		select {
		case <-workCtx.Done():
			break dispatch
		case jobs <- i:
		}
	}
	close(jobs)
	workers.Wait()
	if queryErr != nil {
		return nil, queryErr
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}
