package reliquary_test

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"testing/synctest"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/retrieval"
)

type concurrentIndex struct {
	batchRecordingIndex
	search func(context.Context, reliquary.IndexQuery) ([]*retrieval.Result, error)
}

func (i *concurrentIndex) Search(ctx context.Context, q reliquary.IndexQuery) ([]*retrieval.Result, error) {
	return i.search(ctx, q)
}

type concurrentReranker func(context.Context, string, []*retrieval.Result) ([]float64, error)

func (f concurrentReranker) Rerank(ctx context.Context, q string, r []*retrieval.Result) ([]float64, error) {
	return f(ctx, q, r)
}

func TestSearchConcurrencyBoundsWholePipelineAndAlignsRows(t *testing.T) {
	t.Parallel()
	synctest.Test(t, func(t *testing.T) {
		gates := []chan struct{}{make(chan struct{}), make(chan struct{}), make(chan struct{})}
		entered := make(chan int, 3)
		var active, maximum atomic.Int32
		embedder := &batchRecordingEmbedder{}
		candidate := &retrieval.Result{ID: "shared", Content: "content", Embedding: []float64{1, 1}}
		idx := &concurrentIndex{search: func(_ context.Context, q reliquary.IndexQuery) ([]*retrieval.Result, error) {
			if q.Filter["tenant"] != "original" {
				t.Error("filter ownership")
			}
			q.Filter["tenant"] = "mutated"
			id := int(q.Vector[0]) - 1
			q.Vector[0] = 99
			result := *candidate
			result.ID = fmt.Sprint(id)
			return []*retrieval.Result{&result}, nil
		}}
		reranker := concurrentReranker(func(_ context.Context, _ string, results []*retrieval.Result) ([]float64, error) {
			n := active.Add(1)
			for previous := maximum.Load(); n > previous && !maximum.CompareAndSwap(previous, n); previous = maximum.Load() {
			}
			defer active.Add(-1)
			var id int
			fmt.Sscan(results[0].ID, &id)
			entered <- id
			<-gates[id]
			return []float64{float64(id+1) / 3}, nil
		})
		app := newBatchTestApp(t, embedder, idx)
		done := make(chan struct{})
		var rows [][]*retrieval.Result
		var err error
		go func() {
			rows, err = app.SearchBatch(context.Background(), []string{"same", " ", "same", "third"},
				reliquary.WithSearchConcurrency(9), reliquary.WithSearchConcurrency(2),
				reliquary.WithFilter(map[string]any{"tenant": "original"}), reliquary.WithReranker(reranker),
				reliquary.TopK(1), reliquary.WithMMR(0.5), reliquary.WithExplain())
			close(done)
		}()
		<-entered
		<-entered
		synctest.Wait()
		if maximum.Load() != 2 || len(entered) != 0 {
			t.Fatal("worker bound not applied to full pipeline")
		}
		close(gates[1])
		if id := <-entered; id != 2 {
			t.Fatalf("next job = %d", id)
		}
		close(gates[2])
		synctest.Wait()
		close(gates[0])
		<-done
		if err != nil {
			t.Fatal(err)
		}
		if len(rows) != 4 || rows[1] != nil {
			t.Fatalf("alignment: %#v", rows)
		}
		for i, pos := range []int{0, 2, 3} {
			if len(rows[pos]) != 1 || rows[pos][0].ID != fmt.Sprint(i) {
				t.Fatalf("row %d: %#v", pos, rows[pos])
			}
		}
		rows[0][0].Content = "changed"
		if rows[2][0].Content != "content" || candidate.Content != "content" {
			t.Fatal("result ownership")
		}
		if embedder.calls != 1 || maximum.Load() != 2 {
			t.Fatal("batching or limit changed")
		}
	})
}

func TestSearchConcurrencyCancelsAndJoins(t *testing.T) {
	t.Parallel()
	for _, parentCancel := range []bool{false, true} {
		t.Run(fmt.Sprint(parentCancel), func(t *testing.T) {
			t.Parallel()
			synctest.Test(t, func(t *testing.T) {
				entered := make(chan int, 2)
				fail, siblingExit := make(chan struct{}), make(chan struct{})
				var calls atomic.Int32
				want := errors.New("query failed")
				idx := &concurrentIndex{search: func(ctx context.Context, q reliquary.IndexQuery) ([]*retrieval.Result, error) {
					calls.Add(1)
					id := int(q.Vector[0])
					entered <- id
					if id == 1 && !parentCancel {
						<-fail
						return nil, want
					}
					<-ctx.Done()
					<-siblingExit // Deliberately delay exit to prove the caller joins.
					return nil, ctx.Err()
				}}
				app := newBatchTestApp(t, &batchRecordingEmbedder{}, idx)
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				done := make(chan struct{})
				var rows [][]*retrieval.Result
				var err error
				go func() {
					rows, err = app.SearchBatch(ctx, []string{"one", "two", "three", "four"}, reliquary.WithSearchConcurrency(2))
					close(done)
				}()
				<-entered
				<-entered
				if parentCancel {
					cancel()
				} else {
					close(fail)
				}
				synctest.Wait()
				select {
				case <-done:
					t.Fatal("returned before sibling exit")
				default:
				}
				close(siblingExit)
				<-done
				if parentCancel {
					want = context.Canceled
				}
				if rows != nil || !errors.Is(err, want) || calls.Load() != 2 {
					t.Fatalf("rows = %#v, err = %v, calls = %d", rows, err, calls.Load())
				}
			})
		})
	}
}

func TestSearchConcurrencyValidatesBeforeDispatch(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	idx := &concurrentIndex{search: func(context.Context, reliquary.IndexQuery) ([]*retrieval.Result, error) {
		calls.Add(1)
		return nil, nil
	}}
	app := newBatchTestApp(t, &batchRecordingEmbedder{result: embedding.Result{Vectors: []embedding.Vector{{1, 1}}}}, idx)
	rows, err := app.SearchBatch(context.Background(), []string{"one", "two"}, reliquary.WithSearchConcurrency(2))
	if !errors.Is(err, embedding.ErrInvalidResult) || rows != nil || calls.Load() != 0 {
		t.Fatalf("validation: %#v, %v, calls %d", rows, err, calls.Load())
	}
}

func TestSearchConcurrencyLastOptionCanDisable(t *testing.T) {
	t.Parallel()
	for _, n := range []int{-1, 0, 1} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			t.Parallel()
			idx := &batchRecordingIndex{}
			app := newBatchTestApp(t, &batchRecordingEmbedder{}, idx)
			_, err := app.SearchBatch(context.Background(), []string{"one", "two"}, reliquary.WithSearchConcurrency(4), reliquary.WithSearchConcurrency(n))
			if err != nil || idx.maxActive != 1 || len(idx.queries) != 2 {
				t.Fatalf("sequential: %v, %#v", err, idx)
			}
		})
	}
}

func TestSearchConcurrencyRRFKeepsLanesSequential(t *testing.T) {
	t.Parallel()
	synctest.Test(t, func(t *testing.T) {
		release := make(chan struct{})
		entered := make(chan struct{}, 4)
		var calls atomic.Int32
		idx := &concurrentIndex{search: func(_ context.Context, q reliquary.IndexQuery) ([]*retrieval.Result, error) {
			calls.Add(1)
			entered <- struct{}{}
			<-release
			return []*retrieval.Result{{ID: "hit", Content: "hit", Embedding: []float64{1, 1}}}, nil
		}}
		app := newBatchTestApp(t, &batchRecordingEmbedder{}, idx)
		done := make(chan error, 1)
		go func() {
			_, err := app.SearchBatch(context.Background(), []string{"one", "two"}, reliquary.WithRRF(60), reliquary.WithSearchConcurrency(2))
			done <- err
		}()
		<-entered
		<-entered
		synctest.Wait()
		if calls.Load() != 2 {
			t.Fatalf("concurrent lanes: %d", calls.Load())
		}
		close(release)
		if err := <-done; err != nil {
			t.Fatal(err)
		}
		if calls.Load() != 4 {
			t.Fatalf("RRF calls: %d", calls.Load())
		}
	})
}
