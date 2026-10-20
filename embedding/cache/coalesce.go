package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"strconv"
	"strings"

	"github.com/dotcommander/reliquary/embedding"
)

// ErrCoalescedCallAborted means the flight leader panicked or exited without
// returning normally. The leader retains its original panic or exit behavior.
var ErrCoalescedCallAborted = errors.New("embedding cache: coalesced call aborted")

type flight struct {
	done   chan struct{}
	result embedding.Result
	err    error
}

func requestKey(identity string, request embedding.Request) string {
	var key strings.Builder
	key.WriteString("reliquary:embedding-flight:v1:")
	for _, value := range []string{identity, request.Model.Identity(), strconv.Itoa(int(request.Kind)), strconv.Itoa(len(request.Inputs))} {
		key.WriteString(frame(value))
	}
	for _, input := range request.Inputs {
		key.WriteString(frame(input))
	}
	sum := sha256.Sum256([]byte(key.String()))
	return hex.EncodeToString(sum[:])
}

func (e *Embedder) embedCoalesced(ctx context.Context, request embedding.Request) (result embedding.Result, err error) {
	key := requestKey(e.identity, request)
	e.flightMu.Lock()
	if current, ok := e.flights[key]; ok {
		e.flightMu.Unlock()
		if err := contextError(ctx); err != nil {
			return embedding.Result{}, err
		}
		select {
		case <-ctx.Done():
			return embedding.Result{}, contextError(ctx)
		case <-current.done:
			if err := contextError(ctx); err != nil {
				return embedding.Result{}, err
			}
			if current.err != nil {
				return embedding.Result{}, current.err
			}
			return cloneResult(current.result), nil
		}
	}
	current := &flight{done: make(chan struct{}), err: ErrCoalescedCallAborted}
	if e.flights == nil {
		e.flights = make(map[string]*flight)
	}
	e.flights[key] = current
	e.flightMu.Unlock()
	// Cleanup runs for normal returns, panics and runtime.Goexit. Publishing the
	// private snapshot before closing done synchronizes all follower reads.
	defer func() {
		e.flightMu.Lock()
		delete(e.flights, key)
		close(current.done)
		e.flightMu.Unlock()
	}()
	result, err = e.embed(ctx, request)
	if err == nil {
		current.result = cloneResult(result)
	}
	current.err = err
	return result, err
}
