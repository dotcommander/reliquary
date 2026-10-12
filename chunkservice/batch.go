package chunkservice

import (
	"context"
	"errors"
)

// ProcessBatch chunks each text under the batch's shared parameters,
// sequentially and independently. Item-level failures (empty text, per-text
// validation) become per-item error results; batch-level failures (shared
// params, context cancellation) fail the whole call.
func (s *Service) ProcessBatch(ctx context.Context, req BatchRequest) (*BatchResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := req.Validate(); err != nil {
		return nil, err
	}

	results := make([]BatchResult, len(req.Texts))
	for i, text := range req.Texts {
		resp, err := s.Process(ctx, req.ToChunkRequest(text))
		switch {
		case err == nil:
			results[i] = BatchResult{Index: i, Response: resp}
		case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
			return nil, err
		case IsRequestValidationError(err):
			results[i] = BatchResult{Index: i, Error: err.Error()}
		default:
			// Match the single-request convention: log internally, keep the
			// item payload generic so internals never leak to clients.
			s.log.ErrorContext(ctx, "batch chunk item failed", "index", i, "error", err)
			results[i] = BatchResult{Index: i, Error: "internal error"}
		}
	}
	return &BatchResponse{Results: results}, nil
}
