package jev

import (
	"context"
	"errors"
	"testing"
)

type typedNilClient struct{}

func (c *typedNilClient) Evaluate(context.Context, Call) (Result, error) {
	return Result{}, nil
}

func TestNewRejectsTypedNilClient(t *testing.T) {
	t.Parallel()
	if _, err := New((*typedNilClient)(nil)); !errors.Is(err, ErrNilClient) {
		t.Fatalf("New error = %v, want ErrNilClient", err)
	}
}
