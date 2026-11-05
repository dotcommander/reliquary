package structjudge

import (
	"context"
	"errors"
	"testing"
)

type typedNilClient struct{}

func (c *typedNilClient) Generate(context.Context, Request) ([]byte, error) {
	return nil, nil
}

func TestNewRejectsTypedNilClient(t *testing.T) {
	t.Parallel()
	if _, err := New((*typedNilClient)(nil)); !errors.Is(err, ErrNilClient) {
		t.Fatalf("New error = %v, want ErrNilClient", err)
	}
}
