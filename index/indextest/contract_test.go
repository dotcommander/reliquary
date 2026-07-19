package indextest

import (
	"context"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/index/inmem"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestContractSuite(t *testing.T) {
	t.Parallel()

	Run(t, func() indexcontract.Index {
		return inmem.New()
	})
}

func TestCancelAfterContextHelpers(t *testing.T) {
	t.Parallel()

	ctx := newCancelAfterContext(2)
	if ctx.Err() != nil {
		t.Fatal("expected nil error on first check")
	}
	<-ctx.Done()
	if ctx.Err() != context.Canceled {
		t.Fatalf("expected context.Canceled, got %v", ctx.Err())
	}
	// Check again after canceled flag is true
	if ctx.Err() != context.Canceled {
		t.Fatalf("expected context.Canceled on repeat check, got %v", ctx.Err())
	}

	if ids := resultIDs(nil); ids != "" {
		t.Fatalf("resultIDs(nil) = %q, want empty", ids)
	}
	if ids := resultIDs([]*retrieval.Result{{ID: "a"}}); ids != "a" {
		t.Fatalf("resultIDs = %q, want a", ids)
	}
}
