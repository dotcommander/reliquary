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
	select {
	case <-ctx.Done():
		t.Fatal("Done closed before the configured number of checks")
	default:
	}
	if err := ctx.Err(); err != context.Canceled {
		t.Fatalf("expected context.Canceled on second check, got %v", err)
	}
	select {
	case <-ctx.Done():
	default:
		t.Fatal("Done remained open after cancellation")
	}
	if err := ctx.Err(); err != context.Canceled {
		t.Fatalf("expected context.Canceled on repeat check, got %v", err)
	}

	if ids := resultIDs(nil); ids != "" {
		t.Fatalf("resultIDs(nil) = %q, want empty", ids)
	}
	if ids := resultIDs([]*retrieval.Result{{ID: "a"}}); ids != "a" {
		t.Fatalf("resultIDs = %q, want a", ids)
	}
}
