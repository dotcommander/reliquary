package validate

import (
	"errors"
	"testing"
)

func TestNonEmpty(t *testing.T) {
	t.Parallel()

	if err := NonEmpty("id", "  "); !errors.Is(err, ErrRequired) {
		t.Fatalf("expected ErrRequired, got %v", err)
	}
}

func TestPositive(t *testing.T) {
	t.Parallel()

	if err := Positive("dims", 0); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected ErrInvalid, got %v", err)
	}
}

func TestEqual(t *testing.T) {
	t.Parallel()

	if err := Equal("dims", 3, 4); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected ErrInvalid, got %v", err)
	}
}
