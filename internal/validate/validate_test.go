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
	if err := NonEmpty("id", "valid"); err != nil {
		t.Fatalf("expected nil for valid input, got %v", err)
	}
}

func TestPositive(t *testing.T) {
	t.Parallel()

	if err := Positive("dims", 0); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected ErrInvalid, got %v", err)
	}
	if err := Positive("dims", 10); err != nil {
		t.Fatalf("expected nil for positive input, got %v", err)
	}
}

func TestEqual(t *testing.T) {
	t.Parallel()

	if err := Equal("dims", 3, 4); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected ErrInvalid, got %v", err)
	}
	if err := Equal("dims", 5, 5); err != nil {
		t.Fatalf("expected nil for equal input, got %v", err)
	}
}
