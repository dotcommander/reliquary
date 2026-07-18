// Package validate provides small validation helpers for primitive packages.
package validate

import (
	"errors"
	"fmt"
	"strings"
)

var (
	// ErrRequired reports a missing required value.
	ErrRequired = errors.New("required")
	// ErrInvalid reports a value outside its valid semantic space.
	ErrInvalid = errors.New("invalid")
)

// NonEmpty returns an error when value is empty after trimming whitespace.
func NonEmpty(field, value string) error {
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("%s: %w", field, ErrRequired)
	}
	return nil
}

// Positive returns an error when value is not greater than zero.
func Positive(field string, value int) error {
	if value <= 0 {
		return fmt.Errorf("%s: %w: must be positive", field, ErrInvalid)
	}
	return nil
}

// Equal returns an error when got and want differ.
func Equal[T comparable](field string, got, want T) error {
	if got != want {
		return fmt.Errorf("%s: %w: got %v want %v", field, ErrInvalid, got, want)
	}
	return nil
}
