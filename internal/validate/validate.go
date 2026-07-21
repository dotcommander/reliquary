// Package validate provides small validation helpers for primitive packages.
package validate

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
)

var (
	// ErrRequired reports a missing required value.
	ErrRequired = errors.New("required")
	// ErrInvalid reports a value outside its valid semantic space.
	ErrInvalid = errors.New("invalid")
)

// IsNil reports whether value is nil, including an interface containing a
// typed nil value of a nilable kind.
func IsNil(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice, reflect.UnsafePointer:
		return reflected.IsNil()
	default:
		return false
	}
}

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
