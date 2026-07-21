package validate

import (
	"errors"
	"testing"
	"unsafe"
)

type nilTestInterface interface {
	nilTestMethod()
}

type nilTestPointer struct{}

func (*nilTestPointer) nilTestMethod() {}

func TestIsNil(t *testing.T) {
	t.Parallel()

	var (
		nilChan      chan int
		nilFunc      func()
		nilMap       map[string]int
		nilPointer   *nilTestPointer
		nilSlice     []int
		nilUnsafe    unsafe.Pointer
		nilInterface nilTestInterface = nilPointer
	)
	value := 1
	tests := []struct {
		name  string
		value any
		want  bool
	}{
		{name: "nil", value: nil, want: true},
		{name: "typed nil channel", value: nilChan, want: true},
		{name: "typed nil function", value: nilFunc, want: true},
		{name: "typed nil map", value: nilMap, want: true},
		{name: "typed nil pointer", value: nilPointer, want: true},
		{name: "typed nil slice", value: nilSlice, want: true},
		{name: "typed nil unsafe pointer", value: nilUnsafe, want: true},
		{name: "typed nil through interface", value: nilInterface, want: true},
		{name: "non-nil channel", value: make(chan int), want: false},
		{name: "non-nil function", value: func() {}, want: false},
		{name: "non-nil map", value: map[string]int{}, want: false},
		{name: "non-nil pointer", value: &nilTestPointer{}, want: false},
		{name: "non-nil slice", value: []int{}, want: false},
		{name: "non-nil unsafe pointer", value: unsafe.Pointer(&value), want: false},
		{name: "non-nil scalar", value: 0, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := IsNil(tt.value); got != tt.want {
				t.Fatalf("IsNil(%T) = %v, want %v", tt.value, got, tt.want)
			}
		})
	}
}

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
