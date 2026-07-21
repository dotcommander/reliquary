package embeddings

import (
	"errors"
	"math"
	"testing"
)

func TestValidateResult(t *testing.T) {
	t.Parallel()

	valid := Result{Vectors: []Vector{{1, 0}, {0, 1}}}
	tests := []struct {
		name    string
		request Request
		result  Result
		wantErr bool
	}{
		{
			name:    "valid inferred dimensions",
			request: Request{Inputs: []string{"a", "b"}},
			result:  valid,
		},
		{
			name:    "valid declared dimensions",
			request: Request{Model: ModelRef{Dim: 2}, Inputs: []string{"a", "b"}},
			result:  Result{Model: ModelRef{Dim: 2}, Vectors: valid.Vectors},
		},
		{
			name:    "valid zero magnitude vector",
			request: Request{Inputs: []string{"empty"}},
			result:  Result{Vectors: []Vector{{0, 0}}},
		},
		{
			name:    "valid empty request",
			request: Request{Model: ModelRef{Dim: 2}},
			result:  Result{Model: ModelRef{Dim: 2}},
		},
		{
			name:    "count mismatch",
			request: Request{Inputs: []string{"a", "b"}},
			result:  Result{Vectors: []Vector{{1, 0}}},
			wantErr: true,
		},
		{
			name:    "negative request dimensions",
			request: Request{Model: ModelRef{Dim: -1}},
			result:  Result{},
			wantErr: true,
		},
		{
			name:    "negative result dimensions",
			request: Request{},
			result:  Result{Model: ModelRef{Dim: -1}},
			wantErr: true,
		},
		{
			name:    "conflicting declared dimensions",
			request: Request{Model: ModelRef{Dim: 2}, Inputs: []string{"a"}},
			result:  Result{Model: ModelRef{Dim: 3}, Vectors: []Vector{{1, 0}}},
			wantErr: true,
		},
		{
			name:    "request dimension conflicts with vector",
			request: Request{Model: ModelRef{Dim: 3}, Inputs: []string{"a"}},
			result:  Result{Vectors: []Vector{{1, 0}}},
			wantErr: true,
		},
		{
			name:    "result dimension conflicts with vector",
			request: Request{Inputs: []string{"a"}},
			result:  Result{Model: ModelRef{Dim: 3}, Vectors: []Vector{{1, 0}}},
			wantErr: true,
		},
		{
			name:    "empty vector",
			request: Request{Inputs: []string{"a"}},
			result:  Result{Vectors: []Vector{{}}},
			wantErr: true,
		},
		{
			name:    "ragged vectors",
			request: Request{Inputs: []string{"a", "b"}},
			result:  Result{Vectors: []Vector{{1, 0}, {1}}},
			wantErr: true,
		},
		{
			name:    "NaN",
			request: Request{Inputs: []string{"a"}},
			result:  Result{Vectors: []Vector{{float32(math.NaN())}}},
			wantErr: true,
		},
		{
			name:    "positive infinity",
			request: Request{Inputs: []string{"a"}},
			result:  Result{Vectors: []Vector{{float32(math.Inf(1))}}},
			wantErr: true,
		},
		{
			name:    "negative infinity",
			request: Request{Inputs: []string{"a"}},
			result:  Result{Vectors: []Vector{{float32(math.Inf(-1))}}},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := ValidateResult(tt.request, tt.result)
			if tt.wantErr {
				if !errors.Is(err, ErrInvalidResult) {
					t.Fatalf("ValidateResult() error = %v, want %v", err, ErrInvalidResult)
				}
				return
			}
			if err != nil {
				t.Fatalf("ValidateResult() error = %v", err)
			}
		})
	}
}

func TestValidateDimensions(t *testing.T) {
	t.Parallel()

	err := ValidateDimensions([]Vector{{1, 2}, {3}}, 2)
	if err == nil {
		t.Fatal("expected dimension error")
	}
}

func TestCacheKeyIncludesModel(t *testing.T) {
	t.Parallel()

	input := "same text"
	a := CacheKey(ModelRef{Provider: "local", Name: "a", Dim: 2}, input)
	b := CacheKey(ModelRef{Provider: "local", Name: "b", Dim: 2}, input)
	if a == b {
		t.Fatal("cache key should include model identity")
	}
}

func TestModelRefIdentityGolden(t *testing.T) {
	t.Parallel()

	model := ModelRef{
		Provider: "local",
		Name:     "text-embedding-3-small",
		Version:  "2024-01",
		Revision: "rev:2",
		Dim:      1536,
	}
	want := "modelref:v1:5:local22:text-embedding-3-small7:2024-015:rev:24:1536"
	if got := model.Identity(); got != want {
		t.Fatalf("Identity() = %q, want %q", got, want)
	}
}

func TestModelRefIdentityFramesAmbiguousFields(t *testing.T) {
	t.Parallel()

	a := ModelRef{Provider: "a", Name: "b:c", Dim: 2}
	b := ModelRef{Provider: "a:b", Name: "c", Dim: 2}
	if a.Identity() == b.Identity() {
		t.Fatalf("field framing collision: %q", a.Identity())
	}

	unicode := ModelRef{Provider: "λ", Dim: 2}.Identity()
	if want := "modelref:v1:2:λ0:0:0:1:2"; unicode != want {
		t.Fatalf("byte-length framing = %q, want %q", unicode, want)
	}
}
