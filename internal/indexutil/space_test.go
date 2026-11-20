package indexutil

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestSpaceReconcileLegacy(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		state   Space
		legacy  Space
		want    Space
		wantErr error
	}{
		{
			name: "empty state and legacy",
		},
		{
			name:   "legacy fills empty state",
			legacy: Space{Identity: "legacy", IdentitySet: true, Dimension: 2},
			want:   Space{Identity: "legacy", IdentitySet: true, Dimension: 2},
		},
		{
			name:   "current state is retained when legacy agrees",
			state:  Space{Identity: "current", IdentitySet: true, Dimension: 3},
			legacy: Space{Identity: "current", IdentitySet: true, Dimension: 3},
			want:   Space{Identity: "current", IdentitySet: true, Dimension: 3},
		},
		{
			name:   "legacy fills only missing dimension",
			state:  Space{Identity: "current", IdentitySet: true},
			legacy: Space{Identity: "current", IdentitySet: true, Dimension: 3},
			want:   Space{Identity: "current", IdentitySet: true, Dimension: 3},
		},
		{
			name:   "legacy fills only missing identity",
			state:  Space{Dimension: 3},
			legacy: Space{Identity: "legacy", IdentitySet: true, Dimension: 3},
			want:   Space{Identity: "legacy", IdentitySet: true, Dimension: 3},
		},
		{
			name:  "current state is retained when legacy is empty",
			state: Space{Identity: "current", IdentitySet: true, Dimension: 3},
			want:  Space{Identity: "current", IdentitySet: true, Dimension: 3},
		},
		{
			name:    "identity conflict preserves receiver",
			state:   Space{Identity: "current", IdentitySet: true, Dimension: 2},
			legacy:  Space{Identity: "legacy", IdentitySet: true, Dimension: 2},
			want:    Space{Identity: "current", IdentitySet: true, Dimension: 2},
			wantErr: indexcontract.ErrIdentityMismatch,
		},
		{
			name:    "dimension conflict preserves receiver",
			state:   Space{Identity: "current", IdentitySet: true, Dimension: 2},
			legacy:  Space{Identity: "current", IdentitySet: true, Dimension: 3},
			want:    Space{Identity: "current", IdentitySet: true, Dimension: 2},
			wantErr: indexcontract.ErrDimensionMismatch,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := tt.state
			got, err := tt.state.ReconcileLegacy(tt.legacy)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("ReconcileLegacy error = %v, want errors.Is(_, %v)", err, tt.wantErr)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("ReconcileLegacy result = %#v, want %#v", got, tt.want)
			}
			if !reflect.DeepEqual(tt.state, before) {
				t.Fatalf("ReconcileLegacy mutated receiver: got %#v, want %#v", tt.state, before)
			}
		})
	}
}

func TestSpaceValidateQueryRawVector(t *testing.T) {
	t.Parallel()

	space := Space{Identity: "space-1", IdentitySet: true, Dimension: 2}

	if err := space.ValidateQuery(indexcontract.IndexQuery{
		Identity:  "space-1",
		RawVector: []float32{1, 2, 3},
	}); !errors.Is(err, indexcontract.ErrDimensionMismatch) {
		t.Fatalf("ValidateQuery(raw dim mismatch) error = %v, want ErrDimensionMismatch", err)
	}

	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if err := space.ValidateQuery(indexcontract.IndexQuery{
			Identity:  "space-1",
			RawVector: []float32{1, bad},
		}); err == nil {
			t.Fatalf("ValidateQuery(raw non-finite %v) error = nil, want error", bad)
		}
	}

	if err := space.ValidateQuery(indexcontract.IndexQuery{
		Identity:  "space-1",
		RawVector: []float32{0.5, 0.5},
	}); err != nil {
		t.Fatalf("ValidateQuery(valid raw vector) error = %v", err)
	}
}

func TestSearchScoredCandidatesRejectsRawVectorDimensionMismatch(t *testing.T) {
	t.Parallel()

	items := []*retrieval.Result{
		{ID: "doc1#1", DocumentID: "doc1", Content: "alpha", Embedding: []float64{1, 0}},
	}
	_, err := SearchScoredCandidates(context.Background(), indexcontract.IndexQuery{
		Text:      "alpha",
		RawVector: []float32{1, 0, 0},
	}, items)
	if !errors.Is(err, indexcontract.ErrDimensionMismatch) {
		t.Fatalf("SearchScoredCandidates(raw dim mismatch) error = %v, want ErrDimensionMismatch", err)
	}
}

func TestValidateFilterJSONNumberExponentBound(t *testing.T) {
	t.Parallel()

	for _, bad := range []json.Number{
		"1e100000000",
		"1E100000000",
		"1e-100000000",
		"1e1001",
		"1e-1001",
		"1e9999999999999999999999",
	} {
		if err := ValidateFilter(map[string]any{"n": bad}); err == nil {
			t.Fatalf("ValidateFilter(%q) error = nil, want exponent out of range error", bad)
		}
	}

	for _, okVal := range []json.Number{
		"1e10",
		"1.5e-3",
		"1e1000",
		"1e-1000",
		"42",
	} {
		if err := ValidateFilter(map[string]any{"n": okVal}); err != nil {
			t.Fatalf("ValidateFilter(%q) error = %v, want nil", okVal, err)
		}
	}
}
