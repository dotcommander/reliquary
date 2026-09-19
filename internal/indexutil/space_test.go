package indexutil

import (
	"errors"
	"reflect"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
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
