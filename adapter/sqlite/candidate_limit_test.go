package sqlite

import (
	"encoding/json"
	"slices"
	"testing"
)

func TestBoundCandidateQuery(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name    string
		filter  map[string]any
		limit   int
		bounded bool
	}{
		{name: "no filter", limit: 4, bounded: true},
		{name: "exact scalars", filter: map[string]any{"tenant": "one", "active": true, "removed": nil}, limit: 4, bounded: true},
		{name: "numeric metadata", filter: map[string]any{"value": uint64(9007199254740993)}, limit: 4},
		{name: "JSON number", filter: map[string]any{"value": json.Number("1.5")}, limit: 4},
		{name: "numeric reserved field", filter: map[string]any{"id": 1}, limit: 4, bounded: true},
		{name: "unbounded", limit: 0},
		{name: "negative", limit: -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			const statement = "SELECT id FROM items ORDER BY id"
			args := []any{"existing"}
			got, gotArgs := boundCandidateQuery(statement, args, tc.filter, tc.limit)
			want, wantArgs := statement, []any{"existing"}
			if tc.bounded {
				want += " LIMIT ?"
				wantArgs = append(wantArgs, tc.limit)
			}
			if got != want || !slices.Equal(gotArgs, wantArgs) {
				t.Fatalf("query = %q, args = %v; want %q, %v", got, gotArgs, want, wantArgs)
			}
		})
	}
}
