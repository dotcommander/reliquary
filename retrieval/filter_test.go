package retrieval

import (
	"reflect"
	"testing"
)

func TestFilterInclude(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		filter Filter
		path   string
		want   bool
	}{
		// DefaultFilter: ignored dirs (exact path component)
		{name: "default/node_modules dir", filter: DefaultFilter(), path: "node_modules/x.go", want: false},
		{name: "default/git dir", filter: DefaultFilter(), path: "a/.git/b", want: false},
		{name: "default/vendor dir", filter: DefaultFilter(), path: "vendor/lib.go", want: false},
		// DefaultFilter: ignored extensions
		{name: "default/lock ext", filter: DefaultFilter(), path: "package.lock", want: false},
		{name: "default/sum ext", filter: DefaultFilter(), path: "go.sum", want: false},
		{name: "default/map ext", filter: DefaultFilter(), path: "bundle.js.map", want: false},
		// DefaultFilter: normal file passes
		{name: "default/normal go file", filter: DefaultFilter(), path: "src/main.go", want: true},
		// IncludeExts allowlist
		{name: "includeExts/md excluded", filter: Filter{IncludeExts: []string{".go"}}, path: "x.md", want: false},
		{name: "includeExts/go included", filter: Filter{IncludeExts: []string{".go"}}, path: "x.go", want: true},
		// Extension normalization: "go" (no leading dot) should behave as ".go"
		{name: "includeExts/no-dot go included", filter: Filter{IncludeExts: []string{"go"}}, path: "x.go", want: true},
		{name: "ignoreExts/no-dot lock excluded", filter: Filter{IgnoreExts: []string{"lock"}}, path: "package.lock", want: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := tc.filter.Include(tc.path)
			if got != tc.want {
				t.Errorf("Filter.Include(%q) = %v, want %v", tc.path, got, tc.want)
			}
		})
	}
}

func TestParseCSV(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input string
		want  []string
	}{
		{name: "empty string", input: "", want: nil},
		{name: "only comma", input: ",", want: []string{}},
		{name: "simple list", input: "a,b,c", want: []string{"a", "b", "c"}},
		{name: "whitespace trim and empty drop", input: " a , ,b ", want: []string{"a", "b"}},
		{name: "single value", input: "foo", want: []string{"foo"}},
		{name: "whitespace only", input: "   ", want: nil},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := ParseCSV(tc.input)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("ParseCSV(%q) = %v, want %v", tc.input, got, tc.want)
			}
		})
	}
}
