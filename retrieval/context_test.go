package retrieval

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

type contextTokenCounterFunc func(string) (int, error)

func (f contextTokenCounterFunc) Count(text string) (int, error) {
	return f(text)
}

func runeTokenCounter(text string) (int, error) {
	return len([]rune(text)), nil
}

func TestFormatContextDefaults(t *testing.T) {
	results := []*Result{
		{ID: "one", Content: "alpha", Metadata: map[string]any{"keep": "value"}},
		nil,
		{ID: "empty"},
		{ID: "two", Content: "beta"},
	}
	before := make([]Result, 0, 3)
	for _, result := range results {
		if result != nil {
			before = append(before, *result)
		}
	}

	got, err := FormatContext(results)
	if err != nil {
		t.Fatalf("FormatContext() error = %v", err)
	}
	if want := "alpha\n\nbeta"; got != want {
		t.Fatalf("FormatContext() = %q, want %q", got, want)
	}
	var after []Result
	for _, result := range results {
		if result != nil {
			after = append(after, *result)
		}
	}
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("FormatContext mutated results: %#v", results)
	}
}

func TestFormatContextEmptyInput(t *testing.T) {
	got, err := FormatContext([]*Result{nil, {}, {Content: ""}}, nil)
	if err != nil {
		t.Fatalf("FormatContext() error = %v", err)
	}
	if got != "" {
		t.Fatalf("FormatContext() = %q, want empty output", got)
	}
}

func TestFormatContextHeaderSourceFallbackAndSeparator(t *testing.T) {
	results := []*Result{
		{ID: "id-1", DocumentID: "doc-1", Filename: "one.md", Content: "alpha"},
		{ID: "id-2", DocumentID: "doc-2", Content: "beta"},
		{ID: "id-3", Content: "gamma"},
	}
	got, err := FormatContext(results, WithHeader("Source %s (%s)"), WithSeparator("\n---\n"))
	if err != nil {
		t.Fatalf("FormatContext() error = %v", err)
	}
	want := "Source one.md (one.md)\nalpha\n---\nSource doc-2 (doc-2)\nbeta\n---\nSource id-3 (id-3)\ngamma"
	if got != want {
		t.Fatalf("FormatContext custom = %q, want %q", got, want)
	}
}

func TestFormatContextHeaderGrammar(t *testing.T) {
	result := &Result{
		Filename: "literal-%d-%s-%%",
		Content:  "content",
		Metadata: map[string]any{
			ContextStartLineKey: 7,
			ContextEndLineKey:   9,
		},
	}
	got, err := FormatContext([]*Result{result}, WithHeader("%% %s %d %d %d %q %"))
	if err != nil {
		t.Fatalf("FormatContext() error = %v", err)
	}
	want := "% literal-%d-%s-%% 7 9 %d %q %\ncontent"
	if got != want {
		t.Fatalf("FormatContext() = %q, want %q", got, want)
	}
}

func TestFormatContextLineMetadata(t *testing.T) {
	var decoded map[string]any
	decoder := json.NewDecoder(strings.NewReader(`{"reliquary.context.start_line":12,"reliquary.context.end_line":14}`))
	decoder.UseNumber()
	if err := decoder.Decode(&decoded); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name     string
		metadata map[string]any
		want     string
	}{
		{name: "integers", metadata: map[string]any{ContextStartLineKey: int32(2), ContextEndLineKey: uint64(3)}, want: "2-3"},
		{name: "integral floats", metadata: map[string]any{ContextStartLineKey: float32(4), ContextEndLineKey: float64(5)}, want: "4-5"},
		{name: "json numbers", metadata: decoded, want: "12-14"},
		{name: "json integral decimal", metadata: map[string]any{ContextStartLineKey: json.Number("6.0"), ContextEndLineKey: json.Number("7e0")}, want: "6-7"},
		{name: "json fraction beyond float precision", metadata: map[string]any{ContextStartLineKey: json.Number("1.0000000000000000000000000000000000000000000000000000000000000000000000000000001"), ContextEndLineKey: 2}, want: "0-0"},
		{name: "missing pair", metadata: map[string]any{ContextStartLineKey: 1}, want: "0-0"},
		{name: "fractional", metadata: map[string]any{ContextStartLineKey: 1.5, ContextEndLineKey: 2}, want: "0-0"},
		{name: "nonpositive", metadata: map[string]any{ContextStartLineKey: 0, ContextEndLineKey: 2}, want: "0-0"},
		{name: "wrong type", metadata: map[string]any{ContextStartLineKey: "1", ContextEndLineKey: 2}, want: "0-0"},
		{name: "overflow", metadata: map[string]any{ContextStartLineKey: json.Number("9223372036854775808"), ContextEndLineKey: json.Number("9223372036854775809")}, want: "0-0"},
		{name: "reversed", metadata: map[string]any{ContextStartLineKey: 3, ContextEndLineKey: 2}, want: "0-0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := FormatContext([]*Result{{Content: "body", Metadata: tt.metadata}}, WithHeader("%d-%d"))
			if err != nil {
				t.Fatalf("FormatContext() error = %v", err)
			}
			if want := tt.want + "\nbody"; got != want {
				t.Fatalf("FormatContext() = %q, want %q", got, want)
			}
		})
	}
}

func TestFormatContextTokenBudget(t *testing.T) {
	counter := contextTokenCounterFunc(runeTokenCounter)
	tests := []struct {
		name    string
		results []*Result
		opts    []ContextOption
		want    string
	}{
		{
			name:    "exact fit",
			results: []*Result{{Content: "one"}, {Content: "two"}},
			opts:    []ContextOption{WithMaxTokens(8, counter)},
			want:    "one\n\ntwo",
		},
		{
			name:    "overflow keeps complete prefix",
			results: []*Result{{Content: "one"}, {Content: "two"}},
			opts:    []ContextOption{WithMaxTokens(7, counter)},
			want:    "one",
		},
		{
			name:    "header counts",
			results: []*Result{{ID: "s", Content: "one"}},
			opts:    []ContextOption{WithHeader("[%s]"), WithMaxTokens(5, counter)},
			want:    "",
		},
		{
			name:    "separator counts",
			results: []*Result{{Content: "a"}, {Content: "b"}},
			opts:    []ContextOption{WithSeparator("---"), WithMaxTokens(4, counter)},
			want:    "a",
		},
		{
			name:    "first block overflow",
			results: []*Result{{Content: "large"}, {Content: "x"}},
			opts:    []ContextOption{WithMaxTokens(1, counter)},
			want:    "",
		},
		{
			name:    "does not skip oversized middle block",
			results: []*Result{{Content: "a"}, {Content: "oversized"}, {Content: "b"}},
			opts:    []ContextOption{WithMaxTokens(3, counter)},
			want:    "a",
		},
		{
			name:    "last max option wins",
			results: []*Result{{Content: "abc"}},
			opts:    []ContextOption{WithMaxTokens(1, counter), WithMaxTokens(3, counter)},
			want:    "abc",
		},
		{
			name:    "last separator option wins",
			results: []*Result{{Content: "a"}, {Content: "b"}},
			opts:    []ContextOption{WithSeparator("x"), WithSeparator("|")},
			want:    "a|b",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := FormatContext(tt.results, tt.opts...)
			if err != nil {
				t.Fatalf("FormatContext() error = %v", err)
			}
			if got != tt.want {
				t.Fatalf("FormatContext() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestFormatContextTokenLimitModes(t *testing.T) {
	for _, limit := range []int{-1, 0} {
		called := false
		counter := contextTokenCounterFunc(func(string) (int, error) {
			called = true
			return 0, nil
		})
		got, err := FormatContext([]*Result{{Content: "content"}}, WithMaxTokens(limit, counter))
		if err != nil || got != "" {
			t.Fatalf("FormatContext(limit %d) = %q, %v; want empty output, nil", limit, got, err)
		}
		if called {
			t.Fatalf("counter called for nonpositive limit %d", limit)
		}
	}

	called := false
	unusedCounter := contextTokenCounterFunc(func(string) (int, error) {
		called = true
		return 0, nil
	})
	got, err := FormatContext([]*Result{{Content: "content"}})
	if err != nil || got != "content" {
		t.Fatalf("FormatContext(unbounded) = %q, %v", got, err)
	}
	_ = unusedCounter
	if called {
		t.Fatal("counter called when limit was omitted")
	}

	if got, err := FormatContext(nil, WithMaxTokens(1, nil)); err == nil || got != "" {
		t.Fatalf("FormatContext(nil counter) = %q, %v; want empty output and error", got, err)
	}
}

func TestFormatContextCounterFailuresReturnNoPartialOutput(t *testing.T) {
	wantErr := errors.New("counter failed")
	calls := 0
	counter := contextTokenCounterFunc(func(string) (int, error) {
		calls++
		if calls == 2 {
			return 0, wantErr
		}
		return 1, nil
	})
	got, err := FormatContext([]*Result{{Content: "one"}, {Content: "two"}}, WithMaxTokens(10, counter))
	if !errors.Is(err, wantErr) || got != "" {
		t.Fatalf("FormatContext(counter error) = %q, %v; want empty output and wrapped error", got, err)
	}

	got, err = FormatContext([]*Result{{Content: "one"}}, WithMaxTokens(10, contextTokenCounterFunc(func(string) (int, error) {
		return -1, nil
	})))
	if err == nil || got != "" || !strings.Contains(err.Error(), "negative count") {
		t.Fatalf("FormatContext(negative count) = %q, %v", got, err)
	}
}
