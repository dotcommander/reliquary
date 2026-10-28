package retrieval

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

type byteCounter struct{}

func (byteCounter) Count(s string) (int, error) { return len(s), nil }

func TestContextEnrichment(t *testing.T) {
	t.Parallel()
	original := &Result{ID: "id", Content: " hello ", Metadata: map[string]any{"title": "Title"}}
	inputs := []*Result{nil, {Content: ""}, original, {Content: "skip"}, original, {Content: "later"}, nil}
	calls := 0
	transform := WithContentTransform(func(item ContextItem) (string, error) {
		calls++
		if item.Content == "skip" {
			return "", nil
		}
		return strings.TrimSpace(item.Content), nil
	})
	opts := []ContextOption{transform, WithHeader("ignored"), WithHeaderTemplate("{{.Number}} {{.Result.Metadata.title}} {{.Source}}"), WithCitationTemplate("[{{.Number}}]"), WithMaxTokens(len("[1] 1 Title id\nhello"), byteCounter{})}
	text, summary, err := FormatContextWithSummary(inputs, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if text != "[1] 1 Title id\nhello" {
		t.Fatalf("text = %q", text)
	}
	if !reflect.DeepEqual(summary.IncludedIndices, []int{2}) || !reflect.DeepEqual(summary.SkippedIndices, []int{0, 1, 3, 6}) || !reflect.DeepEqual(summary.OmittedIndices, []int{4, 5}) || summary.InputCount != 7 || summary.TokensUsed == nil || *summary.TokensUsed != len(text) {
		t.Fatalf("summary = %+v", summary)
	}
	if calls != 3 || original.Content != " hello " {
		t.Fatalf("calls=%d original=%q", calls, original.Content)
	}
	plain, err := FormatContext(inputs, opts...)
	if err != nil || plain != text {
		t.Fatalf("FormatContext = %q, %v", plain, err)
	}
}

func TestContextEnrichmentErrorsAndPrecedence(t *testing.T) {
	t.Parallel()
	sentinel := errors.New("callback")
	inputs := []*Result{{Content: "x"}}
	cases := []struct {
		name   string
		opts   []ContextOption
		target error
	}{
		{"malformed header", []ContextOption{WithHeaderTemplate("{{")}, nil},
		{"missing metadata", []ContextOption{WithHeaderTemplate("{{.Result.Metadata.missing}}")}, nil},
		{"malformed citation", []ContextOption{WithCitationTemplate("{{")}, nil},
		{"nil header", []ContextOption{WithHeaderFormatter(nil)}, nil},
		{"nil transform", []ContextOption{WithContentTransform(nil)}, nil},
		{"header callback", []ContextOption{WithHeaderFormatter(func(ContextItem) (string, error) { return "", sentinel })}, sentinel},
		{"transform callback", []ContextOption{WithContentTransform(func(ContextItem) (string, error) { return "", sentinel })}, sentinel},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			text, summary, err := FormatContextWithSummary(inputs, tc.opts...)
			if err == nil || text != "" || !reflect.DeepEqual(summary, FormatSummary{}) || (tc.target != nil && !errors.Is(err, tc.target)) {
				t.Fatalf("%q %+v %v", text, summary, err)
			}
			_, err2 := FormatContext(inputs, tc.opts...)
			if err2 == nil || err.Error() != err2.Error() {
				t.Fatalf("errors differ: %v %v", err, err2)
			}
		})
	}
	text, err := FormatContext(inputs, WithHeaderTemplate("{{"), WithHeaderFormatter(func(item ContextItem) (string, error) { return item.Content, nil }), WithHeader("legacy"), WithCitationTemplate("[{{.Number}}]"))
	if err != nil || text != "[1] legacy\nx" {
		t.Fatalf("%q %v", text, err)
	}
	text, err = FormatContext(inputs, WithCitationTemplate("[{{.Number}}]"), WithHeaderFormatter(func(item ContextItem) (string, error) { return "", nil }))
	if err != nil || text != "[1]\nx" {
		t.Fatalf("%q %v", text, err)
	}
}

func TestContextItemsAndConsecutiveNumbers(t *testing.T) {
	t.Parallel()
	result := &Result{Filename: "source", Content: " raw ", Metadata: map[string]any{ContextStartLineKey: 2, ContextEndLineKey: 4}}
	inputs := []*Result{nil, {Content: ""}, result, {Content: "skip"}, result}
	var transformed, headers []ContextItem
	text, summary, err := FormatContextWithSummary(inputs,
		WithContentTransform(func(item ContextItem) (string, error) {
			transformed = append(transformed, item)
			if item.Content == "skip" {
				return "", nil
			}
			return strings.TrimSpace(item.Content), nil
		}),
		WithHeaderFormatter(func(item ContextItem) (string, error) {
			headers = append(headers, item)
			return "", nil
		}),
		WithCitationTemplate("[{{.Number}}:{{.Content}}]"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if text != "[1:raw]\nraw\n\n[2:raw]\nraw" || !reflect.DeepEqual(summary.IncludedIndices, []int{2, 4}) || !reflect.DeepEqual(summary.SkippedIndices, []int{0, 1, 3}) || summary.TokensUsed != nil {
		t.Fatalf("text=%q summary=%+v", text, summary)
	}
	if len(transformed) != 3 || transformed[0].Content != " raw " || transformed[2].Number != 2 || len(headers) != 2 {
		t.Fatalf("transforms=%+v headers=%+v", transformed, headers)
	}
	for i, item := range headers {
		if item.Result != result || item.InputIndex != []int{2, 4}[i] || item.Number != i+1 || item.Source != "source" || item.StartLine != 2 || item.EndLine != 4 || !item.HasLines || item.Content != "raw" {
			t.Fatalf("header item = %+v", item)
		}
	}
	if result.Content != " raw " {
		t.Fatalf("result changed: %+v", result)
	}
}

func TestContextBudgetTelemetryAndSuffix(t *testing.T) {
	t.Parallel()
	inputs := []*Result{{Content: "a"}, {Content: "b"}, {Content: "c"}, nil, {Content: ""}, {Content: "d"}}
	calls := 0
	headerCalls := 0
	want := "[1] H\na\n\n[2] H\nb"
	text, summary, err := FormatContextWithSummary(inputs,
		WithContentTransform(func(item ContextItem) (string, error) { calls++; return item.Content, nil }),
		WithHeaderFormatter(func(item ContextItem) (string, error) { headerCalls++; return "H", nil }),
		WithCitationTemplate("[{{.Number}}]"), WithMaxTokens(len(want), byteCounter{}),
	)
	if err != nil || text != want || calls != 3 || headerCalls != 3 || !reflect.DeepEqual(summary.IncludedIndices, []int{0, 1}) || !reflect.DeepEqual(summary.OmittedIndices, []int{2, 5}) || !reflect.DeepEqual(summary.SkippedIndices, []int{3, 4}) || summary.TokensUsed == nil || *summary.TokensUsed != len(want) {
		t.Fatalf("text=%q summary=%+v calls=%d err=%v", text, summary, calls, err)
	}
	for _, tc := range []struct {
		name   string
		inputs []*Result
		opts   []ContextOption
	}{
		{"unlimited", inputs[:1], nil},
		{"empty", []*Result{nil, {Content: ""}}, []ContextOption{WithMaxTokens(1, byteCounter{})}},
		{"nonpositive", inputs, []ContextOption{WithMaxTokens(0, nil)}},
		{"first overflow", inputs[:1], []ContextOption{WithMaxTokens(1, byteCounter{}), WithCitationTemplate("[1]")}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, got, err := FormatContextWithSummary(tc.inputs, tc.opts...)
			if err != nil || got.TokensUsed != nil {
				t.Fatalf("summary=%+v err=%v", got, err)
			}
		})
	}
}
