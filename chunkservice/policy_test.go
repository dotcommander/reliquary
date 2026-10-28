package chunkservice

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/dotcommander/reliquary/chunking"
)

func TestValidatePolicyCustomSeparators(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name       string
		separators []string
		profile    SeparatorProfile
		strategy   chunking.Strategy
		invalid    bool
	}{
		{"absent", nil, "", chunking.Recursive, false},
		{"empty", []string{}, "", chunking.Recursive, true},
		{"custom", []string{"|", ";"}, "", chunking.Recursive, false},
		{"fallback only", []string{""}, "", chunking.Recursive, false},
		{"explicit fallback", []string{"|", ""}, "", chunking.Recursive, false},
		{"fallback first", []string{"", "|"}, "", chunking.Recursive, true},
		{"named conflict", []string{"|"}, ProfileDefaultText, chunking.Recursive, true},
		{"wrong strategy", []string{"|"}, "", chunking.WordBoundary, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := Request{ChunkSize: 8, Strategy: tc.strategy, SeparatorProfile: tc.profile, CustomSeparators: tc.separators}
			err := req.ValidatePolicy()
			if (err != nil) != tc.invalid {
				t.Fatalf("ValidatePolicy() = %v, invalid=%v", err, tc.invalid)
			}
			if err != nil && !IsRequestValidationError(err) {
				t.Fatalf("untyped error: %v", err)
			}
			if !tc.invalid && req.Validate() == nil {
				t.Fatal("execution accepted missing text")
			}
		})
	}
}

func TestCustomSeparatorsJSONAndBatch(t *testing.T) {
	t.Parallel()
	for _, separators := range []any{nil, []string{}, []string{"|", ""}} {
		encoded, err := json.Marshal(map[string]any{"texts": []string{"a|b"}, "chunk_size": 2, "strategy": chunking.Recursive, "custom_separators": separators})
		if err != nil {
			t.Fatal(err)
		}
		var req BatchRequest
		if err := json.Unmarshal(encoded, &req); err != nil {
			t.Fatal(err)
		}
		single := req.ToChunkRequest("a|b")
		if !reflect.DeepEqual(req.CustomSeparators, single.CustomSeparators) {
			t.Fatal("batch lost separator policy")
		}
		wantError := separators != nil && len(separators.([]string)) == 0
		if (req.Validate() != nil) != wantError {
			t.Fatalf("batch validation mismatch for %s", encoded)
		}
	}
}

func TestCustomSeparatorsPreserveEmptyJSON(t *testing.T) {
	t.Parallel()
	for _, separators := range [][]string{nil, {}} {
		original := Request{ChunkSize: 8, Strategy: chunking.Recursive, CustomSeparators: separators}
		data, err := json.Marshal(original)
		if err != nil {
			t.Fatal(err)
		}
		var decoded Request
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(decoded.CustomSeparators, separators) {
			t.Fatalf("round trip lost nil/empty distinction: %s", data)
		}
	}
}

func TestCustomSeparatorOrder(t *testing.T) {
	t.Parallel()
	cases := []struct {
		separators []string
		want       []string
	}{
		{[]string{"|", ";"}, []string{"aaa", "bbb;ccc"}},
		{[]string{";", "|"}, []string{"aaa|bbb", "ccc"}},
	}
	for _, tc := range cases {
		chunks, _, err := New().Chunks(context.Background(), Request{Text: "aaa|bbb;ccc", ChunkSize: 7, Strategy: chunking.Recursive, CustomSeparators: tc.separators})
		if err != nil {
			t.Fatal(err)
		}
		got := make([]string, len(chunks))
		for i, c := range chunks {
			got[i] = c.Text
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("cascade %q = %q, want %q", tc.separators, got, tc.want)
		}
	}
}

func TestRecursiveMultilingualSpans(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name, text string
		separators []string
		profile    SeparatorProfile
	}{
		{"custom fallback", "你好世界🌍再见世界", []string{"|"}, ""},
		{"custom unicode", "甲乙。丙丁。戊己。庚辛", []string{"。"}, ""},
		{"named CJK", "甲乙。丙丁。戊己。庚辛", nil, ProfileCJKThai},
		{"named Thai", "ภาษา\u200bไทย\u200bทดสอบ\u200bแบ่ง", nil, ProfileCJKThai},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			before := append([]string(nil), tc.separators...)
			chunks, _, err := New().Chunks(context.Background(), Request{Text: tc.text, ChunkSize: 5, Strategy: chunking.Recursive, CustomSeparators: tc.separators, SeparatorProfile: tc.profile})
			if err != nil {
				t.Fatal(err)
			}
			if len(chunks) < 2 {
				t.Fatalf("expected splitting: %v", chunks)
			}
			if !reflect.DeepEqual(before, tc.separators) {
				t.Fatal("caller separators changed")
			}
			for _, c := range chunks {
				if !utf8.ValidString(c.Text) || c.StartChar < 0 || c.EndChar > len(tc.text) || c.StartChar >= c.EndChar {
					t.Fatalf("invalid span: %+v", c)
				}
				if tc.text[c.StartChar:c.EndChar] != c.Text {
					t.Fatalf("span mismatch: %+v", c)
				}
				if utf8.RuneCountInString(c.Text) > 5 {
					t.Fatalf("size exceeded: %+v", c)
				}
			}
		})
	}
}

func TestFinalTokenCeilings(t *testing.T) {
	t.Parallel()
	exact := "hello world"
	exactCount, err := chunking.CountTokens(exact, string(EncodingCL100K))
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name, text string
		strategy   chunking.Strategy
		limit      int
		preflight  chunking.TokenCounter
	}{
		{"exact", exact, chunking.WordBoundary, exactCount, nil},
		{"unbroken", strings.Repeat("abcdefghij", 40), chunking.WordBoundary, 8, nil},
		{"unicode", strings.Repeat("你好世界🌍。", 20), chunking.Recursive, 8, nil},
		{"semantic", strings.Repeat("One sentence here. Another sentence now. ", 20), chunking.Semantic, 8, nil},
		{"preflight then ceiling", strings.Repeat("One sentence here. Another sentence now. ", 20), chunking.SentenceBoundary, 4, fixedTokenCounter{limit: 20}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			svc := New().WithEmbedder(&multiStubEmbedder{dims: 3})
			chunks, _, err := svc.Chunks(context.Background(), Request{Text: tc.text, ChunkSize: 10000, Strategy: tc.strategy, MaxTokens: tc.limit, PreflightCounter: tc.preflight})
			if err != nil {
				t.Fatal(err)
			}
			if len(chunks) == 0 {
				t.Fatal("missing chunks")
			}
			if tc.name == "exact" && (len(chunks) != 1 || chunks[0].Text != exact) {
				t.Fatalf("exact-limit text changed: %+v", chunks)
			}
			for _, c := range chunks {
				count, err := chunking.CountTokens(c.Text, string(EncodingCL100K))
				if err != nil {
					t.Fatal(err)
				}
				if !utf8.ValidString(c.Text) || count > tc.limit || count != c.TokenCount {
					t.Fatalf("ceiling invalid: actual=%d limit=%d chunk=%+v", count, tc.limit, c)
				}
				if tc.preflight != nil && tc.preflight.CountTokens(c.Text) > tc.preflight.MaxTokens() {
					t.Fatalf("preflight exceeded: %+v", c)
				}
			}
		})
	}
}

func TestZeroFinalTokenCeiling(t *testing.T) {
	t.Parallel()
	text := strings.Repeat("one two three ", 20)
	chunks, _, err := New().Chunks(context.Background(), Request{Text: text, ChunkSize: 10000, Strategy: chunking.WordBoundary})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 {
		t.Fatalf("disabled ceiling split input into %d chunks", len(chunks))
	}
}
