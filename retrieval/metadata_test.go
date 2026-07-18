package retrieval

import (
	"math"
	"reflect"
	"testing"
)

func TestExtractMetadata(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		path     string
		content  string
		wantMeta Metadata
	}{
		{
			name:    "title line recognized and trimmed",
			path:    "doc.md",
			content: "title: My Doc\nsome body text",
			wantMeta: Metadata{
				Title:    "My Doc",
				Headings: nil,
				Path:     "doc.md",
			},
		},
		{
			name:    "markdown headings stripped of hash and trimmed",
			path:    "doc.md",
			content: "# Heading One\n## Heading Two\nbody",
			wantMeta: Metadata{
				Title:    "doc",
				Headings: []string{"Heading One", "Heading Two"},
				Path:     "doc.md",
			},
		},
		{
			name:    "no title line uses path base without extension",
			path:    "docs/guide.md",
			content: "just some content",
			wantMeta: Metadata{
				Title:    "guide",
				Headings: nil,
				Path:     "docs/guide.md",
			},
		},
		{
			name:    "empty content title from path base",
			path:    "docs/reference.txt",
			content: "",
			wantMeta: Metadata{
				Title:    "reference",
				Headings: nil,
				Path:     "docs/reference.txt",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := ExtractMetadata(tc.path, tc.content)
			if got.Title != tc.wantMeta.Title {
				t.Errorf("Title: got %q, want %q", got.Title, tc.wantMeta.Title)
			}
			if got.Path != tc.wantMeta.Path {
				t.Errorf("Path: got %q, want %q", got.Path, tc.wantMeta.Path)
			}
			if !reflect.DeepEqual(got.Headings, tc.wantMeta.Headings) {
				t.Errorf("Headings: got %v, want %v", got.Headings, tc.wantMeta.Headings)
			}
		})
	}
}

func TestMetadataScore(t *testing.T) {
	t.Parallel()

	const eps = 1e-9

	tests := []struct {
		name      string
		meta      Metadata
		query     string
		wantScore float64
	}{
		{
			// queryTerms={"machine"}, Title termSet has "machine" → overlap=1/1=1.0, weight=1.0 → best=1.0
			name:      "query term matches title word",
			meta:      Metadata{Title: "machine learning"},
			query:     "machine",
			wantScore: 1.0,
		},
		{
			// empty query → termSet returns empty map → returns 0 immediately
			name:      "empty query returns zero",
			meta:      Metadata{Title: "machine learning"},
			query:     "",
			wantScore: 0.0,
		},
		{
			// "zebra" has no overlap with "machine learning"
			name:      "no overlap returns zero",
			meta:      Metadata{Title: "machine learning"},
			query:     "zebra",
			wantScore: 0.0,
		},
		{
			// Title empty, Headings=["neural networks"], query="neural"
			// queryTerms={"neural"}, heading termSet has "neural" → overlap=1/1=1.0, weight=0.85 → best=0.85
			name:      "query matches heading only uses heading weight",
			meta:      Metadata{Title: "", Headings: []string{"neural networks"}},
			query:     "neural",
			wantScore: 0.85,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := MetadataScore(tc.query, tc.meta)
			if math.Abs(got-tc.wantScore) > eps {
				t.Errorf("MetadataScore(%q, ...): got %v, want %v", tc.query, got, tc.wantScore)
			}
		})
	}
}
