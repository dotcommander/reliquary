package retrieval

import (
	"path/filepath"
	"strings"
	"unicode"
)

type Metadata struct {
	Title    string
	Headings []string
	Path     string
}

func ExtractMetadata(path string, content string) Metadata {
	meta := Metadata{Path: path, Title: strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))}
	for line := range strings.SplitSeq(content, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		if strings.HasPrefix(trimmed, "title:") || strings.HasPrefix(trimmed, "Title:") {
			_, value, _ := strings.Cut(trimmed, ":")
			meta.Title = strings.Trim(strings.TrimSpace(value), `\"'`)
			continue
		}
		if strings.HasPrefix(trimmed, "#") {
			heading := strings.TrimSpace(strings.TrimLeft(trimmed, "#"))
			if heading != "" {
				meta.Headings = append(meta.Headings, heading)
			}
		}
	}
	return meta
}

func MetadataScore(query string, meta Metadata) float64 {
	queryTerms := termSet(query)
	if len(queryTerms) == 0 {
		return 0
	}
	candidates := []struct {
		text   string
		weight float64
	}{
		{meta.Title, 1.0},
		{strings.Join(meta.Headings, " "), 0.85},
		{filepath.Base(meta.Path), 0.65},
	}
	best := 0.0
	for _, candidate := range candidates {
		score := weightedOverlap(queryTerms, termSet(candidate.text)) * candidate.weight
		if score > best {
			best = score
		}
	}
	return best
}

func termSet(text string) map[string]bool {
	terms := make(map[string]bool)
	for _, term := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}) {
		if len(term) >= 2 {
			terms[term] = true
		}
	}
	return terms
}

func weightedOverlap(queryTerms map[string]bool, candidateTerms map[string]bool) float64 {
	if len(queryTerms) == 0 || len(candidateTerms) == 0 {
		return 0
	}
	matches := 0
	for term := range queryTerms {
		if candidateTerms[term] {
			matches++
		}
	}
	return float64(matches) / float64(len(queryTerms))
}
