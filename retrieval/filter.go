package retrieval

import (
	"path/filepath"
	"slices"
	"strings"
)

type Filter struct {
	IncludeExts []string
	IgnoreDirs  []string
	IgnoreExts  []string
}

func DefaultFilter() Filter {
	return Filter{
		IgnoreDirs: []string{".git", ".hg", ".svn", "node_modules", "vendor", "dist", "build", "__pycache__"},
		IgnoreExts: []string{".lock", ".sum", ".map"},
	}
}

func (f Filter) Include(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	if len(f.IncludeExts) > 0 && !containsNorm(f.IncludeExts, ext) {
		return false
	}
	if containsNorm(f.IgnoreExts, ext) {
		return false
	}
	for part := range strings.SplitSeq(filepath.ToSlash(path), "/") {
		if slices.Contains(f.IgnoreDirs, part) {
			return false
		}
	}
	return true
}

func ParseCSV(value string) []string {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func containsNorm(values []string, target string) bool {
	target = strings.ToLower(target)
	for _, value := range values {
		value = strings.ToLower(strings.TrimSpace(value))
		if value != "" && !strings.HasPrefix(value, ".") {
			value = "." + value
		}
		if value == target {
			return true
		}
	}
	return false
}
