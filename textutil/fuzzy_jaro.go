package textutil

import (
	"github.com/adrg/strutil/metrics"
)

var caseInsensitiveJaroWinkler = &metrics.JaroWinkler{CaseSensitive: false}

// StringSimilarity returns the case-insensitive Jaro-Winkler similarity of a
// and b, a number in [0,1].
func StringSimilarity(a, b string) float64 {
	return caseInsensitiveJaroWinkler.Compare(a, b)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
