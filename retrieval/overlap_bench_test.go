package retrieval

import (
	"strings"
	"testing"
)

func benchTokens(n int) []string {
	words := strings.Repeat("alpha bravo charlie delta echo foxtrot ", 10)
	tokens := strings.Fields(words)
	if n > len(tokens) {
		n = len(tokens)
	}
	return tokens[:n]
}

func BenchmarkOverlapRatio_Map_4x40(b *testing.B) {
	probe, ref := benchTokens(4), benchTokens(40)
	b.ResetTimer()
	for b.Loop() {
		overlapRatio(probe, ref)
	}
}

func BenchmarkOverlapRatioScan_4x40(b *testing.B) {
	probe, ref := benchTokens(4), benchTokens(40)
	b.ResetTimer()
	for b.Loop() {
		overlapRatioScan(probe, ref)
	}
}
