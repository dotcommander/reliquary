package dedup

import (
	"fmt"
	"testing"
)

func BenchmarkIndex(b *testing.B) {
	identity := func(s string) string { return s }
	items := make([]string, 1000)
	for i := range items {
		items[i] = fmt.Sprintf("document number %d with some filler text", i)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		d := NewDetector[string](SimHash, identity)
		d.Index(items)
	}
}

func BenchmarkFindNearDuplicates(b *testing.B) {
	identity := func(s string) string { return s }
	items := make([]string, 1000)
	for i := range items {
		items[i] = fmt.Sprintf("document number %d with some filler text", i)
	}

	d := NewDetector[string](SimHash, identity)
	d.Index(items)

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		d.FindNearDuplicates(3)
	}
}
