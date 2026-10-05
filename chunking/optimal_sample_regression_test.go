package chunking

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestStrategicSampleAllBranchesBoundedRegression(t *testing.T) {
	t.Parallel()
	for _, source := range []string{"", "abc", strings.Repeat("x", 10000), strings.Repeat("猫犬", 5000)} {
		for _, size := range []int{-1, 0, 1, 2, 3, 4, 8, 50, 100, 200, 10000} {
			got := StrategicSample(source, size)
			budget := max(size, 0)
			if utf8.RuneCountInString(got) > budget || !utf8.ValidString(got) {
				t.Fatalf("sample size=%d runes=%d", size, utf8.RuneCountInString(got))
			}
			if size > 0 && utf8.RuneCountInString(source) <= size && got != source {
				t.Fatal("small content changed")
			}
		}
	}
	source := strings.Repeat("a", 1000) + strings.Repeat("b", 1000) + strings.Repeat("c", 1000)
	got := StrategicSample(source, 200)
	if !strings.Contains(got, seamMiddle) || !strings.Contains(got, seamEnd) || !strings.HasPrefix(got, "a") || !strings.HasSuffix(got, "c") {
		t.Fatalf("lost strategic samples: %q", got)
	}
}
