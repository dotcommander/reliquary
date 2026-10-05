package chunking

import "testing"

func TestHardCutBoundedSlidingWindowsRegression(t *testing.T) {
	t.Parallel()
	source := "猫犬鳥魚鹿馬牛羊猿熊"
	for _, overlap := range []int{0, 2, 3, 4, 9} {
		chunks := newHardCutChunker().Chunk(source, 4, overlap)
		assertRegressionChunkBounds(t, source, chunks, 4)
		effective := overlap
		if effective >= 4 {
			effective = 3
		}
		step := 4 - effective
		previousEnd := 0
		for i, c := range chunks {
			wantStart := i * step * 3
			if c.StartChar != wantStart || c.EndChar <= previousEnd {
				t.Fatalf("overlap %d window %d lost sliding provenance: %#v", overlap, i, c)
			}
			previousEnd = c.EndChar
		}
		if previousEnd != len(source) {
			t.Fatalf("lost tail: %d", previousEnd)
		}
	}
}
