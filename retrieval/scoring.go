package retrieval

import "math"

type ScoreBand string

const (
	BandWeak   ScoreBand = "weak"
	BandMedium ScoreBand = "medium"
	BandStrong ScoreBand = "strong"
)

type ScoreComponents struct {
	Semantic float64
	Keyword  float64
	Filename float64
	Metadata float64
}

// CalibratedScore compresses hybrid signals into a stable 0..1 score.
func CalibratedScore(c ScoreComponents) float64 {
	score := 0.62*normalizeCosine(c.Semantic) +
		0.18*clamp01(c.Keyword) +
		0.10*clamp01(c.Filename) +
		0.10*clamp01(c.Metadata)
	return clamp01(score)
}

func Band(score float64) ScoreBand {
	switch {
	case score >= 0.75:
		return BandStrong
	case score >= 0.45:
		return BandMedium
	default:
		return BandWeak
	}
}

func normalizeCosine(score float64) float64 {
	if math.IsNaN(score) {
		return 0
	}
	return clamp01((score + 1) / 2)
}

func clamp01(v float64) float64 {
	switch {
	case v < 0 || math.IsNaN(v):
		return 0
	case v > 1:
		return 1
	default:
		return v
	}
}
