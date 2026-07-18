package retrieval

// WeightOption adjusts a Weights value for NewScorerOpts.
type WeightOption func(*Weights)

// NewScorerOpts builds a Scorer from DefaultWeights plus the given overrides.
// It is the fluent alternative to constructing a Weights struct and calling
// NewScorer.
func NewScorerOpts(opts ...WeightOption) *Scorer {
	w := DefaultWeights()
	for _, opt := range opts {
		opt(&w)
	}
	return NewScorer(w)
}

// Embedding sets the embedding-similarity weight.
func Embedding(v float64) WeightOption { return func(w *Weights) { w.Embedding = v } }

// Keyword sets the keyword-overlap weight.
func Keyword(v float64) WeightOption { return func(w *Weights) { w.Keyword = v } }

// Filename sets the filename-overlap weight.
func Filename(v float64) WeightOption { return func(w *Weights) { w.Filename = v } }

// Recency sets the recency-salience weight.
func Recency(v float64) WeightOption { return func(w *Weights) { w.Recency = v } }

// Importance sets the importance-salience weight.
func Importance(v float64) WeightOption { return func(w *Weights) { w.Importance = v } }
