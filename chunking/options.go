package chunking

// SemanticOption configures SemanticOpts fluently for NewSemantic.
type SemanticOption func(*SemanticOpts)

// NewSemantic is a fluent constructor over NewSemanticChunker: it starts from
// zero-value SemanticOpts (package defaults then apply) and applies the given
// options. Returns ErrNilEmbedder if embedder is nil.
func NewSemantic(embedder BatchEmbedder, opts ...SemanticOption) (*SemanticChunker, error) {
	var o SemanticOpts
	for _, opt := range opts {
		opt(&o)
	}
	return NewSemanticChunker(embedder, o)
}

// WithMaxChunkChars sets the hard ceiling per chunk.
func WithMaxChunkChars(n int) SemanticOption {
	return func(o *SemanticOpts) { o.MaxChunkChars = n }
}

// WithMinChunkChars merges groups smaller than n.
func WithMinChunkChars(n int) SemanticOption {
	return func(o *SemanticOpts) { o.MinChunkChars = n }
}

// WithBreakSensitivity sets the stddev multiplier (higher = fewer breaks).
func WithBreakSensitivity(f float64) SemanticOption {
	return func(o *SemanticOpts) { o.BreakSensitivity = f }
}

// WithSmoothingWindow sets the similarity smoothing window (0 disables).
func WithSmoothingWindow(n int) SemanticOption {
	return func(o *SemanticOpts) { o.SmoothingWindow = n }
}

// WithCoherenceWindow sets the two-sided coherence gate (0 disables).
func WithCoherenceWindow(n int) SemanticOption {
	return func(o *SemanticOpts) { o.CoherenceWindow = n }
}
