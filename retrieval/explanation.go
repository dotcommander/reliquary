package retrieval

// SearchExplanation describes how a retained search candidate moved through
// Reliquary's ranking stages. It is populated only for searches using
// reliquary.WithExplain and is not persistent index data.
type SearchExplanation struct {
	Hybrid          ScoreTrace
	HybridRank      int
	HybridScoreUsed bool
	RRF             *RRFExplanation
	Reranker        *RerankerExplanation
	MMR             *MMRExplanation
	FinalRank       int
}

// RRFExplanation describes reciprocal-rank fusion for one result. A zero lane
// rank and contribution mean that the result was absent from that lane.
type RRFExplanation struct {
	K                   float64
	VectorRank          int
	LexicalRank         int
	VectorContribution  float64
	LexicalContribution float64
	FusedScore          float64
	FusedRank           int
}

// RerankerExplanation describes the observable input and output of an external
// reranker. The Reranker interface does not expose model-internal reasoning.
type RerankerExplanation struct {
	InputRank int
	Score     float64
	Rank      int
}

// MMRExplanation describes one maximal-marginal-relevance selection.
// MaxSimilarity is the maximum positive cosine similarity to a previously
// selected item; missing, orthogonal, and negatively correlated embeddings
// contribute zero. Penalty is signed and is therefore zero or negative.
type MMRExplanation struct {
	Lambda                float64
	Relevance             float64
	MaxSimilarity         float64
	RelevanceContribution float64
	Penalty               float64
	SelectionScore        float64
}
