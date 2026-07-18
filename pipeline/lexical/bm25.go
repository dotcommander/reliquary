package lexical

import "math"

// BM25Params configures local BM25 scoring.
type BM25Params struct {
	K1 float64
	B  float64
}

// DefaultBM25Params returns common BM25 defaults.
func DefaultBM25Params() BM25Params {
	return BM25Params{K1: 1.2, B: 0.75}
}

// BM25Score returns a higher-is-better local lexical score.
//
// It uses Robertson/Sparck Jones IDF with a positive offset:
//
//	log(1 + (N - df + 0.5)/(df + 0.5))
//
// Raw SQLite bm25(), PostgreSQL ts_rank, and other external score spaces are
// intentionally not normalized here; callers should adapt their rank order at
// the application boundary.
func BM25Score(query Query, doc DocumentStats, corpus CorpusStats, params BM25Params) float64 {
	if query.Empty() || doc.Length <= 0 || corpus.DocumentCount <= 0 || corpus.AverageDocumentLength <= 0 {
		return 0
	}
	params = normalizeBM25Params(params)

	var score float64
	for term, queryCount := range query.Terms {
		if queryCount <= 0 {
			continue
		}
		tf := doc.TermFrequency[term]
		df := corpus.DocumentFrequency[term]
		if tf <= 0 || df <= 0 {
			continue
		}

		idf := math.Log(1 + (float64(corpus.DocumentCount-df)+0.5)/(float64(df)+0.5))
		denom := float64(tf) + params.K1*(1-params.B+params.B*float64(doc.Length)/corpus.AverageDocumentLength)
		if denom <= 0 {
			continue
		}
		tfWeight := (float64(tf) * (params.K1 + 1)) / denom
		score += float64(queryCount) * idf * tfWeight
	}
	return score
}

func normalizeBM25Params(params BM25Params) BM25Params {
	defaults := DefaultBM25Params()
	if params.K1 <= 0 {
		params.K1 = defaults.K1
	}
	if params.B < 0 || params.B > 1 || params.B == 0 {
		params.B = defaults.B
	}
	return params
}
