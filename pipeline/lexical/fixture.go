package lexical

import "github.com/dotcommander/reliquary/retrieval"

// Judgment describes expected lexical relevance for one document.
type Judgment struct {
	ID        string
	Relevance float64
	Topic     string
}

// Fixture captures one lexical rank check.
type Fixture struct {
	ID        string
	Query     string
	Judgments []Judgment
}

// EvaluateFixture evaluates lexical candidates for one fixture using the
// shared retrieval metrics package so IR metric semantics stay in one place.
func EvaluateFixture(fixture Fixture, candidates RankedList, k int) retrieval.Metrics {
	relevant := make(map[string]float64, len(fixture.Judgments))
	topics := make(map[string]string, len(fixture.Judgments))
	for _, judgment := range fixture.Judgments {
		if judgment.Relevance > 0 {
			relevant[judgment.ID] = judgment.Relevance
		}
		if judgment.Topic != "" {
			topics[judgment.ID] = judgment.Topic
		}
	}

	results := ToRetrievalResultsWithTopics(candidates, topics)

	return retrieval.Evaluate(retrieval.EvalQuery{
		ID:         fixture.ID,
		Relevant:   relevant,
		TopicByDoc: topics,
	}, results, k)
}
