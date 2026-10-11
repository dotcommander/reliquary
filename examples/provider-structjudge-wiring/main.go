// Command provider-structjudge-wiring demonstrates adapting a structured-output
// LLM completion client (e.g. JSON schema constrained generation) to Reliquary's
// judgment contracts.
//
// It shows:
// 1. Implementing a caller-owned structured generation client (structjudge.Client).
// 2. Constructing adapter/structjudge.New.
// 3. Executing Choice, Score, and Noul judgments through schema-forced completions.
//
// Run from the repository root:
//
//	GOWORK=off go run ./examples/provider-structjudge-wiring
package main

import (
	"context"
	"fmt"
	"strings"

	structjudgeadapter "github.com/dotcommander/reliquary/adapter/structjudge"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
	"github.com/dotcommander/reliquary/judgment"
)

func main() {
	examplekit.Run(run)
}

// schemaCompletionClient simulates a schema-constrained structured output LLM.
type schemaCompletionClient struct{}

func (schemaCompletionClient) Generate(_ context.Context, req structjudgeadapter.Request) ([]byte, error) {
	// The schema is inspectable; the client emits valid JSON conforming to req.Schema.
	schemaStr := string(req.Schema)
	switch {
	case strings.Contains(schemaStr, "distribution"):
		return []byte(`{"selected":"vector_search","distribution":[{"option":"vector_search","probability":0.85},{"option":"lexical_search","probability":0.15}]}`), nil
	case strings.Contains(schemaStr, "value"):
		return []byte(`{"value":96.0,"confidence":0.98}`), nil
	case strings.Contains(schemaStr, "verdict"):
		return []byte(`{"verdict":"yes","confidence":0.94}`), nil
	default:
		return nil, fmt.Errorf("unknown schema: %s", schemaStr)
	}
}

func run(ctx context.Context) error {
	judge, err := structjudgeadapter.New(schemaCompletionClient{})
	if err != nil {
		return fmt.Errorf("new structjudge adapter: %w", err)
	}

	// 1. Schema-forced Choice
	choice, err := judge.Choose(ctx, judgment.ChoiceRequest{
		Task: "Determine retrieval strategy",
		Options: []judgment.Option{
			{ID: "vector_search", Description: "Semantic embedding similarity"},
			{ID: "lexical_search", Description: "BM25 keyword search"},
		},
		Context: []string{"Query: semantic meaning of neural retrieval"},
	})
	if err != nil {
		return fmt.Errorf("choose: %w", err)
	}
	fmt.Printf("Structured Judge Choice: %s\n", choice.Selected)
	for _, p := range choice.Distribution {
		fmt.Printf("  - %s: %.2f\n", p.Option, p.Probability)
	}

	// 2. Schema-forced Score (0-100)
	score, err := judge.Score(ctx, judgment.ScoreRequest{
		Task:    "Score answer quality",
		Target:  "Hybrid search combines dense and sparse representations.",
		Context: []string{"Query: What is hybrid retrieval?"},
	})
	if err != nil {
		return fmt.Errorf("score: %w", err)
	}
	fmt.Printf("\nStructured Judge Score: %.1f/100 (confidence: %.2f)\n", score.Value, score.Confidence)

	// 3. Schema-forced Noul (yes/no)
	noul, err := judge.Ask(ctx, judgment.NoulRequest{
		Question: "Is this document safe to index?",
		Context:  []string{"Public open-source technical reference."},
	})
	if err != nil {
		return fmt.Errorf("noul: %w", err)
	}
	fmt.Printf("\nStructured Judge Noul: %s (confidence: %.2f)\n", noul.Verdict, noul.Confidence)
	return nil
}
