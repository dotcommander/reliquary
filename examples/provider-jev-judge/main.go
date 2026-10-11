// Command provider-jev-judge demonstrates adapting a typed Jev evaluation client
// to Reliquary's judgment contracts (Choice, Score, and Noul).
//
// It shows:
// 1. Implementing a caller-owned Jev client.
// 2. Constructing adapter/jev.New.
// 3. Executing strict typed choices with probability distributions.
// 4. Executing 0-100 bounded numeric scores.
// 5. Executing strict yes/no noul evaluations.
//
// Run from the repository root:
//
//	GOWORK=off go run ./examples/provider-jev-judge
package main

import (
	"context"
	"fmt"

	jevadapter "github.com/dotcommander/reliquary/adapter/jev"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
	"github.com/dotcommander/reliquary/judgment"
)

func main() {
	examplekit.Run(run)
}

// demoJevClient simulates a typed Jev evaluation provider.
type demoJevClient struct{}

func (demoJevClient) Evaluate(_ context.Context, call jevadapter.Call) (jevadapter.Result, error) {
	switch call.Kind {
	case jevadapter.KindChoice:
		return jevadapter.Result{
			Selected: call.Options[0].ID,
			Distribution: []jevadapter.Mass{
				{Option: call.Options[0].ID, Probability: 0.88},
				{Option: call.Options[1].ID, Probability: 0.12},
			},
			Confidence: 0.88,
		}, nil
	case jevadapter.KindScore:
		return jevadapter.Result{
			Value:      94.5,
			Confidence: 0.95,
		}, nil
	case jevadapter.KindNoul:
		return jevadapter.Result{
			Verdict:    "yes",
			Confidence: 0.92,
		}, nil
	default:
		return jevadapter.Result{}, fmt.Errorf("unsupported kind: %s", call.Kind)
	}
}

func run(ctx context.Context) error {
	adapter, err := jevadapter.New(demoJevClient{})
	if err != nil {
		return fmt.Errorf("new jev adapter: %w", err)
	}

	// 1. Typed Choice evaluation
	choice, err := adapter.Choose(ctx, judgment.ChoiceRequest{
		Task: "Route user intent to retrieval or summarization",
		Options: []judgment.Option{
			{ID: "retrieve", Description: "Search indexed knowledge documents"},
			{ID: "summarize", Description: "Summarize existing context"},
		},
		Context: []string{"User query: Where is the documentation for vector indexing?"},
	})
	if err != nil {
		return fmt.Errorf("choose: %w", err)
	}
	fmt.Printf("Jev Choice: %s\n", choice.Selected)
	for _, prob := range choice.Distribution {
		fmt.Printf("  - %s: %.2f\n", prob.Option, prob.Probability)
	}

	// 2. Typed Score evaluation (0 - 100)
	score, err := adapter.Score(ctx, judgment.ScoreRequest{
		Task:    "Evaluate retrieval passage relevance to user query",
		Target:  "SQLite provides an embedded relational database engine.",
		Context: []string{"Query: embedded SQL database"},
	})
	if err != nil {
		return fmt.Errorf("score: %w", err)
	}
	fmt.Printf("\nJev Score: %.1f/100 (confidence: %.2f)\n", score.Value, score.Confidence)

	// 3. Typed Noul (yes/no) evaluation
	noul, err := adapter.Ask(ctx, judgment.NoulRequest{
		Question: "Does the retrieved passage answer the user's question?",
		Context:  []string{"Passage describes SQLite indexing architecture."},
	})
	if err != nil {
		return fmt.Errorf("noul: %w", err)
	}
	fmt.Printf("\nJev Noul verdict: %s (confidence: %.2f)\n", noul.Verdict, noul.Confidence)
	return nil
}
