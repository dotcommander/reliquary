package judgment_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/judgment"
)

func ExampleValidateChoiceRequest() {
	req := judgment.ChoiceRequest{
		Task: "Route user request",
		Options: []judgment.Option{
			{ID: "retrieve", Description: "Search indexed knowledge"},
			{ID: "summarize", Description: "Summarize provided context"},
		},
	}
	err := judgment.ValidateChoiceRequest(req)
	fmt.Println(err == nil)
	// Output: true
}

func ExampleValidateScoreRequest() {
	req := judgment.ScoreRequest{
		Task:   "Assess relevance",
		Target: "SQLite provides an embedded relational database engine.",
	}
	err := judgment.ValidateScoreRequest(req)
	fmt.Println(err == nil)
	// Output: true
}

func ExampleValidateNoulRequest() {
	req := judgment.NoulRequest{
		Question: "Does the text describe database indexing?",
	}
	err := judgment.ValidateNoulRequest(req)
	fmt.Println(err == nil)
	// Output: true
}
