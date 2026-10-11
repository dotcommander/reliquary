package jev_test

import (
	"context"
	"fmt"

	"github.com/dotcommander/reliquary/adapter/jev"
)

type staticJevClient struct{}

func (staticJevClient) Evaluate(_ context.Context, call jev.Call) (jev.Result, error) {
	return jev.Result{
		Selected: "search",
		Distribution: []jev.Mass{
			{Option: "search", Probability: 1.0},
		},
		Confidence: 1.0,
	}, nil
}

func ExampleNew() {
	adapter, err := jev.New(staticJevClient{})
	fmt.Println(adapter != nil, err == nil)
	// Output: true true
}
