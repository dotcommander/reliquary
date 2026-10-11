package structjudge_test

import (
	"context"
	"fmt"

	"github.com/dotcommander/reliquary/adapter/structjudge"
)

type staticStructClient struct{}

func (staticStructClient) Generate(_ context.Context, _ structjudge.Request) ([]byte, error) {
	return []byte(`{"selected":"a","distribution":{"a":1.0},"confidence":1.0}`), nil
}

func ExampleNew() {
	judge, err := structjudge.New(staticStructClient{})
	fmt.Println(judge != nil, err == nil)
	// Output: true true
}
