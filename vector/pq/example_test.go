package pq_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/vector/pq"
)

func ExampleDefaultConfig() {
	cfg := pq.DefaultConfig()
	fmt.Println(cfg.NumSubspaces > 0)
	// Output: true
}
