package clustering_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/vector/clustering"
)

func ExampleNewClusterService() {
	service := clustering.NewClusterService("greedy")
	result, err := service.Cluster([][]float64{{1, 0}, {0.9, 0.1}}, clustering.DefaultClusterOptions())
	fmt.Println(result.K > 0, err == nil)
	// Output: true true
}
