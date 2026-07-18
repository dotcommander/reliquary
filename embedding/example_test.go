package embeddings_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/embedding"
)

func ExampleValidateDimensions() {
	model := embeddings.ModelRef{Provider: "local", Name: "demo", Dim: 3}
	err := embeddings.ValidateDimensions([]embeddings.Vector{{1, 2, 3}}, model.Dim)
	fmt.Println(err == nil)
	// Output: true
}
