package embedding_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/embedding"
)

func ExampleValidateDimensions() {
	model := embedding.ModelRef{Provider: "local", Name: "demo", Dim: 3}
	err := embedding.ValidateDimensions([]embedding.Vector{{1, 2, 3}}, model.Dim)
	fmt.Println(err == nil)
	// Output: true
}
