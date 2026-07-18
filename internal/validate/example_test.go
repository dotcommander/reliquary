package validate_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/internal/validate"
)

func ExampleNonEmpty() {
	err := validate.NonEmpty("id", "doc-1")
	fmt.Println(err == nil)
	// Output: true
}
