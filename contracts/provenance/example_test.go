package provenance_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/contracts/provenance"
	"github.com/dotcommander/reliquary/internal/hash"
)

func ExampleGeneratedArtifact() {
	artifact := provenance.GeneratedArtifact("a1", "markdown", "file://out.md", "importer", hash.SHA256String("body"))
	fmt.Println(artifact.ID, artifact.GeneratedBy)
	// Output: a1 importer
}
