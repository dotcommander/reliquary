package provenance

import (
	"testing"

	"github.com/dotcommander/reliquary/internal/hash"
)

func TestGeneratedArtifact(t *testing.T) {
	t.Parallel()

	digest := hash.SHA256String("output")
	got := GeneratedArtifact("a1", "json", "file://out.json", "test", digest)
	if got.Hash != digest || got.GeneratedBy != "test" || got.GeneratedAt.IsZero() {
		t.Fatalf("unexpected artifact: %#v", got)
	}
}
