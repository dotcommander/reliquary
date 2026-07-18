package hash_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/internal/hash"
)

func ExampleSHA256String() {
	digest := hash.SHA256String("body")
	fmt.Println(digest.Algorithm, digest.String() != "")
	// Output: sha256 true
}

func ExampleHashIdentity() {
	digest := hash.HashIdentity(
		hash.IdentityPart{Kind: "source", InputHash: "source-sha"},
		hash.IdentityPart{Kind: "chunker", Version: "v1", ConfigHash: "chunk-cfg"},
	)
	fmt.Println(digest.Algorithm, digest.String() != "")
	// Output: sha256 true
}
