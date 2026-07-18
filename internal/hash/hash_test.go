package hash

import (
	"strings"
	"testing"
)

func TestSHA256String(t *testing.T) {
	t.Parallel()

	got := SHA256String("corpus")
	if got.String() != "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe" {
		t.Fatalf("unexpected digest: %s", got.String())
	}
}

func TestSHA256Reader(t *testing.T) {
	t.Parallel()

	got, err := SHA256Reader(strings.NewReader("corpus"))
	if err != nil {
		t.Fatalf("SHA256Reader returned error: %v", err)
	}
	if got != SHA256String("corpus") {
		t.Fatalf("reader digest mismatch: got %v want %v", got, SHA256String("corpus"))
	}
}

func TestHashIdentityIsOrderedAndOmitsEmptyFields(t *testing.T) {
	t.Parallel()

	base := HashIdentity(
		IdentityPart{Kind: "chunker", Version: "v1", ConfigHash: "cfg"},
		IdentityPart{Kind: "source", InputHash: "src"},
	)
	withEmpty := HashIdentity(
		IdentityPart{Kind: "chunker", Version: "v1", ConfigHash: "cfg"},
		IdentityPart{},
		IdentityPart{Kind: "source", InputHash: "src"},
	)
	if base != withEmpty {
		t.Fatalf("empty identity part changed digest: %s != %s", base, withEmpty)
	}

	reordered := HashIdentity(
		IdentityPart{Kind: "source", InputHash: "src"},
		IdentityPart{Kind: "chunker", Version: "v1", ConfigHash: "cfg"},
	)
	if base == reordered {
		t.Fatalf("reordered identity produced same digest: %s", base)
	}
	if got := (Identity{{Kind: "chunker", Version: "v1", ConfigHash: "cfg"}}).Digest(); got == (Digest{}) {
		t.Fatal("Identity.Digest returned zero digest")
	}
}
