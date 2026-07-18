package dedup_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/dedup"
)

type Doc struct {
	ID   string
	Body string
}

func ExampleDetector_FindDuplicates() {
	docs := []Doc{
		{ID: "doc1", Body: "The quick brown fox"},
		{ID: "doc2", Body: "the quick brown fox"},
		{ID: "doc3", Body: "Something completely different"},
	}

	// Index using NormalizedHash (ignores case and whitespace differences)
	d := dedup.NewDetector(dedup.NormalizedHash, func(doc Doc) string {
		return doc.Body
	})
	d.Index(docs)

	for _, group := range d.FindDuplicates() {
		fmt.Printf("Duplicate group of %d items:\n", len(group))
		for _, doc := range group {
			fmt.Printf("  - %s: %s\n", doc.ID, doc.Body)
		}
	}
	// Output:
	// Duplicate group of 2 items:
	//   - doc1: The quick brown fox
	//   - doc2: the quick brown fox
}

func ExampleDetector_FindNearDuplicates() {
	docs := []Doc{
		{ID: "doc1", Body: "The quick brown fox jumps over the lazy dog"},
		{ID: "doc2", Body: "The quick brown fox jumps over the lazy dog today"},
		{ID: "doc3", Body: "Different text altogether"},
	}

	// Index using SimHash for near-duplicate (fuzzy similarity) matching
	d := dedup.NewDetector(dedup.SimHash, func(doc Doc) string {
		return doc.Body
	})
	d.Index(docs)

	// Group items within a Hamming distance threshold of 5
	for _, group := range d.FindNearDuplicates(5) {
		fmt.Printf("Near-duplicate cluster of %d items:\n", len(group))
		for _, doc := range group {
			fmt.Printf("  - %s: %s\n", doc.ID, doc.Body)
		}
	}
	// Output:
	// Near-duplicate cluster of 2 items:
	//   - doc1: The quick brown fox jumps over the lazy dog
	//   - doc2: The quick brown fox jumps over the lazy dog today
}
