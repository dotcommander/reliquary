// Command dedup-pipeline demonstrates Reliquary's near-duplicate detection and
// canonical text normalization primitives.
//
// Run from the repository root:
//
//	GOWORK=off go run ./examples/dedup-pipeline
package main

import (
	"context"
	"fmt"

	"github.com/dotcommander/reliquary/dedup"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
)

func main() {
	examplekit.Run(run)
}

type Article struct {
	ID    string
	Title string
	Body  string
}

func run(_ context.Context) error {
	articles := []Article{
		{ID: "art-1", Title: "Go Garbage Collection", Body: "Go uses a concurrent mark-and-sweep garbage collector."},
		{ID: "art-2", Title: "Go GC (Copy)", Body: "go uses a concurrent mark-and-sweep garbage collector."},
		{ID: "art-3", Title: "Go GC (Near-Dup)", Body: "Go uses a concurrent mark-and-sweep garbage collector with short pauses."},
		{ID: "art-4", Title: "Pasta Recipe", Body: "Fresh pasta dough requires high-protein flour and whole eggs."},
	}

	// 1. Normalized Hash deduplication (case and whitespace invariant)
	fmt.Println("=== 1. Normalized Hash (Exact / Normalization Dupes) ===")
	normDetector := dedup.NewDetector(dedup.NormalizedHash, func(a Article) string {
		return a.Body
	})
	normDetector.Index(articles)
	normDupes := normDetector.FindDuplicates()
	fmt.Printf("Found %d duplicate groups:\n", len(normDupes))
	for i, group := range normDupes {
		fmt.Printf("Group %d:\n", i+1)
		for _, item := range group {
			fmt.Printf("  - %s: %q\n", item.ID, item.Body)
		}
	}

	// 2. SimHash near-duplicate clustering
	fmt.Println("\n=== 2. SimHash (Near-Duplicate Fuzzy Clusters) ===")
	simDetector := dedup.NewDetector(dedup.SimHash, func(a Article) string {
		return a.Body
	})
	simDetector.Index(articles)
	// Cluster within Hamming distance threshold of 8
	clusters := simDetector.FindNearDuplicates(8)
	fmt.Printf("Found %d near-duplicate clusters (threshold=8):\n", len(clusters))
	for i, cluster := range clusters {
		fmt.Printf("Cluster %d (%d items):\n", i+1, len(cluster))
		for _, item := range cluster {
			fmt.Printf("  - %s (%s): %q\n", item.ID, item.Title, item.Body)
		}
	}

	// 3. Canonicalize duplicate groups (select survivor)
	fmt.Println("\n=== 3. Canonicalize (Survivor Selection) ===")
	survivors := dedup.Canonicalize(normDupes, func(a, b Article) bool {
		return len(a.Body) > len(b.Body)
	})
	fmt.Printf("Selected %d canonical survivors from %d duplicate groups:\n", len(survivors), len(normDupes))
	for _, item := range survivors {
		fmt.Printf("  - Kept: %s (%q)\n", item.ID, item.Title)
	}
	return nil
}
