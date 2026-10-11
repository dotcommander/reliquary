// Command chunking-strategies demonstrates Reliquary's chunking strategies
// (HeadingAware, SentenceBoundary, SmartBoundary, and Recursive).
//
// Run from the repository root:
//
//	GOWORK=off go run ./examples/chunking-strategies
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/dotcommander/reliquary/chunking"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
)

func main() {
	examplekit.Run(run)
}

func run(_ context.Context) error {
	const markdownSample = `# Reliquary Architecture

Reliquary is a retrieval-only toolkit in Go. It does not own runtime processes.

## Chunking Pipeline

Chunking splits documents into semantic chunks. Boundary preservation keeps sentences whole.
Code blocks are kept intact whenever possible.

## Hybrid Search

Hybrid search fuses lexical BM25 and vector cosine scores using Reciprocal Rank Fusion.
`

	fmt.Println("=== 1. Smart Boundary Chunking ===")
	smartChunker, err := chunking.NewChunker(chunking.SmartBoundary)
	if err != nil {
		return err
	}
	smartChunks := smartChunker.Chunk(markdownSample, 120, 0)
	printChunks(smartChunks)

	fmt.Println("\n=== 2. Sentence Boundary Chunking ===")
	sentenceChunker, err := chunking.NewChunker(chunking.SentenceBoundary)
	if err != nil {
		return err
	}
	sentenceChunks := sentenceChunker.Chunk(markdownSample, 120, 0)
	printChunks(sentenceChunks)

	fmt.Println("\n=== 3. Heading-Aware Chunking ===")
	headingChunker, err := chunking.NewChunker(chunking.HeadingAware)
	if err != nil {
		return err
	}
	headingChunks := headingChunker.Chunk(markdownSample, 150, 0)
	printChunks(headingChunks)

	fmt.Println("\n=== 4. Recursive Separator Chunking ===")
	recursiveChunker := chunking.NewRecursiveChunker(chunking.DefaultTextSeparatorProfile())
	recChunks := recursiveChunker.Chunk(markdownSample, 100, 0)
	printChunks(recChunks)

	return nil
}

func printChunks(chunks []chunking.Chunk) {
	fmt.Printf("Produced %d chunks:\n", len(chunks))
	for i, c := range chunks {
		text := strings.ReplaceAll(strings.TrimSpace(c.Text), "\n", " ")
		if len(text) > 60 {
			text = text[:60] + "..."
		}
		fmt.Printf("  [%d] (len=%d, start=%d, end=%d): %q\n", i+1, len(c.Text), c.StartChar, c.EndChar, text)
	}
}
