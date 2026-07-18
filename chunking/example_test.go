package chunking_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/chunking"
)

func ExampleNewChunker() {
	chunker, err := chunking.NewChunker(chunking.SentenceBoundary)
	if err != nil {
		panic(err)
	}

	chunks := chunker.Chunk("Alpha sentence. Beta sentence. Gamma sentence.", 24, 0)
	fmt.Println(len(chunks) > 0)
	// Output: true
}

func ExampleCJKThaiSeparatorProfile() {
	profile := chunking.CJKThaiSeparatorProfile()
	separators := profile.SeparatorStrings()

	fmt.Println(profile.ID, len(separators) > len(chunking.DefaultTextSeparatorProfile().Separators))
	// Output: cjk_thai true
}
