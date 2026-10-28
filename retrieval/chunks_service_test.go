package retrieval

import (
	"reflect"
	"testing"

	"github.com/dotcommander/reliquary/chunking"
	"github.com/dotcommander/reliquary/document"
)

func TestResultsFromChunksPreservesDocumentMapping(t *testing.T) {
	t.Parallel()
	doc := document.Document{ID: "doc", Title: "source.md", Text: "\ufeffone\r\ntwo\rthree", Metadata: document.Metadata{"tag": "kept", ContextStartLineKey: "900", ContextEndLineKey: "900"}}
	source := document.NormalizeText(doc.Text)
	chunker, err := chunking.NewChunker(chunking.HardCut)
	if err != nil {
		t.Fatal(err)
	}
	got := ResultsFromChunks(doc, source, chunker.Chunk(source, 4, 0))
	want, err := ResultsFromDocuments([]document.Document{doc}, chunking.HardCut, 4, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("shared mapping differs: %v, %v", got, want)
	}
	got[0].Metadata["tag"] = "changed"
	if doc.Metadata["tag"] != "kept" || got[1].Metadata["tag"] != "kept" {
		t.Fatal("metadata aliased")
	}
}
