package document_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/document"
)

func ExampleSpan_Text() {
	text := document.NormalizeText("\ufeffAlpha\r\nBeta")
	span := document.Span{StartByte: 0, EndByte: 5}
	value, ok := span.Text(text)
	fmt.Println(value, ok)
	// Output: Alpha true
}

func ExampleElement_ElementText() {
	source := "# Title\n\nBody text"
	element := document.Element{
		ID:       "el-1",
		Kind:     document.ElementKindTitle,
		Strategy: document.ParserStrategyMarkdown,
		Span:     document.Span{StartByte: 2, EndByte: 7},
		Text:     "fallback title",
	}

	fmt.Println(element.Kind, element.ElementText(source))
	// Output: title Title
}
