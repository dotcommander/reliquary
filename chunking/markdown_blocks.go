package chunking

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/ast"
	"github.com/yuin/goldmark/extension"
	extast "github.com/yuin/goldmark/extension/ast"
	"github.com/yuin/goldmark/text"
)

// Metadata key constants for block metadata population.
const (
	metaKeyHeadingLevel = "heading_level"
	metaKeyLanguage     = "language"
	metaKeyLineCount    = "line_count"
	metaKeyWordCount    = "word_count"
)

// markdownBlock represents a structural block extracted from Markdown via goldmark AST.
type markdownBlock struct {
	text      string            // Raw text content of the block
	blockType string            // "heading", "paragraph", "code", "list", "table", "blockquote"
	level     int               // Heading level 1-6; 0 for non-heading blocks
	startByte int               // Byte offset in original source
	endByte   int               // Byte offset in original source (exclusive)
	metadata  map[string]string // Block-type metadata; nil when not populated
}

// goldmarkParser is a package-level goldmark instance with table extension.
var goldmarkParser = goldmark.New(
	goldmark.WithExtensions(extension.Table),
)

// extractMarkdownBlocks parses src with goldmark and returns structural blocks
// with verbatim byte spans. '#' inside *ast.FencedCodeBlock never reaches the
// *ast.Heading case — fence-gating is structural, not stateful.
func extractMarkdownBlocks(src []byte) []markdownBlock {
	if len(src) == 0 {
		return nil
	}

	doc := goldmarkParser.Parser().Parse(text.NewReader(src))
	extractor := &blockExtractor{
		source: src,
	}
	ast.Walk(doc, extractor.visit)
	for i := range extractor.blocks {
		block := &extractor.blocks[i]
		if block.metadata == nil {
			block.metadata = make(map[string]string)
		}
		block.metadata["type"] = block.blockType
		if _, ok := block.metadata[metaKeyWordCount]; !ok {
			content := block.text
			if block.blockType == "heading" {
				content = strings.TrimLeft(content, "# ")
			}
			block.metadata[metaKeyWordCount] = strconv.Itoa(len(strings.Fields(content)))
		}
	}
	return extractor.blocks
}

// blockExtractor implements the AST visitor pattern for goldmark.
type blockExtractor struct {
	source []byte
	blocks []markdownBlock
}

// visit processes AST nodes and extracts structural blocks.
func (be *blockExtractor) visit(node ast.Node, entering bool) (ast.WalkStatus, error) {
	if !entering {
		return ast.WalkContinue, nil
	}

	switch n := node.(type) {
	case *ast.Heading:
		be.extractHeading(n)
		return ast.WalkSkipChildren, nil
	case *ast.Paragraph:
		be.extractParagraph(n)
		return ast.WalkSkipChildren, nil
	case *ast.FencedCodeBlock:
		be.extractCode(n)
		return ast.WalkSkipChildren, nil
	case *ast.List:
		be.extractList(n)
		return ast.WalkSkipChildren, nil
	case *extast.Table:
		be.extractTable(n)
		return ast.WalkSkipChildren, nil
	case *ast.Blockquote:
		be.extractBlockquote(n)
		return ast.WalkSkipChildren, nil
	}

	return ast.WalkContinue, nil
}

func (be *blockExtractor) extractHeading(n *ast.Heading) {
	content := extractNodeText(n, be.source)
	if strings.TrimSpace(content) == "" {
		return
	}

	// Reconstruct heading text with # prefix for compatibility with existing behavior.
	var buf bytes.Buffer
	for i := 0; i < n.Level; i++ {
		buf.WriteByte('#')
	}
	buf.WriteByte(' ')
	buf.WriteString(content)
	fullText := buf.String()

	// Extend byte span backward to include the # markers.
	// Goldmark's heading Lines() start after the # prefix, but we
	// reconstructed the text to include it, so the span must cover it.
	start, end := nodeByteSpan(n)
	for start > 0 && be.source[start-1] == ' ' {
		start--
	}
	for start > 0 && be.source[start-1] == '#' {
		start--
	}
	level := n.Level

	meta := map[string]string{
		metaKeyHeadingLevel: fmt.Sprintf("%d", level),
	}

	be.blocks = append(be.blocks, markdownBlock{
		text:      fullText,
		blockType: "heading",
		level:     level,
		startByte: start,
		endByte:   end,
		metadata:  meta,
	})
}

func (be *blockExtractor) extractParagraph(n *ast.Paragraph) {
	content := extractNodeText(n, be.source)
	if strings.TrimSpace(content) == "" {
		return
	}

	start, end := nodeByteSpan(n)
	wordCount := len(strings.Fields(content))

	meta := map[string]string{
		metaKeyWordCount: fmt.Sprintf("%d", wordCount),
	}

	be.blocks = append(be.blocks, markdownBlock{
		text:      content,
		blockType: "paragraph",
		level:     0,
		startByte: start,
		endByte:   end,
		metadata:  meta,
	})
}

func (be *blockExtractor) extractCode(n *ast.FencedCodeBlock) {
	content := extractNodeText(n, be.source)
	if strings.TrimSpace(content) == "" {
		return
	}

	// Reconstruct full fenced block including markers.
	var langStr string
	if n.Info != nil {
		langStr = string(n.Info.Value(be.source))
	}
	fullText := "```" + langStr + "\n" + content + "```"

	start, end := nodeByteSpan(n)
	if sourceStart, sourceEnd, ok := fencedCodeSourceSpan(be.source, start, end); ok {
		start, end = sourceStart, sourceEnd
		fullText = string(be.source[start:end])
	} else {
		// Goldmark can normalize indentation or nested block prefixes. Keep
		// reconstructed text, but do not claim source offsets in that case.
		start, end = 0, 0
	}

	trimmedContent := strings.TrimRight(content, "\n")
	lineCount := strings.Count(trimmedContent, "\n") + 1
	meta := map[string]string{
		metaKeyLineCount: fmt.Sprintf("%d", lineCount),
	}
	if langStr != "" {
		meta[metaKeyLanguage] = langStr
	}

	be.blocks = append(be.blocks, markdownBlock{
		text:      fullText,
		blockType: "code",
		level:     0,
		startByte: start,
		endByte:   end,
		metadata:  meta,
	})
}

func (be *blockExtractor) extractList(n *ast.List) {
	var buf bytes.Buffer
	for child := n.FirstChild(); child != nil; child = child.NextSibling() {
		if listItem, ok := child.(*ast.ListItem); ok {
			content := extractNodeText(listItem, be.source)
			if n.IsOrdered() {
				buf.WriteString(fmt.Sprintf("1. %s\n", content))
			} else {
				buf.WriteString(fmt.Sprintf("- %s\n", content))
			}
		}
	}
	content := strings.TrimSpace(buf.String())
	if content == "" {
		return
	}

	start, end := nodeByteSpan(n)

	be.blocks = append(be.blocks, markdownBlock{
		text:      content,
		blockType: "list",
		level:     0,
		startByte: start,
		endByte:   end,
		metadata:  nil,
	})
}

func (be *blockExtractor) extractTable(n *extast.Table) {
	var buf bytes.Buffer
	for child := n.FirstChild(); child != nil; child = child.NextSibling() {
		var cells []string
		for cell := child.FirstChild(); cell != nil; cell = cell.NextSibling() {
			if tableCell, ok := cell.(*extast.TableCell); ok {
				cellContent := extractNodeText(tableCell, be.source)
				cells = append(cells, cellContent)
			}
		}
		if len(cells) > 0 {
			buf.WriteString(fmt.Sprintf("| %s |\n", strings.Join(cells, " | ")))
		}
	}
	content := strings.TrimSpace(buf.String())
	if content == "" {
		return
	}

	start, end := nodeByteSpan(n)

	be.blocks = append(be.blocks, markdownBlock{
		text:      content,
		blockType: "table",
		level:     0,
		startByte: start,
		endByte:   end,
		metadata:  nil,
	})
}

func (be *blockExtractor) extractBlockquote(n *ast.Blockquote) {
	content := extractNodeText(n, be.source)
	if strings.TrimSpace(content) == "" {
		return
	}

	start, end := nodeByteSpan(n)

	be.blocks = append(be.blocks, markdownBlock{
		text:      content,
		blockType: "blockquote",
		level:     0,
		startByte: start,
		endByte:   end,
		metadata:  nil,
	})
}

// fencedCodeSourceSpan inspects only the lines immediately surrounding the
// AST content. It never searches backwards into an unrelated fenced block.
func fencedCodeSourceSpan(source []byte, contentStart, contentEnd int) (int, int, bool) {
	if contentStart <= 0 || contentEnd < contentStart || contentEnd > len(source) {
		return 0, 0, false
	}
	contentLine := bytes.LastIndexByte(source[:contentStart], '\n') + 1
	if contentLine == 0 {
		return 0, 0, false
	}
	openingEnd := contentLine - 1
	openingStart := bytes.LastIndexByte(source[:openingEnd], '\n') + 1
	opening := strings.TrimRight(string(source[openingStart:openingEnd]), "\r")
	marker, count, _, ok := markdownFenceLine(opening)
	if !ok {
		return 0, 0, false
	}

	closingStart := contentEnd
	if closingStart > 0 && source[closingStart-1] != '\n' {
		// A final content line without a newline belongs to an unclosed fence.
		if closingStart != len(source) {
			return 0, 0, false
		}
	}
	end := contentEnd
	if closingStart < len(source) {
		closingEnd := closingStart
		for closingEnd < len(source) && source[closingEnd] != '\n' {
			closingEnd++
		}
		closing := strings.TrimRight(string(source[closingStart:closingEnd]), "\r")
		closeMarker, closeCount, rest, valid := markdownFenceLine(closing)
		if !valid || closeMarker != marker || closeCount < count || strings.TrimSpace(rest) != "" {
			return 0, 0, false
		}
		end = closingEnd
	}
	start := openingStart
	// Chunk text is trimmed by the common emission helper; trim its span too.
	raw := string(source[start:end])
	trimmed := strings.TrimSpace(raw)
	start += strings.Index(raw, trimmed)
	return start, start + len(trimmed), true
}

func markdownFenceLine(line string) (byte, int, string, bool) {
	indent := 0
	for indent < len(line) && line[indent] == ' ' {
		indent++
	}
	if indent > 3 || indent >= len(line) || (line[indent] != '`' && line[indent] != '~') {
		return 0, 0, "", false
	}
	marker := line[indent]
	end := indent
	for end < len(line) && line[end] == marker {
		end++
	}
	if end-indent < 3 {
		return 0, 0, "", false
	}
	return marker, end - indent, line[end:], true
}
