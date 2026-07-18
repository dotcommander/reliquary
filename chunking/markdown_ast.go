package chunking

import (
	"bytes"

	"github.com/yuin/goldmark/ast"
)

// nodeByteSpan returns the byte range [start, end) covering all lines of an
// AST node within the source byte slice. For container nodes (e.g. tables)
// that have no Lines() of their own, it falls back to scanning children.
func nodeByteSpan(node ast.Node) (start, end int) {
	lines := node.Lines()
	if lines.Len() > 0 {
		first := lines.At(0)
		last := lines.At(lines.Len() - 1)
		return int(first.Start), int(last.Stop)
	}
	// Container nodes (tables, etc.) may not have Lines() themselves.
	// Fall back to the span of their first and last children that have lines.
	var firstStart, lastEnd = -1, 0
	for child := node.FirstChild(); child != nil; child = child.NextSibling() {
		cl := child.Lines()
		if cl.Len() == 0 {
			// Recurse into grandchildren.
			cs, ce := nodeByteSpan(child)
			if cs == 0 && ce == 0 {
				continue
			}
			if firstStart < 0 || cs < firstStart {
				firstStart = cs
			}
			if ce > lastEnd {
				lastEnd = ce
			}
			continue
		}
		f := cl.At(0)
		l := cl.At(cl.Len() - 1)
		if firstStart < 0 || int(f.Start) < firstStart {
			firstStart = int(f.Start)
		}
		if int(l.Stop) > lastEnd {
			lastEnd = int(l.Stop)
		}
	}
	return firstStart, lastEnd
}

// extractNodeText extracts text content from an AST node using its Lines().
// For container nodes (e.g. blockquote) with no Lines(), it recursively
// extracts text from block-level children.
func extractNodeText(node ast.Node, source []byte) string {
	lines := node.Lines()
	if lines.Len() > 0 {
		var buf bytes.Buffer
		for i := 0; i < lines.Len(); i++ {
			line := lines.At(i)
			buf.Write(line.Value(source))
		}
		return buf.String()
	}
	// Container node — recurse into block-level children only.
	var buf bytes.Buffer
	for child := node.FirstChild(); child != nil; child = child.NextSibling() {
		if child.Type() == ast.TypeBlock {
			text := extractNodeText(child, source)
			if text != "" {
				buf.WriteString(text)
			}
		}
	}
	return buf.String()
}
