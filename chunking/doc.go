// Package chunking splits text into reusable chunks for search, retrieval,
// summarization, and LLM context assembly.
//
// Each chunk carries StartChar and EndChar byte offsets into the original input
// text, such that original[StartChar:EndChar] == chunk.Text when the text
// appears verbatim. Sub-chunks from section/block fallbacks are rebased via
// adjustChunkSpans rather than cleared. Spans are cleared to 0 only by
// post-processing steps (EnforceHardLimits, EnforceTokenLimits) that cannot
// map the result back to the source.
//
// # Chunk fields
//
// Each Chunk carries structured metadata beyond raw text:
//
//   - Path: section breadcrumb from headings (nil for non-heading strategies)
//   - Metadata: block-type metadata from goldmark parsing (nil for non-goldmark strategies)
//   - ContentHash: first 16 hex characters of SHA-256(text); always set
//
// The Metadata map contains keys like "heading_level", "language", "line_count",
// and "word_count" for blocks parsed via goldmark.
//
// # Prose filtering
//
// FilterProse removes code and table chunks, dropping paragraphs below 5 words
// and headings below 3 words. Chunks without Metadata pass through unchanged.
//
// # Goldmark dependency
//
// The MarkdownAware and HeadingAware strategies use github.com/yuin/goldmark
// for structural parsing. Headings inside fenced code blocks are never
// detected as heading boundaries — fence-gating is structural, not stateful.
//
// # Offset helpers
//
//   - LineForOffset: converts a byte offset to a 1-based line number
//   - Locate: finds byte offsets of a fragment in content (exact then normalized)
//   - FillTokenCounts: populates TokenCount on chunks that lack it
//
// # Error sentinels
//
//   - ErrUnknownStrategy: returned by NewChunker for unrecognized strategy names
//   - ErrNilEmbedder: returned by NewSemanticChunker when embedder is nil
package chunking
