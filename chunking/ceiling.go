package chunking

import (
	"fmt"
	"unicode/utf8"
)

// EnforceTokenCeiling splits any chunk whose token count exceeds maxTokens
// (sentence → word → hard-cut cascade) and fills every chunk's TokenCount so
// results report the effect of the applied ceiling. encoding names a tiktoken
// encoding; empty defaults to cl100k_base. maxTokens <= 0 returns the chunks
// with TokenCount still filled, mirroring the no-ceiling accounting callers
// need for metadata. It composes NewTiktokenTokenizer,
// EnforceTokenLimitsWithTokenizer, and FillTokenCountsWithTokenizer so
// applications apply a post-strategy LLM context budget in one call.
func EnforceTokenCeiling(chunks []Chunk, encoding string, maxTokens int) ([]Chunk, error) {
	tokenizer, err := NewTiktokenTokenizer(encoding)
	if err != nil {
		return nil, fmt.Errorf("create tokenizer: %w", err)
	}
	limited, err := EnforceTokenLimitsWithTokenizer(chunks, tokenizer, maxTokens)
	if err != nil {
		return nil, fmt.Errorf("enforce token ceiling: %w", err)
	}
	if err := FillTokenCountsWithTokenizer(limited, tokenizer); err != nil {
		return nil, fmt.Errorf("count tokens: %w", err)
	}
	return limited, nil
}

// RuneSpan converts a chunk's authoritative byte span
// (source[startByte:endByte] == chunk text) into rune offsets, matching the
// rune basis of character counts reported to users. A span that cannot map
// into source normalizes to (0, 0) rather than being re-derived by substring
// search: an unlocatable span reports "unknown" instead of a plausible but
// wrong location.
func RuneSpan(source string, startByte, endByte int) (int, int) {
	if startByte < 0 || endByte < 0 || endByte < startByte || endByte > len(source) {
		return 0, 0
	}
	return utf8.RuneCountInString(source[:startByte]),
		utf8.RuneCountInString(source[:endByte])
}
