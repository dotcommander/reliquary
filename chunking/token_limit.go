package chunking

import (
	"fmt"
	"strings"
)

// TokenCounter abstracts token counting for budget enforcement.
// Implementations wrap a token encoder (e.g., tiktoken) to provide
// model-specific token budgets.
type TokenCounter interface {
	// CountTokens returns the number of tokens in text.
	CountTokens(text string) int
	// MaxTokens returns the maximum allowed tokens per chunk.
	// Returns 0 to disable token limiting (passthrough).
	MaxTokens() int
}

type tokenizerCounter struct {
	tokenizer Tokenizer
	maxTokens int
	err       error
}

func (c *tokenizerCounter) CountTokens(text string) int {
	if c.err != nil {
		return 0
	}
	count, err := c.tokenizer.Count(text)
	if err != nil {
		c.err = err
		return 0
	}
	return count
}

func (c *tokenizerCounter) MaxTokens() int { return c.maxTokens }

// EnforceTokenLimitsWithTokenizer applies an exact provider/model tokenizer to
// a chunk budget and propagates tokenization failures.
func EnforceTokenLimitsWithTokenizer(chunks []Chunk, tokenizer Tokenizer, maxTokens int) ([]Chunk, error) {
	if tokenizer == nil {
		return nil, fmt.Errorf("tokenizer is nil")
	}
	counter := &tokenizerCounter{tokenizer: tokenizer, maxTokens: maxTokens}
	result := EnforceTokenLimits(chunks, counter)
	if counter.err != nil {
		return nil, fmt.Errorf("count tokens: %w", counter.err)
	}
	return refineExactTokenLimits(result, tokenizer, maxTokens)
}

func refineExactTokenLimits(chunks []Chunk, tokenizer Tokenizer, maxTokens int) ([]Chunk, error) {
	if maxTokens <= 0 {
		return chunks, nil
	}

	result := make([]Chunk, 0, len(chunks))
	for _, chunk := range chunks {
		if strings.TrimSpace(chunk.Text) == "" {
			continue
		}
		count, err := tokenizer.Count(chunk.Text)
		if err != nil {
			return nil, fmt.Errorf("verify chunk %d tokens: %w", chunk.ID, err)
		}
		if count <= maxTokens {
			chunk.ID = len(result)
			result = append(result, chunk)
			continue
		}

		parts, err := splitRunesToExactTokenLimit(chunk.Text, tokenizer, maxTokens)
		if err != nil {
			return nil, fmt.Errorf("split chunk %d: %w", chunk.ID, err)
		}
		for _, part := range parts {
			result = append(result, buildChunk(len(result), part))
		}
	}
	return result, nil
}

func splitRunesToExactTokenLimit(text string, tokenizer Tokenizer, maxTokens int) ([]string, error) {
	runes := []rune(text)
	parts := make([]string, 0, 2)
	for start := 0; start < len(runes); {
		best := start
		for end := start + 1; end <= len(runes); end++ {
			count, err := tokenizer.Count(string(runes[start:end]))
			if err != nil {
				return nil, err
			}
			if count <= maxTokens {
				best = end
			}
		}
		if best == start {
			return nil, fmt.Errorf("no rune prefix fits %d-token limit", maxTokens)
		}
		parts = append(parts, string(runes[start:best]))
		start = best
	}
	return parts, nil
}

// EnforceTokenLimits ensures no chunk exceeds the token counter's MaxTokens
// budget. It layers on top of EnforceHardLimits (character budgets) and
// handles the token dimension.
//
// Oversized chunks are split using cascading boundary logic:
//
//	sentence → word → hard cut
//
// Chunk IDs are rebuilt to be sequential. Empty chunks are dropped.
// Source spans (StartChar/EndChar) are cleared on split sub-chunks since
// the split text cannot be reliably mapped back to the original source.
func EnforceTokenLimits(chunks []Chunk, tc TokenCounter) []Chunk {
	maxTok := tc.MaxTokens()
	if maxTok <= 0 {
		return chunks
	}

	var result []Chunk
	id := 0

	for _, c := range chunks {
		if strings.TrimSpace(c.Text) == "" {
			continue
		}

		// Check actual token count.
		tokens := tc.CountTokens(c.Text)
		if tokens <= maxTok {
			c.ID = id
			result = append(result, c)
			id++
			continue
		}

		// Oversized — split at sentence boundaries, then words, then a
		// conservative rune fallback for indivisible atoms.
		subs := splitChunkToFitTokens(c.Text, maxTok, tc)
		for _, sub := range subs {
			sub = strings.TrimSpace(sub)
			if sub == "" {
				continue
			}
			result = append(result, buildChunk(id, sub))
			id++
		}
	}

	return result
}

// splitChunkToFitTokens splits text into pieces that each fit within maxTok
// tokens. Tries sentence boundaries first, falls back to word boundaries, then
// rune chunks for atoms without usable word boundaries.
func splitChunkToFitTokens(text string, maxTok int, tc TokenCounter) []string {
	sentences := splitIntoSentences(text)
	if len(sentences) > 1 {
		return filterEmptyStrings(accumulateTokenAtoms(sentences, maxTok, tc, func(s string, parts []string) []string {
			return append(parts, splitByTokens(s, maxTok, tc)...)
		}))
	}
	return filterEmptyStrings(splitByTokens(text, maxTok, tc))
}

// splitByTokens splits text at word boundaries to fit within maxTok tokens.
func splitByTokens(text string, maxTok int, tc TokenCounter) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	return accumulateTokenAtoms(words, maxTok, tc, func(w string, parts []string) []string {
		return append(parts, splitAtomToFitTokens(w, maxTok, tc)...)
	})
}

func splitAtomToFitTokens(atom string, maxTok int, tc TokenCounter) []string {
	if atom == "" {
		return nil
	}

	runes := []rune(atom)
	var parts []string
	for start := 0; start < len(runes); {
		best := start
		for end := start + 1; end <= len(runes); end++ {
			if tc.CountTokens(string(runes[start:end])) > maxTok {
				break
			}
			best = end
		}
		if best == start {
			// A single rune can still exceed a pathological counter's budget;
			// emit it to guarantee progress rather than dropping content.
			best = start + 1
		}
		parts = append(parts, string(runes[start:best]))
		start = best
	}
	return parts
}

// accumulateTokenAtoms greedily joins atoms (space-separated) into chunks
// fitting within maxTok tokens. Each atom is tokenized once; a running sum
// avoids O(n²) re-encoding. The +1 on the running count approximates the
// space separator token. onOverflow is called when a single atom exceeds
// maxTok on its own.
func accumulateTokenAtoms(atoms []string, maxTok int, tc TokenCounter, onOverflow func(string, []string) []string) []string {
	var parts []string
	var buf strings.Builder
	running := 0

	for _, atom := range atoms {
		atomTok := tc.CountTokens(atom)
		extra := atomTok
		if buf.Len() > 0 {
			extra++ // space separator
		}

		if running+extra > maxTok && buf.Len() > 0 {
			parts = append(parts, strings.TrimSpace(buf.String()))
			buf.Reset()
			running = 0
			extra = atomTok // no separator at start of new chunk
		}

		if atomTok > maxTok {
			parts = onOverflow(atom, parts)
			buf.Reset()
			running = 0
			continue
		}

		if buf.Len() > 0 {
			buf.WriteByte(' ')
		}
		buf.WriteString(atom)
		running += extra
	}

	if buf.Len() > 0 {
		parts = append(parts, strings.TrimSpace(buf.String()))
	}

	return parts
}

func filterEmptyStrings(ss []string) []string {
	var result []string
	for _, s := range ss {
		if strings.TrimSpace(s) != "" {
			result = append(result, s)
		}
	}
	return result
}

// ChunkWithTokenLimit composes any Chunker with model-specific token-budget
// enforcement. It runs the base chunker, then applies EnforceTokenLimits.
//
// If c is nil, returns nil. If tc is nil, returns the base chunks unchanged.
// This is the recommended way to combine a boundary strategy (MarkdownAware,
// HeadingAware, SmartBoundary, etc.) with a token ceiling.
func ChunkWithTokenLimit(c Chunker, text string, size int, overlap int, tc TokenCounter) []Chunk {
	if c == nil {
		return nil
	}
	chunks := c.Chunk(text, size, overlap)
	if tc == nil {
		return chunks
	}
	return EnforceTokenLimits(chunks, tc)
}
