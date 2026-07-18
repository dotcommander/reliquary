package dedup

import (
	"crypto/sha256"
	"fmt"
	"hash/fnv"
	"regexp"
	"strings"
	"unicode"
)

var (
	whitespaceRegex  = regexp.MustCompile(`\s+`)
	nonAlphanumRegex = regexp.MustCompile(`[^a-z0-9\s]`)
)

// HashingStrategy defines different approaches to content hashing
type HashingStrategy string

const (
	// SimpleHash - Basic content hash
	SimpleHash HashingStrategy = "simple"
	// NormalizedHash - Normalized text hash (whitespace, case insensitive)
	NormalizedHash HashingStrategy = "normalized"
	// SemanticHash - Content-aware hash (structure preserved)
	SemanticHash HashingStrategy = "semantic"
	// SimHash - Locality-sensitive hash (Charikar 2002) for similarity detection
	SimHash HashingStrategy = "simhash"
)

// ContentHasher provides fast duplicate detection capabilities
type ContentHasher struct {
	strategy HashingStrategy
	// shingleSize is the character n-gram size used by SimHash. Default 3.
	shingleSize int
	// simHashBits is the number of significant SimHash bits (1..64). Default 64.
	// Output is always %016x-encoded; values <64 zero the high bits, keeping
	// the hex-string length constant so Detector.hammingDistance stays correct.
	simHashBits int
}

// Default SimHash tuning parameters (preserve historical behavior).
const (
	defaultShingleSize = 3
	defaultSimHashBits = 64
	maxSimHashBits     = 64
)

// NewContentHasher creates a new content hasher with the specified strategy.
// SimHash shingle size and bit width default to 3-grams and 64 bits; override
// with WithSimHashOptions.
func NewContentHasher(strategy HashingStrategy) *ContentHasher {
	return &ContentHasher{
		strategy:    strategy,
		shingleSize: defaultShingleSize,
		simHashBits: defaultSimHashBits,
	}
}

// WithSimHashOptions configures the SimHash shingle (n-gram) size and bit
// width, returning the receiver for chaining (matching Detector.WithOrdering).
// Invalid values are clamped to the supported range rather than erroring,
// matching the package's tolerant input handling: shingleSize is clamped to
// a minimum of 1; bits is clamped to the range 1..64 (the internal FNV-64 /
// uint64 pipeline and Detector.hammingDistance cannot represent wider hashes).
func (ch *ContentHasher) WithSimHashOptions(shingleSize, bits int) *ContentHasher {
	if shingleSize < 1 {
		shingleSize = 1
	}
	if bits < 1 {
		bits = 1
	}
	if bits > maxSimHashBits {
		bits = maxSimHashBits
	}
	ch.shingleSize = shingleSize
	ch.simHashBits = bits
	return ch
}

// SupportsNearDuplicate reports whether the strategy produces
// locality-sensitive hashes usable by FindNearDuplicates.
func (ch *ContentHasher) SupportsNearDuplicate() bool {
	return ch.strategy == SimHash
}

// HashContent generates a hash for the given content using the configured strategy
func (ch *ContentHasher) HashContent(content string) string {
	switch ch.strategy {
	case NormalizedHash:
		return ch.normalizedHash(content)
	case SemanticHash:
		return ch.semanticHash(content)
	case SimHash:
		return ch.simHash(content)
	default:
		return ch.simpleHash(content)
	}
}

// simpleHash creates a basic content hash
func (ch *ContentHasher) simpleHash(content string) string {
	h := sha256.Sum256([]byte(content))
	// Truncate to first 16 hex chars (64 bits). Trades collision resistance
	// (birthday bound ~2^32 inputs) for compact, comparable hash keys.
	return fmt.Sprintf("%x", h)[:16]
}

// normalizedHash creates a hash of normalized content (whitespace/case insensitive)
func (ch *ContentHasher) normalizedHash(content string) string {
	// Normalize whitespace and case
	normalized := strings.ToLower(content)
	normalized = whitespaceRegex.ReplaceAllString(normalized, " ")
	normalized = strings.TrimSpace(normalized)

	h := sha256.Sum256([]byte(normalized))
	return fmt.Sprintf("%x", h)[:16]
}

// semanticHash creates a content-aware hash preserving structure
func (ch *ContentHasher) semanticHash(content string) string {
	// Extract semantic elements
	lines := strings.Split(content, "\n")
	var semanticParts []string

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Preserve markdown structure
		if strings.HasPrefix(line, "#") {
			semanticParts = append(semanticParts, "HEADER:"+line)
		} else if strings.HasPrefix(line, "-") || strings.HasPrefix(line, "*") {
			semanticParts = append(semanticParts, "LIST:"+strings.TrimSpace(line[1:]))
		} else if strings.HasPrefix(line, "```") {
			semanticParts = append(semanticParts, "CODE_BLOCK")
		} else {
			// Regular content — include short lines so all-short-line docs
			// don't collide to sha256("").
			words := strings.Fields(strings.ToLower(line))
			if len(words) > 0 {
				semanticParts = append(semanticParts, "TEXT:"+strings.Join(words, " "))
			}
		}
	}

	semanticContent := strings.Join(semanticParts, "|")
	h := sha256.Sum256([]byte(semanticContent))
	return fmt.Sprintf("%x", h)[:16]
}

// simHash creates a similarity-sensitive hash using shingles
func (ch *ContentHasher) simHash(content string) string {
	// Create character shingles (3-grams)
	shingles := ch.createShingles(content, ch.shingleSize)

	// Count shingle occurrences for frequency weighting.
	features := make(map[string]int)
	for _, shingle := range shingles {
		features[shingle]++
	}

	// SimHash bit vector, weighting each feature by its occurrence count.
	// Width is ch.simHashBits (1..64); the %016x encoding below is unchanged so
	// the hex-string length stays constant for Detector.hammingDistance.
	width := ch.simHashBits
	if width < 1 || width > maxSimHashBits {
		width = defaultSimHashBits
	}
	hashBits := make([]int, width)

	for feature, count := range features {
		h := fnv.New64a()
		h.Write([]byte(feature))
		hash := h.Sum64()

		// Add weighted bits from hash
		for i := 0; i < width; i++ {
			if hash&(1<<uint(i)) != 0 {
				hashBits[i] += count
			} else {
				hashBits[i] -= count
			}
		}
	}

	// Convert to final hash
	var finalHash uint64
	for i := 0; i < width; i++ {
		if hashBits[i] > 0 {
			finalHash |= 1 << uint(i)
		}
	}

	return fmt.Sprintf("%016x", finalHash)
}

// createShingles creates n-gram shingles from content
func (ch *ContentHasher) createShingles(content string, n int) []string {
	// Normalize content
	content = strings.ToLower(content)
	content = nonAlphanumRegex.ReplaceAllString(content, "")
	content = whitespaceRegex.ReplaceAllString(content, " ")

	var shingles []string
	runes := []rune(content)

	for i := 0; i <= len(runes)-n; i++ {
		shingle := string(runes[i : i+n])
		if !unicode.IsSpace(runes[i]) && !unicode.IsSpace(runes[i+n-1]) {
			shingles = append(shingles, shingle)
		}
	}

	return shingles
}
