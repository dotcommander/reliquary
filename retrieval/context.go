package retrieval

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"reflect"
	"strings"
)

// ContextStartLineKey is the retrieval-owned metadata key for a result's
// inclusive, one-based starting source line.
const ContextStartLineKey = "reliquary.context.start_line"

// ContextEndLineKey is the retrieval-owned metadata key for a result's
// inclusive, one-based ending source line.
const ContextEndLineKey = "reliquary.context.end_line"

// ContextTokenCounter counts tokens using the caller's provider and model
// policy.
type ContextTokenCounter interface {
	Count(text string) (int, error)
}

// ContextOption configures FormatContext.
type ContextOption func(*contextConfig)

type contextConfig struct {
	header    string
	separator string
	maxTokens int
	counter   ContextTokenCounter
	limitSet  bool
}

// WithHeader adds a header before each result's content. The supported
// placeholders are %s for the source, the first two %d placeholders for the
// inclusive source line range, and %% for a literal percent sign.
func WithHeader(template string) ContextOption {
	return func(c *contextConfig) {
		c.header = template
	}
}

// WithSeparator replaces the default blank-line separator between results.
func WithSeparator(separator string) ContextOption {
	return func(c *contextConfig) {
		c.separator = separator
	}
}

// WithMaxTokens limits formatted context to a contiguous prefix of complete
// result blocks. A positive limit requires a non-nil counter. A nonpositive
// limit produces empty output without invoking the counter.
func WithMaxTokens(maxTokens int, counter ContextTokenCounter) ContextOption {
	return func(c *contextConfig) {
		c.maxTokens = maxTokens
		c.counter = counter
		c.limitSet = true
	}
}

// FormatContext renders non-empty retrieval results in order. It adds no prompt
// instructions, escaping, or filtering; by default blocks are joined by one
// blank line.
func FormatContext(results []*Result, opts ...ContextOption) (string, error) {
	cfg := contextConfig{separator: "\n\n"}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	if cfg.limitSet {
		if cfg.maxTokens <= 0 {
			return "", nil
		}
		if cfg.counter == nil {
			return "", fmt.Errorf("retrieval: positive context token limit requires a token counter")
		}
	}

	var formatted strings.Builder
	blocks := 0
	for _, result := range results {
		if result == nil || result.Content == "" {
			continue
		}

		block := formatContextBlock(result, cfg.header)
		tentative := block
		if blocks > 0 {
			tentative = formatted.String() + cfg.separator + block
		}
		if cfg.limitSet {
			count, err := cfg.counter.Count(tentative)
			if err != nil {
				return "", fmt.Errorf("retrieval: count formatted context tokens: %w", err)
			}
			if count < 0 {
				return "", fmt.Errorf("retrieval: count formatted context tokens: negative count %d", count)
			}
			if count > cfg.maxTokens {
				break
			}
		}

		if blocks > 0 {
			formatted.WriteString(cfg.separator)
		}
		formatted.WriteString(block)
		blocks++
	}

	return formatted.String(), nil
}

func formatContextBlock(result *Result, header string) string {
	if header == "" {
		return result.Content
	}
	source := result.Filename
	if source == "" {
		source = result.DocumentID
	}
	if source == "" {
		source = result.ID
	}
	startLine, endLine := contextLineRange(result.Metadata)
	return formatContextHeader(header, source, startLine, endLine) + "\n" + result.Content
}

func formatContextHeader(template, source string, startLine, endLine int) string {
	var formatted strings.Builder
	formatted.Grow(len(template))
	linePlaceholder := 0
	for i := 0; i < len(template); {
		if template[i] != '%' || i+1 == len(template) {
			formatted.WriteByte(template[i])
			i++
			continue
		}

		switch template[i+1] {
		case '%':
			formatted.WriteByte('%')
			i += 2
		case 's':
			formatted.WriteString(source)
			i += 2
		case 'd':
			switch linePlaceholder {
			case 0:
				fmt.Fprint(&formatted, startLine)
			case 1:
				fmt.Fprint(&formatted, endLine)
			default:
				formatted.WriteString("%d")
			}
			linePlaceholder++
			i += 2
		default:
			formatted.WriteByte('%')
			i++
		}
	}
	return formatted.String()
}

func contextLineRange(metadata map[string]any) (int, int) {
	if metadata == nil {
		return 0, 0
	}
	start, startOK := positiveIntegral(metadata[ContextStartLineKey])
	end, endOK := positiveIntegral(metadata[ContextEndLineKey])
	if !startOK || !endOK || end < start {
		return 0, 0
	}
	return start, end
}

func positiveIntegral(value any) (int, bool) {
	if number, ok := value.(json.Number); ok {
		return positiveJSONInteger(number)
	}
	if value == nil {
		return 0, false
	}

	maxInt := uint64(^uint(0) >> 1)
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		n := v.Int()
		if n <= 0 || uint64(n) > maxInt {
			return 0, false
		}
		return int(n), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		n := v.Uint()
		if n == 0 || n > maxInt {
			return 0, false
		}
		return int(n), true
	case reflect.Float32, reflect.Float64:
		n := v.Float()
		if n <= 0 || math.IsInf(n, 0) || math.IsNaN(n) || math.Trunc(n) != n || n > float64(maxInt) {
			return 0, false
		}
		converted := int(n)
		if converted <= 0 {
			return 0, false
		}
		return converted, true
	default:
		return 0, false
	}
}

func positiveJSONInteger(number json.Number) (int, bool) {
	raw := number.String()
	if !json.Valid([]byte(raw)) {
		return 0, false
	}
	parsed, ok := new(big.Rat).SetString(raw)
	if !ok || parsed.Sign() <= 0 || parsed.Denom().Cmp(big.NewInt(1)) != 0 {
		return 0, false
	}
	maxInt := new(big.Int).SetUint64(uint64(^uint(0) >> 1))
	if parsed.Num().Cmp(maxInt) > 0 {
		return 0, false
	}
	return int(parsed.Num().Int64()), true
}
