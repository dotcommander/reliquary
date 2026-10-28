package retrieval

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"reflect"
	"strings"
	"text/template"
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
	header          string
	separator       string
	maxTokens       int
	counter         ContextTokenCounter
	limitSet        bool
	headerMode      int
	headerFormatter func(ContextItem) (string, error)
	citation        string
	citationSet     bool
	transform       func(ContextItem) (string, error)
	transformSet    bool
}

// WithHeader adds a header before each result's content. The supported
// placeholders are %s for the source, the first two %d placeholders for the
// inclusive source line range, and %% for a literal percent sign.
func WithHeader(template string) ContextOption {
	return func(c *contextConfig) {
		c.header = template
		c.headerMode = 0
	}
}

// ContextItem describes a passage being formatted. Result and its referenced data
// must be treated as read-only by callbacks.
type ContextItem struct {
	Result             *Result
	InputIndex         int
	Number             int
	Source             string
	StartLine, EndLine int
	HasLines           bool
	Content            string
}

// FormatSummary maps emitted, omitted, and skipped passages to input positions.
// TokensUsed is present only when the accepted output was counted.
type FormatSummary struct {
	InputCount      int
	IncludedIndices []int
	OmittedIndices  []int
	SkippedIndices  []int
	TokensUsed      *int
}

// WithHeaderTemplate selects a text/template header; the last header option wins.
func WithHeaderTemplate(text string) ContextOption {
	return func(c *contextConfig) { c.header = text; c.headerMode = 1 }
}

// WithHeaderFormatter selects a callback to render each header.
func WithHeaderFormatter(formatter func(ContextItem) (string, error)) ContextOption {
	return func(c *contextConfig) { c.headerFormatter = formatter; c.headerMode = 2 }
}

// WithCitationTemplate renders an optional citation before each passage.
func WithCitationTemplate(text string) ContextOption {
	return func(c *contextConfig) { c.citation = text; c.citationSet = true }
}

// WithContentTransform selects the last supplied content transformation.
func WithContentTransform(transform func(ContextItem) (string, error)) ContextOption {
	return func(c *contextConfig) { c.transform = transform; c.transformSet = true }
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
	text, _, err := FormatContextWithSummary(results, opts...)
	return text, err
}

// FormatContextWithSummary formats passages and reports their input positions.
func FormatContextWithSummary(results []*Result, opts ...ContextOption) (string, FormatSummary, error) {
	cfg := contextConfig{separator: "\n\n"}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	summary := FormatSummary{InputCount: len(results)}
	if cfg.limitSet && cfg.maxTokens <= 0 {
		classifySuffix(&summary, results, 0)
		return "", summary, nil
	}
	if cfg.limitSet && cfg.counter == nil {
		return "", FormatSummary{}, fmt.Errorf("retrieval: positive context token limit requires a token counter")
	}
	if cfg.headerMode == 2 && cfg.headerFormatter == nil {
		return "", FormatSummary{}, fmt.Errorf("retrieval: nil header formatter")
	}
	if cfg.transformSet && cfg.transform == nil {
		return "", FormatSummary{}, fmt.Errorf("retrieval: nil content transform")
	}
	var header, citation *template.Template
	var err error
	if cfg.headerMode == 1 {
		header, err = template.New("header").Option("missingkey=error").Parse(cfg.header)
		if err != nil {
			return "", FormatSummary{}, fmt.Errorf("retrieval: parse header template: %w", err)
		}
	}
	if cfg.citationSet {
		citation, err = template.New("citation").Option("missingkey=error").Parse(cfg.citation)
		if err != nil {
			return "", FormatSummary{}, fmt.Errorf("retrieval: parse citation template: %w", err)
		}
	}
	var formatted strings.Builder
	for i, result := range results {
		if result == nil || result.Content == "" {
			summary.SkippedIndices = append(summary.SkippedIndices, i)
			continue
		}
		source := result.Filename
		if source == "" {
			source = result.DocumentID
		}
		if source == "" {
			source = result.ID
		}
		start, end := contextLineRange(result.Metadata)
		item := ContextItem{Result: result, InputIndex: i, Number: len(summary.IncludedIndices) + 1, Source: source, StartLine: start, EndLine: end, HasLines: start != 0, Content: result.Content}
		if cfg.transformSet {
			item.Content, err = cfg.transform(item)
			if err != nil {
				return "", FormatSummary{}, fmt.Errorf("retrieval: transform content at input index %d: %w", i, err)
			}
			if item.Content == "" {
				summary.SkippedIndices = append(summary.SkippedIndices, i)
				continue
			}
		}
		var h, cite string
		switch cfg.headerMode {
		case 0:
			h = formatContextHeader(cfg.header, source, start, end)
		case 1:
			h, err = executeContextTemplate(header, item)
		case 2:
			h, err = cfg.headerFormatter(item)
		}
		if err != nil {
			return "", FormatSummary{}, fmt.Errorf("retrieval: format header at input index %d: %w", i, err)
		}
		if citation != nil {
			cite, err = executeContextTemplate(citation, item)
			if err != nil {
				return "", FormatSummary{}, fmt.Errorf("retrieval: format citation at input index %d: %w", i, err)
			}
		}
		if cite != "" {
			if h != "" {
				h = cite + " " + h
			} else {
				h = cite
			}
		}
		block := item.Content
		if h != "" {
			block = h + "\n" + block
		}
		tentative := block
		if len(summary.IncludedIndices) > 0 {
			tentative = formatted.String() + cfg.separator + block
		}
		if cfg.limitSet {
			count, countErr := cfg.counter.Count(tentative)
			if countErr != nil {
				return "", FormatSummary{}, fmt.Errorf("retrieval: count formatted context tokens: %w", countErr)
			}
			if count < 0 {
				return "", FormatSummary{}, fmt.Errorf("retrieval: count formatted context tokens: negative count %d", count)
			}
			if count > cfg.maxTokens {
				summary.OmittedIndices = append(summary.OmittedIndices, i)
				classifySuffix(&summary, results, i+1)
				break
			}
			summary.TokensUsed = &count
		}
		if len(summary.IncludedIndices) > 0 {
			formatted.WriteString(cfg.separator)
		}
		formatted.WriteString(block)
		summary.IncludedIndices = append(summary.IncludedIndices, i)
	}
	return formatted.String(), summary, nil
}

func classifySuffix(summary *FormatSummary, results []*Result, start int) {
	for i := start; i < len(results); i++ {
		if results[i] == nil || results[i].Content == "" {
			summary.SkippedIndices = append(summary.SkippedIndices, i)
		} else {
			summary.OmittedIndices = append(summary.OmittedIndices, i)
		}
	}
}

func executeContextTemplate(t *template.Template, item ContextItem) (string, error) {
	var out strings.Builder
	if err := t.Execute(&out, item); err != nil {
		return "", err
	}
	return out.String(), nil
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
