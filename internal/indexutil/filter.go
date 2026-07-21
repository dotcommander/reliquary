package indexutil

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"strconv"

	"github.com/dotcommander/reliquary/retrieval"
)

// ValidateFilter enforces the backend-independent JSON-scalar filter contract.
func ValidateFilter(filter map[string]any) error {
	for key, value := range filter {
		if _, ok, err := jsonNumber(value); err != nil {
			return fmt.Errorf("filter %q: %w", key, err)
		} else if ok {
			continue
		}
		switch value.(type) {
		case nil, string, bool:
		default:
			return fmt.Errorf("filter %q must be scalar JSON", key)
		}
	}
	return nil
}

// MatchesFilter reports whether item matches every filter entry. Callers must
// validate the filter first with ValidateFilter.
func MatchesFilter(item *retrieval.Result, filter map[string]any) bool {
	for key, want := range filter {
		var got any
		var present bool
		switch key {
		case "id":
			got, present = item.ID, true
		case "document_id":
			got, present = item.DocumentID, true
		case "filename":
			got, present = item.Filename, true
		default:
			got, present = item.Metadata[key]
		}
		if !present || !scalarEqual(got, want) {
			return false
		}
	}
	return true
}

// IsJSONNumber reports whether value is one of the accepted numeric filter
// types. It assumes ValidateFilter has already accepted the value.
func IsJSONNumber(value any) bool {
	_, ok, _ := jsonNumber(value)
	return ok
}

func scalarEqual(got, want any) bool {
	if want == nil {
		return got == nil
	}
	if wantNumber, ok, err := jsonNumber(want); ok && err == nil {
		gotNumber, gotOK, gotErr := jsonNumber(got)
		return gotOK && gotErr == nil && wantNumber.Cmp(gotNumber) == 0
	}
	switch want := want.(type) {
	case string:
		got, ok := got.(string)
		return ok && got == want
	case bool:
		got, ok := got.(bool)
		return ok && got == want
	default:
		return false
	}
}

func jsonNumber(value any) (*big.Rat, bool, error) {
	var text string
	switch value := value.(type) {
	case int:
		text = strconv.FormatInt(int64(value), 10)
	case int8:
		text = strconv.FormatInt(int64(value), 10)
	case int16:
		text = strconv.FormatInt(int64(value), 10)
	case int32:
		text = strconv.FormatInt(int64(value), 10)
	case int64:
		text = strconv.FormatInt(value, 10)
	case uint:
		text = strconv.FormatUint(uint64(value), 10)
	case uint8:
		text = strconv.FormatUint(uint64(value), 10)
	case uint16:
		text = strconv.FormatUint(uint64(value), 10)
	case uint32:
		text = strconv.FormatUint(uint64(value), 10)
	case uint64:
		text = strconv.FormatUint(value, 10)
	case float32:
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, true, fmt.Errorf("numeric value must be finite")
		}
		text = strconv.FormatFloat(float64(value), 'g', -1, 32)
	case float64:
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, true, fmt.Errorf("numeric value must be finite")
		}
		text = strconv.FormatFloat(value, 'g', -1, 64)
	case json.Number:
		if _, err := json.Marshal(value); err != nil {
			return nil, true, fmt.Errorf("invalid JSON number %q: %w", value, err)
		}
		text = value.String()
	default:
		return nil, false, nil
	}
	number, ok := new(big.Rat).SetString(text)
	if !ok {
		return nil, true, fmt.Errorf("invalid JSON number %q", text)
	}
	return number, true, nil
}
