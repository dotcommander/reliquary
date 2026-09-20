// Package judgment defines provider-neutral typed evaluation contracts.
//
// Judgment replaces free-form generation with three strict result shapes:
// Choice (a definitive pick plus a probability distribution over an
// enumerable option set), Score (a finite 0-100 rating), and Noul (a strict
// yes/no verdict). Providers, prompts, caches, and escalation policy stay in
// adapters or applications; this package owns only the typed contracts and
// their validation.
package judgment

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
)

// ErrInvalidRequest reports a malformed judgment request.
var ErrInvalidRequest = errors.New("judgment: invalid request")

// ErrInvalidResult reports a malformed judgment result.
var ErrInvalidResult = errors.New("judgment: invalid result")

// distributionTolerance bounds accumulated rounding error when a complete
// choice distribution must sum to one.
const distributionTolerance = 1e-6

// Probability is a probability mass in [0, 1].
type Probability float64

// Valid reports whether p is finite and within [0, 1].
func (p Probability) Valid() bool {
	return !math.IsNaN(float64(p)) && !math.IsInf(float64(p), 0) && p >= 0 && p <= 1
}

// Option is one enumerable choice target. ID is the stable identifier that
// judged results reference; Description is optional supporting detail.
type Option struct {
	ID          string
	Description string
}

// ChoiceRequest asks a Chooser to pick exactly one option from Options.
type ChoiceRequest struct {
	// Task describes the decision being routed.
	Task string
	// Context carries the retrieved facts the judgment is based on.
	Context []string
	// Options enumerates the allowed targets. A choice request needs at least
	// two options with unique, non-blank IDs.
	Options []Option
}

// OptionProbability is the probability mass one choice option received.
type OptionProbability struct {
	Option      string
	Probability Probability
}

// Choice is a validated routing decision: the selected option plus a complete
// probability distribution over the request's option set.
type Choice struct {
	Selected     string
	Distribution []OptionProbability
}

// ScoreRequest asks a Scorer to rate one target on a fixed 0-100 scale.
type ScoreRequest struct {
	Task    string
	Target  string
	Context []string
}

// Score is a validated rating: a finite value in [0, 100] plus the judge's
// calibrated confidence in that value.
type Score struct {
	Value      float64
	Confidence Probability
}

// NoulRequest asks an Asker one strict yes/no question.
type NoulRequest struct {
	Question string
	Context  []string
}

// Verdict is a strict gate outcome.
type Verdict string

const (
	VerdictYes Verdict = "yes"
	VerdictNo  Verdict = "no"
)

// Valid reports whether v is one of the defined verdicts.
func (v Verdict) Valid() bool { return v == VerdictYes || v == VerdictNo }

// Bool returns the verdict as a boolean gate value. Unknown verdicts are
// false; ValidateNoul rejects them.
func (v Verdict) Bool() bool { return v == VerdictYes }

// Noul is a validated gate answer: a strict verdict plus the judge's
// calibrated confidence in that verdict.
type Noul struct {
	Verdict    Verdict
	Confidence Probability
}

// Yes reports whether the gate verdict is yes.
func (n Noul) Yes() bool { return n.Verdict == VerdictYes }

// Chooser returns a definitive pick from an enumerable option set together
// with a complete probability distribution over that set.
type Chooser interface {
	Choose(ctx context.Context, request ChoiceRequest) (Choice, error)
}

// Scorer returns a numeric 0-100 rating for one target given context.
type Scorer interface {
	Score(ctx context.Context, request ScoreRequest) (Score, error)
}

// Asker answers a strict yes/no question with a calibrated probability.
type Asker interface {
	Ask(ctx context.Context, request NoulRequest) (Noul, error)
}

// Judge is a provider-neutral evaluator implementing all three judgment
// primitives. Implementations may compose the individual interfaces.
type Judge interface {
	Chooser
	Scorer
	Asker
}

// ValidateChoice verifies a request/result pair: the request names at least
// two uniquely identified options, the selected option belongs to the
// request's option set, the distribution assigns exactly one probability in
// [0, 1] to every request option, the distribution sums to one within
// tolerance, and the selected option holds the modal (highest) probability.
func ValidateChoice(request ChoiceRequest, result Choice) error {
	if err := validateChoiceRequest(request); err != nil {
		return err
	}
	options := make(map[string]struct{}, len(request.Options))
	for _, option := range request.Options {
		options[option.ID] = struct{}{}
	}
	if _, ok := options[result.Selected]; !ok {
		return fmt.Errorf("%w: selected option %q is not in the request option set", ErrInvalidResult, result.Selected)
	}
	if len(result.Distribution) != len(request.Options) {
		return fmt.Errorf("%w: distribution has %d entries for %d options", ErrInvalidResult, len(result.Distribution), len(request.Options))
	}
	assigned := make(map[string]Probability, len(result.Distribution))
	var total Probability
	for _, entry := range result.Distribution {
		if _, ok := options[entry.Option]; !ok {
			return fmt.Errorf("%w: distribution entry %q is not in the request option set", ErrInvalidResult, entry.Option)
		}
		if _, duplicate := assigned[entry.Option]; duplicate {
			return fmt.Errorf("%w: distribution assigns option %q more than once", ErrInvalidResult, entry.Option)
		}
		if !entry.Probability.Valid() {
			return fmt.Errorf("%w: option %q has probability %v outside [0, 1]", ErrInvalidResult, entry.Option, float64(entry.Probability))
		}
		assigned[entry.Option] = entry.Probability
		total += entry.Probability
	}
	if math.Abs(float64(total)-1) > distributionTolerance {
		return fmt.Errorf("%w: distribution sums to %v, not 1", ErrInvalidResult, float64(total))
	}
	selected := assigned[result.Selected]
	for option, probability := range assigned {
		if probability > selected {
			return fmt.Errorf("%w: selected option %q is not the modal option; %q has probability %v", ErrInvalidResult, result.Selected, option, float64(probability))
		}
	}
	return nil
}

// ValidateScore verifies a request/result pair: the request names a task and
// target, the value is finite and within [0, 100], and confidence is a valid
// probability.
func ValidateScore(request ScoreRequest, result Score) error {
	if err := validateScoreRequest(request); err != nil {
		return err
	}
	if math.IsNaN(result.Value) || math.IsInf(result.Value, 0) {
		return fmt.Errorf("%w: score value is not finite", ErrInvalidResult)
	}
	if result.Value < 0 || result.Value > 100 {
		return fmt.Errorf("%w: score value %v outside [0, 100]", ErrInvalidResult, result.Value)
	}
	if !result.Confidence.Valid() {
		return fmt.Errorf("%w: confidence %v outside [0, 1]", ErrInvalidResult, float64(result.Confidence))
	}
	return nil
}

// ValidateNoul verifies a request/result pair: the request asks a question
// and the result carries a known verdict with a valid confidence probability.
func ValidateNoul(request NoulRequest, result Noul) error {
	if err := validateNoulRequest(request); err != nil {
		return err
	}
	if !result.Verdict.Valid() {
		return fmt.Errorf("%w: unknown verdict %q", ErrInvalidResult, result.Verdict)
	}
	if !result.Confidence.Valid() {
		return fmt.Errorf("%w: confidence %v outside [0, 1]", ErrInvalidResult, float64(result.Confidence))
	}
	return nil
}

func validateChoiceRequest(request ChoiceRequest) error {
	if strings.TrimSpace(request.Task) == "" {
		return fmt.Errorf("%w: task must not be blank", ErrInvalidRequest)
	}
	if len(request.Options) < 2 {
		return fmt.Errorf("%w: choice needs at least 2 options, got %d", ErrInvalidRequest, len(request.Options))
	}
	seen := make(map[string]struct{}, len(request.Options))
	for _, option := range request.Options {
		if strings.TrimSpace(option.ID) == "" {
			return fmt.Errorf("%w: option ID must not be blank", ErrInvalidRequest)
		}
		if _, duplicate := seen[option.ID]; duplicate {
			return fmt.Errorf("%w: duplicate option ID %q", ErrInvalidRequest, option.ID)
		}
		seen[option.ID] = struct{}{}
	}
	return nil
}

func validateScoreRequest(request ScoreRequest) error {
	if strings.TrimSpace(request.Task) == "" {
		return fmt.Errorf("%w: task must not be blank", ErrInvalidRequest)
	}
	if strings.TrimSpace(request.Target) == "" {
		return fmt.Errorf("%w: target must not be blank", ErrInvalidRequest)
	}
	return nil
}

func validateNoulRequest(request NoulRequest) error {
	if strings.TrimSpace(request.Question) == "" {
		return fmt.Errorf("%w: question must not be blank", ErrInvalidRequest)
	}
	return nil
}
