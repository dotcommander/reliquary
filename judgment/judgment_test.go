package judgment

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"
)

func validChoiceRequest() ChoiceRequest {
	return ChoiceRequest{
		Task: "route the incoming request",
		Options: []Option{
			{ID: "search"},
			{ID: "summarize"},
			{ID: "translate"},
		},
	}
}

func distribution(entries ...OptionProbability) []OptionProbability {
	return entries
}

func TestValidateChoice(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		request ChoiceRequest
		result  Choice
		wantErr error
	}{
		{
			name:    "valid modal selection",
			request: validChoiceRequest(),
			result: Choice{
				Selected: "search",
				Distribution: distribution(
					OptionProbability{Option: "search", Probability: 0.5},
					OptionProbability{Option: "summarize", Probability: 0.3},
					OptionProbability{Option: "translate", Probability: 0.2},
				),
			},
		},
		{
			name:    "valid uniform selection",
			request: validChoiceRequest(),
			result: Choice{
				Selected: "summarize",
				Distribution: distribution(
					OptionProbability{Option: "search", Probability: 1.0 / 3.0},
					OptionProbability{Option: "summarize", Probability: 1.0 / 3.0},
					OptionProbability{Option: "translate", Probability: 1.0 / 3.0},
				),
			},
		},
		{
			name: "valid two-option certainty",
			request: ChoiceRequest{
				Task:    "pick a lane",
				Options: []Option{{ID: "a"}, {ID: "b"}},
			},
			result: Choice{
				Selected: "b",
				Distribution: distribution(
					OptionProbability{Option: "a", Probability: 0},
					OptionProbability{Option: "b", Probability: 1},
				),
			},
		},
		{
			name:    "selected not modal",
			request: validChoiceRequest(),
			result: Choice{
				Selected: "translate",
				Distribution: distribution(
					OptionProbability{Option: "search", Probability: 0.5},
					OptionProbability{Option: "summarize", Probability: 0.3},
					OptionProbability{Option: "translate", Probability: 0.2},
				),
			},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "selected not in option set",
			request: validChoiceRequest(),
			result: Choice{
				Selected: "delete",
				Distribution: distribution(
					OptionProbability{Option: "search", Probability: 0.5},
					OptionProbability{Option: "summarize", Probability: 0.3},
					OptionProbability{Option: "translate", Probability: 0.2},
				),
			},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "distribution length mismatch",
			request: validChoiceRequest(),
			result: Choice{
				Selected: "search",
				Distribution: distribution(
					OptionProbability{Option: "search", Probability: 1},
				),
			},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "distribution entry not in option set",
			request: validChoiceRequest(),
			result: Choice{
				Selected: "search",
				Distribution: distribution(
					OptionProbability{Option: "search", Probability: 0.4},
					OptionProbability{Option: "delete", Probability: 0.4},
					OptionProbability{Option: "summarize", Probability: 0.2},
				),
			},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "distribution assigns option twice",
			request: validChoiceRequest(),
			result: Choice{
				Selected: "search",
				Distribution: distribution(
					OptionProbability{Option: "search", Probability: 0.4},
					OptionProbability{Option: "search", Probability: 0.4},
					OptionProbability{Option: "summarize", Probability: 0.2},
				),
			},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "negative probability",
			request: validChoiceRequest(),
			result: Choice{
				Selected: "search",
				Distribution: distribution(
					OptionProbability{Option: "search", Probability: 1.1},
					OptionProbability{Option: "summarize", Probability: -0.1},
					OptionProbability{Option: "translate", Probability: 0},
				),
			},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "probability above one",
			request: validChoiceRequest(),
			result: Choice{
				Selected: "search",
				Distribution: distribution(
					OptionProbability{Option: "search", Probability: 1.2},
					OptionProbability{Option: "summarize", Probability: -0.2},
					OptionProbability{Option: "translate", Probability: 0},
				),
			},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "probability is NaN",
			request: validChoiceRequest(),
			result: Choice{
				Selected: "search",
				Distribution: distribution(
					OptionProbability{Option: "search", Probability: Probability(math.NaN())},
					OptionProbability{Option: "summarize", Probability: 1},
					OptionProbability{Option: "translate", Probability: 0},
				),
			},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "distribution does not sum to one",
			request: validChoiceRequest(),
			result: Choice{
				Selected: "search",
				Distribution: distribution(
					OptionProbability{Option: "search", Probability: 0.5},
					OptionProbability{Option: "summarize", Probability: 0.3},
					OptionProbability{Option: "translate", Probability: 0.1},
				),
			},
			wantErr: ErrInvalidResult,
		},
		{
			name: "blank task",
			request: ChoiceRequest{
				Options: []Option{{ID: "a"}, {ID: "b"}},
			},
			result:  Choice{},
			wantErr: ErrInvalidRequest,
		},
		{
			name: "single option",
			request: ChoiceRequest{
				Task:    "pick",
				Options: []Option{{ID: "a"}},
			},
			result:  Choice{},
			wantErr: ErrInvalidRequest,
		},
		{
			name: "blank option ID",
			request: ChoiceRequest{
				Task:    "pick",
				Options: []Option{{ID: "a"}, {ID: " "}},
			},
			result:  Choice{},
			wantErr: ErrInvalidRequest,
		},
		{
			name: "duplicate option ID",
			request: ChoiceRequest{
				Task:    "pick",
				Options: []Option{{ID: "a"}, {ID: "a"}},
			},
			result:  Choice{},
			wantErr: ErrInvalidRequest,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			err := ValidateChoice(test.request, test.result)
			if test.wantErr == nil {
				if err != nil {
					t.Fatalf("ValidateChoice(%+v) unexpected error: %v", test.result, err)
				}
				return
			}
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("ValidateChoice(%+v) error = %v, want %v", test.result, err, test.wantErr)
			}
		})
	}
}

func TestValidateScore(t *testing.T) {
	t.Parallel()

	validRequest := ScoreRequest{Task: "rank", Target: "doc-1"}
	tests := []struct {
		name    string
		request ScoreRequest
		result  Score
		wantErr error
	}{
		{
			name:    "valid zero bound",
			request: validRequest,
			result:  Score{Value: 0, Confidence: 0.9},
		},
		{
			name:    "valid hundred bound",
			request: validRequest,
			result:  Score{Value: 100, Confidence: 1},
		},
		{
			name:    "valid midpoint",
			request: validRequest,
			result:  Score{Value: 42.5, Confidence: 0},
		},
		{
			name:    "value below zero",
			request: validRequest,
			result:  Score{Value: -0.1},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "value above hundred",
			request: validRequest,
			result:  Score{Value: 100.1},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "value is NaN",
			request: validRequest,
			result:  Score{Value: math.NaN()},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "value is infinity",
			request: validRequest,
			result:  Score{Value: math.Inf(1)},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "confidence below zero",
			request: validRequest,
			result:  Score{Value: 50, Confidence: -0.1},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "confidence above one",
			request: validRequest,
			result:  Score{Value: 50, Confidence: 1.1},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "confidence is NaN",
			request: validRequest,
			result:  Score{Value: 50, Confidence: Probability(math.NaN())},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "blank task",
			request: ScoreRequest{Target: "doc-1"},
			wantErr: ErrInvalidRequest,
		},
		{
			name:    "blank target",
			request: ScoreRequest{Task: "rank"},
			wantErr: ErrInvalidRequest,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			err := ValidateScore(test.request, test.result)
			if test.wantErr == nil {
				if err != nil {
					t.Fatalf("ValidateScore(%+v) unexpected error: %v", test.result, err)
				}
				return
			}
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("ValidateScore(%+v) error = %v, want %v", test.result, err, test.wantErr)
			}
		})
	}
}

func TestValidateNoul(t *testing.T) {
	t.Parallel()

	validRequest := NoulRequest{Question: "is the build green?"}
	tests := []struct {
		name    string
		request NoulRequest
		result  Noul
		wantErr error
	}{
		{
			name:    "valid yes verdict",
			request: validRequest,
			result:  Noul{Verdict: VerdictYes, Confidence: 0.97},
		},
		{
			name:    "valid no verdict",
			request: validRequest,
			result:  Noul{Verdict: VerdictNo, Confidence: 0.2},
		},
		{
			name:    "unknown verdict",
			request: validRequest,
			result:  Noul{Verdict: Verdict("maybe")},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "empty verdict",
			request: validRequest,
			result:  Noul{},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "confidence below zero",
			request: validRequest,
			result:  Noul{Verdict: VerdictYes, Confidence: -0.1},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "confidence above one",
			request: validRequest,
			result:  Noul{Verdict: VerdictNo, Confidence: 1.5},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "confidence is infinity",
			request: validRequest,
			result:  Noul{Verdict: VerdictYes, Confidence: Probability(math.Inf(1))},
			wantErr: ErrInvalidResult,
		},
		{
			name:    "blank question",
			request: NoulRequest{Question: "  "},
			wantErr: ErrInvalidRequest,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			err := ValidateNoul(test.request, test.result)
			if test.wantErr == nil {
				if err != nil {
					t.Fatalf("ValidateNoul(%+v) unexpected error: %v", test.result, err)
				}
				return
			}
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("ValidateNoul(%+v) error = %v, want %v", test.result, err, test.wantErr)
			}
		})
	}
}

func TestProbabilityValid(t *testing.T) {
	t.Parallel()

	tests := []struct {
		value Probability
		valid bool
	}{
		{Probability(0), true},
		{Probability(1), true},
		{Probability(0.5), true},
		{Probability(-0.0000001), false},
		{Probability(1.0000001), false},
		{Probability(math.NaN()), false},
		{Probability(math.Inf(1)), false},
		{Probability(math.Inf(-1)), false},
	}
	for _, test := range tests {
		if got := test.value.Valid(); got != test.valid {
			t.Errorf("Probability(%v).Valid() = %v, want %v", float64(test.value), got, test.valid)
		}
	}
}

func TestVerdict(t *testing.T) {
	t.Parallel()

	if !VerdictYes.Valid() || !VerdictNo.Valid() {
		t.Error("defined verdicts must be valid")
	}
	if Verdict("maybe").Valid() {
		t.Error("unknown verdict must be invalid")
	}
	if !VerdictYes.Bool() || VerdictNo.Bool() {
		t.Error("verdict boolean mapping is wrong")
	}
	if !(Noul{Verdict: VerdictYes}).Yes() || (Noul{Verdict: VerdictNo}).Yes() {
		t.Error("Noul.Yes must mirror the verdict")
	}
}

// stubJudge is a deterministic Judge used to prove the contract interfaces
// compile and round-trip validated results without any provider.
type stubJudge struct{}

func (stubJudge) Choose(_ context.Context, request ChoiceRequest) (Choice, error) {
	share := 1.0 / float64(2*len(request.Options))
	probabilities := make([]OptionProbability, len(request.Options))
	for i, option := range request.Options {
		mass := Probability(share)
		if i == 0 {
			mass = Probability(0.5 + share)
		}
		probabilities[i] = OptionProbability{Option: option.ID, Probability: mass}
	}
	return Choice{Selected: request.Options[0].ID, Distribution: probabilities}, nil
}

func (stubJudge) Score(_ context.Context, _ ScoreRequest) (Score, error) {
	return Score{Value: 80, Confidence: 0.9}, nil
}

func (stubJudge) Ask(_ context.Context, _ NoulRequest) (Noul, error) {
	return Noul{Verdict: VerdictYes, Confidence: 0.99}, nil
}

var _ Judge = stubJudge{}

func TestJudgeInterfaces(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	var judge Judge = stubJudge{}

	choice, err := judge.Choose(ctx, validChoiceRequest())
	if err != nil {
		t.Fatalf("Choose unexpected error: %v", err)
	}
	if err := ValidateChoice(validChoiceRequest(), choice); err != nil {
		t.Fatalf("stub choice does not validate: %v", err)
	}
	if choice.Selected != "search" {
		t.Errorf("Choose selected %q, want %q", choice.Selected, "search")
	}

	scoreRequest := ScoreRequest{Task: "rank", Target: "doc-1"}
	score, err := judge.Score(ctx, scoreRequest)
	if err != nil {
		t.Fatalf("Score unexpected error: %v", err)
	}
	if err := ValidateScore(scoreRequest, score); err != nil {
		t.Fatalf("stub score does not validate: %v", err)
	}

	noulRequest := NoulRequest{Question: "is the build green?"}
	answer, err := judge.Ask(ctx, noulRequest)
	if err != nil {
		t.Fatalf("Ask unexpected error: %v", err)
	}
	if err := ValidateNoul(noulRequest, answer); err != nil {
		t.Fatalf("stub noul does not validate: %v", err)
	}
}

func ExampleValidateChoice() {
	request := ChoiceRequest{
		Task: "route the incoming request",
		Options: []Option{
			{ID: "search"},
			{ID: "summarize"},
		},
	}
	result := Choice{
		Selected: "search",
		Distribution: []OptionProbability{
			{Option: "search", Probability: 0.75},
			{Option: "summarize", Probability: 0.25},
		},
	}
	if err := ValidateChoice(request, result); err != nil {
		fmt.Println("invalid choice:", err)
		return
	}
	fmt.Println("selected:", result.Selected)
	// Output: selected: search
}

func ExampleValidateNoul() {
	request := NoulRequest{Question: "is the build green?"}
	result := Noul{Verdict: VerdictYes, Confidence: 0.97}
	if err := ValidateNoul(request, result); err != nil {
		fmt.Println("invalid noul:", err)
		return
	}
	fmt.Println("gate open:", result.Yes())
	// Output: gate open: true
}
