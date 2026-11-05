package judgment

import (
	"context"
	"errors"
	"testing"
)

type typedNilJudge struct{}

func (j *typedNilJudge) Choose(context.Context, ChoiceRequest) (Choice, error) {
	return Choice{}, nil
}

func (j *typedNilJudge) Score(context.Context, ScoreRequest) (Score, error) {
	return Score{}, nil
}

func (j *typedNilJudge) Ask(context.Context, NoulRequest) (Noul, error) {
	return Noul{}, nil
}

type typedNilMemory struct{}

func (m *typedNilMemory) GetChoice(Scope, ChoiceRequest) (Choice, bool) { return Choice{}, false }
func (m *typedNilMemory) PutChoice(Scope, ChoiceRequest, Choice) error  { return nil }
func (m *typedNilMemory) GetScore(Scope, ScoreRequest) (Score, bool)    { return Score{}, false }
func (m *typedNilMemory) PutScore(Scope, ScoreRequest, Score) error     { return nil }
func (m *typedNilMemory) GetNoul(Scope, NoulRequest) (Noul, bool)       { return Noul{}, false }
func (m *typedNilMemory) PutNoul(Scope, NoulRequest, Noul) error        { return nil }

func TestNewRouterRejectsTypedNilCollaborators(t *testing.T) {
	t.Parallel()
	valid := &typedNilJudge{}
	thresholds := Thresholds{Choice: 0.5, Score: 0.5, Noul: 0.5}
	cases := []struct {
		name       string
		reflex     Judge
		deliberate Judge
	}{
		{"typed-nil reflex", (*typedNilJudge)(nil), valid},
		{"typed-nil deliberate", valid, (*typedNilJudge)(nil)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewRouter(tc.reflex, tc.deliberate, &typedNilMemory{}, thresholds)
			if !errors.Is(err, ErrInvalidPolicy) {
				t.Fatalf("NewRouter error = %v, want ErrInvalidPolicy", err)
			}
		})
	}
}
