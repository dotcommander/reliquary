package structjudge

import (
	"context"
	"errors"
	"testing"

	"github.com/dotcommander/reliquary/judgment"
)

type typedNilClient struct{}

func (c *typedNilClient) Generate(context.Context, Request) ([]byte, error) {
	return nil, nil
}

func TestNewRejectsTypedNilClient(t *testing.T) {
	t.Parallel()
	if _, err := New((*typedNilClient)(nil)); !errors.Is(err, ErrNilClient) {
		t.Fatalf("New error = %v, want ErrNilClient", err)
	}
}

type staticClient struct {
	raw []byte
}

func (c staticClient) Generate(context.Context, Request) ([]byte, error) {
	return c.raw, nil
}

func TestAdapterRejectsNullAndMissingRequiredFields(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	choiceReq := judgment.ChoiceRequest{
		Task: "pick",
		Options: []judgment.Option{
			{ID: "a"},
			{ID: "b"},
		},
	}
	scoreReq := judgment.ScoreRequest{Task: "rate", Target: "item"}
	noulReq := judgment.NoulRequest{Question: "ok?"}

	badChoicePayloads := []string{
		`null`,
		`{}`,
		`{"selected":"a"}`,
		`{"distribution":[{"option":"a","probability":1},{"option":"b","probability":0}]}`,
		`{"selected":null,"distribution":[{"option":"a","probability":1},{"option":"b","probability":0}]}`,
		`{"selected":"a","distribution":null}`,
		`{"selected":"a","distribution":[{"option":"a"},{"option":"b","probability":0}]}`,
		`{"selected":"a","distribution":[{"probability":1},{"option":"b","probability":0}]}`,
		`{"selected":"a","distribution":[null,{"option":"b","probability":0}]}`,
	}
	for _, raw := range badChoicePayloads {
		adapter, err := New(staticClient{raw: []byte(raw)})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, err := adapter.Choose(ctx, choiceReq); !errors.Is(err, ErrInvalidResponse) {
			t.Errorf("Choose(%s) error = %v, want ErrInvalidResponse", raw, err)
		}
	}

	badScorePayloads := []string{
		`null`,
		`{}`,
		`{"value":50}`,
		`{"confidence":0.8}`,
		`{"value":null,"confidence":0.8}`,
		`{"value":50,"confidence":null}`,
	}
	for _, raw := range badScorePayloads {
		adapter, err := New(staticClient{raw: []byte(raw)})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, err := adapter.Score(ctx, scoreReq); !errors.Is(err, ErrInvalidResponse) {
			t.Errorf("Score(%s) error = %v, want ErrInvalidResponse", raw, err)
		}
	}

	badNoulPayloads := []string{
		`null`,
		`{}`,
		`{"verdict":"yes"}`,
		`{"confidence":0.8}`,
		`{"verdict":null,"confidence":0.8}`,
		`{"verdict":"yes","confidence":null}`,
	}
	for _, raw := range badNoulPayloads {
		adapter, err := New(staticClient{raw: []byte(raw)})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, err := adapter.Ask(ctx, noulReq); !errors.Is(err, ErrInvalidResponse) {
			t.Errorf("Ask(%s) error = %v, want ErrInvalidResponse", raw, err)
		}
	}

	// Explicit zero values must still be accepted.
	scoreZero, err := New(staticClient{raw: []byte(`{"value":0,"confidence":0}`)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	gotScore, err := scoreZero.Score(ctx, scoreReq)
	if err != nil || gotScore.Value != 0 || gotScore.Confidence != 0 {
		t.Fatalf("Score(zero) = (%+v, %v), want zero score with nil error", gotScore, err)
	}

	noulZero, err := New(staticClient{raw: []byte(`{"verdict":"no","confidence":0}`)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	gotNoul, err := noulZero.Ask(ctx, noulReq)
	if err != nil || gotNoul.Verdict != judgment.VerdictNo || gotNoul.Confidence != 0 {
		t.Fatalf("Ask(zero confidence) = (%+v, %v), want no/0 with nil error", gotNoul, err)
	}
}
