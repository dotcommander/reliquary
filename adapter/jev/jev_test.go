package jev

import (
	"context"
	"errors"
	"testing"

	"github.com/dotcommander/reliquary/judgment"
)

// errStub is a sentinel the stub client replays to prove provider errors
// propagate unchanged.
var errStub = errors.New("stub: jev unavailable")

type ctxKey struct{}

// stubClient is a deterministic in-memory Jev client. It records every call
// and context it receives and replays a fixed result or error, proving the
// adapter needs no transport of its own.
type stubClient struct {
	result Result
	err    error

	calls []Call
	ctxs  []context.Context
}

var _ Client = (*stubClient)(nil)

func (s *stubClient) Evaluate(ctx context.Context, call Call) (Result, error) {
	s.calls = append(s.calls, call)
	s.ctxs = append(s.ctxs, ctx)
	if s.err != nil {
		return Result{}, s.err
	}
	return s.result, nil
}

func stubChoiceRequest() judgment.ChoiceRequest {
	return judgment.ChoiceRequest{
		Task:    "route the incoming request",
		Context: []string{"facts: build is green", "facts: tests pass"},
		Options: []judgment.Option{
			{ID: "search"},
			{ID: "summarize", Description: "condense the retrieved context"},
			{ID: "translate"},
		},
	}
}

func stubChoiceResult() Result {
	return Result{
		Selected: "search",
		Distribution: []Mass{
			{Option: "search", Probability: 0.5},
			{Option: "summarize", Probability: 0.3},
			{Option: "translate", Probability: 0.2},
		},
	}
}

func TestNewRejectsNilClient(t *testing.T) {
	t.Parallel()

	if _, err := New(nil); !errors.Is(err, ErrNilClient) {
		t.Fatalf("New(nil) error = %v, want ErrNilClient", err)
	}

	adapter, err := New(&stubClient{})
	if err != nil {
		t.Fatalf("New(stub) unexpected error: %v", err)
	}
	if adapter == nil {
		t.Fatal("New(stub) returned a nil adapter")
	}
}

func TestChooseMapping(t *testing.T) {
	t.Parallel()

	client := &stubClient{result: stubChoiceResult()}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	request := stubChoiceRequest()
	ctx := context.WithValue(context.Background(), ctxKey{}, "choose-worker")

	choice, err := adapter.Choose(ctx, request)
	if err != nil {
		t.Fatalf("Choose unexpected error: %v", err)
	}
	if choice.Selected != "search" {
		t.Errorf("Choose selected %q, want %q", choice.Selected, "search")
	}
	want := []judgment.OptionProbability{
		{Option: "search", Probability: 0.5},
		{Option: "summarize", Probability: 0.3},
		{Option: "translate", Probability: 0.2},
	}
	if len(choice.Distribution) != len(want) {
		t.Fatalf("Choose distribution length = %d, want %d", len(choice.Distribution), len(want))
	}
	for i, entry := range choice.Distribution {
		if entry != want[i] {
			t.Errorf("Choose distribution[%d] = %+v, want %+v", i, entry, want[i])
		}
	}
	if err := judgment.ValidateChoice(request, choice); err != nil {
		t.Errorf("mapped choice does not validate: %v", err)
	}

	if len(client.calls) != 1 {
		t.Fatalf("client received %d calls, want 1", len(client.calls))
	}
	call := client.calls[0]
	if call.Kind != KindChoice {
		t.Errorf("call.Kind = %q, want %q", call.Kind, KindChoice)
	}
	if call.Task != request.Task {
		t.Errorf("call.Task = %q, want %q", call.Task, request.Task)
	}
	if call.Target != "" || call.Question != "" {
		t.Errorf("choice call must not carry score or noul fields: %+v", call)
	}
	if len(call.Context) != len(request.Context) || call.Context[0] != request.Context[0] || call.Context[1] != request.Context[1] {
		t.Errorf("call.Context = %v, want %v", call.Context, request.Context)
	}
	if &call.Context[0] == &request.Context[0] {
		t.Error("call.Context aliases the caller's slice instead of a copy")
	}
	if len(call.Options) != len(request.Options) {
		t.Fatalf("call.Options length = %d, want %d", len(call.Options), len(request.Options))
	}
	for i, option := range call.Options {
		if option.ID != request.Options[i].ID || option.Description != request.Options[i].Description {
			t.Errorf("call.Options[%d] = %+v, want %+v", i, option, request.Options[i])
		}
	}
	if got := client.ctxs[0].Value(ctxKey{}); got != "choose-worker" {
		t.Errorf("call ctx did not propagate, got %v", got)
	}
}

func TestScoreMapping(t *testing.T) {
	t.Parallel()

	client := &stubClient{result: Result{Value: 87.5, Confidence: 0.82}}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	request := judgment.ScoreRequest{
		Task:    "rank the snippet",
		Target:  "doc-1",
		Context: []string{"facts: doc-1 answers the question"},
	}
	ctx := context.WithValue(context.Background(), ctxKey{}, "score-worker")

	score, err := adapter.Score(ctx, request)
	if err != nil {
		t.Fatalf("Score unexpected error: %v", err)
	}
	if score.Value != 87.5 {
		t.Errorf("Score value = %v, want 87.5", score.Value)
	}
	if score.Confidence != 0.82 {
		t.Errorf("Score confidence = %v, want 0.82", score.Confidence)
	}
	if err := judgment.ValidateScore(request, score); err != nil {
		t.Errorf("mapped score does not validate: %v", err)
	}

	if len(client.calls) != 1 {
		t.Fatalf("client received %d calls, want 1", len(client.calls))
	}
	call := client.calls[0]
	if call.Kind != KindScore {
		t.Errorf("call.Kind = %q, want %q", call.Kind, KindScore)
	}
	if call.Task != request.Task {
		t.Errorf("call.Task = %q, want %q", call.Task, request.Task)
	}
	if call.Target != request.Target {
		t.Errorf("call.Target = %q, want %q", call.Target, request.Target)
	}
	if call.Options != nil || call.Question != "" {
		t.Errorf("score call must not carry choice or noul fields: %+v", call)
	}
	if len(call.Context) != 1 || call.Context[0] != request.Context[0] {
		t.Errorf("call.Context = %v, want %v", call.Context, request.Context)
	}
	if got := client.ctxs[0].Value(ctxKey{}); got != "score-worker" {
		t.Errorf("call ctx did not propagate, got %v", got)
	}
}

func TestAskMapping(t *testing.T) {
	t.Parallel()

	client := &stubClient{result: Result{Verdict: "no", Confidence: 0.91}}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	request := judgment.NoulRequest{
		Question: "is the build broken?",
		Context:  []string{"facts: build is green"},
	}
	ctx := context.WithValue(context.Background(), ctxKey{}, "ask-worker")

	answer, err := adapter.Ask(ctx, request)
	if err != nil {
		t.Fatalf("Ask unexpected error: %v", err)
	}
	if answer.Verdict != judgment.VerdictNo {
		t.Errorf("Ask verdict = %q, want %q", answer.Verdict, judgment.VerdictNo)
	}
	if answer.Confidence != 0.91 {
		t.Errorf("Ask confidence = %v, want 0.91", answer.Confidence)
	}
	if err := judgment.ValidateNoul(request, answer); err != nil {
		t.Errorf("mapped noul does not validate: %v", err)
	}

	if len(client.calls) != 1 {
		t.Fatalf("client received %d calls, want 1", len(client.calls))
	}
	call := client.calls[0]
	if call.Kind != KindNoul {
		t.Errorf("call.Kind = %q, want %q", call.Kind, KindNoul)
	}
	if call.Task != "" || call.Target != "" || call.Options != nil {
		t.Errorf("noul call must not carry choice or score fields: %+v", call)
	}
	if call.Question != request.Question {
		t.Errorf("call.Question = %q, want %q", call.Question, request.Question)
	}
	if len(call.Context) != 1 || call.Context[0] != request.Context[0] {
		t.Errorf("call.Context = %v, want %v", call.Context, request.Context)
	}
	if got := client.ctxs[0].Value(ctxKey{}); got != "ask-worker" {
		t.Errorf("call ctx did not propagate, got %v", got)
	}
}

func TestAskWithoutContextSendsNilContext(t *testing.T) {
	t.Parallel()

	client := &stubClient{result: Result{Verdict: "yes", Confidence: 1}}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	request := judgment.NoulRequest{Question: "is the build green?"}

	if _, err := adapter.Ask(context.Background(), request); err != nil {
		t.Fatalf("Ask unexpected error: %v", err)
	}
	if client.calls[0].Context != nil {
		t.Errorf("call.Context = %v, want nil", client.calls[0].Context)
	}
}

func TestJudgeComposition(t *testing.T) {
	t.Parallel()

	client := &stubClient{result: stubChoiceResult()}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	var judge judgment.Judge = adapter

	ctx := context.Background()
	if _, err := judge.Choose(ctx, stubChoiceRequest()); err != nil {
		t.Fatalf("Judge.Choose unexpected error: %v", err)
	}

	client.result = Result{Value: 42, Confidence: 0.5}
	if _, err := judge.Score(ctx, judgment.ScoreRequest{Task: "rank", Target: "doc-1"}); err != nil {
		t.Fatalf("Judge.Score unexpected error: %v", err)
	}

	client.result = Result{Verdict: "yes", Confidence: 0.99}
	if _, err := judge.Ask(ctx, judgment.NoulRequest{Question: "is the build green?"}); err != nil {
		t.Fatalf("Judge.Ask unexpected error: %v", err)
	}
	if len(client.calls) != 3 {
		t.Errorf("client received %d calls, want 3", len(client.calls))
	}
}

func TestClientErrorPropagates(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		invoke  func(*Adapter, context.Context) error
		wantErr error
	}{
		{
			name: "choose",
			invoke: func(adapter *Adapter, ctx context.Context) error {
				_, err := adapter.Choose(ctx, stubChoiceRequest())
				return err
			},
			wantErr: errStub,
		},
		{
			name: "score",
			invoke: func(adapter *Adapter, ctx context.Context) error {
				_, err := adapter.Score(ctx, judgment.ScoreRequest{Task: "rank", Target: "doc-1"})
				return err
			},
			wantErr: errStub,
		},
		{
			name: "ask",
			invoke: func(adapter *Adapter, ctx context.Context) error {
				_, err := adapter.Ask(ctx, judgment.NoulRequest{Question: "is the build green?"})
				return err
			},
			wantErr: errStub,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			client := &stubClient{err: errStub}
			adapter, err := New(client)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if err := test.invoke(adapter, context.Background()); !errors.Is(err, test.wantErr) {
				t.Fatalf("error = %v, want %v", err, test.wantErr)
			}
		})
	}
}

func TestInvalidResultsRejected(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		result  Result
		invoke  func(*Adapter, context.Context, Result) error
		wantErr error
	}{
		{
			name:   "choice distribution does not sum to one",
			result: Result{Selected: "search", Distribution: []Mass{{Option: "search", Probability: 0.6}, {Option: "summarize", Probability: 0.35}, {Option: "translate", Probability: 0.15}}},
			invoke: func(adapter *Adapter, ctx context.Context, _ Result) error {
				_, err := adapter.Choose(ctx, stubChoiceRequest())
				return err
			},
			wantErr: judgment.ErrInvalidResult,
		},
		{
			name:   "choice selected not in option set",
			result: Result{Selected: "delete", Distribution: []Mass{{Option: "search", Probability: 1}}},
			invoke: func(adapter *Adapter, ctx context.Context, _ Result) error {
				_, err := adapter.Choose(ctx, stubChoiceRequest())
				return err
			},
			wantErr: judgment.ErrInvalidResult,
		},
		{
			name:   "score value above hundred",
			result: Result{Value: 100.5, Confidence: 0.9},
			invoke: func(adapter *Adapter, ctx context.Context, _ Result) error {
				_, err := adapter.Score(ctx, judgment.ScoreRequest{Task: "rank", Target: "doc-1"})
				return err
			},
			wantErr: judgment.ErrInvalidResult,
		},
		{
			name:   "score confidence above one",
			result: Result{Value: 50, Confidence: 1.5},
			invoke: func(adapter *Adapter, ctx context.Context, _ Result) error {
				_, err := adapter.Score(ctx, judgment.ScoreRequest{Task: "rank", Target: "doc-1"})
				return err
			},
			wantErr: judgment.ErrInvalidResult,
		},
		{
			name:   "noul unknown verdict",
			result: Result{Verdict: "maybe", Confidence: 0.9},
			invoke: func(adapter *Adapter, ctx context.Context, _ Result) error {
				_, err := adapter.Ask(ctx, judgment.NoulRequest{Question: "is the build green?"})
				return err
			},
			wantErr: judgment.ErrInvalidResult,
		},
		{
			name:  "blank choice task",
			invoke: func(adapter *Adapter, ctx context.Context, result Result) error {
				request := stubChoiceRequest()
				request.Task = ""
				_, err := adapter.Choose(ctx, request)
				return err
			},
			wantErr: judgment.ErrInvalidRequest,
		},
		{
			name:  "single choice option",
			invoke: func(adapter *Adapter, ctx context.Context, result Result) error {
				request := judgment.ChoiceRequest{Task: "pick", Options: []judgment.Option{{ID: "a"}}}
				_, err := adapter.Choose(ctx, request)
				return err
			},
			wantErr: judgment.ErrInvalidRequest,
		},
		{
			name:  "blank noul question",
			invoke: func(adapter *Adapter, ctx context.Context, result Result) error {
				_, err := adapter.Ask(ctx, judgment.NoulRequest{Question: " "})
				return err
			},
			wantErr: judgment.ErrInvalidRequest,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			client := &stubClient{result: test.result}
			adapter, err := New(client)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if err := test.invoke(adapter, context.Background(), test.result); !errors.Is(err, test.wantErr) {
				t.Fatalf("error = %v, want %v", err, test.wantErr)
			}
		})
	}
}

func TestCallIsIndependentOfCallerMutation(t *testing.T) {
	t.Parallel()

	client := &stubClient{result: stubChoiceResult()}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	request := stubChoiceRequest()

	if _, err := adapter.Choose(context.Background(), request); err != nil {
		t.Fatalf("Choose unexpected error: %v", err)
	}

	request.Context[0] = "mutated after the call"
	request.Options[0].ID = "mutated after the call"

	call := client.calls[0]
	if call.Context[0] != "facts: build is green" {
		t.Errorf("call.Context was mutated through the caller slice: %v", call.Context)
	}
	if call.Options[0].ID != "search" {
		t.Errorf("call.Options was mutated through the caller slice: %v", call.Options)
	}
}
