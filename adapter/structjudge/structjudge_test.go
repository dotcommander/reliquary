package structjudge

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/dotcommander/reliquary/judgment"
)

// scriptedClient answers Generate from a fixed payload (or error) and
// captures the last request it served.
type scriptedClient struct {
	payload []byte
	err     error

	calls    int
	lastReq  Request
	lastKind string
}

func (c *scriptedClient) Generate(_ context.Context, request Request) ([]byte, error) {
	c.calls++
	c.lastReq = request
	return c.payload, c.err
}

func choicePayload() []byte {
	return []byte(`{
		"selected": "search",
		"distribution": [
			{"option": "search", "probability": 0.7},
			{"option": "summarize", "probability": 0.2},
			{"option": "translate", "probability": 0.1}
		]
	}`)
}

func choiceRequest() judgment.ChoiceRequest {
	return judgment.ChoiceRequest{
		Task: "route the user request",
		Options: []judgment.Option{
			{ID: "search", Description: "retrieve passages"},
			{ID: "summarize", Description: "condense passages"},
			{ID: "translate"},
		},
		Context: []string{"the corpus holds go internals notes"},
	}
}

func TestNewRejectsNilClient(t *testing.T) {
	t.Parallel()

	if _, err := New(nil); !errors.Is(err, ErrNilClient) {
		t.Fatalf("New(nil) error = %v, want ErrNilClient", err)
	}
}

func TestChooseMapsValidChoice(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{payload: choicePayload()}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New unexpected error: %v", err)
	}

	choice, err := adapter.Choose(context.Background(), choiceRequest())
	if err != nil {
		t.Fatalf("Choose unexpected error: %v", err)
	}
	if choice.Selected != "search" {
		t.Fatalf("selected = %q, want %q", choice.Selected, "search")
	}
	if len(choice.Distribution) != 3 {
		t.Fatalf("distribution entries = %d, want 3", len(choice.Distribution))
	}
	if got := choice.Distribution[0]; got.Option != "search" || got.Probability != 0.7 {
		t.Fatalf("distribution[0] = %+v, want search/0.7", got)
	}
	if client.calls != 1 {
		t.Fatalf("client calls = %d, want 1", client.calls)
	}
}

func TestChooseInstructionCarriesRequest(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{payload: choicePayload()}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New unexpected error: %v", err)
	}
	if _, err := adapter.Choose(context.Background(), choiceRequest()); err != nil {
		t.Fatalf("Choose unexpected error: %v", err)
	}

	instruction := client.lastReq.Instruction
	for _, want := range []string{
		"Task: route the user request",
		"- search: retrieve passages",
		"- summarize: condense passages",
		"- translate\n", // bare option without description
		"- the corpus holds go internals notes",
		`"selected"`,
	} {
		if !strings.Contains(instruction, want) {
			t.Fatalf("instruction missing %q:\n%s", want, instruction)
		}
	}

	var schema map[string]any
	if err := json.Unmarshal(client.lastReq.Schema, &schema); err != nil {
		t.Fatalf("choice schema is not valid JSON: %v", err)
	}
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("choice schema has no properties object: %v", schema)
	}
	for _, required := range []string{"selected", "distribution"} {
		if _, ok := properties[required]; !ok {
			t.Fatalf("choice schema properties missing %q", required)
		}
	}
}

func TestScoreMapsValidScore(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{payload: []byte(`{"value": 72, "confidence": 0.62}`)}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New unexpected error: %v", err)
	}

	request := judgment.ScoreRequest{Task: "grade the summary", Target: "summary-7", Context: []string{"the summary omits the punchline"}}
	score, err := adapter.Score(context.Background(), request)
	if err != nil {
		t.Fatalf("Score unexpected error: %v", err)
	}
	if score.Value != 72 || score.Confidence != 0.62 {
		t.Fatalf("score = %+v, want 72/0.62", score)
	}

	instruction := client.lastReq.Instruction
	for _, want := range []string{"Task: grade the summary", "Target: summary-7", "- the summary omits the punchline"} {
		if !strings.Contains(instruction, want) {
			t.Fatalf("instruction missing %q:\n%s", want, instruction)
		}
	}
}

func TestAskMapsValidNoul(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{payload: []byte(`{"verdict": "yes", "confidence": 0.93}`)}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New unexpected error: %v", err)
	}

	request := judgment.NoulRequest{Question: "does go use a concurrent collector", Context: []string{"go gc runs concurrently"}}
	noul, err := adapter.Ask(context.Background(), request)
	if err != nil {
		t.Fatalf("Ask unexpected error: %v", err)
	}
	if noul.Verdict != judgment.VerdictYes || noul.Confidence != 0.93 {
		t.Fatalf("noul = %+v, want yes/0.93", noul)
	}
	if !strings.Contains(client.lastReq.Instruction, "Question: does go use a concurrent collector") {
		t.Fatalf("instruction missing question:\n%s", client.lastReq.Instruction)
	}

	var schema map[string]any
	if err := json.Unmarshal(client.lastReq.Schema, &schema); err != nil {
		t.Fatalf("noul schema is not valid JSON: %v", err)
	}
	properties := schema["properties"].(map[string]any)
	verdict := properties["verdict"].(map[string]any)
	enum, ok := verdict["enum"].([]any)
	if !ok || len(enum) != 2 {
		t.Fatalf("noul verdict schema enum = %v, want yes/no", verdict["enum"])
	}
}

func TestClientErrorPropagates(t *testing.T) {
	t.Parallel()

	cause := errors.New("gateway unreachable")
	client := &scriptedClient{err: cause}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New unexpected error: %v", err)
	}

	_, err = adapter.Ask(context.Background(), judgment.NoulRequest{Question: "any question"})
	if !errors.Is(err, ErrGenerate) || !errors.Is(err, cause) {
		t.Fatalf("Ask error = %v, want ErrGenerate wrapping cause", err)
	}
}

func TestInvalidJSONRejected(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{payload: []byte(`{"verdict": `)}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New unexpected error: %v", err)
	}

	_, err = adapter.Ask(context.Background(), judgment.NoulRequest{Question: "any question"})
	if !errors.Is(err, ErrInvalidResponse) {
		t.Fatalf("Ask error = %v, want ErrInvalidResponse", err)
	}
}

func TestOutOfRangeScoreRejected(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{payload: []byte(`{"value": 150, "confidence": 0.9}`)}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New unexpected error: %v", err)
	}

	_, err = adapter.Score(context.Background(), judgment.ScoreRequest{Task: "grade", Target: "x"})
	if !errors.Is(err, ErrInvalidResponse) {
		t.Fatalf("Score error = %v, want ErrInvalidResponse", err)
	}
}

func TestNonModalChoiceRejected(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{payload: []byte(`{
		"selected": "summarize",
		"distribution": [
			{"option": "search", "probability": 0.7},
			{"option": "summarize", "probability": 0.3}
		]
	}`)}
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New unexpected error: %v", err)
	}

	_, err = adapter.Choose(context.Background(), judgment.ChoiceRequest{
		Task:    "route",
		Options: []judgment.Option{{ID: "search"}, {ID: "summarize"}},
	})
	if !errors.Is(err, ErrInvalidResponse) {
		t.Fatalf("Choose error = %v, want ErrInvalidResponse", err)
	}
}

func TestAdapterSatisfiesJudge(t *testing.T) {
	t.Parallel()

	var judge judgment.Judge = mustAdapter(t, &scriptedClient{payload: choicePayload()})
	if judge == nil {
		t.Fatal("adapter is nil")
	}
}

func mustAdapter(t *testing.T, client Client) *Adapter {
	t.Helper()
	adapter, err := New(client)
	if err != nil {
		t.Fatalf("New unexpected error: %v", err)
	}
	return adapter
}
