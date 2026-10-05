// Package structjudge adapts an injected structured-output completion
// client to Reliquary's judgment contracts.
//
// Structured-output providers force a JSON Schema on their responses,
// which is exactly what the strict judgment primitives need: a Choice
// distribution that sums to one, a Score in [0, 100], a Noul verdict
// from a two-value enum. This adapter owns the per-primitive response
// schemas, renders each judgment request into one instruction, forwards
// it to the injected Client, maps the JSON response back, and validates
// the pair with the judgment package. Transport, credentials,
// endpoints, retries, and model choice belong entirely to the injected
// Client; the adapter performs no network, file, or environment I/O and
// retains no state beyond the client reference.
//
// The motivating client is a multi-provider LLM gateway (for example
// Wormhole's structured builder), wired with a few lines of caller
// glue:
//
//	type wormholeJudge struct{ w *wormhole.Wormhole }
//
//	func (g wormholeJudge) Generate(ctx context.Context, r structjudge.Request) ([]byte, error) {
//		resp, err := g.w.Structured().Model("…").Prompt(r.Instruction).Schema(r.Schema).Do(ctx)
//		// … extract the JSON payload from resp …
//	}
package structjudge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/dotcommander/reliquary/judgment"
)

// ErrNilClient reports construction with a missing client.
var ErrNilClient = errors.New("structjudge adapter: client must not be nil")

// ErrGenerate reports a client-side generation failure.
var ErrGenerate = errors.New("structjudge adapter: generate")

// ErrInvalidResponse reports a response that is not valid JSON for the
// primitive's schema or fails the judgment contract.
var ErrInvalidResponse = errors.New("structjudge adapter: invalid response")

// Request is one structured completion: the full instruction text and
// the JSON Schema the response must satisfy.
type Request struct {
	// Instruction is the complete prompt for the judgment.
	Instruction string
	// Schema is the JSON Schema the response must satisfy.
	Schema []byte
}

// Client generates one schema-forced JSON completion. Implementations
// own the provider, transport, credentials, and retry policy.
type Client interface {
	Generate(ctx context.Context, request Request) ([]byte, error)
}

// Adapter maps judgment primitives onto an injected Client.
type Adapter struct {
	client Client
}

// New constructs an Adapter over client.
func New(client Client) (*Adapter, error) {
	if client == nil {
		return nil, ErrNilClient
	}
	return &Adapter{client: client}, nil
}

var _ judgment.Judge = (*Adapter)(nil)

// Choice response schema: the selected option plus a probability
// distribution over the option set.
var choiceSchema = []byte(`{
	"type": "object",
	"properties": {
		"selected": {"type": "string"},
		"distribution": {
			"type": "array",
			"minItems": 1,
			"items": {
				"type": "object",
				"properties": {
					"option": {"type": "string"},
					"probability": {"type": "number", "minimum": 0, "maximum": 1}
				},
				"required": ["option", "probability"]
			}
		}
	},
	"required": ["selected", "distribution"]
}`)

// Score response schema: a 0-100 value plus calibrated confidence.
var scoreSchema = []byte(`{
	"type": "object",
	"properties": {
		"value": {"type": "number", "minimum": 0, "maximum": 100},
		"confidence": {"type": "number", "minimum": 0, "maximum": 1}
	},
	"required": ["value", "confidence"]
}`)

// Noul response schema: a strict yes/no verdict plus confidence.
var noulSchema = []byte(`{
	"type": "object",
	"properties": {
		"verdict": {"type": "string", "enum": ["yes", "no"]},
		"confidence": {"type": "number", "minimum": 0, "maximum": 1}
	},
	"required": ["verdict", "confidence"]
}`)

type wireChoice struct {
	Selected     string `json:"selected"`
	Distribution []struct {
		Option      string  `json:"option"`
		Probability float64 `json:"probability"`
	} `json:"distribution"`
}

type wireScore struct {
	Value      float64 `json:"value"`
	Confidence float64 `json:"confidence"`
}

type wireNoul struct {
	Verdict    string  `json:"verdict"`
	Confidence float64 `json:"confidence"`
}

// Choose renders the choice request into one instruction, generates a
// schema-forced response, and validates the mapped choice.
func (a *Adapter) Choose(ctx context.Context, request judgment.ChoiceRequest) (judgment.Choice, error) {
	if err := judgment.ValidateChoiceRequest(request); err != nil {
		return judgment.Choice{}, err
	}
	var payload wireChoice
	if err := a.generate(ctx, choiceInstruction(request), choiceSchema, &payload); err != nil {
		return judgment.Choice{}, err
	}
	result := judgment.Choice{Selected: payload.Selected}
	result.Distribution = make([]judgment.OptionProbability, len(payload.Distribution))
	for i, mass := range payload.Distribution {
		result.Distribution[i] = judgment.OptionProbability{
			Option:      mass.Option,
			Probability: judgment.Probability(mass.Probability),
		}
	}
	if err := judgment.ValidateChoice(request, result); err != nil {
		return judgment.Choice{}, fmt.Errorf("%w: choice: %w", ErrInvalidResponse, err)
	}
	return result, nil
}

// Score renders the score request into one instruction, generates a
// schema-forced response, and validates the mapped score.
func (a *Adapter) Score(ctx context.Context, request judgment.ScoreRequest) (judgment.Score, error) {
	if err := judgment.ValidateScoreRequest(request); err != nil {
		return judgment.Score{}, err
	}
	var payload wireScore
	if err := a.generate(ctx, scoreInstruction(request), scoreSchema, &payload); err != nil {
		return judgment.Score{}, err
	}
	result := judgment.Score{Value: payload.Value, Confidence: judgment.Probability(payload.Confidence)}
	if err := judgment.ValidateScore(request, result); err != nil {
		return judgment.Score{}, fmt.Errorf("%w: score: %w", ErrInvalidResponse, err)
	}
	return result, nil
}

// Ask renders the noul request into one instruction, generates a
// schema-forced response, and validates the mapped verdict.
func (a *Adapter) Ask(ctx context.Context, request judgment.NoulRequest) (judgment.Noul, error) {
	if err := judgment.ValidateNoulRequest(request); err != nil {
		return judgment.Noul{}, err
	}
	var payload wireNoul
	if err := a.generate(ctx, noulInstruction(request), noulSchema, &payload); err != nil {
		return judgment.Noul{}, err
	}
	result := judgment.Noul{Verdict: judgment.Verdict(payload.Verdict), Confidence: judgment.Probability(payload.Confidence)}
	if err := judgment.ValidateNoul(request, result); err != nil {
		return judgment.Noul{}, fmt.Errorf("%w: noul: %w", ErrInvalidResponse, err)
	}
	return result, nil
}

// generate forwards one instruction plus schema to the client and
// decodes the JSON response into out.
func (a *Adapter) generate(ctx context.Context, instruction string, schema []byte, out any) error {
	raw, err := a.client.Generate(ctx, Request{Instruction: instruction, Schema: schema})
	if err != nil {
		return fmt.Errorf("%w: %w", ErrGenerate, err)
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("%w: decode: %w", ErrInvalidResponse, err)
	}
	return nil
}

// choiceInstruction renders the task, option set, and context with the
// response contract.
func choiceInstruction(request judgment.ChoiceRequest) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Task: %s\n", request.Task)
	b.WriteString("Options:\n")
	for _, option := range request.Options {
		if option.Description == "" {
			fmt.Fprintf(&b, "- %s\n", option.ID)
		} else {
			fmt.Fprintf(&b, "- %s: %s\n", option.ID, option.Description)
		}
	}
	writeContext(&b, request.Context)
	b.WriteString("Pick exactly one option. Respond as JSON with \"selected\" set to the picked option's ID and \"distribution\" holding the probability of every listed option, summing to 1. The selected option must carry the highest probability.\n")
	return b.String()
}

// scoreInstruction renders the task, target, and context with the
// response contract.
func scoreInstruction(request judgment.ScoreRequest) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Task: %s\nTarget: %s\n", request.Task, request.Target)
	writeContext(&b, request.Context)
	b.WriteString("Rate the target from 0 to 100. Respond as JSON with \"value\" in [0, 100] and \"confidence\" in [0, 1] holding your calibrated confidence in that value.\n")
	return b.String()
}

// noulInstruction renders the question and context with the response
// contract.
func noulInstruction(request judgment.NoulRequest) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Question: %s\n", request.Question)
	writeContext(&b, request.Context)
	b.WriteString("Answer strictly yes or no. Respond as JSON with \"verdict\" set to \"yes\" or \"no\" and \"confidence\" in [0, 1] holding your calibrated confidence in the verdict.\n")
	return b.String()
}

func writeContext(b *strings.Builder, context []string) {
	if len(context) == 0 {
		return
	}
	b.WriteString("Context:\n")
	for _, fact := range context {
		fmt.Fprintf(b, "- %s\n", fact)
	}
}
