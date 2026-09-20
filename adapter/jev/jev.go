// Package jev adapts an injected Jev evaluation client to Reliquary's
// judgment contracts.
//
// Jev executes typed evaluations that return strict probabilities and enums
// instead of generated prose. This adapter owns only the mapping between the
// provider-neutral judgment types and the Jev call/result shapes: it
// translates each judgment request into one typed Call, forwards it to the
// injected Client, maps the provider-shaped Result back, and validates the
// pair with the judgment package. Transport, credentials, endpoints, retry
// policy, and persistent state belong entirely to the injected Client; the
// adapter performs no network, file, or environment I/O and retains no state
// beyond the client reference.
package jev

import (
	"context"
	"errors"
	"fmt"

	"github.com/dotcommander/reliquary/judgment"
)

// ErrNilClient reports construction with a missing Jev client.
var ErrNilClient = errors.New("jev adapter: client must not be nil")

// Kind identifies the Jev primitive a Call requests.
type Kind string

const (
	// KindChoice requests a definitive pick plus a probability distribution
	// over the call's option set.
	KindChoice Kind = "choice"
	// KindScore requests a 0-100 rating for the call's target.
	KindScore Kind = "score"
	// KindNoul requests one strict yes/no verdict for the call's question.
	KindNoul Kind = "noul"
)

// Option is one enumerable choice target in a Jev call.
type Option struct {
	ID          string
	Description string
}

// Mass is the probability mass one choice option received from Jev.
type Mass struct {
	Option      string
	Probability float64
}

// Call is one typed Jev evaluation request. Kind selects the primitive and
// which fields Jev reads: choice uses Task, Context, and Options; score uses
// Task, Context, and Target; noul uses Question and Context.
type Call struct {
	Kind     Kind
	Task     string
	Context  []string
	Options  []Option
	Target   string
	Question string
}

// Result is one typed Jev evaluation response. Choice fills Selected and
// Distribution; score fills Value and Confidence; noul fills Verdict and
// Confidence.
type Result struct {
	Selected     string
	Distribution []Mass
	Value        float64
	Verdict      string
	Confidence   float64
}

// Client is the Jev provider boundary this adapter wraps. Implementations
// own transport, credentials, and retry policy; they perform exactly one
// typed evaluation per call so the adapter stays free of provider state.
type Client interface {
	Evaluate(ctx context.Context, call Call) (Result, error)
}

// Adapter maps an injected Client onto Reliquary's judgment contracts. The
// zero value is unusable; construct it with New.
type Adapter struct {
	client Client
}

var (
	_ judgment.Chooser = (*Adapter)(nil)
	_ judgment.Scorer  = (*Adapter)(nil)
	_ judgment.Asker   = (*Adapter)(nil)
	_ judgment.Judge   = (*Adapter)(nil)
)

// New constructs the adapter over an injected Jev client. It performs no
// network, file, or environment I/O and stores nothing beyond the client.
func New(client Client) (*Adapter, error) {
	if client == nil {
		return nil, ErrNilClient
	}
	return &Adapter{client: client}, nil
}

// Choose routes a choice request through the injected client, maps the
// provider result onto a judgment Choice, and validates the pair.
func (a *Adapter) Choose(ctx context.Context, request judgment.ChoiceRequest) (judgment.Choice, error) {
	call := Call{
		Kind:    KindChoice,
		Task:    request.Task,
		Context: cloneContext(request.Context),
		Options: cloneOptions(request.Options),
	}
	result, err := a.client.Evaluate(ctx, call)
	if err != nil {
		return judgment.Choice{}, fmt.Errorf("jev adapter: choose: %w", err)
	}
	choice := judgment.Choice{Selected: result.Selected}
	if result.Distribution != nil {
		choice.Distribution = make([]judgment.OptionProbability, len(result.Distribution))
		for i, entry := range result.Distribution {
			choice.Distribution[i] = judgment.OptionProbability{
				Option:      entry.Option,
				Probability: judgment.Probability(entry.Probability),
			}
		}
	}
	if err := judgment.ValidateChoice(request, choice); err != nil {
		return judgment.Choice{}, fmt.Errorf("jev adapter: %w", err)
	}
	return choice, nil
}

// Score routes a score request through the injected client, maps the
// provider result onto a judgment Score, and validates the pair.
func (a *Adapter) Score(ctx context.Context, request judgment.ScoreRequest) (judgment.Score, error) {
	call := Call{
		Kind:    KindScore,
		Task:    request.Task,
		Context: cloneContext(request.Context),
		Target:  request.Target,
	}
	result, err := a.client.Evaluate(ctx, call)
	if err != nil {
		return judgment.Score{}, fmt.Errorf("jev adapter: score: %w", err)
	}
	score := judgment.Score{
		Value:      result.Value,
		Confidence: judgment.Probability(result.Confidence),
	}
	if err := judgment.ValidateScore(request, score); err != nil {
		return judgment.Score{}, fmt.Errorf("jev adapter: %w", err)
	}
	return score, nil
}

// Ask routes a strict yes/no question through the injected client, maps the
// provider result onto a judgment Noul, and validates the pair.
func (a *Adapter) Ask(ctx context.Context, request judgment.NoulRequest) (judgment.Noul, error) {
	call := Call{
		Kind:     KindNoul,
		Context:  cloneContext(request.Context),
		Question: request.Question,
	}
	result, err := a.client.Evaluate(ctx, call)
	if err != nil {
		return judgment.Noul{}, fmt.Errorf("jev adapter: ask: %w", err)
	}
	noul := judgment.Noul{
		Verdict:    judgment.Verdict(result.Verdict),
		Confidence: judgment.Probability(result.Confidence),
	}
	if err := judgment.ValidateNoul(request, noul); err != nil {
		return judgment.Noul{}, fmt.Errorf("jev adapter: %w", err)
	}
	return noul, nil
}

func cloneContext(lines []string) []string {
	if lines == nil {
		return nil
	}
	cloned := make([]string, len(lines))
	copy(cloned, lines)
	return cloned
}

func cloneOptions(options []judgment.Option) []Option {
	if options == nil {
		return nil
	}
	cloned := make([]Option, len(options))
	for i, option := range options {
		cloned[i] = Option{ID: option.ID, Description: option.Description}
	}
	return cloned
}
