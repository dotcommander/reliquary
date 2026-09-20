// Command judgment-quickstart wires the Embed→Judge stack end to end.
//
// Tier 0: the reliquary facade embeds a small knowledge corpus and
// retrieves the passages every judgment is based on, and the
// judgment/memory store caches each served judgment for exact-hit reuse.
// Tier 1: a fast reflex Jev judge answers the typed evaluation
// primitives. Tier 2: a deliberative Jev judge takes over whenever the
// reflex confidence falls below the calibrated threshold.
//
// The repeated request at the end resolves as an exact-hit memory answer
// with zero provider calls. Both Jev providers are deterministic scripted
// stand-ins; real deployments inject a jev.Client that owns transport and
// credentials.
//
// Run from the repo root:
//
//	GOWORK=off go run ./examples/judgment-quickstart
package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/adapter/jev"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
	"github.com/dotcommander/reliquary/judgment"
	"github.com/dotcommander/reliquary/judgment/memory"
)

func main() {
	examplekit.Run(run)
}

func run(ctx context.Context) error {
	// Tier 0: embed the knowledge corpus once, retrieve context per
	// request.
	app := reliquary.Quickstart()
	if _, err := app.Ingest(ctx, corpus()...); err != nil {
		return err
	}

	query := "how does Go reclaim memory"
	found, err := app.Search(ctx, query, reliquary.TopK(2))
	if err != nil {
		return err
	}
	facts := make([]string, 0, len(found))
	for _, result := range found {
		facts = append(facts, strings.TrimSpace(result.Content))
	}
	fmt.Printf("Query: %s\n", query)
	for _, fact := range facts {
		fmt.Printf("- context: %s\n", fact)
	}

	// Judgment memory: the shared exact-hit cache for served judgments.
	store, err := memory.New(
		memory.WithTTL(30*time.Minute),
		memory.WithCapacity(1024),
		memory.WithDisableTouchOnHit(),
	)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	// Tier 1 reflex and Tier 2 deliberative Jev judges over injected
	// clients.
	reflexClient := &scriptedJev{
		choice: jev.Result{
			Selected:     "search",
			Distribution: []jev.Mass{
				{Option: "search", Probability: 0.55},
				{Option: "summarize", Probability: 0.27},
				{Option: "translate", Probability: 0.18},
			},
		},
		score: jev.Result{Value: 72, Confidence: 0.62},
		noul:  jev.Result{Verdict: "yes", Confidence: 0.61},
	}
	deliberateClient := &scriptedJev{
		choice: jev.Result{
			Selected:     "search",
			Distribution: []jev.Mass{
				{Option: "search", Probability: 0.92},
				{Option: "summarize", Probability: 0.05},
				{Option: "translate", Probability: 0.03},
			},
		},
		score: jev.Result{Value: 91.5, Confidence: 0.95},
		noul:  jev.Result{Verdict: "no", Confidence: 0.93},
	}
	reflex, err := jev.New(reflexClient)
	if err != nil {
		return err
	}
	deliberate, err := jev.New(deliberateClient)
	if err != nil {
		return err
	}

	router, err := judgment.NewRouter(reflex, deliberate, &scopeMemory{store: store}, judgment.Thresholds{
		Choice: 0.8,
		Score:  0.8,
		Noul:   0.8,
	})
	if err != nil {
		return err
	}
	scope := judgment.Scope{Tenant: "demo", Schema: "router.v1"}

	// Choice: route the question over an enumerable option set. The reflex
	// confidence 0.55 falls below the 0.8 threshold, so the deliberative
	// judge answers instead.
	choiceRequest := judgment.ChoiceRequest{
		Task:    "route the developer question",
		Context: facts,
		Options: []judgment.Option{
			{ID: "search", Description: "answer from the retrieved knowledge"},
			{ID: "summarize", Description: "condense the retrieved knowledge"},
			{ID: "translate", Description: "translate the retrieved knowledge"},
		},
	}
	choice, err := router.Choose(ctx, scope, choiceRequest)
	if err != nil {
		return err
	}
	fmt.Printf("Choice: %s (route %s, confidence %.3f)\n", choice.Judgment.Selected, choice.Route, choice.Confidence)
	for _, entry := range choice.Judgment.Distribution {
		fmt.Printf("  %s: %.3f\n", entry.Option, entry.Probability)
	}

	// Score: rank the top retrieved passage. Again the reflex confidence
	// escalates to the deliberative judge.
	score, err := router.Score(ctx, scope, judgment.ScoreRequest{
		Task:    "rank the top retrieved passage",
		Target:  found[0].ID,
		Context: facts,
	})
	if err != nil {
		return err
	}
	fmt.Printf("Score: %.1f (route %s, confidence %.3f)\n", score.Judgment.Value, score.Route, score.Confidence)

	// Noul: a strict gate. Deliberation flips the reflex verdict.
	gate, err := router.Ask(ctx, scope, judgment.NoulRequest{
		Question: "does the top retrieved passage answer the question?",
		Context:  facts,
	})
	if err != nil {
		return err
	}
	fmt.Printf("Gate: %s (route %s, confidence %.3f)\n", gate.Judgment.Verdict, gate.Route, gate.Confidence)

	// The identical choice request is now an exact-hit memory answer with
	// zero provider calls.
	repeat, err := router.Choose(ctx, scope, choiceRequest)
	if err != nil {
		return err
	}
	fmt.Printf("Repeat choice: %s (route %s, confidence %.3f)\n", repeat.Judgment.Selected, repeat.Route, repeat.Confidence)

	metrics := store.Metrics()
	fmt.Printf("Judgment memory: %d hits, %d misses\n", metrics.Hits, metrics.Misses)
	fmt.Printf("Provider calls: reflex %d, deliberate %d\n", reflexClient.calls, deliberateClient.calls)
	return nil
}

// scopeMemory adapts the judgment/memory Store to judgment.Router's
// judgment.Memory seam: every lookup derives its store key from the
// routing scope plus a deterministic fingerprint of the complete request,
// so only identical tenant, schema, primitive, and request content hit.
type scopeMemory struct {
	store *memory.Store
}

var _ judgment.Memory = (*scopeMemory)(nil)

func (m *scopeMemory) GetChoice(scope judgment.Scope, request judgment.ChoiceRequest) (judgment.Choice, bool) {
	key := memory.Key{
		Tenant:      scope.Tenant,
		Schema:      scope.Schema,
		Primitive:   memory.PrimitiveChoice,
		Fingerprint: memory.FingerprintChoice(request),
	}
	return m.store.GetChoice(key)
}

func (m *scopeMemory) PutChoice(scope judgment.Scope, request judgment.ChoiceRequest, result judgment.Choice) error {
	key := memory.Key{
		Tenant:      scope.Tenant,
		Schema:      scope.Schema,
		Primitive:   memory.PrimitiveChoice,
		Fingerprint: memory.FingerprintChoice(request),
	}
	return m.store.PutChoice(key, result)
}

func (m *scopeMemory) GetScore(scope judgment.Scope, request judgment.ScoreRequest) (judgment.Score, bool) {
	key := memory.Key{
		Tenant:      scope.Tenant,
		Schema:      scope.Schema,
		Primitive:   memory.PrimitiveScore,
		Fingerprint: memory.FingerprintScore(request),
	}
	return m.store.GetScore(key)
}

func (m *scopeMemory) PutScore(scope judgment.Scope, request judgment.ScoreRequest, result judgment.Score) error {
	key := memory.Key{
		Tenant:      scope.Tenant,
		Schema:      scope.Schema,
		Primitive:   memory.PrimitiveScore,
		Fingerprint: memory.FingerprintScore(request),
	}
	return m.store.PutScore(key, result)
}

func (m *scopeMemory) GetNoul(scope judgment.Scope, request judgment.NoulRequest) (judgment.Noul, bool) {
	key := memory.Key{
		Tenant:      scope.Tenant,
		Schema:      scope.Schema,
		Primitive:   memory.PrimitiveNoul,
		Fingerprint: memory.FingerprintNoul(request),
	}
	return m.store.GetNoul(key)
}

func (m *scopeMemory) PutNoul(scope judgment.Scope, request judgment.NoulRequest, result judgment.Noul) error {
	key := memory.Key{
		Tenant:      scope.Tenant,
		Schema:      scope.Schema,
		Primitive:   memory.PrimitiveNoul,
		Fingerprint: memory.FingerprintNoul(request),
	}
	return m.store.PutNoul(key, result)
}

// scriptedJev is a deterministic stand-in for a Jev evaluation runtime: it
// answers every typed call from fixed scripted results and counts calls.
type scriptedJev struct {
	choice jev.Result
	score  jev.Result
	noul   jev.Result

	calls int
}

var _ jev.Client = (*scriptedJev)(nil)

func (s *scriptedJev) Evaluate(_ context.Context, call jev.Call) (jev.Result, error) {
	s.calls++
	switch call.Kind {
	case jev.KindChoice:
		return s.choice, nil
	case jev.KindScore:
		return s.score, nil
	case jev.KindNoul:
		return s.noul, nil
	default:
		return jev.Result{}, fmt.Errorf("scripted jev: unknown kind %q", call.Kind)
	}
}

func corpus() []document.Document {
	return []document.Document{
		{
			ID:     "go-gc",
			Title:  "go-garbage-collection.md",
			Format: document.FormatMarkdown,
			Text:   "Go uses a concurrent garbage collector to reclaim unreachable memory while keeping pauses short.",
		},
		{
			ID:     "pasta",
			Title:  "fresh-pasta.md",
			Format: document.FormatMarkdown,
			Text:   "Fresh pasta dough rests before rolling so tagliatelle holds sauce and keeps its bite.",
		},
		{
			ID:     "stars",
			Title:  "neutron-stars.md",
			Format: document.FormatMarkdown,
			Text:   "Neutron stars form from collapsed stellar cores and pack enormous mass into a tiny radius.",
		},
	}
}
