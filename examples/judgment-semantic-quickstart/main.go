// Command judgment-semantic-quickstart wires semantic judgment memory
// over a local OpenAI-compatible embedding server.
//
// Tier 0: a quantized local embedding model (default
// nomicai-modernbert-embed-base-4bit) behind an OpenAI-compatible
// /v1/embeddings endpoint embeds every judgment input. Tier 1: a reflex
// Jev judge answers the typed primitives. Tier 2: below-threshold
// reflex confidence escalates to a deliberative Jev judge. The
// judgment/memory SemanticStore caches served judgments keyed by their
// input embedding, so a later paraphrase of the same question is
// answered from memory — same judgment, zero provider calls — as long
// as its embedding stays within the similarity threshold.
//
// The judgment.Memory router contract carries no context; this example
// owns one context from run() and reuses it for the store's embed
// calls. Both Jev providers are deterministic scripted stand-ins; real
// deployments inject a jev.Client that owns transport and credentials.
//
// The caller owns the embedding server process. Start it first
// (defaults target http://localhost:8000/v1), then run from the repo
// root:
//
//	GOWORK=off go run ./examples/judgment-semantic-quickstart
//
// Override with EMBED_SERVER_URL, EMBED_SERVER_API_KEY, EMBED_MODEL,
// EMBED_DIM, and EMBED_THRESHOLD.
package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/dotcommander/reliquary/adapter/jev"
	"github.com/dotcommander/reliquary/adapter/openai"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
	"github.com/dotcommander/reliquary/judgment"
	"github.com/dotcommander/reliquary/judgment/memory"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func main() {
	examplekit.Run(run)
}

func run(ctx context.Context) error {
	baseURL := getenv("EMBED_SERVER_URL", "http://localhost:8000/v1")
	apiKey := getenv("EMBED_SERVER_API_KEY", "local")
	model := getenv("EMBED_MODEL", "nomicai-modernbert-embed-base-4bit")
	dims := getenvInt("EMBED_DIM", 768)
	threshold := getenvFloat("EMBED_THRESHOLD", 0.85)

	client := openaisdk.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	)
	embedder, err := openai.New(client, openai.Config{
		Model:          model,
		Dimensions:     dims,
		DocumentPrefix: "search_document: ",
		QueryPrefix:    "search_query: ",
	})
	if err != nil {
		return err
	}

	// Semantic judgment memory: exact fingerprint hits stay the fast
	// path; a paraphrase whose embedding lands within the threshold of a
	// stored input reuses that judgment without any provider call.
	store, err := memory.NewSemantic(
		memory.SemanticConfig{
			Embedder:  embedder,
			Model:     embedding.ModelRef{Provider: "openai", Name: model, Dim: dims},
			Threshold: threshold,
		},
		memory.WithTTL(30*time.Minute),
		memory.WithCapacity(1024),
		memory.WithDisableTouchOnHit(),
	)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	// Tier 1 reflex and Tier 2 deliberative Jev judges over scripted
	// clients. The reflex verdict's 0.58 confidence sits below the 0.70
	// escalation threshold, so first-time questions escalate.
	reflexClient := &scriptedJev{
		noul: jev.Result{Verdict: "yes", Confidence: 0.58},
	}
	deliberateClient := &scriptedJev{
		noul: jev.Result{Verdict: "yes", Confidence: 0.93},
	}
	reflex, err := jev.New(reflexClient)
	if err != nil {
		return err
	}
	deliberate, err := jev.New(deliberateClient)
	if err != nil {
		return err
	}

	mem := &semanticMemory{store: store, ctx: ctx}
	router, err := judgment.NewRouter(reflex, deliberate, mem, judgment.Thresholds{
		Choice: 0.70,
		Score:  0.70,
		Noul:   0.70,
	})
	if err != nil {
		return err
	}

	scope := judgment.Scope{Tenant: "demo", Schema: "judgment-semantic-quickstart/v1"}

	fmt.Printf("Server: %s (%s, %d dims, threshold %.2f)\n\n", baseURL, model, dims, threshold)

	// Ask 1 — miss: the reflex answers below threshold, the deliberative
	// judge takes over, and the served verdict is cached with its input
	// embedding. The first embed call happens here, so surface the
	// server hint on failure.
	original := judgment.NoulRequest{
		Question: "does Go reclaim memory with a concurrent collector",
		Context:  []string{"Go uses a concurrent garbage collector to reclaim unreachable memory while keeping pauses short."},
	}
	routed, err := router.Ask(ctx, scope, original)
	if err != nil {
		return fmt.Errorf("ask 1 (is the embedding server running at %s?): %w", baseURL, err)
	}
	report(1, "original", routed, mem)

	// Ask 2 — paraphrase: a different fingerprint (so no exact hit), but
	// its embedding lands within the threshold of ask 1's stored input,
	// so the cached verdict is reused with zero provider calls.
	paraphrase := judgment.NoulRequest{
		Question: "is Go memory reclaimed by a collector that runs concurrently",
		Context:  []string{"Go uses a concurrent garbage collector to reclaim unreachable memory while keeping pauses short."},
	}
	routed, err = router.Ask(ctx, scope, paraphrase)
	if err != nil {
		return fmt.Errorf("ask 2: %w", err)
	}
	report(2, "paraphrase", routed, mem)

	// Ask 3 — unrelated: neither an exact nor a semantic hit, so the
	// reflex (and then the deliberative) judge is consulted again.
	unrelated := judgment.NoulRequest{
		Question: "does fresh pasta dough rest before rolling",
		Context:  []string{"Fresh pasta dough rests before rolling so tagliatelle holds sauce and keeps its bite."},
	}
	routed, err = router.Ask(ctx, scope, unrelated)
	if err != nil {
		return fmt.Errorf("ask 3: %w", err)
	}
	report(3, "unrelated", routed, mem)

	metrics := store.Metrics()
	fmt.Printf("\nStore metrics: %d insertion(s), %d semantic hit(s), %d miss(es), %d exact hit(s)\n",
		metrics.Insertions, metrics.SemanticHits, metrics.Misses, metrics.ExactHits)
	fmt.Printf("Provider calls: reflex %d, deliberative %d\n", reflexClient.calls, deliberateClient.calls)
	fmt.Println("The paraphrase was answered from semantic memory: same judgment, zero provider cost.")
	return nil
}

func report(step int, label string, routed judgment.Routed[judgment.Noul], mem *semanticMemory) {
	switch {
	case mem.semantic:
		fmt.Printf("Ask %d (%s): route=%s verdict=%s confidence=%.2f [semantic hit, similarity %.3f]\n",
			step, label, routed.Route, routed.Judgment.Verdict, routed.Judgment.Confidence, mem.similarity)
	default:
		fmt.Printf("Ask %d (%s): route=%s verdict=%s confidence=%.2f\n",
			step, label, routed.Route, routed.Judgment.Verdict, routed.Judgment.Confidence)
	}
}

// semanticMemory adapts the SemanticStore to the router's exact-hit
// judgment.Memory contract. The store's lookup embeds the request's
// natural text, which the fingerprint alone cannot reconstruct, so each
// method derives that text from the request and remembers the most
// recent hit for the demo's reporting.
type semanticMemory struct {
	store *memory.SemanticStore
	ctx   context.Context

	semantic   bool
	similarity float32
}

var _ judgment.Memory = (*semanticMemory)(nil)

func (m *semanticMemory) GetNoul(scope judgment.Scope, request judgment.NoulRequest) (judgment.Noul, bool) {
	m.semantic = false
	key := memory.Key{
		Tenant:      scope.Tenant,
		Schema:      scope.Schema,
		Primitive:   memory.PrimitiveNoul,
		Fingerprint: memory.FingerprintNoul(request),
	}
	noul, hit, err := m.store.GetNoul(m.ctx, key, noulText(request))
	if err != nil || hit.Kind == memory.HitNone {
		return judgment.Noul{}, false
	}
	m.semantic = hit.Kind == memory.HitSemantic
	m.similarity = hit.Similarity
	return noul, true
}

func (m *semanticMemory) PutNoul(scope judgment.Scope, request judgment.NoulRequest, result judgment.Noul) error {
	key := memory.Key{
		Tenant:      scope.Tenant,
		Schema:      scope.Schema,
		Primitive:   memory.PrimitiveNoul,
		Fingerprint: memory.FingerprintNoul(request),
	}
	return m.store.PutNoul(m.ctx, key, noulText(request), result)
}

func (m *semanticMemory) GetChoice(scope judgment.Scope, request judgment.ChoiceRequest) (judgment.Choice, bool) {
	m.semantic = false
	key := memory.Key{
		Tenant:      scope.Tenant,
		Schema:      scope.Schema,
		Primitive:   memory.PrimitiveChoice,
		Fingerprint: memory.FingerprintChoice(request),
	}
	choice, hit, err := m.store.GetChoice(m.ctx, key, choiceText(request))
	if err != nil || hit.Kind == memory.HitNone {
		return judgment.Choice{}, false
	}
	m.semantic = hit.Kind == memory.HitSemantic
	m.similarity = hit.Similarity
	return choice, true
}

func (m *semanticMemory) PutChoice(scope judgment.Scope, request judgment.ChoiceRequest, result judgment.Choice) error {
	key := memory.Key{
		Tenant:      scope.Tenant,
		Schema:      scope.Schema,
		Primitive:   memory.PrimitiveChoice,
		Fingerprint: memory.FingerprintChoice(request),
	}
	return m.store.PutChoice(m.ctx, key, choiceText(request), result)
}

func (m *semanticMemory) GetScore(scope judgment.Scope, request judgment.ScoreRequest) (judgment.Score, bool) {
	m.semantic = false
	key := memory.Key{
		Tenant:      scope.Tenant,
		Schema:      scope.Schema,
		Primitive:   memory.PrimitiveScore,
		Fingerprint: memory.FingerprintScore(request),
	}
	score, hit, err := m.store.GetScore(m.ctx, key, scoreText(request))
	if err != nil || hit.Kind == memory.HitNone {
		return judgment.Score{}, false
	}
	m.semantic = hit.Kind == memory.HitSemantic
	m.similarity = hit.Similarity
	return score, true
}

func (m *semanticMemory) PutScore(scope judgment.Scope, request judgment.ScoreRequest, result judgment.Score) error {
	key := memory.Key{
		Tenant:      scope.Tenant,
		Schema:      scope.Schema,
		Primitive:   memory.PrimitiveScore,
		Fingerprint: memory.FingerprintScore(request),
	}
	return m.store.PutScore(m.ctx, key, scoreText(request), result)
}

func noulText(request judgment.NoulRequest) string {
	return strings.TrimSpace(strings.Join(append([]string{request.Question}, request.Context...), "\n"))
}

func choiceText(request judgment.ChoiceRequest) string {
	parts := make([]string, 0, 2+len(request.Options)+len(request.Context))
	parts = append(parts, request.Task)
	for _, option := range request.Options {
		parts = append(parts, option.ID+": "+option.Description)
	}
	parts = append(parts, request.Context...)
	return strings.Join(parts, "\n")
}

func scoreText(request judgment.ScoreRequest) string {
	return strings.TrimSpace(strings.Join(append([]string{request.Task, request.Target}, request.Context...), "\n"))
}

// scriptedJev is a deterministic stand-in for a Jev evaluation runtime:
// it answers every typed call from fixed scripted results and counts calls.
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

func getenv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func getenvInt(key string, fallback int) int {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil {
			return parsed
		}
	}
	return fallback
}

func getenvFloat(key string, fallback float64) float64 {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.ParseFloat(value, 64); err == nil {
			return parsed
		}
	}
	return fallback
}
