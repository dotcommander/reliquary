package judgment

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
)

// recordingJudge is a deterministic Judge that counts calls per primitive
// and returns scripted results.
type recordingJudge struct {
	choice Choice
	score  Score
	noul   Noul
	err    error

	chooseCalls int
	scoreCalls  int
	askCalls    int
}

func (j *recordingJudge) Choose(_ context.Context, _ ChoiceRequest) (Choice, error) {
	j.chooseCalls++
	if j.err != nil {
		return Choice{}, j.err
	}
	return j.choice, nil
}

func (j *recordingJudge) Score(_ context.Context, _ ScoreRequest) (Score, error) {
	j.scoreCalls++
	if j.err != nil {
		return Score{}, j.err
	}
	return j.score, nil
}

func (j *recordingJudge) Ask(_ context.Context, _ NoulRequest) (Noul, error) {
	j.askCalls++
	if j.err != nil {
		return Noul{}, j.err
	}
	return j.noul, nil
}

var _ Judge = (*recordingJudge)(nil)

// modalChoice builds a choice over validChoiceRequest's option set that
// validates whenever mass is at least one third: the selected option
// "search" carries mass and the remainder splits evenly.
func modalChoice(mass Probability) Choice {
	rest := Probability((1 - float64(mass)) / 2)
	return Choice{
		Selected: "search",
		Distribution: distribution(
			OptionProbability{Option: "search", Probability: mass},
			OptionProbability{Option: "summarize", Probability: rest},
			OptionProbability{Option: "translate", Probability: rest},
		),
	}
}

func scoreFixture() ScoreRequest { return ScoreRequest{Task: "rank", Target: "doc-1"} }

func noulFixture() NoulRequest { return NoulRequest{Question: "is the build green?"} }

func escalationScope() Scope { return Scope{Tenant: "acme", Schema: "v1"} }

// mapMemory is an exact-hit Memory keyed by scope plus full request
// content, mirroring the memory package's key semantics for routing tests.
type mapMemory struct {
	choices map[string]Choice
	scores  map[string]Score
	nouls   map[string]Noul

	gets   int
	puts   int
	putErr error
}

func newMapMemory() *mapMemory {
	return &mapMemory{
		choices: map[string]Choice{},
		scores:  map[string]Score{},
		nouls:   map[string]Noul{},
	}
}

func choiceMemoryKey(scope Scope, request ChoiceRequest) string {
	ids := make([]string, len(request.Options))
	for i, option := range request.Options {
		ids[i] = option.ID
	}
	parts := append([]string{scope.Tenant, scope.Schema, request.Task, strings.Join(ids, ",")}, request.Context...)
	return strings.Join(parts, "\x1f")
}

func scoreMemoryKey(scope Scope, request ScoreRequest) string {
	parts := append([]string{scope.Tenant, scope.Schema, request.Task, request.Target}, request.Context...)
	return strings.Join(parts, "\x1f")
}

func noulMemoryKey(scope Scope, request NoulRequest) string {
	parts := append([]string{scope.Tenant, scope.Schema, request.Question}, request.Context...)
	return strings.Join(parts, "\x1f")
}

func (m *mapMemory) GetChoice(scope Scope, request ChoiceRequest) (Choice, bool) {
	m.gets++
	result, ok := m.choices[choiceMemoryKey(scope, request)]
	return result, ok
}

func (m *mapMemory) PutChoice(scope Scope, request ChoiceRequest, result Choice) error {
	m.puts++
	if m.putErr != nil {
		return m.putErr
	}
	m.choices[choiceMemoryKey(scope, request)] = result
	return nil
}

func (m *mapMemory) GetScore(scope Scope, request ScoreRequest) (Score, bool) {
	m.gets++
	result, ok := m.scores[scoreMemoryKey(scope, request)]
	return result, ok
}

func (m *mapMemory) PutScore(scope Scope, request ScoreRequest, result Score) error {
	m.puts++
	if m.putErr != nil {
		return m.putErr
	}
	m.scores[scoreMemoryKey(scope, request)] = result
	return nil
}

func (m *mapMemory) GetNoul(scope Scope, request NoulRequest) (Noul, bool) {
	m.gets++
	result, ok := m.nouls[noulMemoryKey(scope, request)]
	return result, ok
}

func (m *mapMemory) PutNoul(scope Scope, request NoulRequest, result Noul) error {
	m.puts++
	if m.putErr != nil {
		return m.putErr
	}
	m.nouls[noulMemoryKey(scope, request)] = result
	return nil
}

var _ Memory = (*mapMemory)(nil)

func TestThresholdsValidate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		thresholds Thresholds
		wantErr    error
	}{
		{
			name:       "calibrated per primitive",
			thresholds: Thresholds{Choice: 0.9, Score: 0.5, Noul: 0.3},
		},
		{
			name:       "zero disables escalation",
			thresholds: Thresholds{},
		},
		{
			name:       "one escalates short of certainty",
			thresholds: Thresholds{Choice: 1, Score: 1, Noul: 1},
		},
		{
			name:       "choice threshold above one",
			thresholds: Thresholds{Choice: 1.1, Score: 0.5, Noul: 0.5},
			wantErr:    ErrInvalidPolicy,
		},
		{
			name:       "score threshold negative",
			thresholds: Thresholds{Choice: 0.5, Score: -0.1, Noul: 0.5},
			wantErr:    ErrInvalidPolicy,
		},
		{
			name:       "noul threshold NaN",
			thresholds: Thresholds{Choice: 0.5, Score: 0.5, Noul: Probability(math.NaN())},
			wantErr:    ErrInvalidPolicy,
		},
		{
			name:       "noul threshold infinite",
			thresholds: Thresholds{Choice: 0.5, Score: 0.5, Noul: Probability(math.Inf(1))},
			wantErr:    ErrInvalidPolicy,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			err := test.thresholds.Validate()
			if test.wantErr == nil {
				if err != nil {
					t.Fatalf("Thresholds%+v.Validate() unexpected error: %v", test.thresholds, err)
				}
				return
			}
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("Thresholds%+v.Validate() error = %v, want %v", test.thresholds, err, test.wantErr)
			}
		})
	}
}

func TestNewRouterRejectsInvalidPolicy(t *testing.T) {
	t.Parallel()

	valid := Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8}
	newRouter := func() (*Router, error) {
		return NewRouter(&recordingJudge{}, &recordingJudge{}, newMapMemory(), valid)
	}

	router, err := newRouter()
	if err != nil {
		t.Fatalf("NewRouter unexpected error: %v", err)
	}
	if router == nil {
		t.Fatal("NewRouter returned nil router without error")
	}
	if got := router.Thresholds(); got != valid {
		t.Errorf("Router.Thresholds() = %+v, want %+v", got, valid)
	}

	tests := []struct {
		name string
		call func() (*Router, error)
	}{
		{
			name: "nil reflex provider",
			call: func() (*Router, error) {
				return NewRouter(nil, &recordingJudge{}, newMapMemory(), valid)
			},
		},
		{
			name: "nil deliberative provider",
			call: func() (*Router, error) {
				return NewRouter(&recordingJudge{}, nil, newMapMemory(), valid)
			},
		},
		{
			name: "nil memory",
			call: func() (*Router, error) {
				return NewRouter(&recordingJudge{}, &recordingJudge{}, nil, valid)
			},
		},
		{
			name: "invalid thresholds",
			call: func() (*Router, error) {
				return NewRouter(&recordingJudge{}, &recordingJudge{}, newMapMemory(), Thresholds{Choice: 1.5})
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			router, err := test.call()
			if router != nil {
				t.Fatalf("NewRouter returned non-nil router %+v alongside error", router)
			}
			if !errors.Is(err, ErrInvalidPolicy) {
				t.Fatalf("NewRouter error = %v, want %v", err, ErrInvalidPolicy)
			}
		})
	}
}

func TestRouterCacheHitSkipsProviders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		run  func(t *testing.T)
	}{
		{
			name: "choice exact hit serves cached judgment",
			run: func(t *testing.T) {
				t.Parallel()

				scope := escalationScope()
				request := validChoiceRequest()
				seeded := modalChoice(0.72)
				mem := newMapMemory()
				mem.choices[choiceMemoryKey(scope, request)] = seeded
				reflex, deliberate := &recordingJudge{}, &recordingJudge{}
				router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.99, Score: 0.99, Noul: 0.99})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}

				for call := 1; call <= 2; call++ {
					routed, err := router.Choose(context.Background(), scope, request)
					if err != nil {
						t.Fatalf("call %d unexpected error: %v", call, err)
					}
					if routed.Route != RouteCache {
						t.Fatalf("call %d route = %q, want %q", call, routed.Route, RouteCache)
					}
					if routed.Judgment.Selected != seeded.Selected {
						t.Fatalf("call %d selected = %q, want %q", call, routed.Judgment.Selected, seeded.Selected)
					}
					if len(routed.Judgment.Distribution) != len(seeded.Distribution) {
						t.Fatalf("call %d distribution length = %d, want %d", call, len(routed.Judgment.Distribution), len(seeded.Distribution))
					}
					if routed.Confidence != Probability(0.72) {
						t.Fatalf("call %d confidence = %v, want 0.72", call, float64(routed.Confidence))
					}
				}
				if reflex.chooseCalls != 0 || deliberate.chooseCalls != 0 {
					t.Fatalf("provider calls on cache hit: reflex=%d deliberate=%d", reflex.chooseCalls, deliberate.chooseCalls)
				}
				if mem.puts != 0 {
					t.Fatalf("cache hits must not write memory, puts = %d", mem.puts)
				}
				if mem.gets != 2 {
					t.Fatalf("gets = %d, want 2", mem.gets)
				}
			},
		},
		{
			name: "score exact hit serves cached judgment",
			run: func(t *testing.T) {
				t.Parallel()

				scope := escalationScope()
				request := scoreFixture()
				seeded := Score{Value: 62, Confidence: 0.66}
				mem := newMapMemory()
				mem.scores[scoreMemoryKey(scope, request)] = seeded
				reflex, deliberate := &recordingJudge{}, &recordingJudge{}
				router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.99, Score: 0.99, Noul: 0.99})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}

				routed, err := router.Score(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("Score unexpected error: %v", err)
				}
				if routed.Route != RouteCache || routed.Judgment.Value != seeded.Value || routed.Confidence != seeded.Confidence {
					t.Fatalf("cache-hit score = %+v, want cached %+v on route %q", routed, seeded, RouteCache)
				}
				if reflex.scoreCalls != 0 || deliberate.scoreCalls != 0 {
					t.Fatalf("provider calls on cache hit: reflex=%d deliberate=%d", reflex.scoreCalls, deliberate.scoreCalls)
				}
				if mem.puts != 0 {
					t.Fatalf("cache hits must not write memory, puts = %d", mem.puts)
				}
			},
		},
		{
			name: "noul exact hit serves cached judgment",
			run: func(t *testing.T) {
				t.Parallel()

				scope := escalationScope()
				request := noulFixture()
				seeded := Noul{Verdict: VerdictNo, Confidence: 0.71}
				mem := newMapMemory()
				mem.nouls[noulMemoryKey(scope, request)] = seeded
				reflex, deliberate := &recordingJudge{}, &recordingJudge{}
				router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.99, Score: 0.99, Noul: 0.99})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}

				routed, err := router.Ask(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("Ask unexpected error: %v", err)
				}
				if routed.Route != RouteCache || routed.Judgment.Verdict != seeded.Verdict || routed.Confidence != seeded.Confidence {
					t.Fatalf("cache-hit noul = %+v, want cached %+v on route %q", routed, seeded, RouteCache)
				}
				if reflex.askCalls != 0 || deliberate.askCalls != 0 {
					t.Fatalf("provider calls on cache hit: reflex=%d deliberate=%d", reflex.askCalls, deliberate.askCalls)
				}
				if mem.puts != 0 {
					t.Fatalf("cache hits must not write memory, puts = %d", mem.puts)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, test.run)
	}
}

func TestRouterMissRoutesThroughReflexAndCaches(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		run  func(t *testing.T)
	}{
		{
			name: "choice miss is judged then cached",
			run: func(t *testing.T) {
				t.Parallel()

				scope := escalationScope()
				request := validChoiceRequest()
				mem := newMapMemory()
				reflex := &recordingJudge{choice: modalChoice(0.5)}
				deliberate := &recordingJudge{choice: modalChoice(0.99)}
				router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.5, Score: 0.5, Noul: 0.5})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}

				first, err := router.Choose(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("first Choose unexpected error: %v", err)
				}
				if first.Route != RouteReflex || first.Confidence != Probability(0.5) {
					t.Fatalf("first routed = %+v, want reflex at threshold confidence 0.5", first)
				}
				if deliberate.chooseCalls != 0 {
					t.Fatalf("deliberative provider called at-threshold choice: %d calls", deliberate.chooseCalls)
				}
				if mem.puts != 1 {
					t.Fatalf("puts = %d, want 1", mem.puts)
				}

				second, err := router.Choose(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("second Choose unexpected error: %v", err)
				}
				if second.Route != RouteCache {
					t.Fatalf("second route = %q, want %q", second.Route, RouteCache)
				}
				if second.Judgment.Selected != first.Judgment.Selected || second.Confidence != first.Confidence {
					t.Fatalf("second routed = %+v, want cached %+v", second, first)
				}
				if reflex.chooseCalls != 1 || deliberate.chooseCalls != 0 {
					t.Fatalf("calls after repeat: reflex=%d deliberate=%d, want 1 and 0", reflex.chooseCalls, deliberate.chooseCalls)
				}
			},
		},
		{
			name: "score miss is judged then cached",
			run: func(t *testing.T) {
				t.Parallel()

				scope := escalationScope()
				request := scoreFixture()
				mem := newMapMemory()
				reflex := &recordingJudge{score: Score{Value: 50, Confidence: 0.5}}
				deliberate := &recordingJudge{score: Score{Value: 10, Confidence: 0.99}}
				router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.5, Score: 0.5, Noul: 0.5})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}

				first, err := router.Score(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("first Score unexpected error: %v", err)
				}
				if first.Route != RouteReflex || first.Judgment.Value != 50 || first.Confidence != Probability(0.5) {
					t.Fatalf("first routed = %+v, want reflex value 50 at threshold confidence", first)
				}
				if deliberate.scoreCalls != 0 {
					t.Fatalf("deliberative provider called at-threshold score: %d calls", deliberate.scoreCalls)
				}

				second, err := router.Score(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("second Score unexpected error: %v", err)
				}
				if second.Route != RouteCache || second.Judgment.Value != 50 {
					t.Fatalf("second routed = %+v, want cached reflex value 50", second)
				}
				if reflex.scoreCalls != 1 || deliberate.scoreCalls != 0 {
					t.Fatalf("calls after repeat: reflex=%d deliberate=%d, want 1 and 0", reflex.scoreCalls, deliberate.scoreCalls)
				}
			},
		},
		{
			name: "noul miss is judged then cached",
			run: func(t *testing.T) {
				t.Parallel()

				scope := escalationScope()
				request := noulFixture()
				mem := newMapMemory()
				reflex := &recordingJudge{noul: Noul{Verdict: VerdictYes, Confidence: 0.5}}
				deliberate := &recordingJudge{noul: Noul{Verdict: VerdictNo, Confidence: 0.99}}
				router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.5, Score: 0.5, Noul: 0.5})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}

				first, err := router.Ask(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("first Ask unexpected error: %v", err)
				}
				if first.Route != RouteReflex || first.Judgment.Verdict != VerdictYes || first.Confidence != Probability(0.5) {
					t.Fatalf("first routed = %+v, want reflex yes at threshold confidence", first)
				}
				if deliberate.askCalls != 0 {
					t.Fatalf("deliberative provider called at-threshold noul: %d calls", deliberate.askCalls)
				}

				second, err := router.Ask(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("second Ask unexpected error: %v", err)
				}
				if second.Route != RouteCache || second.Judgment.Verdict != VerdictYes {
					t.Fatalf("second routed = %+v, want cached reflex yes", second)
				}
				if reflex.askCalls != 1 || deliberate.askCalls != 0 {
					t.Fatalf("calls after repeat: reflex=%d deliberate=%d, want 1 and 0", reflex.askCalls, deliberate.askCalls)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, test.run)
	}
}

func TestRouterRoutesOnlyExactScopeAndRequest(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		run  func(t *testing.T)
	}{
		{
			name: "choice is scoped by tenant, schema, and request",
			run: func(t *testing.T) {
				t.Parallel()

				scope := escalationScope()
				request := validChoiceRequest()
				otherScope := Scope{Tenant: "beta", Schema: "v1"}
				otherRequest := validChoiceRequest()
				otherRequest.Task = "route a different task"
				mem := newMapMemory()
				reflex := &recordingJudge{choice: modalChoice(0.9)}
				router, err := NewRouter(reflex, &recordingJudge{}, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}
				ctx := context.Background()

				if _, err := router.Choose(ctx, scope, request); err != nil {
					t.Fatalf("prime Choose unexpected error: %v", err)
				}

				otherTenant, err := router.Choose(ctx, otherScope, request)
				if err != nil {
					t.Fatalf("other-tenant Choose unexpected error: %v", err)
				}
				if otherTenant.Route != RouteReflex {
					t.Fatalf("other tenant route = %q, want %q", otherTenant.Route, RouteReflex)
				}

				otherRequestRouted, err := router.Choose(ctx, scope, otherRequest)
				if err != nil {
					t.Fatalf("other-request Choose unexpected error: %v", err)
				}
				if otherRequestRouted.Route != RouteReflex {
					t.Fatalf("other request route = %q, want %q", otherRequestRouted.Route, RouteReflex)
				}

				original, err := router.Choose(ctx, scope, request)
				if err != nil {
					t.Fatalf("original Choose unexpected error: %v", err)
				}
				if original.Route != RouteCache {
					t.Fatalf("original scope and request route = %q, want %q", original.Route, RouteCache)
				}
				if reflex.chooseCalls != 3 {
					t.Fatalf("reflex calls = %d, want 3 (misses only)", reflex.chooseCalls)
				}
			},
		},
		{
			name: "score is scoped by tenant, schema, and request",
			run: func(t *testing.T) {
				t.Parallel()

				scope := escalationScope()
				request := scoreFixture()
				otherScope := Scope{Tenant: "acme", Schema: "v2"}
				otherRequest := scoreFixture()
				otherRequest.Target = "doc-2"
				mem := newMapMemory()
				reflex := &recordingJudge{score: Score{Value: 70, Confidence: 0.9}}
				router, err := NewRouter(reflex, &recordingJudge{}, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}
				ctx := context.Background()

				if _, err := router.Score(ctx, scope, request); err != nil {
					t.Fatalf("prime Score unexpected error: %v", err)
				}

				otherSchema, err := router.Score(ctx, otherScope, request)
				if err != nil {
					t.Fatalf("other-schema Score unexpected error: %v", err)
				}
				if otherSchema.Route != RouteReflex {
					t.Fatalf("other schema route = %q, want %q", otherSchema.Route, RouteReflex)
				}

				otherTarget, err := router.Score(ctx, scope, otherRequest)
				if err != nil {
					t.Fatalf("other-target Score unexpected error: %v", err)
				}
				if otherTarget.Route != RouteReflex {
					t.Fatalf("other target route = %q, want %q", otherTarget.Route, RouteReflex)
				}

				original, err := router.Score(ctx, scope, request)
				if err != nil {
					t.Fatalf("original Score unexpected error: %v", err)
				}
				if original.Route != RouteCache {
					t.Fatalf("original scope and request route = %q, want %q", original.Route, RouteCache)
				}
				if reflex.scoreCalls != 3 {
					t.Fatalf("reflex calls = %d, want 3 (misses only)", reflex.scoreCalls)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, test.run)
	}
}

func TestRouterEscalatesBelowThresholdToDeliberative(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		run  func(t *testing.T)
	}{
		{
			name: "choice forwards below-threshold confidence",
			run: func(t *testing.T) {
				t.Parallel()

				scope := escalationScope()
				request := validChoiceRequest()
				mem := newMapMemory()
				reflex := &recordingJudge{choice: modalChoice(0.5)}
				deliberate := &recordingJudge{choice: modalChoice(0.95)}
				router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}

				routed, err := router.Choose(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("Choose unexpected error: %v", err)
				}
				if routed.Route != RouteDeliberate {
					t.Fatalf("route = %q, want %q", routed.Route, RouteDeliberate)
				}
				if reflex.chooseCalls != 1 || deliberate.chooseCalls != 1 {
					t.Fatalf("calls: reflex=%d deliberate=%d, want 1 and 1", reflex.chooseCalls, deliberate.chooseCalls)
				}
				if routed.Confidence != Probability(0.95) || routed.Judgment.Selected != "search" {
					t.Fatalf("routed = %+v, want deliberated choice at 0.95", routed)
				}
				if mem.puts != 1 {
					t.Fatalf("puts = %d, want 1 (deliberated answer cached)", mem.puts)
				}

				repeat, err := router.Choose(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("repeat Choose unexpected error: %v", err)
				}
				if repeat.Route != RouteCache || repeat.Confidence != Probability(0.95) {
					t.Fatalf("repeat routed = %+v, want cached deliberated answer", repeat)
				}
				if deliberate.chooseCalls != 1 {
					t.Fatalf("deliberative calls after repeat = %d, want 1", deliberate.chooseCalls)
				}
			},
		},
		{
			name: "score forwards below-threshold confidence",
			run: func(t *testing.T) {
				t.Parallel()

				scope := escalationScope()
				request := scoreFixture()
				mem := newMapMemory()
				reflex := &recordingJudge{score: Score{Value: 40, Confidence: 0.7}}
				deliberate := &recordingJudge{score: Score{Value: 90, Confidence: 0.96}}
				router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}

				routed, err := router.Score(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("Score unexpected error: %v", err)
				}
				if routed.Route != RouteDeliberate || routed.Judgment.Value != 90 || routed.Confidence != Probability(0.96) {
					t.Fatalf("routed = %+v, want deliberated score 90 at 0.96", routed)
				}
				if reflex.scoreCalls != 1 || deliberate.scoreCalls != 1 {
					t.Fatalf("calls: reflex=%d deliberate=%d, want 1 and 1", reflex.scoreCalls, deliberate.scoreCalls)
				}

				repeat, err := router.Score(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("repeat Score unexpected error: %v", err)
				}
				if repeat.Route != RouteCache || repeat.Judgment.Value != 90 {
					t.Fatalf("repeat routed = %+v, want cached deliberated score", repeat)
				}
				if deliberate.scoreCalls != 1 {
					t.Fatalf("deliberative calls after repeat = %d, want 1", deliberate.scoreCalls)
				}
			},
		},
		{
			name: "noul forwards below-threshold confidence",
			run: func(t *testing.T) {
				t.Parallel()

				scope := escalationScope()
				request := noulFixture()
				mem := newMapMemory()
				reflex := &recordingJudge{noul: Noul{Verdict: VerdictYes, Confidence: 0.79}}
				deliberate := &recordingJudge{noul: Noul{Verdict: VerdictNo, Confidence: 0.99}}
				router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}

				routed, err := router.Ask(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("Ask unexpected error: %v", err)
				}
				if routed.Route != RouteDeliberate || routed.Judgment.Verdict != VerdictNo || routed.Confidence != Probability(0.99) {
					t.Fatalf("routed = %+v, want deliberated no at 0.99", routed)
				}
				if reflex.askCalls != 1 || deliberate.askCalls != 1 {
					t.Fatalf("calls: reflex=%d deliberate=%d, want 1 and 1", reflex.askCalls, deliberate.askCalls)
				}

				repeat, err := router.Ask(context.Background(), scope, request)
				if err != nil {
					t.Fatalf("repeat Ask unexpected error: %v", err)
				}
				if repeat.Route != RouteCache || repeat.Judgment.Verdict != VerdictNo {
					t.Fatalf("repeat routed = %+v, want cached deliberated verdict", repeat)
				}
				if deliberate.askCalls != 1 {
					t.Fatalf("deliberative calls after repeat = %d, want 1", deliberate.askCalls)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, test.run)
	}
}

func TestRouterThresholdsArePerPrimitive(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		thresholds Thresholds
		wantChoice Route
		wantScore  Route
		wantNoul   Route
	}{
		{
			name:       "only choice escalates",
			thresholds: Thresholds{Choice: 0.9, Score: 0.5, Noul: 0.3},
			wantChoice: RouteDeliberate,
			wantScore:  RouteReflex,
			wantNoul:   RouteReflex,
		},
		{
			name:       "score and noul escalate while choice stays reflex",
			thresholds: Thresholds{Choice: 0.4, Score: 0.9, Noul: 0.9},
			wantChoice: RouteReflex,
			wantScore:  RouteDeliberate,
			wantNoul:   RouteDeliberate,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			scope := escalationScope()
			ctx := context.Background()
			mem := newMapMemory()
			reflex := &recordingJudge{
				choice: modalChoice(0.5),
				score:  Score{Value: 60, Confidence: 0.6},
				noul:   Noul{Verdict: VerdictNo, Confidence: 0.35},
			}
			deliberate := &recordingJudge{
				choice: modalChoice(0.97),
				score:  Score{Value: 55, Confidence: 0.97},
				noul:   Noul{Verdict: VerdictYes, Confidence: 0.97},
			}
			router, err := NewRouter(reflex, deliberate, mem, test.thresholds)
			if err != nil {
				t.Fatalf("NewRouter unexpected error: %v", err)
			}
			if got := router.Thresholds(); got != test.thresholds {
				t.Fatalf("Router.Thresholds() = %+v, want %+v", got, test.thresholds)
			}

			routedChoice, err := router.Choose(ctx, scope, validChoiceRequest())
			if err != nil {
				t.Fatalf("Choose unexpected error: %v", err)
			}
			if routedChoice.Route != test.wantChoice {
				t.Errorf("choice route = %q, want %q", routedChoice.Route, test.wantChoice)
			}

			routedScore, err := router.Score(ctx, scope, scoreFixture())
			if err != nil {
				t.Fatalf("Score unexpected error: %v", err)
			}
			if routedScore.Route != test.wantScore {
				t.Errorf("score route = %q, want %q", routedScore.Route, test.wantScore)
			}

			routedNoul, err := router.Ask(ctx, scope, noulFixture())
			if err != nil {
				t.Fatalf("Ask unexpected error: %v", err)
			}
			if routedNoul.Route != test.wantNoul {
				t.Errorf("noul route = %q, want %q", routedNoul.Route, test.wantNoul)
			}

			wantDeliberate := map[Route]int{RouteDeliberate: 1, RouteReflex: 0}
			if deliberate.chooseCalls != wantDeliberate[test.wantChoice] {
				t.Errorf("deliberative choose calls = %d, want %d", deliberate.chooseCalls, wantDeliberate[test.wantChoice])
			}
			if deliberate.scoreCalls != wantDeliberate[test.wantScore] {
				t.Errorf("deliberative score calls = %d, want %d", deliberate.scoreCalls, wantDeliberate[test.wantScore])
			}
			if deliberate.askCalls != wantDeliberate[test.wantNoul] {
				t.Errorf("deliberative ask calls = %d, want %d", deliberate.askCalls, wantDeliberate[test.wantNoul])
			}
			if reflex.chooseCalls != 1 || reflex.scoreCalls != 1 || reflex.askCalls != 1 {
				t.Errorf("reflex calls = choose:%d score:%d ask:%d, want 1 each", reflex.chooseCalls, reflex.scoreCalls, reflex.askCalls)
			}
		})
	}
}

func TestRouterEscalationIsStrictlyBelow(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		threshold  Probability
		confidence Probability
		wantRoute  Route
	}{
		{
			name:       "confidence equal to threshold stays reflex",
			threshold:  0.5,
			confidence: 0.5,
			wantRoute:  RouteReflex,
		},
		{
			name:       "confidence just below threshold escalates",
			threshold:  0.5,
			confidence: 0.4999,
			wantRoute:  RouteDeliberate,
		},
		{
			name:       "zero threshold never escalates",
			threshold:  0,
			confidence: 0,
			wantRoute:  RouteReflex,
		},
		{
			name:       "certain confidence survives a threshold of one",
			threshold:  1,
			confidence: 1,
			wantRoute:  RouteReflex,
		},
		{
			name:       "threshold of one escalates near-certain confidence",
			threshold:  1,
			confidence: 0.999,
			wantRoute:  RouteDeliberate,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			scope := escalationScope()
			mem := newMapMemory()
			reflex := &recordingJudge{score: Score{Value: 50, Confidence: test.confidence}}
			deliberate := &recordingJudge{score: Score{Value: 88, Confidence: 0.9}}
			router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.5, Score: test.threshold, Noul: 0.5})
			if err != nil {
				t.Fatalf("NewRouter unexpected error: %v", err)
			}

			routed, err := router.Score(context.Background(), scope, scoreFixture())
			if err != nil {
				t.Fatalf("Score unexpected error: %v", err)
			}
			if routed.Route != test.wantRoute {
				t.Fatalf("route = %q, want %q", routed.Route, test.wantRoute)
			}
			if reflex.scoreCalls != 1 {
				t.Fatalf("reflex calls = %d, want 1", reflex.scoreCalls)
			}
			wantDeliberate := 0
			if test.wantRoute == RouteDeliberate {
				wantDeliberate = 1
			}
			if deliberate.scoreCalls != wantDeliberate {
				t.Fatalf("deliberative calls = %d, want %d", deliberate.scoreCalls, wantDeliberate)
			}
		})
	}
}

func TestRouterValidatesInputBeforeAnyTraffic(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		call func(ctx context.Context, router *Router) error
	}{
		{
			name: "choice with one option",
			call: func(ctx context.Context, router *Router) error {
				_, err := router.Choose(ctx, escalationScope(), ChoiceRequest{Task: "pick", Options: []Option{{ID: "a"}}})
				return err
			},
		},
		{
			name: "score with blank target",
			call: func(ctx context.Context, router *Router) error {
				_, err := router.Score(ctx, escalationScope(), ScoreRequest{Task: "rank"})
				return err
			},
		},
		{
			name: "noul with blank question",
			call: func(ctx context.Context, router *Router) error {
				_, err := router.Ask(ctx, escalationScope(), NoulRequest{Question: " "})
				return err
			},
		},
		{
			name: "blank tenant scope",
			call: func(ctx context.Context, router *Router) error {
				_, err := router.Choose(ctx, Scope{Schema: "v1"}, validChoiceRequest())
				return err
			},
		},
		{
			name: "blank schema scope",
			call: func(ctx context.Context, router *Router) error {
				_, err := router.Score(ctx, Scope{Tenant: "acme", Schema: "\t"}, scoreFixture())
				return err
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			mem := newMapMemory()
			reflex, deliberate := &recordingJudge{}, &recordingJudge{}
			router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.5, Score: 0.5, Noul: 0.5})
			if err != nil {
				t.Fatalf("NewRouter unexpected error: %v", err)
			}

			err = test.call(context.Background(), router)
			if !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("error = %v, want %v", err, ErrInvalidRequest)
			}
			if reflex.chooseCalls+reflex.scoreCalls+reflex.askCalls != 0 {
				t.Fatalf("reflex was called for invalid input")
			}
			if deliberate.chooseCalls+deliberate.scoreCalls+deliberate.askCalls != 0 {
				t.Fatalf("deliberative provider was called for invalid input")
			}
			if mem.gets != 0 || mem.puts != 0 {
				t.Fatalf("memory traffic for invalid input: gets=%d puts=%d", mem.gets, mem.puts)
			}
		})
	}
}

func TestRouterValidatesProviderResults(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		run  func(t *testing.T)
	}{
		{
			name: "reflex choice with incomplete distribution is rejected",
			run: func(t *testing.T) {
				t.Parallel()

				mem := newMapMemory()
				reflex := &recordingJudge{choice: Choice{
					Selected: "search",
					Distribution: distribution(
						OptionProbability{Option: "search", Probability: 0.5},
						OptionProbability{Option: "summarize", Probability: 0.3},
						OptionProbability{Option: "translate", Probability: 0.1},
					),
				}}
				deliberate := &recordingJudge{choice: modalChoice(0.99)}
				router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}

				_, err = router.Choose(context.Background(), escalationScope(), validChoiceRequest())
				if !errors.Is(err, ErrInvalidResult) {
					t.Fatalf("error = %v, want %v", err, ErrInvalidResult)
				}
				if deliberate.chooseCalls != 0 {
					t.Fatalf("invalid reflex result must not escalate, deliberate calls = %d", deliberate.chooseCalls)
				}
				if mem.puts != 0 {
					t.Fatalf("invalid reflex result must not be cached, puts = %d", mem.puts)
				}
			},
		},
		{
			name: "reflex score above the scale is rejected",
			run: func(t *testing.T) {
				t.Parallel()

				mem := newMapMemory()
				reflex := &recordingJudge{score: Score{Value: 101, Confidence: 0.9}}
				deliberate := &recordingJudge{score: Score{Value: 50, Confidence: 0.9}}
				router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}

				_, err = router.Score(context.Background(), escalationScope(), scoreFixture())
				if !errors.Is(err, ErrInvalidResult) {
					t.Fatalf("error = %v, want %v", err, ErrInvalidResult)
				}
				if deliberate.scoreCalls != 0 || mem.puts != 0 {
					t.Fatalf("invalid score escalated (%d) or cached (%d)", deliberate.scoreCalls, mem.puts)
				}
			},
		},
		{
			name: "deliberative noul with unknown verdict is rejected",
			run: func(t *testing.T) {
				t.Parallel()

				mem := newMapMemory()
				reflex := &recordingJudge{noul: Noul{Verdict: VerdictYes, Confidence: 0.1}}
				deliberate := &recordingJudge{noul: Noul{Verdict: Verdict("maybe"), Confidence: 0.9}}
				router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
				if err != nil {
					t.Fatalf("NewRouter unexpected error: %v", err)
				}

				_, err = router.Ask(context.Background(), escalationScope(), noulFixture())
				if !errors.Is(err, ErrInvalidResult) {
					t.Fatalf("error = %v, want %v", err, ErrInvalidResult)
				}
				if deliberate.askCalls != 1 {
					t.Fatalf("deliberative calls = %d, want 1", deliberate.askCalls)
				}
				if mem.puts != 0 {
					t.Fatalf("invalid deliberative result must not be cached, puts = %d", mem.puts)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, test.run)
	}
}

func TestRouterRejectsCorruptCacheHit(t *testing.T) {
	t.Parallel()

	scope := escalationScope()
	request := validChoiceRequest()
	mem := newMapMemory()
	mem.choices[choiceMemoryKey(scope, request)] = Choice{
		Selected: "ghost",
		Distribution: distribution(
			OptionProbability{Option: "search", Probability: 0.6},
			OptionProbability{Option: "summarize", Probability: 0.2},
			OptionProbability{Option: "translate", Probability: 0.2},
		),
	}
	reflex, deliberate := &recordingJudge{}, &recordingJudge{}
	router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
	if err != nil {
		t.Fatalf("NewRouter unexpected error: %v", err)
	}

	_, err = router.Choose(context.Background(), scope, request)
	if !errors.Is(err, ErrInvalidResult) {
		t.Fatalf("error = %v, want %v wrapped from cache validation", err, ErrInvalidResult)
	}
	if reflex.chooseCalls != 0 || deliberate.chooseCalls != 0 {
		t.Fatalf("providers called despite corrupt cache: reflex=%d deliberate=%d", reflex.chooseCalls, deliberate.chooseCalls)
	}
	if mem.puts != 0 {
		t.Fatalf("corrupt cache must not be rewritten, puts = %d", mem.puts)
	}
}

func TestRouterProviderErrorPropagation(t *testing.T) {
	t.Parallel()

	t.Run("reflex error fails the routing", func(t *testing.T) {
		t.Parallel()

		mem := newMapMemory()
		reflexFailure := errors.New("reflex unavailable")
		reflex := &recordingJudge{err: reflexFailure}
		deliberate := &recordingJudge{choice: modalChoice(0.99)}
		router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
		if err != nil {
			t.Fatalf("NewRouter unexpected error: %v", err)
		}

		_, err = router.Choose(context.Background(), escalationScope(), validChoiceRequest())
		if !errors.Is(err, reflexFailure) {
			t.Fatalf("error = %v, want wrapped %v", err, reflexFailure)
		}
		if deliberate.chooseCalls != 0 || mem.puts != 0 {
			t.Fatalf("failed reflex escalated (%d) or cached (%d)", deliberate.chooseCalls, mem.puts)
		}
	})

	t.Run("deliberative error fails the routing", func(t *testing.T) {
		t.Parallel()

		mem := newMapMemory()
		reflex := &recordingJudge{choice: modalChoice(0.5)}
		deliberativeFailure := errors.New("deliberation unavailable")
		deliberate := &recordingJudge{err: deliberativeFailure}
		router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
		if err != nil {
			t.Fatalf("NewRouter unexpected error: %v", err)
		}

		_, err = router.Choose(context.Background(), escalationScope(), validChoiceRequest())
		if !errors.Is(err, deliberativeFailure) {
			t.Fatalf("error = %v, want wrapped %v", err, deliberativeFailure)
		}
		if mem.puts != 0 {
			t.Fatalf("failed escalation must not cache, puts = %d", mem.puts)
		}
	})
}

func TestRouterPutFailureStillServesJudgment(t *testing.T) {
	t.Parallel()

	t.Run("reflex answer served alongside cache error", func(t *testing.T) {
		t.Parallel()

		mem := newMapMemory()
		cacheFailure := errors.New("cache write failed")
		mem.putErr = cacheFailure
		reflex := &recordingJudge{choice: modalChoice(0.9)}
		deliberate := &recordingJudge{}
		router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
		if err != nil {
			t.Fatalf("NewRouter unexpected error: %v", err)
		}

		routed, err := router.Choose(context.Background(), escalationScope(), validChoiceRequest())
		if !errors.Is(err, cacheFailure) {
			t.Fatalf("error = %v, want wrapped %v", err, cacheFailure)
		}
		if routed.Route != RouteReflex || routed.Judgment.Selected != "search" || routed.Confidence != Probability(0.9) {
			t.Fatalf("routed = %+v, want the served reflex judgment alongside the error", routed)
		}
	})

	t.Run("deliberated answer served alongside cache error", func(t *testing.T) {
		t.Parallel()

		mem := newMapMemory()
		cacheFailure := errors.New("cache write failed")
		mem.putErr = cacheFailure
		reflex := &recordingJudge{choice: modalChoice(0.5)}
		deliberate := &recordingJudge{choice: modalChoice(0.9)}
		router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
		if err != nil {
			t.Fatalf("NewRouter unexpected error: %v", err)
		}

		routed, err := router.Choose(context.Background(), escalationScope(), validChoiceRequest())
		if !errors.Is(err, cacheFailure) {
			t.Fatalf("error = %v, want wrapped %v", err, cacheFailure)
		}
		if routed.Route != RouteDeliberate || routed.Confidence != Probability(0.9) {
			t.Fatalf("routed = %+v, want the served deliberated judgment alongside the error", routed)
		}
	})
}

func ExampleRouter_Choose() {
	scope := Scope{Tenant: "demo", Schema: "v1"}
	mem := newMapMemory()
	reflex := &recordingJudge{choice: modalChoice(0.55)}
	deliberate := &recordingJudge{choice: modalChoice(0.97)}
	router, err := NewRouter(reflex, deliberate, mem, Thresholds{Choice: 0.8, Score: 0.8, Noul: 0.8})
	if err != nil {
		fmt.Println("router error:", err)
		return
	}

	// The reflex confidence 0.55 falls below the choice threshold 0.8, so
	// the request forwards to the deliberative provider.
	first, err := router.Choose(context.Background(), scope, validChoiceRequest())
	if err != nil {
		fmt.Println("routing error:", err)
		return
	}
	fmt.Println(first.Route, first.Judgment.Selected, first.Confidence)

	// The identical request is now an exact cache hit.
	second, _ := router.Choose(context.Background(), scope, validChoiceRequest())
	fmt.Println(second.Route)
	// Output:
	// deliberate search 0.97
	// cache
}
