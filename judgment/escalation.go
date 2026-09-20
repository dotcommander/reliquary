package judgment

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// ErrInvalidPolicy reports an escalation router constructed with a nil
// collaborator or out-of-range thresholds.
var ErrInvalidPolicy = errors.New("judgment: invalid escalation policy")

// Route names the path that served a judgment.
type Route string

const (
	// RouteCache served the judgment from exact-hit memory without any
	// provider call.
	RouteCache Route = "cache"
	// RouteReflex served the judgment from the reflex provider because its
	// confidence met the primitive's threshold.
	RouteReflex Route = "reflex"
	// RouteDeliberate served the judgment from the deliberative provider
	// after the reflex confidence fell below the threshold.
	RouteDeliberate Route = "deliberate"
)

// Thresholds are the per-primitive escalation policy values. A reflex
// judgment escalates to the deliberative provider when its confidence
// falls strictly below its primitive's threshold: a threshold of zero
// never escalates, and a threshold of one escalates every judgment short
// of perfect confidence. Each primitive carries its own threshold so
// routing policy is calibrated separately for choices, scores, and gates.
type Thresholds struct {
	// Choice is the minimum selected-option probability a reflex choice
	// must reach to be served without deliberation.
	Choice Probability
	// Score is the minimum confidence a reflex score must reach.
	Score Probability
	// Noul is the minimum confidence a reflex verdict must reach.
	Noul Probability
}

// Validate reports whether every threshold is a valid probability.
func (t Thresholds) Validate() error {
	if !t.Choice.Valid() {
		return fmt.Errorf("%w: choice threshold %v outside [0, 1]", ErrInvalidPolicy, float64(t.Choice))
	}
	if !t.Score.Valid() {
		return fmt.Errorf("%w: score threshold %v outside [0, 1]", ErrInvalidPolicy, float64(t.Score))
	}
	if !t.Noul.Valid() {
		return fmt.Errorf("%w: noul threshold %v outside [0, 1]", ErrInvalidPolicy, float64(t.Noul))
	}
	return nil
}

// Scope identifies the judgment-sharing boundary for one routing call:
// cached entries are shared only within one tenant and one schema
// version.
type Scope struct {
	// Tenant isolates one customer or deployment's judgments.
	Tenant string
	// Schema versions the judging pipeline; changing prompts, models, or
	// option sets bumps the schema and invalidates older entries.
	Schema string
}

func (s Scope) validate() error {
	if strings.TrimSpace(s.Tenant) == "" {
		return fmt.Errorf("%w: scope tenant must not be blank", ErrInvalidRequest)
	}
	if strings.TrimSpace(s.Schema) == "" {
		return fmt.Errorf("%w: scope schema must not be blank", ErrInvalidRequest)
	}
	return nil
}

// Memory is the exact-hit judgment memory the router consults before any
// provider call. Get methods report whether an entry exists for exactly
// this scope and request; Put methods store a validated judgment for
// later exact hits. The judgment/memory package's Store provides this
// contract through a thin adapter that derives the cache key's primitive
// and request fingerprint.
type Memory interface {
	// GetChoice returns the cached choice for the exact scope and request.
	GetChoice(scope Scope, request ChoiceRequest) (Choice, bool)
	// PutChoice caches a validated choice for the scope and request.
	PutChoice(scope Scope, request ChoiceRequest, result Choice) error
	// GetScore returns the cached score for the exact scope and request.
	GetScore(scope Scope, request ScoreRequest) (Score, bool)
	// PutScore caches a validated score for the scope and request.
	PutScore(scope Scope, request ScoreRequest, result Score) error
	// GetNoul returns the cached gate answer for the exact scope and
	// request.
	GetNoul(scope Scope, request NoulRequest) (Noul, bool)
	// PutNoul caches a validated gate answer for the scope and request.
	PutNoul(scope Scope, request NoulRequest, result Noul) error
}

// Routed is a served judgment plus the routing metadata that produced it.
type Routed[T any] struct {
	// Judgment is the validated result being served.
	Judgment T
	// Route names the path that served the judgment.
	Route Route
	// Confidence is the governing confidence of the served judgment: the
	// selected option's probability for a choice, or the result's
	// calibrated confidence for a score or verdict.
	Confidence Probability
}

// Router is the cache-then-provider escalation policy for the judgment
// primitives. Every request is routed through memory first: an exact hit
// is served without provider calls, and a miss is judged by the fast
// reflex provider (Tier 1). When the reflex confidence falls strictly
// below the per-primitive threshold, the request is forwarded to the
// injected deliberative provider (Tier 2), whose validated answer is
// served instead. Escalation is one hop: a deliberative answer is served
// once it validates, whatever its own confidence. Fresh answers are
// cached for their scope; when caching is the only failure, the served
// judgment is returned alongside the error.
//
// Provider results are validated before they are served, escalated, or
// cached, and cache hits are re-validated against the request so a
// corrupt entry surfaces instead of silently answering. A Router holds no
// mutable state and is safe for concurrent use; collaborators must be
// safe for their own concurrent use.
type Router struct {
	reflex      Judge
	deliberate  Judge
	memory      Memory
	thresholds  Thresholds
}

// NewRouter creates a Router over the injected providers and memory.
// reflex is the fast Tier-1 judge, deliberate is the heavy Tier-2 judge
// that below-threshold confidences forward to, and thresholds carries the
// per-primitive escalation policy. All collaborators are required and
// caller-owned; the router owns no transport, credentials, or persistent
// state.
func NewRouter(reflex, deliberate Judge, mem Memory, thresholds Thresholds) (*Router, error) {
	if reflex == nil {
		return nil, fmt.Errorf("%w: reflex provider must not be nil", ErrInvalidPolicy)
	}
	if deliberate == nil {
		return nil, fmt.Errorf("%w: deliberative provider must not be nil", ErrInvalidPolicy)
	}
	if mem == nil {
		return nil, fmt.Errorf("%w: memory must not be nil", ErrInvalidPolicy)
	}
	if err := thresholds.Validate(); err != nil {
		return nil, err
	}
	return &Router{
		reflex:     reflex,
		deliberate: deliberate,
		memory:     mem,
		thresholds: thresholds,
	}, nil
}

// Thresholds returns the router's per-primitive escalation policy.
func (r *Router) Thresholds() Thresholds { return r.thresholds }

// Choose routes one choice request through cache-then-provider
// escalation: an exact memory hit is served without provider calls;
// otherwise the reflex provider judges the request, and when the selected
// option's probability falls strictly below the Choice threshold the
// request is forwarded to the deliberative provider, whose validated
// answer is served. The served answer is cached for the scope; when
// caching is the only failure, the routed choice is returned alongside
// the error.
func (r *Router) Choose(ctx context.Context, scope Scope, request ChoiceRequest) (Routed[Choice], error) {
	if err := scope.validate(); err != nil {
		return Routed[Choice]{}, err
	}
	if err := validateChoiceRequest(request); err != nil {
		return Routed[Choice]{}, err
	}
	if cached, ok := r.memory.GetChoice(scope, request); ok {
		if err := ValidateChoice(request, cached); err != nil {
			return Routed[Choice]{}, fmt.Errorf("judgment: cached choice does not validate: %w", err)
		}
		return Routed[Choice]{Judgment: cached, Route: RouteCache, Confidence: choiceConfidence(cached)}, nil
	}
	result, err := r.reflex.Choose(ctx, request)
	if err != nil {
		return Routed[Choice]{}, fmt.Errorf("judgment: reflex choice: %w", err)
	}
	if err := ValidateChoice(request, result); err != nil {
		return Routed[Choice]{}, fmt.Errorf("judgment: reflex choice does not validate: %w", err)
	}
	routed := Routed[Choice]{Judgment: result, Route: RouteReflex, Confidence: choiceConfidence(result)}
	if routed.Confidence < r.thresholds.Choice {
		deliberated, err := r.deliberate.Choose(ctx, request)
		if err != nil {
			return Routed[Choice]{}, fmt.Errorf("judgment: deliberative choice: %w", err)
		}
		if err := ValidateChoice(request, deliberated); err != nil {
			return Routed[Choice]{}, fmt.Errorf("judgment: deliberative choice does not validate: %w", err)
		}
		routed = Routed[Choice]{Judgment: deliberated, Route: RouteDeliberate, Confidence: choiceConfidence(deliberated)}
		result = deliberated
	}
	if err := r.memory.PutChoice(scope, request, result); err != nil {
		return routed, fmt.Errorf("judgment: caching choice: %w", err)
	}
	return routed, nil
}

// Score routes one score request through cache-then-provider escalation:
// an exact memory hit is served without provider calls; otherwise the
// reflex provider judges the request, and when the result's confidence
// falls strictly below the Score threshold the request is forwarded to
// the deliberative provider, whose validated answer is served. The served
// answer is cached for the scope; when caching is the only failure, the
// routed score is returned alongside the error.
func (r *Router) Score(ctx context.Context, scope Scope, request ScoreRequest) (Routed[Score], error) {
	if err := scope.validate(); err != nil {
		return Routed[Score]{}, err
	}
	if err := validateScoreRequest(request); err != nil {
		return Routed[Score]{}, err
	}
	if cached, ok := r.memory.GetScore(scope, request); ok {
		if err := ValidateScore(request, cached); err != nil {
			return Routed[Score]{}, fmt.Errorf("judgment: cached score does not validate: %w", err)
		}
		return Routed[Score]{Judgment: cached, Route: RouteCache, Confidence: cached.Confidence}, nil
	}
	result, err := r.reflex.Score(ctx, request)
	if err != nil {
		return Routed[Score]{}, fmt.Errorf("judgment: reflex score: %w", err)
	}
	if err := ValidateScore(request, result); err != nil {
		return Routed[Score]{}, fmt.Errorf("judgment: reflex score does not validate: %w", err)
	}
	routed := Routed[Score]{Judgment: result, Route: RouteReflex, Confidence: result.Confidence}
	if routed.Confidence < r.thresholds.Score {
		deliberated, err := r.deliberate.Score(ctx, request)
		if err != nil {
			return Routed[Score]{}, fmt.Errorf("judgment: deliberative score: %w", err)
		}
		if err := ValidateScore(request, deliberated); err != nil {
			return Routed[Score]{}, fmt.Errorf("judgment: deliberative score does not validate: %w", err)
		}
		routed = Routed[Score]{Judgment: deliberated, Route: RouteDeliberate, Confidence: deliberated.Confidence}
		result = deliberated
	}
	if err := r.memory.PutScore(scope, request, result); err != nil {
		return routed, fmt.Errorf("judgment: caching score: %w", err)
	}
	return routed, nil
}

// Ask routes one yes/no question through cache-then-provider escalation:
// an exact memory hit is served without provider calls; otherwise the
// reflex provider judges the question, and when the verdict's confidence
// falls strictly below the Noul threshold the question is forwarded to
// the deliberative provider, whose validated answer is served. The served
// answer is cached for the scope; when caching is the only failure, the
// routed verdict is returned alongside the error.
func (r *Router) Ask(ctx context.Context, scope Scope, request NoulRequest) (Routed[Noul], error) {
	if err := scope.validate(); err != nil {
		return Routed[Noul]{}, err
	}
	if err := validateNoulRequest(request); err != nil {
		return Routed[Noul]{}, err
	}
	if cached, ok := r.memory.GetNoul(scope, request); ok {
		if err := ValidateNoul(request, cached); err != nil {
			return Routed[Noul]{}, fmt.Errorf("judgment: cached noul does not validate: %w", err)
		}
		return Routed[Noul]{Judgment: cached, Route: RouteCache, Confidence: cached.Confidence}, nil
	}
	result, err := r.reflex.Ask(ctx, request)
	if err != nil {
		return Routed[Noul]{}, fmt.Errorf("judgment: reflex noul: %w", err)
	}
	if err := ValidateNoul(request, result); err != nil {
		return Routed[Noul]{}, fmt.Errorf("judgment: reflex noul does not validate: %w", err)
	}
	routed := Routed[Noul]{Judgment: result, Route: RouteReflex, Confidence: result.Confidence}
	if routed.Confidence < r.thresholds.Noul {
		deliberated, err := r.deliberate.Ask(ctx, request)
		if err != nil {
			return Routed[Noul]{}, fmt.Errorf("judgment: deliberative noul: %w", err)
		}
		if err := ValidateNoul(request, deliberated); err != nil {
			return Routed[Noul]{}, fmt.Errorf("judgment: deliberative noul does not validate: %w", err)
		}
		routed = Routed[Noul]{Judgment: deliberated, Route: RouteDeliberate, Confidence: deliberated.Confidence}
		result = deliberated
	}
	if err := r.memory.PutNoul(scope, request, result); err != nil {
		return routed, fmt.Errorf("judgment: caching noul: %w", err)
	}
	return routed, nil
}

// choiceConfidence returns the probability mass a validated choice
// assigned to its selected (modal) option.
func choiceConfidence(result Choice) Probability {
	for _, entry := range result.Distribution {
		if entry.Option == result.Selected {
			return entry.Probability
		}
	}
	return 0
}
