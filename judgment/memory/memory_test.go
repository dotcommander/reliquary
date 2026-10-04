package memory

import (
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/dotcommander/reliquary/judgment"
)

func newTestStore(t *testing.T, opts ...Option) *Store {
	t.Helper()
	store, err := New(append([]Option{WithTTL(time.Hour)}, opts...)...)
	if err != nil {
		t.Fatalf("New unexpected error: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return store
}

func with(base Key, mutate func(*Key)) Key {
	key := base
	mutate(&key)
	return key
}

func choiceFixtureRequest() judgment.ChoiceRequest {
	return judgment.ChoiceRequest{
		Task: "route the incoming request",
		Options: []judgment.Option{
			{ID: "search"},
			{ID: "summarize"},
		},
		Context: []string{"retrieved facts"},
	}
}

func choiceFixture() judgment.Choice {
	return judgment.Choice{
		Selected: "search",
		Distribution: []judgment.OptionProbability{
			{Option: "search", Probability: 0.7},
			{Option: "summarize", Probability: 0.3},
		},
	}
}

func baseChoiceKey() Key {
	return Key{Tenant: "acme", Schema: "router.v2", Primitive: PrimitiveChoice, Fingerprint: "fp-choice"}
}

func TestNewRequiresPositiveLifetime(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		opts []Option
	}{
		{name: "no options", opts: nil},
		{name: "zero lifetime", opts: []Option{WithTTL(0)}},
		{name: "negative lifetime", opts: []Option{WithTTL(-time.Second)}},
		{name: "other options only", opts: []Option{WithCapacity(8)}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			store, err := New(test.opts...)
			if err == nil {
				_ = store.Close()
				t.Fatal("New must reject a non-positive default lifetime")
			}
		})
	}
}

func TestChoiceExactHitScope(t *testing.T) {
	t.Parallel()

	store := newTestStore(t)
	key := baseChoiceKey()
	result := choiceFixture()

	if err := store.PutChoice(key, result); err != nil {
		t.Fatalf("PutChoice unexpected error: %v", err)
	}

	hit, ok := store.GetChoice(key)
	if !ok {
		t.Fatal("identical tenant, schema, primitive, and fingerprint must hit")
	}
	if !reflect.DeepEqual(hit, result) {
		t.Errorf("hit returned %+v, want the cached judgment %+v", hit, result)
	}

	variants := []struct {
		name string
		key  Key
	}{
		{"different tenant", with(key, func(k *Key) { k.Tenant = "other" })},
		{"different schema", with(key, func(k *Key) { k.Schema = "router.v3" })},
		{"different primitive", with(key, func(k *Key) { k.Primitive = PrimitiveScore })},
		{"different fingerprint", with(key, func(k *Key) { k.Fingerprint = "fp-other" })},
	}
	for _, variant := range variants {
		t.Run(variant.name, func(t *testing.T) {
			t.Parallel()

			if _, ok := store.GetChoice(variant.key); ok {
				t.Error("a key differing in any component must miss")
			}
		})
	}

	// The stored entry holds a choice, so the other primitives must miss
	// even under the exact same key.
	if _, ok := store.GetScore(key); ok {
		t.Error("GetScore must miss an entry stored as a choice")
	}
	if _, ok := store.GetNoul(key); ok {
		t.Error("GetNoul must miss an entry stored as a choice")
	}
}

func TestScoreExactHitScope(t *testing.T) {
	t.Parallel()

	store := newTestStore(t)
	key := Key{
		Tenant:      "acme",
		Schema:      "ranker.v1",
		Primitive:   PrimitiveScore,
		Fingerprint: FingerprintScore(judgment.ScoreRequest{Task: "rank", Target: "doc-1"}),
	}
	result := judgment.Score{Value: 87.5, Confidence: 0.9}

	if err := store.PutScore(key, result); err != nil {
		t.Fatalf("PutScore unexpected error: %v", err)
	}

	hit, ok := store.GetScore(key)
	if !ok {
		t.Fatal("identical key must hit")
	}
	if hit != result {
		t.Errorf("hit returned %+v, want %+v", hit, result)
	}

	for _, variant := range []struct {
		name string
		key  Key
	}{
		{"different tenant", with(key, func(k *Key) { k.Tenant = "other" })},
		{"different schema", with(key, func(k *Key) { k.Schema = "ranker.v2" })},
		{"different primitive", with(key, func(k *Key) { k.Primitive = PrimitiveNoul })},
		{"different fingerprint", with(key, func(k *Key) { k.Fingerprint = "fp-other" })},
	} {
		t.Run(variant.name, func(t *testing.T) {
			t.Parallel()

			if _, ok := store.GetScore(variant.key); ok {
				t.Error("a key differing in any component must miss")
			}
		})
	}

	if _, ok := store.GetChoice(key); ok {
		t.Error("GetChoice must miss an entry stored as a score")
	}
}

func TestNoulExactHitScope(t *testing.T) {
	t.Parallel()

	store := newTestStore(t)
	key := Key{
		Tenant:      "acme",
		Schema:      "gate.v1",
		Primitive:   PrimitiveNoul,
		Fingerprint: FingerprintNoul(judgment.NoulRequest{Question: "is the build green?"}),
	}
	result := judgment.Noul{Verdict: judgment.VerdictYes, Confidence: 0.97}

	if err := store.PutNoul(key, result); err != nil {
		t.Fatalf("PutNoul unexpected error: %v", err)
	}

	hit, ok := store.GetNoul(key)
	if !ok {
		t.Fatal("identical key must hit")
	}
	if hit != result {
		t.Errorf("hit returned %+v, want %+v", hit, result)
	}

	if _, ok := store.GetChoice(key); ok {
		t.Error("GetChoice must miss an entry stored as a noul")
	}
	if _, ok := store.GetScore(key); ok {
		t.Error("GetScore must miss an entry stored as a noul")
	}
}

func TestPutChoiceValidatesKey(t *testing.T) {
	t.Parallel()

	testPutValidatesKey(t, PrimitiveChoice, func(s *Store, k Key) error {
		return s.PutChoice(k, choiceFixture())
	})
}

func TestPutScoreValidatesKey(t *testing.T) {
	t.Parallel()

	testPutValidatesKey(t, PrimitiveScore, func(s *Store, k Key) error {
		return s.PutScore(k, judgment.Score{Value: 50, Confidence: 0.5})
	})
}

func TestPutNoulValidatesKey(t *testing.T) {
	t.Parallel()

	testPutValidatesKey(t, PrimitiveNoul, func(s *Store, k Key) error {
		return s.PutNoul(k, judgment.Noul{Verdict: judgment.VerdictNo, Confidence: 0.5})
	})
}

func testPutValidatesKey(t *testing.T, primitive Primitive, put func(*Store, Key) error) {
	t.Helper()

	store := newTestStore(t)
	base := Key{Tenant: "acme", Schema: "router.v2", Primitive: primitive, Fingerprint: "fp"}
	other := PrimitiveChoice
	if primitive == PrimitiveChoice {
		other = PrimitiveScore
	}

	tests := []struct {
		name   string
		mutate func(*Key)
	}{
		{"blank tenant", func(k *Key) { k.Tenant = "" }},
		{"whitespace tenant", func(k *Key) { k.Tenant = "  " }},
		{"blank schema", func(k *Key) { k.Schema = "" }},
		{"blank fingerprint", func(k *Key) { k.Fingerprint = " " }},
		{"unknown primitive", func(k *Key) { k.Primitive = Primitive("maybe") }},
		{"primitive mismatch", func(k *Key) { k.Primitive = other }},
		{"valid key", func(*Key) {}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			key := with(base, test.mutate)
			err := put(store, key)
			if test.name == "valid key" {
				if err != nil {
					t.Fatalf("valid key must be accepted, got %v", err)
				}
				return
			}
			if !errors.Is(err, ErrInvalidKey) {
				t.Fatalf("error = %v, want ErrInvalidKey", err)
			}
		})
	}
}

func TestFingerprintChoice(t *testing.T) {
	t.Parallel()

	base := choiceFixtureRequest()
	tests := []struct {
		name   string
		mutate func(*judgment.ChoiceRequest)
	}{
		{"identical request", func(*judgment.ChoiceRequest) {}},
		{"different task", func(r *judgment.ChoiceRequest) { r.Task = "rank the request" }},
		{"different option id", func(r *judgment.ChoiceRequest) { r.Options[0].ID = "browse" }},
		{"different option description", func(r *judgment.ChoiceRequest) { r.Options[0].Description = "full text search" }},
		{"different option set size", func(r *judgment.ChoiceRequest) { r.Options = r.Options[:1] }},
		{"swapped option order", func(r *judgment.ChoiceRequest) { r.Options[0], r.Options[1] = r.Options[1], r.Options[0] }},
		{"extra context", func(r *judgment.ChoiceRequest) { r.Context = append(r.Context, "late fact") }},
		{"different context content", func(r *judgment.ChoiceRequest) { r.Context[0] = "other facts" }},
	}
	fingerprint := FingerprintChoice(base)
	if len(fingerprint) != 64 {
		t.Errorf("fingerprint is %d chars, want 64 hex digits", len(fingerprint))
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			request := choiceFixtureRequest()
			test.mutate(&request)
			other := FingerprintChoice(request)
			if test.name == "identical request" {
				if other != fingerprint {
					t.Error("identical requests must fingerprint identically")
				}
				return
			}
			if other == fingerprint {
				t.Error("different requests must fingerprint differently")
			}
		})
	}
}

func TestFingerprintScore(t *testing.T) {
	t.Parallel()

	base := judgment.ScoreRequest{Task: "rank", Target: "doc-1", Context: []string{"facts"}}
	tests := []struct {
		name   string
		mutate func(*judgment.ScoreRequest)
	}{
		{"identical request", func(*judgment.ScoreRequest) {}},
		{"different task", func(r *judgment.ScoreRequest) { r.Task = "grade" }},
		{"different target", func(r *judgment.ScoreRequest) { r.Target = "doc-2" }},
		{"different context", func(r *judgment.ScoreRequest) { r.Context[0] = "other facts" }},
		{"extra context", func(r *judgment.ScoreRequest) { r.Context = append(r.Context, "late fact") }},
	}
	fingerprint := FingerprintScore(base)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			// Clone the context slice: parallel subtests mutate index 0, and a
			// shallow copy of base would share the backing array with siblings.
			request := base
			request.Context = slices.Clone(base.Context)
			test.mutate(&request)
			other := FingerprintScore(request)
			if test.name == "identical request" {
				if other != fingerprint {
					t.Error("identical requests must fingerprint identically")
				}
				return
			}
			if other == fingerprint {
				t.Error("different requests must fingerprint differently")
			}
		})
	}
}

func TestFingerprintNoul(t *testing.T) {
	t.Parallel()

	base := judgment.NoulRequest{Question: "is the build green?", Context: []string{"facts"}}
	tests := []struct {
		name   string
		mutate func(*judgment.NoulRequest)
	}{
		{"identical request", func(*judgment.NoulRequest) {}},
		{"different question", func(r *judgment.NoulRequest) { r.Question = "is the build red?" }},
		{"different context", func(r *judgment.NoulRequest) { r.Context[0] = "other facts" }},
	}
	fingerprint := FingerprintNoul(base)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			// Clone the context slice: parallel subtests mutate index 0, and a
			// shallow copy of base would share the backing array with siblings.
			request := base
			request.Context = slices.Clone(base.Context)
			test.mutate(&request)
			other := FingerprintNoul(request)
			if test.name == "identical request" {
				if other != fingerprint {
					t.Error("identical requests must fingerprint identically")
				}
				return
			}
			if other == fingerprint {
				t.Error("different requests must fingerprint differently")
			}
		})
	}
}

func TestFingerprintsArePrimitiveScoped(t *testing.T) {
	t.Parallel()

	choice := FingerprintChoice(choiceFixtureRequest())
	score := FingerprintScore(judgment.ScoreRequest{
		Task:    "route the incoming request",
		Target:  "search",
		Context: []string{"retrieved facts"},
	})
	noul := FingerprintNoul(judgment.NoulRequest{
		Question: "route the incoming request",
		Context:  []string{"retrieved facts"},
	})
	if choice == score || choice == noul || score == noul {
		t.Error("fingerprints must stay primitive-scoped")
	}
}

func TestFingerprintScopedCacheHit(t *testing.T) {
	t.Parallel()

	store := newTestStore(t)
	request := choiceFixtureRequest()
	key := with(baseChoiceKey(), func(k *Key) { k.Fingerprint = FingerprintChoice(request) })
	if err := store.PutChoice(key, choiceFixture()); err != nil {
		t.Fatalf("PutChoice unexpected error: %v", err)
	}

	if _, ok := store.GetChoice(key); !ok {
		t.Error("the fingerprint of an identical request must hit")
	}

	changed := choiceFixtureRequest()
	changed.Context = append([]string(nil), changed.Context...)
	changed.Context = append(changed.Context, "late fact")
	changedKey := with(key, func(k *Key) { k.Fingerprint = FingerprintChoice(changed) })
	if _, ok := store.GetChoice(changedKey); ok {
		t.Error("a changed request must change the fingerprint and miss")
	}
}

func TestStoreDefendsCachedValues(t *testing.T) {
	t.Parallel()

	store := newTestStore(t)
	key := baseChoiceKey()
	result := choiceFixture()
	if err := store.PutChoice(key, result); err != nil {
		t.Fatalf("PutChoice unexpected error: %v", err)
	}

	result.Distribution[0].Probability = 0
	if hit, ok := store.GetChoice(key); !ok || !reflect.DeepEqual(hit, choiceFixture()) {
		t.Errorf("caller mutations must not corrupt the cached judgment, got %+v", hit)
	}

	hit, ok := store.GetChoice(key)
	if !ok {
		t.Fatal("expected hit")
	}
	hit.Distribution[1].Probability = 0
	if again, ok := store.GetChoice(key); !ok || !reflect.DeepEqual(again, choiceFixture()) {
		t.Errorf("returned-value mutations must not corrupt the cached judgment, got %+v", again)
	}
}

func TestDeleteAndLen(t *testing.T) {
	t.Parallel()

	store := newTestStore(t)
	choiceKey := baseChoiceKey()
	scoreKey := with(choiceKey, func(k *Key) { k.Primitive = PrimitiveScore; k.Fingerprint = "fp-score" })
	noulKey := with(choiceKey, func(k *Key) { k.Primitive = PrimitiveNoul; k.Fingerprint = "fp-noul" })

	if err := store.PutChoice(choiceKey, choiceFixture()); err != nil {
		t.Fatalf("PutChoice unexpected error: %v", err)
	}
	if err := store.PutScore(scoreKey, judgment.Score{Value: 50, Confidence: 0.5}); err != nil {
		t.Fatalf("PutScore unexpected error: %v", err)
	}
	if err := store.PutNoul(noulKey, judgment.Noul{Verdict: judgment.VerdictNo, Confidence: 0.5}); err != nil {
		t.Fatalf("PutNoul unexpected error: %v", err)
	}
	if store.Len() != 3 {
		t.Errorf("Len() = %d, want 3", store.Len())
	}

	store.Delete(choiceKey)
	if _, ok := store.GetChoice(choiceKey); ok {
		t.Error("deleted entry must miss")
	}
	if store.Len() != 2 {
		t.Errorf("Len() after delete = %d, want 2", store.Len())
	}
	if _, ok := store.GetScore(scoreKey); !ok {
		t.Error("delete must be key-scoped")
	}
}

func TestMetrics(t *testing.T) {
	t.Parallel()

	store := newTestStore(t)
	key := baseChoiceKey()
	if err := store.PutChoice(key, choiceFixture()); err != nil {
		t.Fatalf("PutChoice unexpected error: %v", err)
	}
	if _, ok := store.GetChoice(key); !ok {
		t.Fatal("expected hit")
	}
	missKey := with(key, func(k *Key) { k.Fingerprint = "fp-miss" })
	if _, ok := store.GetChoice(missKey); ok {
		t.Fatal("expected miss")
	}

	metrics := store.Metrics()
	if metrics.Insertions != 1 || metrics.Hits != 1 || metrics.Misses != 1 || metrics.Updates != 0 {
		t.Errorf("metrics = %+v, want one insertion, hit, and miss", metrics)
	}
}

func TestCapacityEviction(t *testing.T) {
	t.Parallel()

	store := newTestStore(t, WithCapacity(1))
	first := with(baseChoiceKey(), func(k *Key) { k.Fingerprint = "fp-first" })
	second := with(baseChoiceKey(), func(k *Key) { k.Fingerprint = "fp-second" })
	if err := store.PutChoice(first, choiceFixture()); err != nil {
		t.Fatalf("PutChoice unexpected error: %v", err)
	}
	if err := store.PutChoice(second, choiceFixture()); err != nil {
		t.Fatalf("PutChoice unexpected error: %v", err)
	}

	if _, ok := store.GetChoice(first); ok {
		t.Error("capacity eviction must remove the oldest entry")
	}
	if _, ok := store.GetChoice(second); !ok {
		t.Error("the newest entry must survive capacity eviction")
	}
	if metrics := store.Metrics(); metrics.Evictions != 1 {
		t.Errorf("evictions = %d, want 1", metrics.Evictions)
	}
}

func TestTTLEvictionWithCallerConfiguredLifetime(t *testing.T) {
	t.Parallel()

	synctest.Test(t, func(t *testing.T) {
		store, err := New(WithTTL(30*time.Minute), WithDisableTouchOnHit())
		if err != nil {
			t.Fatalf("New unexpected error: %v", err)
		}
		defer store.Close() // must stop the cleanup goroutine inside the bubble

		key := baseChoiceKey()
		if err := store.PutChoice(key, choiceFixture()); err != nil {
			t.Fatalf("PutChoice unexpected error: %v", err)
		}
		if _, ok := store.GetChoice(key); !ok {
			t.Fatal("fresh entry must hit before its lifetime elapses")
		}

		time.Sleep(31 * time.Minute) // fake clock: past the 30m lifetime
		synctest.Wait()              // let the ttlcache cleanup loop run

		if _, ok := store.GetChoice(key); ok {
			t.Error("entry must miss after the caller-configured lifetime")
		}
		if metrics := store.Metrics(); metrics.Evictions == 0 {
			t.Error("ttlcache must count the expired entry as evicted")
		}
		if store.Len() != 0 {
			t.Errorf("Len() = %d, want 0 after eviction", store.Len())
		}
	})
}

func TestEntryLifetimeOverridesDefault(t *testing.T) {
	t.Parallel()

	synctest.Test(t, func(t *testing.T) {
		store, err := New(WithTTL(time.Hour), WithDisableTouchOnHit())
		if err != nil {
			t.Fatalf("New unexpected error: %v", err)
		}
		defer store.Close()

		short := with(baseChoiceKey(), func(k *Key) { k.Fingerprint = "fp-short" })
		long := with(baseChoiceKey(), func(k *Key) { k.Fingerprint = "fp-long" })
		ignored := with(baseChoiceKey(), func(k *Key) { k.Fingerprint = "fp-ignored" })
		if err := store.PutChoice(short, choiceFixture(), WithEntryTTL(30*time.Minute)); err != nil {
			t.Fatalf("PutChoice(short) unexpected error: %v", err)
		}
		if err := store.PutChoice(long, choiceFixture()); err != nil {
			t.Fatalf("PutChoice(long) unexpected error: %v", err)
		}
		if err := store.PutChoice(ignored, choiceFixture(), WithEntryTTL(0)); err != nil {
			t.Fatalf("PutChoice(ignored) unexpected error: %v", err)
		}

		time.Sleep(45 * time.Minute) // fake clock: past 30m, within 1h
		synctest.Wait()

		if _, ok := store.GetChoice(short); ok {
			t.Error("short entry must expire after its entry lifetime")
		}
		if _, ok := store.GetChoice(long); !ok {
			t.Error("default-lifetime entry must still hit")
		}
		if _, ok := store.GetChoice(ignored); !ok {
			t.Error("non-positive entry lifetime must fall back to the default")
		}
		if metrics := store.Metrics(); metrics.Evictions != 1 {
			t.Errorf("evictions = %d, want 1", metrics.Evictions)
		}
		if store.Len() != 2 {
			t.Errorf("Len() = %d, want 2 after eviction", store.Len())
		}
	})
}

func TestHitExtendsLifetimeByDefault(t *testing.T) {
	t.Parallel()

	synctest.Test(t, func(t *testing.T) {
		store, err := New(WithTTL(time.Hour))
		if err != nil {
			t.Fatalf("New unexpected error: %v", err)
		}
		defer store.Close()

		key := baseChoiceKey()
		if err := store.PutChoice(key, choiceFixture()); err != nil {
			t.Fatalf("PutChoice unexpected error: %v", err)
		}

		time.Sleep(30 * time.Minute) // fake clock
		if _, ok := store.GetChoice(key); !ok {
			t.Fatal("entry must hit at t=30m")
		}

		time.Sleep(45 * time.Minute) // t=75m: past the original 60m expiry
		if _, ok := store.GetChoice(key); !ok {
			t.Error("a hit must extend the entry lifetime")
		}

		time.Sleep(61 * time.Minute) // t=136m: past the touched 135m expiry
		if _, ok := store.GetChoice(key); ok {
			t.Error("extended entry must eventually expire")
		}
	})
}

func TestDisabledTouchKeepsDeadline(t *testing.T) {
	t.Parallel()

	synctest.Test(t, func(t *testing.T) {
		store, err := New(WithTTL(time.Hour), WithDisableTouchOnHit())
		if err != nil {
			t.Fatalf("New unexpected error: %v", err)
		}
		defer store.Close()

		key := baseChoiceKey()
		if err := store.PutChoice(key, choiceFixture()); err != nil {
			t.Fatalf("PutChoice unexpected error: %v", err)
		}

		time.Sleep(30 * time.Minute) // fake clock
		if _, ok := store.GetChoice(key); !ok {
			t.Fatal("entry must hit at t=30m")
		}

		time.Sleep(45 * time.Minute) // t=75m: past the untouched 60m expiry
		if _, ok := store.GetChoice(key); ok {
			t.Error("hits must not extend the entry when touch is disabled")
		}
	})
}

func TestConcurrentAccess(t *testing.T) {
	t.Parallel()

	store := newTestStore(t, WithCapacity(64))

	var wg sync.WaitGroup
	for worker := range 8 {
		wg.Add(1)
		go func(worker int) {
			// exits after its fixed op count completes
			defer wg.Done()
			key := with(baseChoiceKey(), func(k *Key) {
				k.Primitive = PrimitiveScore
				k.Fingerprint = fmt.Sprintf("fp-%d", worker%4)
			})
			result := judgment.Score{Value: float64(worker), Confidence: 0.5}
			for range 100 {
				_ = store.PutScore(key, result)
				_, _ = store.GetScore(key)
			}
		}(worker)
	}
	wg.Wait()

	if store.Len() > 4 {
		t.Errorf("Len() = %d, want at most 4 distinct keys", store.Len())
	}
}

func ExampleNew() {
	store, err := New(WithTTL(10 * time.Minute))
	if err != nil {
		fmt.Println("new store:", err)
		return
	}
	defer store.Close()

	request := judgment.ChoiceRequest{
		Task:    "route the incoming request",
		Options: []judgment.Option{{ID: "search"}, {ID: "summarize"}},
	}
	key := Key{
		Tenant:      "acme",
		Schema:      "router.v2",
		Primitive:   PrimitiveChoice,
		Fingerprint: FingerprintChoice(request),
	}
	result := judgment.Choice{
		Selected: "search",
		Distribution: []judgment.OptionProbability{
			{Option: "search", Probability: 0.7},
			{Option: "summarize", Probability: 0.3},
		},
	}
	if err := store.PutChoice(key, result); err != nil {
		fmt.Println("put:", err)
		return
	}
	if hit, ok := store.GetChoice(key); ok {
		fmt.Println("cache hit:", hit.Selected)
	}
	// Output: cache hit: search
}
