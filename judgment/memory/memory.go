// Package memory provides the exact-hit judgment memory for the
// Embed→Judge stack: validated judgment.Choice, judgment.Score, and
// judgment.Noul results cached under four-part keys (tenant, schema,
// primitive, request fingerprint) and evicted after caller-configured
// lifetimes by github.com/jellydator/ttlcache/v3.
//
// Lookup is exact-match only: an entry is shared when the tenant, schema
// version, primitive, and fingerprint of the lookup key are all identical
// to the stored key. Semantic near-duplicate routing is out of scope. The
// store does not coalesce concurrent misses for one key; callers that need
// fill coalescing should provide it around the store.
//
// The zero value is not usable; construct with New. New starts one
// background goroutine that evicts expired entries; stop it with Close.
// Results are trusted as validated: pass them through the judgment
// package's ValidateChoice, ValidateScore, and ValidateNoul before
// storing.
package memory

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/dotcommander/reliquary/internal/hash"
	"github.com/dotcommander/reliquary/judgment"
	"github.com/jellydator/ttlcache/v3"
)

// Primitive names which judgment primitive a cached entry holds.
type Primitive string

const (
	// PrimitiveChoice marks a cached judgment.Choice.
	PrimitiveChoice Primitive = "choice"
	// PrimitiveScore marks a cached judgment.Score.
	PrimitiveScore Primitive = "score"
	// PrimitiveNoul marks a cached judgment.Noul.
	PrimitiveNoul Primitive = "noul"
)

// Valid reports whether p is one of the defined judgment primitives.
func (p Primitive) Valid() bool {
	switch p {
	case PrimitiveChoice, PrimitiveScore, PrimitiveNoul:
		return true
	default:
		return false
	}
}

// Key identifies one cacheable judgment. Entries are shared only when
// Tenant, Schema, Primitive, and Fingerprint are all identical.
type Key struct {
	// Tenant isolates one customer or deployment's judgments.
	Tenant string
	// Schema versions the judging pipeline; changing prompts, models, or
	// option sets bumps the schema and invalidates older entries.
	Schema string
	// Primitive is the judgment primitive the entry holds.
	Primitive Primitive
	// Fingerprint identifies the exact request content, typically from
	// FingerprintChoice, FingerprintScore, or FingerprintNoul.
	Fingerprint string
}

// ErrInvalidKey reports a Put key that cannot identify a judgment: a blank
// tenant, schema, or fingerprint, or a primitive that is unknown or does
// not match the stored result type.
var ErrInvalidKey = errors.New("judgment/memory: invalid cache key")

// Option configures a Store at construction.
type Option func(*config)

type config struct {
	ttl               time.Duration
	capacity          uint64
	disableTouchOnHit bool
}

// WithTTL sets the default lifetime of cached judgments. It is required
// and must be positive; entries otherwise live forever in memory.
func WithTTL(lifetime time.Duration) Option {
	return func(c *config) { c.ttl = lifetime }
}

// WithCapacity limits the store to at most limit entries, evicting the
// least recently used entry when full. Zero (the default) is unlimited.
func WithCapacity(limit uint64) Option {
	return func(c *config) { c.capacity = limit }
}

// WithDisableTouchOnHit stops hits from extending an entry's lifetime.
func WithDisableTouchOnHit() Option {
	return func(c *config) { c.disableTouchOnHit = true }
}

// PutOption tunes one insertion.
type PutOption func(*putConfig)

type putConfig struct {
	ttl time.Duration
}

// WithEntryTTL caches this judgment for the given lifetime instead of the
// store default. Non-positive lifetimes are ignored.
func WithEntryTTL(lifetime time.Duration) PutOption {
	return func(c *putConfig) {
		if lifetime > 0 {
			c.ttl = lifetime
		}
	}
}

// Metrics reports lifetime store counters.
type Metrics struct {
	Insertions uint64
	Updates    uint64
	Hits       uint64
	Misses     uint64
	Evictions  uint64
}

// entry is the cached value. Exactly one result field is populated,
// matching the primitive the key was stored under.
type entry struct {
	choice *judgment.Choice
	score  *judgment.Score
	noul   *judgment.Noul
}

// Store is an exact-hit judgment cache backed by ttlcache.
type Store struct {
	cache     *ttlcache.Cache[Key, entry]
	closeOnce sync.Once
}

// New creates a Store and starts its background eviction goroutine, which
// exits when Close is called.
func New(opts ...Option) (*Store, error) {
	var cfg config
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.ttl <= 0 {
		return nil, fmt.Errorf("judgment/memory: WithTTL lifetime must be positive, got %v", cfg.ttl)
	}
	cacheOpts := []ttlcache.Option[Key, entry]{
		ttlcache.WithTTL[Key, entry](cfg.ttl),
	}
	if cfg.capacity > 0 {
		cacheOpts = append(cacheOpts, ttlcache.WithCapacity[Key, entry](cfg.capacity))
	}
	if cfg.disableTouchOnHit {
		cacheOpts = append(cacheOpts, ttlcache.WithDisableTouchOnHit[Key, entry]())
	}
	cache := ttlcache.New[Key, entry](cacheOpts...)
	store := &Store{cache: cache}
	// Cleanup goroutine: ttlcache.Start blocks until Close calls Stop.
	go cache.Start()
	return store, nil
}

// GetChoice returns the cached choice for key. ok is false when no entry
// exists, the entry expired, or the entry holds another primitive. Unless
// disabled, a hit extends the entry's lifetime.
func (s *Store) GetChoice(key Key) (judgment.Choice, bool) {
	item := s.cache.Get(key)
	if item == nil {
		return judgment.Choice{}, false
	}
	if stored := item.Value().choice; stored != nil {
		return copyChoice(*stored), true
	}
	return judgment.Choice{}, false
}

// PutChoice caches result under key with the store default lifetime unless
// an entry lifetime is given. Keys must carry PrimitiveChoice.
func (s *Store) PutChoice(key Key, result judgment.Choice, opts ...PutOption) error {
	if err := validateKey(key, PrimitiveChoice); err != nil {
		return err
	}
	stored := copyChoice(result)
	s.cache.Set(key, entry{choice: &stored}, putTTL(opts))
	return nil
}

// GetScore returns the cached score for key with GetChoice's miss rules.
func (s *Store) GetScore(key Key) (judgment.Score, bool) {
	item := s.cache.Get(key)
	if item == nil {
		return judgment.Score{}, false
	}
	if stored := item.Value().score; stored != nil {
		return *stored, true
	}
	return judgment.Score{}, false
}

// PutScore caches result under key. Keys must carry PrimitiveScore.
func (s *Store) PutScore(key Key, result judgment.Score, opts ...PutOption) error {
	if err := validateKey(key, PrimitiveScore); err != nil {
		return err
	}
	stored := result
	s.cache.Set(key, entry{score: &stored}, putTTL(opts))
	return nil
}

// GetNoul returns the cached gate answer for key with GetChoice's miss
// rules.
func (s *Store) GetNoul(key Key) (judgment.Noul, bool) {
	item := s.cache.Get(key)
	if item == nil {
		return judgment.Noul{}, false
	}
	if stored := item.Value().noul; stored != nil {
		return *stored, true
	}
	return judgment.Noul{}, false
}

// PutNoul caches result under key. Keys must carry PrimitiveNoul.
func (s *Store) PutNoul(key Key, result judgment.Noul, opts ...PutOption) error {
	if err := validateKey(key, PrimitiveNoul); err != nil {
		return err
	}
	stored := result
	s.cache.Set(key, entry{noul: &stored}, putTTL(opts))
	return nil
}

// Delete removes the entry for key, whether or not it has expired.
func (s *Store) Delete(key Key) {
	s.cache.Delete(key)
}

// Len reports the number of unexpired entries.
func (s *Store) Len() int {
	return s.cache.Len()
}

// Metrics returns the store's lifetime counters.
func (s *Store) Metrics() Metrics {
	m := s.cache.Metrics()
	return Metrics{
		Insertions: m.Insertions,
		Updates:    m.Updates,
		Hits:       m.Hits,
		Misses:     m.Misses,
		Evictions:  m.Evictions,
	}
}

// Close stops the background eviction goroutine. Entries stay readable
// until they expire lazily. Close is idempotent and always returns nil.
func (s *Store) Close() error {
	s.closeOnce.Do(func() { s.cache.Stop() })
	return nil
}

func validateKey(key Key, primitive Primitive) error {
	if strings.TrimSpace(key.Tenant) == "" {
		return fmt.Errorf("%w: tenant must not be blank", ErrInvalidKey)
	}
	if strings.TrimSpace(key.Schema) == "" {
		return fmt.Errorf("%w: schema must not be blank", ErrInvalidKey)
	}
	if !key.Primitive.Valid() {
		return fmt.Errorf("%w: unknown primitive %q", ErrInvalidKey, key.Primitive)
	}
	if key.Primitive != primitive {
		return fmt.Errorf("%w: key primitive %q does not match the %q result", ErrInvalidKey, key.Primitive, primitive)
	}
	if strings.TrimSpace(key.Fingerprint) == "" {
		return fmt.Errorf("%w: fingerprint must not be blank", ErrInvalidKey)
	}
	return nil
}

func putTTL(opts []PutOption) time.Duration {
	var cfg putConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.ttl > 0 {
		return cfg.ttl
	}
	return ttlcache.DefaultTTL
}

// copyChoice detaches the distribution slice so callers and the store
// cannot observe each other's mutations.
func copyChoice(result judgment.Choice) judgment.Choice {
	if result.Distribution == nil {
		return result
	}
	distribution := make([]judgment.OptionProbability, len(result.Distribution))
	copy(distribution, result.Distribution)
	result.Distribution = distribution
	return result
}

// FingerprintChoice returns a deterministic SHA-256 fingerprint of the
// complete choice request: task, options (ID and description, in order),
// and context entries (in order). Identical requests fingerprint
// identically; any difference yields a different fingerprint. The request
// itself is not validated.
func FingerprintChoice(request judgment.ChoiceRequest) string {
	parts := make([]hash.IdentityPart, 0, 1+len(request.Options)+len(request.Context))
	parts = append(parts, hash.IdentityPart{Kind: "judgment.choice", Value: request.Task})
	for _, option := range request.Options {
		parts = append(parts, hash.IdentityPart{Kind: "option", ID: option.ID, Value: option.Description})
	}
	parts = appendContext(parts, request.Context)
	return hash.HashIdentity(parts...).Hex
}

// FingerprintScore returns a deterministic SHA-256 fingerprint of the
// complete score request: task, target, and context entries (in order).
func FingerprintScore(request judgment.ScoreRequest) string {
	parts := make([]hash.IdentityPart, 0, 2+len(request.Context))
	parts = append(parts,
		hash.IdentityPart{Kind: "judgment.score", Value: request.Task},
		hash.IdentityPart{Kind: "target", Value: request.Target},
	)
	parts = appendContext(parts, request.Context)
	return hash.HashIdentity(parts...).Hex
}

// FingerprintNoul returns a deterministic SHA-256 fingerprint of the
// complete noul request: question and context entries (in order).
func FingerprintNoul(request judgment.NoulRequest) string {
	parts := make([]hash.IdentityPart, 0, 1+len(request.Context))
	parts = append(parts, hash.IdentityPart{Kind: "judgment.noul", Value: request.Question})
	parts = appendContext(parts, request.Context)
	return hash.HashIdentity(parts...).Hex
}

func appendContext(parts []hash.IdentityPart, context []string) []hash.IdentityPart {
	for i, fact := range context {
		parts = append(parts, hash.IdentityPart{Kind: "context", ID: strconv.Itoa(i), Value: fact})
	}
	return parts
}
