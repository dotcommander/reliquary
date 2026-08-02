package vectors

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
)

func kneeConfigFromEnv(getenv func(string) string) (kneeConfig, error) {
	dimensions, err := kneePositiveInt(kneeDimEnv, getenv(kneeDimEnv), defaultKneeDimensions)
	if err != nil {
		return kneeConfig{}, err
	}
	queries, err := kneePositiveInt(kneeQueriesEnv, getenv(kneeQueriesEnv), defaultKneeQueries)
	if err != nil {
		return kneeConfig{}, err
	}
	concurrency, err := kneePositiveInt(kneeConcurrencyEnv, getenv(kneeConcurrencyEnv), defaultKneeConcurrency)
	if err != nil {
		return kneeConfig{}, err
	}
	sizes, err := kneePositiveIntList(kneeSizesEnv, getenv(kneeSizesEnv), defaultKneeSizes)
	if err != nil {
		return kneeConfig{}, err
	}
	candidateLimit, err := kneePositiveInt(
		kneeCandidatesEnv,
		getenv(kneeCandidatesEnv),
		defaultKneeCandidateLimit,
	)
	if err != nil {
		return kneeConfig{}, err
	}
	strictEnabled, err := kneeBool(kneeStrictEnv, getenv(kneeStrictEnv))
	if err != nil {
		return kneeConfig{}, err
	}
	minRecallAt1, minRecallAt1Set, err := kneeRecallThreshold(kneeMinRecallAt1Env, getenv(kneeMinRecallAt1Env))
	if err != nil {
		return kneeConfig{}, err
	}
	minRecallAt5, minRecallAt5Set, err := kneeRecallThreshold(kneeMinRecallAt5Env, getenv(kneeMinRecallAt5Env))
	if err != nil {
		return kneeConfig{}, err
	}
	maxP95Ratio, maxP95RatioSet, err := kneePositiveFloat(kneeMaxP95RatioEnv, getenv(kneeMaxP95RatioEnv))
	if err != nil {
		return kneeConfig{}, err
	}
	maxP95MS, maxP95MSSet, err := kneePositiveFloat(kneeMaxP95MSEnv, getenv(kneeMaxP95MSEnv))
	if err != nil {
		return kneeConfig{}, err
	}

	thresholdSet := minRecallAt1Set || minRecallAt5Set || maxP95RatioSet || maxP95MSSet
	if strictEnabled && !thresholdSet {
		return kneeConfig{}, fmt.Errorf("%s: strict mode requires at least one approximate threshold", kneeStrictEnv)
	}
	if !strictEnabled && thresholdSet {
		thresholdName := kneeMinRecallAt1Env
		switch {
		case minRecallAt1Set:
			thresholdName = kneeMinRecallAt1Env
		case minRecallAt5Set:
			thresholdName = kneeMinRecallAt5Env
		case maxP95RatioSet:
			thresholdName = kneeMaxP95RatioEnv
		case maxP95MSSet:
			thresholdName = kneeMaxP95MSEnv
		}
		return kneeConfig{}, fmt.Errorf("%s: requires %s=1", thresholdName, kneeStrictEnv)
	}

	return kneeConfig{
		Dimensions:     dimensions,
		Queries:        queries,
		Concurrency:    concurrency,
		Sizes:          sizes,
		CandidateLimit: candidateLimit,
		Strict: kneeThresholds{
			Enabled:      strictEnabled,
			MinRecallAt1: minRecallAt1,
			MinRecallAt5: minRecallAt5,
			MaxP95Ratio:  maxP95Ratio,
			MaxP95MS:     maxP95MS,
		},
	}, nil
}

func kneeBool(name, raw string) (bool, error) {
	switch raw {
	case "", "0":
		return false, nil
	case "1":
		return true, nil
	default:
		return false, fmt.Errorf("%s: must be unset, 0, or 1", name)
	}
}

func kneePositiveInt(name, raw string, fallback int) (int, error) {
	if raw == "" {
		return fallback, nil
	}
	raw = strings.TrimSpace(raw)
	value, err := strconv.ParseInt(raw, 10, strconv.IntSize)
	if err != nil {
		return 0, fmt.Errorf("%s: parse positive base-10 integer: %w", name, err)
	}
	if value <= 0 {
		return 0, fmt.Errorf("%s: must be positive", name)
	}
	return int(value), nil
}

func kneePositiveIntList(name, raw string, fallback []int) ([]int, error) {
	if raw == "" {
		return slices.Clone(fallback), nil
	}
	parts := strings.Split(raw, ",")
	values := make([]int, 0, len(parts))
	seen := make(map[int]struct{}, len(parts))
	for index, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil, fmt.Errorf("%s: element %d is empty", name, index)
		}
		value, err := strconv.ParseInt(part, 10, strconv.IntSize)
		if err != nil {
			return nil, fmt.Errorf("%s: parse element %d as a positive base-10 integer: %w", name, index, err)
		}
		if value <= 0 {
			return nil, fmt.Errorf("%s: element %d must be positive", name, index)
		}
		intValue := int(value)
		if _, exists := seen[intValue]; exists {
			return nil, fmt.Errorf("%s: duplicate value %d", name, intValue)
		}
		seen[intValue] = struct{}{}
		values = append(values, intValue)
	}
	return values, nil
}

func kneeRecallThreshold(name, raw string) (float64, bool, error) {
	value, set, err := kneeOptionalFloat(name, raw)
	if err != nil {
		return 0, false, err
	}
	if set && (value <= 0 || value > 1) {
		return 0, false, fmt.Errorf("%s: must be in (0,1]", name)
	}
	return value, set, nil
}

func kneePositiveFloat(name, raw string) (float64, bool, error) {
	value, set, err := kneeOptionalFloat(name, raw)
	if err != nil {
		return 0, false, err
	}
	if set && value <= 0 {
		return 0, false, fmt.Errorf("%s: must be positive", name)
	}
	return value, set, nil
}

func kneeOptionalFloat(name, raw string) (float64, bool, error) {
	if raw == "" {
		return 0, false, nil
	}
	raw = strings.TrimSpace(raw)
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, false, fmt.Errorf("%s: parse number: %w", name, err)
	}
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, false, fmt.Errorf("%s: must be finite", name)
	}
	return value, true, nil
}
