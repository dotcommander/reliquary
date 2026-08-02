package vectors

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

func validateKneeRun(strict kneeThresholds, results []kneeResult) error {
	var exact, binary *kneeResult
	for index := range results {
		result := &results[index]
		if result.QueryErrors != 0 {
			return fmt.Errorf("profile %s field QueryErrors: got %d, want 0", result.Profile.ID, result.QueryErrors)
		}
		switch result.Profile.Kind {
		case IndexKindExact:
			if exact != nil {
				return fmt.Errorf("profile exact: duplicate result")
			}
			exact = result
		case IndexKindBinary:
			if binary != nil {
				return fmt.Errorf("profile binary: duplicate result")
			}
			binary = result
		}
	}
	if exact == nil {
		return fmt.Errorf("profile exact: missing result")
	}
	if binary == nil {
		return fmt.Errorf("profile binary: missing result")
	}
	if exact.RecallAt1 < 1 {
		return fmt.Errorf("profile %s field RecallAt1: got %.6f, want 1.0", exact.Profile.ID, exact.RecallAt1)
	}
	if exact.RecallAt5 < 1 {
		return fmt.Errorf("profile %s field RecallAt5: got %.6f, want 1.0", exact.Profile.ID, exact.RecallAt5)
	}
	if !strict.Enabled {
		return nil
	}
	if strict.MinRecallAt1 > 0 && binary.RecallAt1 < strict.MinRecallAt1 {
		return fmt.Errorf(
			"profile %s field RecallAt1: observed %.6f < configured threshold %.6f",
			binary.Profile.ID,
			binary.RecallAt1,
			strict.MinRecallAt1,
		)
	}
	if strict.MinRecallAt5 > 0 && binary.RecallAt5 < strict.MinRecallAt5 {
		return fmt.Errorf(
			"profile %s field RecallAt5: observed %.6f < configured threshold %.6f",
			binary.Profile.ID,
			binary.RecallAt5,
			strict.MinRecallAt5,
		)
	}
	if strict.MaxP95Ratio > 0 {
		limit := exact.Latency.P95MS * strict.MaxP95Ratio
		if binary.Latency.P95MS > limit {
			return fmt.Errorf(
				"profile %s field P95MS: observed %.6f > exact %.6f * configured threshold %.6f",
				binary.Profile.ID,
				binary.Latency.P95MS,
				exact.Latency.P95MS,
				strict.MaxP95Ratio,
			)
		}
	}
	if strict.MaxP95MS > 0 && binary.Latency.P95MS > strict.MaxP95MS {
		return fmt.Errorf(
			"profile %s field P95MS: observed %.6f > configured threshold %.6f",
			binary.Profile.ID,
			binary.Latency.P95MS,
			strict.MaxP95MS,
		)
	}
	return nil
}

func TestKneeRunValidation(t *testing.T) {
	t.Parallel()
	exact := kneeResult{
		Profile:   ANNProfile{ID: "exact", Kind: IndexKindExact},
		RecallAt1: 1,
		RecallAt5: 1,
		Latency:   LatencySummary{P95MS: 10},
	}
	binary := kneeResult{
		Profile:   ANNProfile{ID: "binary-candidates-2", Kind: IndexKindBinary},
		RecallAt1: 0.8,
		RecallAt5: 0.9,
		Latency:   LatencySummary{P95MS: 5},
	}
	if err := validateKneeRun(kneeThresholds{}, []kneeResult{exact, binary}); err != nil {
		t.Fatalf("valid non-strict run: %v", err)
	}

	tests := []struct {
		name    string
		results []kneeResult
	}{
		{name: "missing exact", results: []kneeResult{binary}},
		{name: "missing binary", results: []kneeResult{exact}},
		{
			name: "exact query error",
			results: []kneeResult{
				withKneeResult(exact, func(result *kneeResult) { result.QueryErrors = 1 }),
				binary,
			},
		},
		{
			name: "binary query error",
			results: []kneeResult{
				exact,
				withKneeResult(binary, func(result *kneeResult) { result.QueryErrors = 1 }),
			},
		},
		{
			name: "exact recall at one",
			results: []kneeResult{
				withKneeResult(exact, func(result *kneeResult) { result.RecallAt1 = 0.99 }),
				binary,
			},
		},
		{
			name: "exact recall at five",
			results: []kneeResult{
				withKneeResult(exact, func(result *kneeResult) { result.RecallAt5 = 0.99 }),
				binary,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if err := validateKneeRun(kneeThresholds{}, test.results); err == nil {
				t.Fatal("validateKneeRun() error = nil, want error")
			}
		})
	}

	disabledThreshold := kneeThresholds{MinRecallAt1: 0.9}
	if err := validateKneeRun(disabledThreshold, []kneeResult{exact, binary}); err != nil {
		t.Fatalf("disabled binary threshold was applied: %v", err)
	}
	enabledThreshold := disabledThreshold
	enabledThreshold.Enabled = true
	if err := validateKneeRun(enabledThreshold, []kneeResult{exact, binary}); err == nil {
		t.Fatal("enabled binary threshold was not applied")
	}
}

func TestKneeStrictThresholds(t *testing.T) {
	t.Parallel()
	exact := kneeResult{
		Profile:   ANNProfile{ID: "exact", Kind: IndexKindExact},
		RecallAt1: 1,
		RecallAt5: 1,
		Latency:   LatencySummary{P95MS: 10},
	}
	binary := kneeResult{
		Profile:   ANNProfile{ID: "binary-candidates-2", Kind: IndexKindBinary},
		RecallAt1: 0.9,
		RecallAt5: 0.8,
		Latency:   LatencySummary{P95MS: 5},
	}
	equality := kneeThresholds{
		Enabled:      true,
		MinRecallAt1: 0.9,
		MinRecallAt5: 0.8,
		MaxP95Ratio:  0.5,
		MaxP95MS:     5,
	}
	if err := validateKneeRun(equality, []kneeResult{exact, binary}); err != nil {
		t.Fatalf("equality thresholds: %v", err)
	}

	tests := []struct {
		name      string
		threshold kneeThresholds
	}{
		{
			name:      "recall at one just below",
			threshold: kneeThresholds{Enabled: true, MinRecallAt1: math.Nextafter(0.9, 1)},
		},
		{
			name:      "recall at five just below",
			threshold: kneeThresholds{Enabled: true, MinRecallAt5: math.Nextafter(0.8, 1)},
		},
		{
			name:      "p95 ratio just above",
			threshold: kneeThresholds{Enabled: true, MaxP95Ratio: math.Nextafter(0.5, 0)},
		},
		{
			name:      "p95 milliseconds just above",
			threshold: kneeThresholds{Enabled: true, MaxP95MS: math.Nextafter(5, 0)},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			err := validateKneeRun(test.threshold, []kneeResult{exact, binary})
			if err == nil ||
				!strings.Contains(err.Error(), binary.Profile.ID) ||
				!strings.Contains(err.Error(), "configured threshold") {
				t.Fatalf("error = %v, want profile and configured threshold", err)
			}
		})
	}
}

func withKneeResult(result kneeResult, mutate func(*kneeResult)) kneeResult {
	mutate(&result)
	return result
}
