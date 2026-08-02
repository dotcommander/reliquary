package vectors

import (
	"slices"
	"strings"
	"testing"
)

func TestKneeConfigDefaultsAndOverrides(t *testing.T) {
	t.Parallel()
	defaults, err := kneeConfigFromEnv(func(string) string { return "" })
	if err != nil {
		t.Fatalf("kneeConfigFromEnv(defaults): %v", err)
	}
	if defaults.Dimensions != defaultKneeDimensions ||
		defaults.Queries != defaultKneeQueries ||
		defaults.Concurrency != defaultKneeConcurrency ||
		defaults.CandidateLimit != defaultKneeCandidateLimit ||
		!slices.Equal(defaults.Sizes, defaultKneeSizes) ||
		defaults.Strict.Enabled {
		t.Fatalf("default config = %#v", defaults)
	}
	defaults.Sizes[0] = 99
	if defaultKneeSizes[0] != 1000 {
		t.Fatal("default sizes share mutable storage with parsed config")
	}

	values := map[string]string{
		kneeDimEnv:          "64",
		kneeQueriesEnv:      "17",
		kneeConcurrencyEnv:  "4",
		kneeSizesEnv:        "10, 20",
		kneeCandidatesEnv:   "8",
		kneeStrictEnv:       "1",
		kneeMinRecallAt1Env: "0.99",
		kneeMinRecallAt5Env: "0.98",
		kneeMaxP95RatioEnv:  "0.90",
		kneeMaxP95MSEnv:     "12.5",
	}
	overrides, err := kneeConfigFromEnv(func(name string) string { return values[name] })
	if err != nil {
		t.Fatalf("kneeConfigFromEnv(overrides): %v", err)
	}
	if overrides.Dimensions != 64 ||
		overrides.Queries != 17 ||
		overrides.Concurrency != 4 ||
		overrides.CandidateLimit != 8 ||
		!slices.Equal(overrides.Sizes, []int{10, 20}) ||
		overrides.Strict != (kneeThresholds{
			Enabled:      true,
			MinRecallAt1: 0.99,
			MinRecallAt5: 0.98,
			MaxP95Ratio:  0.90,
			MaxP95MS:     12.5,
		}) {
		t.Fatalf("override config = %#v", overrides)
	}
}

func TestKneeConfigRejectsInvalidValues(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		values  map[string]string
		wantEnv string
	}{
		{name: "enable flag", values: map[string]string{kneeEnableEnv: "true"}, wantEnv: kneeEnableEnv},
		{name: "enable whitespace", values: map[string]string{kneeEnableEnv: " "}, wantEnv: kneeEnableEnv},
		{name: "dimension syntax", values: map[string]string{kneeDimEnv: "1.5"}, wantEnv: kneeDimEnv},
		{name: "dimension whitespace", values: map[string]string{kneeDimEnv: " "}, wantEnv: kneeDimEnv},
		{name: "dimension zero", values: map[string]string{kneeDimEnv: "0"}, wantEnv: kneeDimEnv},
		{name: "queries negative", values: map[string]string{kneeQueriesEnv: "-1"}, wantEnv: kneeQueriesEnv},
		{name: "concurrency syntax", values: map[string]string{kneeConcurrencyEnv: "x"}, wantEnv: kneeConcurrencyEnv},
		{name: "candidates zero", values: map[string]string{kneeCandidatesEnv: "0"}, wantEnv: kneeCandidatesEnv},
		{name: "sizes empty element", values: map[string]string{kneeSizesEnv: "1,,2"}, wantEnv: kneeSizesEnv},
		{name: "sizes whitespace element", values: map[string]string{kneeSizesEnv: "1, ,2"}, wantEnv: kneeSizesEnv},
		{name: "sizes whitespace only", values: map[string]string{kneeSizesEnv: " "}, wantEnv: kneeSizesEnv},
		{name: "sizes trailing comma", values: map[string]string{kneeSizesEnv: "1,2,"}, wantEnv: kneeSizesEnv},
		{name: "sizes sign only", values: map[string]string{kneeSizesEnv: "1,+"}, wantEnv: kneeSizesEnv},
		{name: "sizes duplicate", values: map[string]string{kneeSizesEnv: "1,2,1"}, wantEnv: kneeSizesEnv},
		{name: "sizes nonpositive", values: map[string]string{kneeSizesEnv: "-1"}, wantEnv: kneeSizesEnv},
		{name: "strict flag", values: map[string]string{kneeStrictEnv: "yes"}, wantEnv: kneeStrictEnv},
		{
			name:    "strict without threshold",
			values:  map[string]string{kneeStrictEnv: "1"},
			wantEnv: kneeStrictEnv,
		},
		{
			name:    "threshold without strict",
			values:  map[string]string{kneeMinRecallAt1Env: "0.9"},
			wantEnv: kneeMinRecallAt1Env,
		},
		{
			name: "recall zero",
			values: map[string]string{
				kneeStrictEnv:       "1",
				kneeMinRecallAt1Env: "0",
			},
			wantEnv: kneeMinRecallAt1Env,
		},
		{
			name: "recall above one",
			values: map[string]string{
				kneeStrictEnv:       "1",
				kneeMinRecallAt5Env: "1.1",
			},
			wantEnv: kneeMinRecallAt5Env,
		},
		{
			name: "recall nonfinite",
			values: map[string]string{
				kneeStrictEnv:       "1",
				kneeMinRecallAt1Env: "NaN",
			},
			wantEnv: kneeMinRecallAt1Env,
		},
		{
			name: "ratio nonpositive",
			values: map[string]string{
				kneeStrictEnv:      "1",
				kneeMaxP95RatioEnv: "-1",
			},
			wantEnv: kneeMaxP95RatioEnv,
		},
		{
			name: "milliseconds nonfinite",
			values: map[string]string{
				kneeStrictEnv:   "1",
				kneeMaxP95MSEnv: "+Inf",
			},
			wantEnv: kneeMaxP95MSEnv,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			getenv := func(name string) string { return test.values[name] }
			var err error
			if test.wantEnv == kneeEnableEnv {
				_, err = kneeBool(kneeEnableEnv, getenv(kneeEnableEnv))
			} else {
				_, err = kneeConfigFromEnv(getenv)
			}
			if err == nil || !strings.Contains(err.Error(), test.wantEnv) {
				t.Fatalf("error = %v, want failure naming %s", err, test.wantEnv)
			}
		})
	}
}
