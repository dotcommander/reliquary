package vectors

import (
	"os"
	"strings"
	"testing"
	"time"
)

const (
	kneeEnableEnv       = "RELIQUARY_VECTOR_INDEX_KNEE"
	kneeDimEnv          = "RELIQUARY_VECTOR_INDEX_KNEE_DIM"
	kneeQueriesEnv      = "RELIQUARY_VECTOR_INDEX_KNEE_QUERIES"
	kneeConcurrencyEnv  = "RELIQUARY_VECTOR_INDEX_KNEE_CONCURRENCY"
	kneeSizesEnv        = "RELIQUARY_VECTOR_INDEX_KNEE_SIZES"
	kneeCandidatesEnv   = "RELIQUARY_VECTOR_INDEX_KNEE_CANDIDATES"
	kneeStrictEnv       = "RELIQUARY_VECTOR_INDEX_KNEE_STRICT"
	kneeMinRecallAt1Env = "RELIQUARY_VECTOR_INDEX_KNEE_APPROX_MIN_RECALL_AT_1"
	kneeMinRecallAt5Env = "RELIQUARY_VECTOR_INDEX_KNEE_APPROX_MIN_RECALL_AT_5"
	kneeMaxP95RatioEnv  = "RELIQUARY_VECTOR_INDEX_KNEE_APPROX_MAX_P95_RATIO"
	kneeMaxP95MSEnv     = "RELIQUARY_VECTOR_INDEX_KNEE_APPROX_MAX_P95_MS"

	defaultKneeDimensions     = 512
	defaultKneeQueries        = 25
	defaultKneeConcurrency    = 1
	defaultKneeCandidateLimit = 100
	kneeResultLimit           = 5
	kneeMinimumSimilarity     = -1
	kneeMemoryLabel           = "payload_bytes"
)

var defaultKneeSizes = []int{1000, 3000}

type kneeSearch func([]float32) ([]SearchResult, bool)

type kneeThresholds struct {
	Enabled      bool
	MinRecallAt1 float64
	MinRecallAt5 float64
	MaxP95Ratio  float64
	MaxP95MS     float64
}

type kneeConfig struct {
	Dimensions     int
	Queries        int
	Concurrency    int
	Sizes          []int
	CandidateLimit int
	Strict         kneeThresholds
}

type kneeResult struct {
	Profile     ANNProfile
	RecallAt1   float64
	RecallAt5   float64
	Latency     LatencySummary
	QPS         float64
	BuildMS     float64
	QueryErrors int
}

type kneeOutputRow struct {
	Vectors int
	Result  kneeResult
}

type kneeFixture struct {
	vectors      [][]float32
	blobs        [][]byte
	arena        []byte
	chunks       []IndexChunk
	groups       []string
	chunkIndexes []int
}

type kneeObservation struct {
	results []SearchResult
	ok      bool
	elapsed time.Duration
}

func TestVectorIndexKneeHarness(t *testing.T) {
	t.Parallel()
	enabled, err := kneeBool(kneeEnableEnv, os.Getenv(kneeEnableEnv))
	if err != nil {
		t.Fatal(err)
	}
	if !enabled {
		t.Skip("set RELIQUARY_VECTOR_INDEX_KNEE=1 to run vector index knee harness")
	}

	config, err := kneeConfigFromEnv(os.Getenv)
	if err != nil {
		t.Fatal(err)
	}

	rows := make([]kneeOutputRow, 0, len(config.Sizes)*2)
	for _, vectorCount := range config.Sizes {
		exactProfile, err := newKneeProfile(IndexKindExact, vectorCount, config.Dimensions, 0)
		if err != nil {
			t.Fatal(err)
		}
		binaryProfile, err := newKneeProfile(
			IndexKindBinary,
			vectorCount,
			config.Dimensions,
			config.CandidateLimit,
		)
		if err != nil {
			t.Fatal(err)
		}

		fixture := newKneeFixture(vectorCount, config.Dimensions)
		queries := newKneeQueries(config.Queries, config.Dimensions)

		exactStart := time.Now()
		exact, exactReport := NewExactIndexChecked(config.Dimensions, fixture.chunks, fixture.arena)
		exactBuild := time.Since(exactStart)
		if err := validateKneeBuildReport("exact", exactReport, vectorCount); err != nil {
			t.Fatal(err)
		}

		binaryStart := time.Now()
		binary, binaryReport := NewBinaryIndexChecked(
			fixture.blobs,
			fixture.groups,
			fixture.chunkIndexes,
			config.Dimensions,
		)
		binaryBuild := time.Since(binaryStart)
		if err := validateKneeBuildReport("binary", binaryReport, vectorCount); err != nil {
			t.Fatal(err)
		}

		truth, err := kneeGroundTruth(exact, queries)
		if err != nil {
			t.Fatalf("vector count %d: %v", vectorCount, err)
		}

		exactSearch := func(query []float32) ([]SearchResult, bool) {
			return exact.Search(query, kneeResultLimit, kneeMinimumSimilarity)
		}
		exactResult, err := measureKneeProfile(
			exactProfile,
			exactSearch,
			queries,
			truth,
			config.Concurrency,
			exactBuild,
		)
		if err != nil {
			t.Fatalf("measure exact profile: %v", err)
		}

		binarySearch := func(query []float32) ([]SearchResult, bool) {
			candidates, ok := binary.SearchCandidatesLimit(query, config.CandidateLimit)
			if !ok {
				return nil, false
			}
			keys := make([]IndexKey, len(candidates))
			for index, candidate := range candidates {
				keys[index] = IndexKey{Group: candidate.Group, ChunkIndex: candidate.ChunkIndex}
			}
			return exact.SearchKeys(query, kneeResultLimit, kneeMinimumSimilarity, keys)
		}
		binaryResult, err := measureKneeProfile(
			binaryProfile,
			binarySearch,
			queries,
			truth,
			config.Concurrency,
			binaryBuild,
		)
		if err != nil {
			t.Fatalf("measure binary profile: %v", err)
		}

		results := []kneeResult{exactResult, binaryResult}
		if err := validateKneeRun(config.Strict, results); err != nil {
			t.Fatalf("vector count %d: %v", vectorCount, err)
		}
		rows = append(rows,
			kneeOutputRow{Vectors: vectorCount, Result: exactResult},
			kneeOutputRow{Vectors: vectorCount, Result: binaryResult},
		)
	}

	for _, line := range strings.Split(strings.TrimSuffix(formatKneeResults(rows), "\n"), "\n") {
		t.Log(line)
	}
}
