package vectors

import (
	"fmt"
	"sync/atomic"
	"testing"
	"time"
)

func TestKneeRecall(t *testing.T) {
	t.Parallel()
	result := func(group string, chunk int) SearchResult {
		return SearchResult{Group: group, ChunkIndex: chunk}
	}
	tests := []struct {
		name    string
		results []SearchResult
		truth   []SearchResult
		limit   int
		want    float64
	}{
		{
			name:    "exact identity",
			results: []SearchResult{result("a", 0), result("b", 0)},
			truth:   []SearchResult{result("a", 0), result("b", 0)},
			limit:   2,
			want:    1,
		},
		{
			name:    "partial overlap independent of order",
			results: []SearchResult{result("c", 0), result("a", 0)},
			truth:   []SearchResult{result("a", 0), result("b", 0)},
			limit:   2,
			want:    0.5,
		},
		{
			name:    "duplicate result counts once",
			results: []SearchResult{result("a", 0), result("a", 0)},
			truth:   []SearchResult{result("a", 0), result("b", 0)},
			limit:   2,
			want:    0.5,
		},
		{
			name:    "duplicate oracle identity counts once",
			results: []SearchResult{result("a", 0)},
			truth:   []SearchResult{result("a", 0), result("a", 0)},
			limit:   2,
			want:    1,
		},
		{
			name:    "corpus smaller than five",
			results: []SearchResult{result("a", 0)},
			truth:   []SearchResult{result("a", 0)},
			limit:   5,
			want:    1,
		},
		{
			name:    "empty oracle",
			results: []SearchResult{result("a", 0)},
			limit:   5,
			want:    0,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if got := kneeRecall(test.results, test.truth, test.limit); got != test.want {
				t.Fatalf("kneeRecall() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestKneeNearestRankPercentiles(t *testing.T) {
	t.Parallel()
	if _, err := kneeLatencySummary(nil); err == nil {
		t.Fatal("kneeLatencySummary(nil) error = nil, want error")
	}
	tests := []struct {
		name    string
		samples []time.Duration
		wantMin float64
		wantP50 float64
		wantP95 float64
		wantMax float64
	}{
		{
			name:    "singleton",
			samples: []time.Duration{5 * time.Millisecond},
			wantMin: 5,
			wantP50: 5,
			wantP95: 5,
			wantMax: 5,
		},
		{
			name: "even",
			samples: []time.Duration{
				4 * time.Millisecond,
				1 * time.Millisecond,
				3 * time.Millisecond,
				2 * time.Millisecond,
			},
			wantMin: 1,
			wantP50: 2,
			wantP95: 4,
			wantMax: 4,
		},
		{
			name: "odd",
			samples: []time.Duration{
				5 * time.Millisecond,
				1 * time.Millisecond,
				4 * time.Millisecond,
				2 * time.Millisecond,
				3 * time.Millisecond,
			},
			wantMin: 1,
			wantP50: 3,
			wantP95: 5,
			wantMax: 5,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := kneeLatencySummary(test.samples)
			if err != nil {
				t.Fatalf("kneeLatencySummary(): %v", err)
			}
			if got.Samples != len(test.samples) ||
				got.MinMS != test.wantMin ||
				got.P50MS != test.wantP50 ||
				got.P95MS != test.wantP95 ||
				got.MaxMS != test.wantMax {
				t.Fatalf("summary = %#v", got)
			}
		})
	}
}

func TestKneeFormatResults(t *testing.T) {
	t.Parallel()
	rows := []kneeOutputRow{
		{
			Vectors: 10,
			Result: kneeResult{
				Profile: ANNProfile{
					ID:             "exact",
					MemoryEstimate: MemoryEstimate{Bytes: 80, Label: kneeMemoryLabel},
				},
				RecallAt1: 1,
				RecallAt5: 1,
				Latency:   LatencySummary{P50MS: 1.25, P95MS: 2.5},
				QPS:       100.25,
				BuildMS:   3.75,
			},
		},
		{
			Vectors: 10,
			Result: kneeResult{
				Profile: ANNProfile{
					ID:             "binary-candidates-2",
					MemoryEstimate: MemoryEstimate{Bytes: 48, Label: kneeMemoryLabel},
				},
				RecallAt1:   0.5,
				RecallAt5:   0.75,
				Latency:     LatencySummary{P50MS: 0.25, P95MS: 0.5},
				QPS:         250.75,
				BuildMS:     1.125,
				QueryErrors: 1,
			},
		},
	}
	want := "" +
		"profile\tvectors\trecall@1\trecall@5\tp50_ms\tp95_ms\tqps\tbuild_ms\tpayload_bytes\terrors\n" +
		"exact\t10\t1.000\t1.000\t1.250\t2.500\t100.2\t3.750\t80\t0\n" +
		"binary-candidates-2\t10\t0.500\t0.750\t0.250\t0.500\t250.8\t1.125\t48\t1\n"
	if got := formatKneeResults(rows); got != want {
		t.Fatalf("formatKneeResults() =\n%s\nwant:\n%s", got, want)
	}
}

func TestKneeConcurrentSearch(t *testing.T) {
	t.Parallel()
	const (
		queryCount = 12
		workers    = 3
	)
	queries := make([][]float32, queryCount)
	truth := make([][]SearchResult, queryCount)
	for index := range queryCount {
		queries[index] = []float32{float32(index)}
		truth[index] = []SearchResult{{Group: fmt.Sprintf("row-%02d", index)}}
	}

	var active, maximum atomic.Int64
	counts := make([]atomic.Int64, queryCount)
	search := func(query []float32) ([]SearchResult, bool) {
		index := int(query[0])
		counts[index].Add(1)
		now := active.Add(1)
		for {
			seen := maximum.Load()
			if now <= seen || maximum.CompareAndSwap(seen, now) {
				break
			}
		}
		defer active.Add(-1)
		if index == 5 {
			return nil, false
		}
		return []SearchResult{{Group: fmt.Sprintf("row-%02d", index)}}, true
	}
	profile := ANNProfile{ID: "test", Kind: IndexKindBinary}
	result, err := measureKneeProfile(profile, search, queries, truth, workers, time.Millisecond)
	if err != nil {
		t.Fatalf("measureKneeProfile(): %v", err)
	}
	if result.QueryErrors != 1 {
		t.Fatalf("QueryErrors = %d, want 1", result.QueryErrors)
	}
	if maximum.Load() > workers {
		t.Fatalf("maximum workers = %d, want <= %d", maximum.Load(), workers)
	}
	for index := range counts {
		if counts[index].Load() != 1 {
			t.Fatalf("query %d calls = %d, want 1", index, counts[index].Load())
		}
	}

	observations, _ := runKneeSearches(
		func(query []float32) ([]SearchResult, bool) {
			index := int(query[0])
			return []SearchResult{{Group: fmt.Sprintf("row-%02d", index)}}, true
		},
		queries,
		queryCount+5,
	)
	for index, observation := range observations {
		if len(observation.results) != 1 ||
			observation.results[0].Group != fmt.Sprintf("row-%02d", index) {
			t.Fatalf("observation %d = %#v, want original query order", index, observation)
		}
	}
}
