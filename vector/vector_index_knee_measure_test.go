package vectors

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"time"
)

func kneeGroundTruth(exact *ExactIndex, queries [][]float32) ([][]SearchResult, error) {
	truth := make([][]SearchResult, len(queries))
	for index, query := range queries {
		results, ok := exact.Search(query, kneeResultLimit, kneeMinimumSimilarity)
		if !ok {
			return nil, fmt.Errorf("exact oracle query %d: search failed", index)
		}
		if len(results) == 0 {
			return nil, fmt.Errorf("exact oracle query %d: empty result", index)
		}
		truth[index] = results
	}
	return truth, nil
}

func measureKneeProfile(
	profile ANNProfile,
	search kneeSearch,
	queries [][]float32,
	truth [][]SearchResult,
	concurrency int,
	buildTime time.Duration,
) (kneeResult, error) {
	observations, batchElapsed := runKneeSearches(search, queries, concurrency)
	if batchElapsed <= 0 {
		return kneeResult{}, fmt.Errorf("profile %s batch elapsed: must be positive", profile.ID)
	}
	if len(observations) != len(truth) {
		return kneeResult{}, fmt.Errorf(
			"profile %s observations: got %d, want %d",
			profile.ID,
			len(observations),
			len(truth),
		)
	}

	latencies := make([]time.Duration, len(observations))
	var recallAt1, recallAt5 float64
	var queryErrors int
	for index, observation := range observations {
		latencies[index] = observation.elapsed
		if !observation.ok {
			queryErrors++
			continue
		}
		recallAt1 += kneeRecall(observation.results, truth[index], 1)
		recallAt5 += kneeRecall(observation.results, truth[index], kneeResultLimit)
	}
	latency, err := kneeLatencySummary(latencies)
	if err != nil {
		return kneeResult{}, fmt.Errorf("profile %s latency: %w", profile.ID, err)
	}
	queryCount := float64(len(queries))
	return kneeResult{
		Profile:     profile,
		RecallAt1:   recallAt1 / queryCount,
		RecallAt5:   recallAt5 / queryCount,
		Latency:     latency,
		QPS:         queryCount / batchElapsed.Seconds(),
		BuildMS:     durationMilliseconds(buildTime),
		QueryErrors: queryErrors,
	}, nil
}

func runKneeSearches(search kneeSearch, queries [][]float32, concurrency int) ([]kneeObservation, time.Duration) {
	observations := make([]kneeObservation, len(queries))
	workers := min(concurrency, len(queries))
	if workers <= 1 {
		startBatch := time.Now()
		for index, query := range queries {
			start := time.Now()
			results, ok := search(query)
			observations[index] = kneeObservation{
				results: results,
				ok:      ok,
				elapsed: time.Since(start),
			}
		}
		return observations, time.Since(startBatch)
	}

	jobs := make(chan int)
	startBatch := time.Now()
	var group sync.WaitGroup
	for range workers {
		group.Go(func() {
			for index := range jobs {
				start := time.Now()
				results, ok := search(queries[index])
				observations[index] = kneeObservation{
					results: results,
					ok:      ok,
					elapsed: time.Since(start),
				}
			}
		})
	}
	for index := range queries {
		jobs <- index
	}
	close(jobs)
	group.Wait()
	return observations, time.Since(startBatch)
}

func kneeRecall(results, truth []SearchResult, limit int) float64 {
	truthLimit := min(limit, len(truth))
	if truthLimit == 0 {
		return 0
	}
	wanted := make(map[IndexKey]struct{}, truthLimit)
	for _, result := range truth[:truthLimit] {
		wanted[IndexKey{Group: result.Group, ChunkIndex: result.ChunkIndex}] = struct{}{}
	}
	have := make(map[IndexKey]struct{}, min(limit, len(results)))
	for _, result := range results[:min(limit, len(results))] {
		have[IndexKey{Group: result.Group, ChunkIndex: result.ChunkIndex}] = struct{}{}
	}
	hits := 0
	for key := range have {
		if _, ok := wanted[key]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(wanted))
}

func kneeLatencySummary(samples []time.Duration) (LatencySummary, error) {
	if len(samples) == 0 {
		return LatencySummary{}, fmt.Errorf("samples: empty")
	}
	sorted := slices.Clone(samples)
	slices.Sort(sorted)
	return LatencySummary{
		Samples: len(sorted),
		MinMS:   durationMilliseconds(sorted[0]),
		P50MS:   durationMilliseconds(kneeNearestRank(sorted, 0.50)),
		P95MS:   durationMilliseconds(kneeNearestRank(sorted, 0.95)),
		MaxMS:   durationMilliseconds(sorted[len(sorted)-1]),
	}, nil
}

func kneeNearestRank(sorted []time.Duration, percentile float64) time.Duration {
	rank := int(math.Ceil(percentile*float64(len(sorted)))) - 1
	rank = max(0, min(rank, len(sorted)-1))
	return sorted[rank]
}

func durationMilliseconds(value time.Duration) float64 {
	return float64(value) / float64(time.Millisecond)
}

func formatKneeResults(rows []kneeOutputRow) string {
	var output strings.Builder
	output.WriteString("profile\tvectors\trecall@1\trecall@5\tp50_ms\tp95_ms\tqps\tbuild_ms\tpayload_bytes\terrors\n")
	for _, row := range rows {
		result := row.Result
		fmt.Fprintf(
			&output,
			"%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.1f\t%.3f\t%d\t%d\n",
			result.Profile.ID,
			row.Vectors,
			result.RecallAt1,
			result.RecallAt5,
			result.Latency.P50MS,
			result.Latency.P95MS,
			result.QPS,
			result.BuildMS,
			result.Profile.MemoryEstimate.Bytes,
			result.QueryErrors,
		)
	}
	return output.String()
}
