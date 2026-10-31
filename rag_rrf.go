package reliquary

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/internal/indexutil"
	pipelinelexical "github.com/dotcommander/reliquary/pipeline/lexical"
	"github.com/dotcommander/reliquary/retrieval"
)

func (a *App) searchEmbeddedRRF(ctx context.Context, query string, vector embedding.Vector, cfg searchConfig) ([]*retrieval.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	vectorItems, err := a.index.Search(ctx, IndexQuery{
		Identity: a.indexIdentity,
		Vector:   slices.Clone(vector),
		Limit:    cfg.candidateLimit,
		Filter:   cloneSearchFilter(cfg.filter),
	})
	if err != nil {
		return nil, fmt.Errorf("reliquary search vector lane: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	vectorRanked, vectorByID, err := cloneRRFLane("vector", vectorItems)
	if err != nil {
		return nil, err
	}
	lexicalItems, err := a.index.Search(ctx, IndexQuery{
		Identity: a.indexIdentity,
		Text:     query,
		Limit:    cfg.candidateLimit,
		Filter:   cloneSearchFilter(cfg.filter),
	})
	if err != nil {
		return nil, fmt.Errorf("reliquary search lexical lane: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	lexicalRanked, lexicalByID, err := cloneRRFLane("lexical", lexicalItems)
	if err != nil {
		return nil, err
	}
	inputs := make([]pipelinelexical.FusionInput, 0, 2)
	nonemptyLaneCount := 0
	if len(vectorRanked) > 0 {
		inputs = append(inputs, pipelinelexical.FusionInput{Source: "vector", Candidates: vectorRanked})
		nonemptyLaneCount++
	}
	if len(lexicalRanked) > 0 {
		inputs = append(inputs, pipelinelexical.FusionInput{Source: "lexical", Candidates: lexicalRanked})
		nonemptyLaneCount++
	}
	if nonemptyLaneCount == 0 {
		return nil, nil
	}

	fused := pipelinelexical.FuseRRFByID(inputs, pipelinelexical.FusionOptions{K: cfg.rrfK})
	ranked := make([]*retrieval.Result, 0, len(fused))
	rrfScores := make(map[string]float64, len(fused))
	normalize := (cfg.rrfK + 1) / float64(nonemptyLaneCount)
	for _, candidate := range fused {
		item := vectorByID[candidate.ID]
		if item == nil {
			item = lexicalByID[candidate.ID]
		}
		if item == nil {
			return nil, fmt.Errorf("reliquary search RRF: missing payload for result %q", candidate.ID)
		}
		rrfScores[candidate.ID] = candidate.Score * normalize
		ranked = append(ranked, item)
	}

	// RRF owns the order and fused score; hybrid scoring supplies diagnostics.
	// Only explained searches need the hybrid sort and trace ranks.
	var explanations explanationMap
	if cfg.explain {
		ranked, explanations = scoreCandidates(a.weights, vector, query, ranked, true, false)
		byID := make(map[string]*retrieval.Result, len(ranked))
		for _, item := range ranked {
			byID[item.ID] = item
		}
		for i, candidate := range fused {
			ranked[i] = byID[candidate.ID]
		}
	} else {
		retrieval.NewScorer(a.weights).ScoreBatch(retrieval.EmbeddingVector(vector), query, ranked)
	}
	var vectorRanks, lexicalRanks map[string]int
	if cfg.explain {
		vectorRanks = rrfRanks(vectorRanked)
		lexicalRanks = rrfRanks(lexicalRanked)
	}
	for i, candidate := range fused {
		ranked[i].CombinedScore = rrfScores[candidate.ID]
		if explanation := explanations[ranked[i]]; explanation != nil {
			vectorRank := vectorRanks[candidate.ID]
			lexicalRank := lexicalRanks[candidate.ID]
			explanation.RRF = &retrieval.RRFExplanation{
				K:                   cfg.rrfK,
				VectorRank:          vectorRank,
				LexicalRank:         lexicalRank,
				VectorContribution:  normalizedRRFContribution(cfg.rrfK, vectorRank, normalize),
				LexicalContribution: normalizedRRFContribution(cfg.rrfK, lexicalRank, normalize),
				FusedScore:          rrfScores[candidate.ID],
				FusedRank:           i + 1,
			}
		}
	}
	return finishSearch(ctx, query, cfg, ranked, explanations)
}

func rrfRanks(ranked pipelinelexical.RankedList) map[string]int {
	ranks := make(map[string]int, len(ranked))
	for i, candidate := range ranked {
		if _, exists := ranks[candidate.ID]; !exists {
			ranks[candidate.ID] = i + 1
		}
	}
	return ranks
}

func normalizedRRFContribution(k float64, rank int, normalize float64) float64 {
	if rank == 0 {
		return 0
	}
	return normalize / (k + float64(rank))
}

func cloneRRFLane(source string, items []*retrieval.Result) (pipelinelexical.RankedList, map[string]*retrieval.Result, error) {
	ranked := make(pipelinelexical.RankedList, 0, len(items))
	byID := make(map[string]*retrieval.Result, len(items))
	for _, item := range items {
		if item == nil {
			continue
		}
		if strings.TrimSpace(item.ID) == "" {
			return nil, nil, fmt.Errorf("reliquary search RRF: %s lane returned a blank result ID", source)
		}
		ranked = append(ranked, pipelinelexical.Candidate{ID: item.ID, Source: source})
		if _, exists := byID[item.ID]; !exists {
			byID[item.ID] = indexutil.Clone(item)
		}
	}
	return ranked, byID, nil
}
