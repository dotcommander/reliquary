// Package retrieval is a hybrid scoring and reranking layer for local retrieval
// pipelines.
//
// It sits above github.com/dotcommander/reliquary/vector (cosine primitives),
// github.com/dotcommander/reliquary/chunking (text splitting), and the
// provider-neutral embeddings contract. Callers still own model/provider
// execution; retrieval only adapts embedding vectors into its scoring space.
//
// # Pipeline shape
//
// chunk (chunking) → embed (caller) → Rerank → MMR → Evaluate
//
// Optional diagnostics and calibration layers can be added around that path:
// ScoreReference for cohort-relative score percentiles, EvaluateSegments for
// slice-level metrics, EvaluateLayers for candidate/rerank/diversification
// attribution, EvaluateRun for fixture-backed golden reports, and TuneWeights
// for deterministic weight/lambda grid search.
//
// # Scoring pipelines
//
// Two paths are available depending on whether corpus context is present:
//
//   - Scorer / Rerank: corpus-aware batch scoring. Rerank runs min-max
//     calibration across the whole result set so ranking is relative to the
//     corpus. Build with NewScorer(DefaultWeights()) and call Rerank with the
//     query embedding, query text, and all candidate Results.
//     RerankWithTrace follows the same scoring path and returns rank-aligned
//     diagnostics for raw scores, calibrated scores, weights, and signal
//     contributions. RerankWithReference applies an explicit ScoreReference
//     after reranking when callers need stable cohort-relative percentiles.
//
//   - CalibratedScore: single-document scoring with fixed weights
//     (0.62 cosine / 0.18 keyword / 0.10 filename / 0.10 metadata). Independent
//     of corpus distribution; useful when scoring one result in isolation.
//
// AdaptiveWeights adjusts the weight mix by query token count when a more
// query-length-aware Scorer is preferred over DefaultWeights.
//
// # Diversification
//
// MMR (Maximal Marginal Relevance) re-orders a ranked list to balance
// relevance against redundancy. The lambda parameter slides between
// pure relevance (lambda=1) and pure diversity (lambda=0).
//
// # Evaluation and tuning
//
// Evaluate computes RecallAtK, PrecisionAtK, MRR, NDCGAtK, and UniqueTopicAtK
// from a ranked result list against expected relevant document IDs.
// EvaluateSegments reports the same metrics by caller-owned segment keys with
// sample and hit counts. EvaluateLayers separates candidate generation,
// reranking, diversification, and final top-k metrics so regressions can be
// localized. EvaluateRun evaluates captured run outputs against golden fixture
// judgments and reports aggregate, per-query, segment, and layer metrics.
// TuneWeights runs a deterministic grid over precomputed ScoreSignals and
// optional MMR lambdas, rejects configs that miss floor constraints, and returns
// the best remaining configuration.
//
// # Supporting building blocks
//
//   - Filter: path inclusion/exclusion by extension or prefix
//   - ExtractMetadata / MetadataScore: title and heading signal from file path
//     and content
//   - ScoreReference: sorted reference cohorts for stable percentile scoring
package retrieval
