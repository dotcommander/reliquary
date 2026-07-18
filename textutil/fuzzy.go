package textutil

import (
	"strings"
	"unicode"
)

const (
	PersonAliasMinScore      = 0.80
	PersonFuzzyTokenMinScore = 0.88
	LongTermMinLen           = 7
	LongTermShortMaxDistance = 1
	LongTermLongMaxDistance  = 2
	LongTermLongLen          = 10
	PhraseContainmentScore   = 0.96
)

type MatchReason string

const (
	ReasonNone          MatchReason = ""
	ReasonExactPhrase   MatchReason = "exact_phrase"
	ReasonAliasPhrase   MatchReason = "alias_phrase"
	ReasonTokenCoverage MatchReason = "token_coverage"
	ReasonFuzzyToken    MatchReason = "fuzzy_token"
)

type AliasMatch struct {
	Score  float64
	Reason MatchReason
}

type StopTermFunc func(string) bool

func AliasQueryScore(query, canonical string, aliases []string, stop StopTermFunc) AliasMatch {
	best := phraseScore(query, canonical, stop)
	for _, alias := range aliases {
		score := phraseScore(query, alias, stop)
		if score.Score > best.Score {
			best = score
			if best.Reason == ReasonExactPhrase {
				best.Reason = ReasonAliasPhrase
			}
		}
	}
	return best
}

func PhraseTerms(s string, stop StopTermFunc) []string {
	var terms []string
	for _, raw := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if len(raw) < 2 || (stop != nil && stop(raw)) {
			continue
		}
		terms = append(terms, raw)
	}
	return terms
}

func TextTerms(text string) []string {
	seen := map[string]struct{}{}
	var terms []string
	for _, raw := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if len(raw) < 4 {
			continue
		}
		if _, ok := seen[raw]; ok {
			continue
		}
		seen[raw] = struct{}{}
		terms = append(terms, raw)
	}
	return terms
}

func LongTermNearMatch(term string, textTerms []string) bool {
	term = strings.TrimSuffix(strings.ToLower(term), "s")
	if len(term) < LongTermMinLen {
		return false
	}
	maxDistance := LongTermShortMaxDistance
	if len([]rune(term)) >= LongTermLongLen {
		maxDistance = LongTermLongMaxDistance
	}
	for _, candidate := range textTerms {
		candidate = strings.TrimSuffix(candidate, "s")
		if len(candidate) < LongTermMinLen || absInt(len([]rune(term))-len([]rune(candidate))) > maxDistance {
			continue
		}
		if boundedEditDistance([]rune(term), []rune(candidate), maxDistance) <= maxDistance {
			return true
		}
	}
	return false
}

func phraseScore(query, target string, stop StopTermFunc) AliasMatch {
	queryTerms := PhraseTerms(query, stop)
	targetTerms := PhraseTerms(target, stop)
	if len(queryTerms) == 0 || len(targetTerms) == 0 {
		return AliasMatch{}
	}
	queryPhrase := strings.Join(queryTerms, " ")
	targetPhrase := strings.Join(targetTerms, " ")
	if queryPhrase == targetPhrase {
		return AliasMatch{Score: 1, Reason: ReasonExactPhrase}
	}
	if containsTermSequence(queryTerms, targetTerms) {
		return AliasMatch{Score: PhraseContainmentScore, Reason: ReasonAliasPhrase}
	}

	best := AliasMatch{}
	if len(targetTerms) == 1 && len(queryTerms) >= len(targetTerms) {
		for i := 0; i <= len(queryTerms)-len(targetTerms); i++ {
			window := strings.Join(queryTerms[i:i+len(targetTerms)], " ")
			_, fuzzy := termSimilar(window, targetPhrase)
			if !fuzzy {
				continue
			}
			if score := StringSimilarity(window, targetPhrase); score > best.Score {
				best = AliasMatch{Score: score, Reason: ReasonFuzzyToken}
			}
		}
	}
	coverage, fuzzy := termCoverage(queryTerms, targetTerms)
	if coverage > best.Score {
		best = AliasMatch{Score: coverage, Reason: ReasonTokenCoverage}
		if fuzzy {
			best.Reason = ReasonFuzzyToken
		}
	}
	if boundaryNameCoverage(queryTerms, targetTerms) && best.Score < 0.84 {
		best = AliasMatch{Score: 0.84, Reason: ReasonTokenCoverage}
	}
	return best
}

func containsTermSequence(queryTerms, targetTerms []string) bool {
	if len(queryTerms) < len(targetTerms) {
		return false
	}
	for i := 0; i <= len(queryTerms)-len(targetTerms); i++ {
		matched := true
		for j, target := range targetTerms {
			if queryTerms[i+j] != target {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}

func boundaryNameCoverage(queryTerms, targetTerms []string) bool {
	if len(queryTerms) < 2 || len(targetTerms) < 3 {
		return false
	}
	first := targetTerms[0]
	last := targetTerms[len(targetTerms)-1]
	var firstHit, lastHit bool
	for _, query := range queryTerms {
		exact, fuzzy := termSimilar(query, first)
		firstHit = firstHit || exact || fuzzy
		exact, fuzzy = termSimilar(query, last)
		lastHit = lastHit || exact || fuzzy
	}
	return firstHit && lastHit
}

func termCoverage(queryTerms, targetTerms []string) (float64, bool) {
	var hits int
	var usedFuzzy bool
	for _, target := range targetTerms {
		for _, query := range queryTerms {
			exact, fuzzy := termSimilar(query, target)
			if exact || fuzzy {
				hits++
				usedFuzzy = usedFuzzy || fuzzy
				break
			}
		}
	}
	return float64(hits) / float64(len(targetTerms)), usedFuzzy
}

func termSimilar(a, b string) (exact, fuzzy bool) {
	if a == b {
		return true, false
	}
	if len(a) < 4 || len(b) < 4 {
		return false, false
	}
	maxDistance := LongTermShortMaxDistance
	if minInt(len([]rune(a)), len([]rune(b))) >= LongTermLongLen {
		maxDistance = LongTermLongMaxDistance
	}
	if absInt(len([]rune(a))-len([]rune(b))) > maxDistance {
		return false, false
	}
	return false, StringSimilarity(a, b) >= PersonFuzzyTokenMinScore
}

func boundedEditDistance(a, b []rune, maxDistance int) int {
	if len(a) == 0 {
		return len(b)
	}
	if len(b) == 0 {
		return len(a)
	}
	if absInt(len(a)-len(b)) > maxDistance {
		return maxDistance + 1
	}
	prev := make([]int, len(b)+1)
	curr := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		curr[0] = i
		rowMin := curr[0]
		for j := 1; j <= len(b); j++ {
			cost := 0
			if a[i-1] != b[j-1] {
				cost = 1
			}
			curr[j] = minInt(prev[j]+1, minInt(curr[j-1]+1, prev[j-1]+cost))
			if curr[j] < rowMin {
				rowMin = curr[j]
			}
		}
		if rowMin > maxDistance {
			return maxDistance + 1
		}
		prev, curr = curr, prev
	}
	return prev[len(b)]
}
