package dedup

// Canonicalize reduces each duplicate group to a single representative. For each
// group it keeps the element for which better(candidate, current) reports the
// candidate as preferable, returning one survivor per group in input order.
// Empty groups are skipped; a singleton group yields that element unchanged.
// Pair this with Detector.FindDuplicateGroups / FindNearDuplicates.
func Canonicalize[T any](groups [][]T, better func(a, b T) bool) []T {
	return CanonicalizeWith(groups, better, nil)
}

// CanonicalizeWith is Canonicalize plus a merge step: for every non-winner in a
// group, merge(winner, loser) is invoked before the loser is discarded — e.g. to
// fold a duplicate's metadata into the survivor. With a pointer element type,
// merge can mutate the winner in place. merge may be nil.
func CanonicalizeWith[T any](groups [][]T, better func(a, b T) bool, merge func(winner, loser T)) []T {
	survivors := make([]T, 0, len(groups))

	for _, group := range groups {
		if len(group) == 0 {
			continue
		}

		// Fold to the best element; on ties keep the earlier (first-seen wins).
		winner := group[0]
		winnerIdx := 0
		for i := 1; i < len(group); i++ {
			if better(group[i], winner) {
				winner = group[i]
				winnerIdx = i
			}
		}

		if merge != nil {
			for i, loser := range group {
				if i == winnerIdx {
					continue
				}
				merge(winner, loser)
			}
		}

		survivors = append(survivors, winner)
	}

	return survivors
}
