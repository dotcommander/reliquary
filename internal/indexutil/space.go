package indexutil

import (
	"fmt"
	"math"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

// Space is the reset-only identity and embedding-dimension latch shared by
// Index implementations. IdentitySet distinguishes an unestablished identity
// from an established empty identity.
type Space struct {
	Identity    string
	IdentitySet bool
	Dimension   int
}

// ValidateResults returns the space produced by accepting items. The receiver
// is unchanged when validation fails. The first non-nil result establishes the
// identity, and the first result with an embedding establishes the dimension.
func (s Space) ValidateResults(items []*retrieval.Result) (Space, error) {
	next := s
	for _, item := range items {
		if item == nil {
			continue
		}
		if !next.IdentitySet {
			next.Identity = item.IndexIdentity
			next.IdentitySet = true
		} else if item.IndexIdentity != next.Identity {
			return s, fmt.Errorf("%w: index has %q, item %q has %q", indexcontract.ErrIdentityMismatch, next.Identity, item.ID, item.IndexIdentity)
		}
		if len(item.Embedding) == 0 {
			continue
		}
		if next.Dimension == 0 {
			next.Dimension = len(item.Embedding)
		} else if len(item.Embedding) != next.Dimension {
			return s, fmt.Errorf("%w: index has %d dimensions, item %q has %d", indexcontract.ErrDimensionMismatch, next.Dimension, item.ID, len(item.Embedding))
		}
		for _, value := range item.Embedding {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return s, fmt.Errorf("reliquary index: item %q embedding values must be finite", item.ID)
			}
		}
	}
	return next, nil
}

// ValidateQuery rejects reads against a different established index space.
func (s Space) ValidateQuery(query indexcontract.IndexQuery) error {
	if s.IdentitySet && query.Identity != s.Identity {
		return fmt.Errorf("%w: index has %q, query has %q", indexcontract.ErrIdentityMismatch, s.Identity, query.Identity)
	}
	if s.Dimension > 0 && len(query.Vector) > 0 && len(query.Vector) != s.Dimension {
		return fmt.Errorf("%w: index has %d dimensions, query has %d", indexcontract.ErrDimensionMismatch, s.Dimension, len(query.Vector))
	}
	for _, value := range query.Vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("reliquary index: query vector values must be finite")
		}
	}
	return nil
}
