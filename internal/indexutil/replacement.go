package indexutil

import (
	"fmt"
	"strings"

	indexcontract "github.com/dotcommander/reliquary/index"
)

// ValidateReplacements enforces the document-replacement batch invariants
// shared by every built-in Index implementation.
func ValidateReplacements(replacements []indexcontract.DocumentReplacement) error {
	seen := make(map[string]struct{}, len(replacements))
	seenResultIDs := make(map[string]struct{})
	for _, replacement := range replacements {
		if err := ValidateDocumentID(replacement.DocumentID); err != nil {
			return err
		}
		if _, exists := seen[replacement.DocumentID]; exists {
			return fmt.Errorf("%w: %q", indexcontract.ErrDuplicateDocumentID, replacement.DocumentID)
		}
		seen[replacement.DocumentID] = struct{}{}
		for _, item := range replacement.Results {
			if item == nil {
				continue
			}
			if item.ID == "" {
				return fmt.Errorf("reliquary index: empty item ID")
			}
			if _, exists := seenResultIDs[item.ID]; exists {
				return fmt.Errorf("%w: duplicate %q", indexcontract.ErrResultIDConflict, item.ID)
			}
			seenResultIDs[item.ID] = struct{}{}
			if item.DocumentID != replacement.DocumentID {
				return fmt.Errorf("reliquary index: result %q belongs to document %q, not replacement %q", item.ID, item.DocumentID, replacement.DocumentID)
			}
		}
	}
	return nil
}

// ValidateDocumentID rejects empty or whitespace-only document identifiers.
func ValidateDocumentID(documentID string) error {
	if strings.TrimSpace(documentID) == "" {
		return indexcontract.ErrInvalidDocumentID
	}
	return nil
}
