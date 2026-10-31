package sqlite

import "github.com/dotcommander/reliquary/internal/indexutil"

// boundCandidateQuery pushes the scan bound into SQL only when SQL filtering
// is exact. Numeric metadata predicates deliberately preselect by type alone;
// their exact-value comparison must happen before spending the candidate limit.
func boundCandidateQuery(statement string, args []any, filter map[string]any, limit int) (string, []any) {
	if limit <= 0 {
		return statement, args
	}
	for key, value := range filter {
		switch key {
		case "id", "document_id", "filename":
			continue // Reserved fields are checked exactly (or rejected) in SQL.
		}
		if indexutil.IsJSONNumber(value) {
			return statement, args
		}
	}
	return statement + " LIMIT ?", append(args, limit)
}
