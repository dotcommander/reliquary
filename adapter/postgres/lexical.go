package postgres

import (
	"fmt"
	"strings"

	indexcontract "github.com/dotcommander/reliquary/index"
)

// lexicalCandidateQuery uses PostgreSQL full-text ranking only to select a
// bounded candidate set. Final scoring still uses Reliquary's ranking. Callers
// with large corpora may create a matching GIN expression index themselves;
// Migrate does not create one or interpolate a configuration into DDL.
func (i *Index) lexicalCandidateQuery(statement string, where []string, args []any, query indexcontract.IndexQuery) (string, []any, error) {
	args = append(args, i.textSearchConfig, query.Text)
	configArg, textArg := len(args)-1, len(args)
	vector := fmt.Sprintf(`to_tsvector($%d::regconfig, filename || ' ' || content)`, configArg)
	search := fmt.Sprintf(`plainto_tsquery($%d::regconfig, $%d)`, configArg, textArg)
	where = append(where, vector+` @@ `+search)
	statement += ` WHERE ` + strings.Join(where, ` AND `)
	statement += ` ORDER BY ts_rank_cd(` + vector + `, ` + search + `) DESC, id ASC`
	args = append(args, max(i.lexicalCandidateLimit, query.Limit))
	statement += fmt.Sprintf(` LIMIT $%d`, len(args))
	return statement, args, nil
}
