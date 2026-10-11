package postgres_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/adapter/postgres"
	"github.com/jackc/pgx/v5/pgxpool"
)

func ExampleNew() {
	// Construction validates table configuration without database I/O.
	pool := &pgxpool.Pool{}
	idx, err := postgres.New(pool, postgres.Config{Table: "reliquary_index"})
	fmt.Println(idx != nil, err == nil)
	// Output: true true
}
