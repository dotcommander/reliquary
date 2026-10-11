package sqlite_test

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/dotcommander/reliquary/adapter/sqlite"
	_ "modernc.org/sqlite"
)

func ExampleNew() {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		panic(err)
	}
	defer db.Close()
	db.SetMaxOpenConns(1)

	idx, err := sqlite.New(db, sqlite.Config{
		Table:          "reliquary_index",
		CandidateLimit: 50,
	})
	if err != nil {
		panic(err)
	}

	err = idx.Migrate(context.Background())
	fmt.Println(idx != nil, err == nil)
	// Output: true true
}
