package main

import (
	"context"
	"log"
	"os"

	postgresadapter "github.com/dotcommander/reliquary/adapter/postgres"
	"github.com/jackc/pgx/v5/pgxpool"
)

func main() {
	if err := run(context.Background()); err != nil {
		log.Print(err)
		os.Exit(1)
	}
}

func run(ctx context.Context) error {
	pool, err := pgxpool.New(ctx, os.Getenv("DATABASE_URL"))
	if err != nil {
		return err
	}
	defer pool.Close()

	index, err := postgresadapter.New(pool, postgresadapter.Config{Table: "reliquary_index"})
	if err != nil {
		return err
	}
	if err := index.Migrate(ctx); err != nil {
		return err
	}
	return nil
}
