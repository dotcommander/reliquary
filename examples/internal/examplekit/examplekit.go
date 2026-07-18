// Package examplekit holds demo-only helpers for examples.
//
// Keep this package out of public Reliquary APIs. It exists so examples show the
// library shape, not temp-dir, env, output-capture, or httptest ceremony.
package examplekit

import (
	"context"
	"fmt"
	"os"
)

// Run is the preferred edge for examples: keep example logic in run(ctx) error
// and translate failures to process exit only in main.
func Run(fn func(context.Context) error) {
	if err := fn(context.Background()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
