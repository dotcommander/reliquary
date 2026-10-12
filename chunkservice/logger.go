package chunkservice

import (
	"context"
	"log/slog"
)

// discardLogger drops every record without allocating; it keeps library
// consumers silent unless the wiring injects a real logger.
func discardLogger() *slog.Logger {
	return slog.New(discardHandler{})
}

// discardHandler drops every record without allocating.
type discardHandler struct{}

func (discardHandler) Enabled(context.Context, slog.Level) bool  { return false }
func (discardHandler) Handle(context.Context, slog.Record) error { return nil }
func (h discardHandler) WithAttrs([]slog.Attr) slog.Handler      { return h }
func (h discardHandler) WithGroup(string) slog.Handler           { return h }
