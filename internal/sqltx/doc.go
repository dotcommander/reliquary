// Package sqltx provides generic transaction lifecycle semantics shared by
// storage adapters. It centralizes begin, rollback, commit, error, and panic
// behavior without importing concrete SQL drivers.
package sqltx
