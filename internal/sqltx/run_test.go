package sqltx

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRun_Success(t *testing.T) {
	t.Parallel()

	var committed, rolledBack bool
	err := Run(
		func() (string, error) { return "tx", nil },
		func(string) error { rolledBack = true; return nil },
		func(string) error { committed = true; return nil },
		func(string) error { return nil },
	)
	require.NoError(t, err)
	require.True(t, committed, "commit must run on success")
	require.False(t, rolledBack, "rollback must not run on success")
}

func TestRun_BeginFailure(t *testing.T) {
	t.Parallel()

	beginErr := errors.New("boom")
	var fnRan, commitRan, rollbackRan bool
	err := Run(
		func() (string, error) { return "", beginErr },
		func(string) error { rollbackRan = true; return nil },
		func(string) error { commitRan = true; return nil },
		func(string) error { fnRan = true; return nil },
	)
	require.ErrorIs(t, err, beginErr)
	require.Contains(t, err.Error(), "begin transaction")
	require.False(t, fnRan, "fn must not run when begin fails")
	require.False(t, commitRan, "commit must not run when begin fails")
	require.False(t, rollbackRan, "rollback must not run when begin fails")
}

func TestRun_FnErrorTriggersRollback(t *testing.T) {
	t.Parallel()

	fnErr := errors.New("fn failed")
	var rolledBack, committed bool
	err := Run(
		func() (string, error) { return "tx", nil },
		func(string) error { rolledBack = true; return nil },
		func(string) error { committed = true; return nil },
		func(string) error { return fnErr },
	)
	require.ErrorIs(t, err, fnErr)
	require.True(t, rolledBack, "rollback must run when fn errors")
	require.False(t, committed, "commit must not run when fn errors")
}

func TestRun_RollbackFailureJoinsError(t *testing.T) {
	t.Parallel()

	fnErr := errors.New("fn failed")
	rbErr := errors.New("rollback exploded")
	err := Run(
		func() (string, error) { return "tx", nil },
		func(string) error { return rbErr },
		func(string) error { return nil },
		func(string) error { return fnErr },
	)
	require.ErrorIs(t, err, fnErr, "original fn error must be preserved")
	require.ErrorIs(t, err, rbErr, "rollback error must be joined in")
	require.Contains(t, err.Error(), "rollback failed")
}

func TestRun_CommitFailureWrapped(t *testing.T) {
	t.Parallel()

	commitErr := errors.New("commit exploded")
	err := Run(
		func() (string, error) { return "tx", nil },
		func(string) error { return nil },
		func(string) error { return commitErr },
		func(string) error { return nil },
	)
	require.ErrorIs(t, err, commitErr)
	require.Contains(t, err.Error(), "commit transaction")
}

func TestRun_PanicTriggersRollbackAndRepanics(t *testing.T) {
	t.Parallel()

	var rolledBack bool
	require.PanicsWithValue(t, "kaboom", func() {
		_ = Run(
			func() (string, error) { return "tx", nil },
			func(string) error { rolledBack = true; return nil },
			func(string) error { return nil },
			func(string) error { panic("kaboom") },
		)
	})
	require.True(t, rolledBack, "rollback must run when fn panics")
}
