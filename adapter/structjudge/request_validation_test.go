package structjudge

import (
	"context"
	"errors"
	"testing"

	"github.com/dotcommander/reliquary/judgment"
)

func TestRejectInvalidRequestsBeforeClient(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name   string
		invoke func(*Adapter) error
	}{
		{"choice blank task", func(a *Adapter) error {
			_, err := a.Choose(context.Background(), judgment.ChoiceRequest{Options: []judgment.Option{{ID: "a"}, {ID: "b"}}})
			return err
		}},
		{"choice missing options", func(a *Adapter) error {
			_, err := a.Choose(context.Background(), judgment.ChoiceRequest{Task: "route"})
			return err
		}},
		{"choice duplicate options", func(a *Adapter) error {
			_, err := a.Choose(context.Background(), judgment.ChoiceRequest{Task: "route", Options: []judgment.Option{{ID: "a"}, {ID: "a"}}})
			return err
		}},
		{"choice blank option", func(a *Adapter) error {
			_, err := a.Choose(context.Background(), judgment.ChoiceRequest{Task: "route", Options: []judgment.Option{{ID: "a"}, {ID: " "}}})
			return err
		}},
		{"score blank task", func(a *Adapter) error {
			_, err := a.Score(context.Background(), judgment.ScoreRequest{Target: "document"})
			return err
		}},
		{"score blank target", func(a *Adapter) error {
			_, err := a.Score(context.Background(), judgment.ScoreRequest{Task: "rate", Target: " \t"})
			return err
		}},
		{"noul blank question", func(a *Adapter) error {
			_, err := a.Ask(context.Background(), judgment.NoulRequest{Question: " \n"})
			return err
		}},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			providerErr := errors.New("provider must not be called")
			client := &scriptedClient{err: providerErr}
			adapter, err := New(client)
			if err != nil {
				t.Fatal(err)
			}
			err = tt.invoke(adapter)
			if !errors.Is(err, judgment.ErrInvalidRequest) {
				t.Fatalf("error = %v, want ErrInvalidRequest", err)
			}
			if errors.Is(err, ErrInvalidResponse) {
				t.Fatalf("request misclassified as response error: %v", err)
			}
			if errors.Is(err, providerErr) {
				t.Fatalf("provider reached: %v", err)
			}
			if calls := client.calls; calls != 0 {
				t.Fatalf("client calls = %d, want 0", calls)
			}
		})
	}
}

func TestValidRequestsPreserveProviderErrors(t *testing.T) {
	t.Parallel()
	for _, tt := range []struct {
		name   string
		invoke func(*Adapter) error
	}{
		{"choice", func(a *Adapter) error {
			_, err := a.Choose(context.Background(), judgment.ChoiceRequest{Task: "route", Options: []judgment.Option{{ID: "a"}, {ID: "b"}}})
			return err
		}},
		{"score", func(a *Adapter) error {
			_, err := a.Score(context.Background(), judgment.ScoreRequest{Task: "rate", Target: "document"})
			return err
		}},
		{"noul", func(a *Adapter) error {
			_, err := a.Ask(context.Background(), judgment.NoulRequest{Question: "ready?"})
			return err
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			providerErr := errors.New("provider unavailable")
			client := &scriptedClient{err: providerErr}
			adapter, err := New(client)
			if err != nil {
				t.Fatal(err)
			}
			err = tt.invoke(adapter)
			if !errors.Is(err, providerErr) || errors.Is(err, judgment.ErrInvalidRequest) {
				t.Fatalf("provider error = %v", err)
			}
			if calls := client.calls; calls != 1 {
				t.Fatalf("client calls = %d, want 1", calls)
			}
		})
	}
}
