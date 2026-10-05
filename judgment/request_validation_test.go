package judgment

import (
	"errors"
	"testing"
)

func TestValidateChoiceRequest(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		request ChoiceRequest
		invalid bool
	}{
		{"valid", ChoiceRequest{Task: "route", Options: []Option{{ID: "a"}, {ID: "b"}}}, false},
		{"blank task", ChoiceRequest{Task: " \t", Options: []Option{{ID: "a"}, {ID: "b"}}}, true},
		{"no options", ChoiceRequest{Task: "route"}, true},
		{"one option", ChoiceRequest{Task: "route", Options: []Option{{ID: "a"}}}, true},
		{"blank option", ChoiceRequest{Task: "route", Options: []Option{{ID: "a"}, {ID: " \t"}}}, true},
		{"duplicate option", ChoiceRequest{Task: "route", Options: []Option{{ID: "a"}, {ID: "a"}}}, true},
		{"untrimmed distinct IDs", ChoiceRequest{Task: "route", Options: []Option{{ID: "a"}, {ID: " a"}}}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := ValidateChoiceRequest(tt.request)
			if tt.invalid {
				if !errors.Is(err, ErrInvalidRequest) {
					t.Fatalf("error = %v, want ErrInvalidRequest", err)
				}
			} else if err != nil {
				t.Fatalf("valid request: %v", err)
			}
		})
	}
}

func TestValidateScoreRequest(t *testing.T) {
	t.Parallel()
	for _, tt := range []struct {
		name    string
		request ScoreRequest
		invalid bool
	}{
		{"valid", ScoreRequest{Task: "rate", Target: "document"}, false},
		{"blank task", ScoreRequest{Task: " \n", Target: "document"}, true},
		{"blank target", ScoreRequest{Task: "rate", Target: " \t"}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := ValidateScoreRequest(tt.request)
			if tt.invalid {
				if !errors.Is(err, ErrInvalidRequest) {
					t.Fatalf("error = %v, want ErrInvalidRequest", err)
				}
			} else if err != nil {
				t.Fatalf("valid request: %v", err)
			}
		})
	}
}

func TestValidateNoulRequest(t *testing.T) {
	t.Parallel()
	for _, tt := range []struct {
		name    string
		request NoulRequest
		invalid bool
	}{
		{"valid", NoulRequest{Question: "ready?"}, false},
		{"blank question", NoulRequest{Question: " \n\t"}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := ValidateNoulRequest(tt.request)
			if tt.invalid {
				if !errors.Is(err, ErrInvalidRequest) {
					t.Fatalf("error = %v, want ErrInvalidRequest", err)
				}
			} else if err != nil {
				t.Fatalf("valid request: %v", err)
			}
		})
	}
}
