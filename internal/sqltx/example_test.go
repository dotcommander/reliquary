package sqltx

import "fmt"

func ExampleRun() {
	committed := false
	err := Run(
		func() (string, error) { return "tx", nil },
		func(string) error { return nil },
		func(string) error {
			committed = true
			return nil
		},
		func(string) error { return nil },
	)
	fmt.Println(err == nil, committed)
	// Output: true true
}
