package indextest

import (
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/index/inmem"
)

func TestTextContractWithInMemoryIndex(t *testing.T) {
	t.Parallel()
	runTextContract(t, func() indexcontract.Index { return inmem.New() })
}
