package inmem_test

import (
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/index/indextest"
	"github.com/dotcommander/reliquary/index/inmem"
)

func TestIndexContract(t *testing.T) {
	t.Parallel()
	indextest.Run(t, func() indexcontract.Index { return inmem.New() })
}
