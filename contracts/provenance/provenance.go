// Package provenance defines source, artifact, claim, and lineage primitives.
package provenance

import (
	"time"

	"github.com/dotcommander/reliquary/internal/hash"
)

// Hash identifies source or artifact bytes.
type Hash = hash.Digest

// Source describes an input controlled outside reliquary.
type Source struct {
	ID       string
	Kind     string
	URI      string
	Hash     Hash
	Metadata map[string]string
}

// Artifact describes a generated or stored output.
type Artifact struct {
	ID          string
	Kind        string
	URI         string
	Hash        Hash
	GeneratedBy string
	GeneratedAt time.Time
	Metadata    map[string]string
}

// Claim records an assertion and the artifacts or sources that support it.
type Claim struct {
	ID          string
	Text        string
	Subject     string
	Confidence  float64
	SourceIDs   []string
	ArtifactIDs []string
	Metadata    map[string]string
}

// Link connects an input source to an output artifact.
type Link struct {
	SourceID   string
	ArtifactID string
	Relation   string
}

// Lineage records source-to-output relationships.
type Lineage struct {
	ID        string
	Sources   []Source
	Artifacts []Artifact
	Claims    []Claim
	Links     []Link
}

// GeneratedArtifact returns an artifact with UTC generated metadata.
func GeneratedArtifact(id, kind, uri, generator string, digest Hash) Artifact {
	return Artifact{
		ID:          id,
		Kind:        kind,
		URI:         uri,
		Hash:        digest,
		GeneratedBy: generator,
		GeneratedAt: time.Now().UTC(),
	}
}
