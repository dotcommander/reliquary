package vectors

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	supporthash "github.com/dotcommander/reliquary/internal/hash"
)

// IndexManifestVersion is the current IndexManifest schema version.
const IndexManifestVersion = 1

// IndexKind identifies the index implementation described by a manifest.
type IndexKind string

const (
	IndexKindExact  IndexKind = "exact"
	IndexKindBinary IndexKind = "binary"
)

// IndexManifestIdentity describes compatibility inputs that can make a vector
// index stale when they change. Zero-valued fields are treated as unspecified
// by Validate so callers can enforce only the identity dimensions they own.
type IndexManifestIdentity struct {
	Kind               IndexKind
	Dims               int
	EmbeddingModelID   string
	EmbeddingModelHash string
	MediansHash        string
	IndexProfileHash   string
	ChunkStrategy      string
	ChunkSize          int
	ChunkOverlap       int
}

// IndexManifest records the compatibility identity and provenance for a built
// vector index. It is intentionally data-only so callers can persist it in any
// format they already use.
type IndexManifest struct {
	Version            int
	CreatedAt          time.Time
	Kind               IndexKind
	Dims               int
	IndexedRows        int
	EmbeddingModelID   string
	EmbeddingModelHash string
	MediansHash        string
	IndexProfileHash   string
	ChunkStrategy      string
	ChunkSize          int
	ChunkOverlap       int
}

// NewIndexManifest builds a manifest from caller-owned identity fields.
func NewIndexManifest(identity IndexManifestIdentity, indexedRows int) IndexManifest {
	return IndexManifest{
		Version:            IndexManifestVersion,
		CreatedAt:          time.Now().UTC(),
		Kind:               identity.Kind,
		Dims:               identity.Dims,
		IndexedRows:        indexedRows,
		EmbeddingModelID:   identity.EmbeddingModelID,
		EmbeddingModelHash: identity.EmbeddingModelHash,
		MediansHash:        identity.MediansHash,
		IndexProfileHash:   identity.IndexProfileHash,
		ChunkStrategy:      identity.ChunkStrategy,
		ChunkSize:          identity.ChunkSize,
		ChunkOverlap:       identity.ChunkOverlap,
	}
}

// Manifest returns a compatibility manifest for the exact index, merging
// caller-owned embedding/chunking identity with index-owned dimensions and row count.
func (idx *ExactIndex) Manifest(identity IndexManifestIdentity) IndexManifest {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	identity.Kind = IndexKindExact
	identity.Dims = idx.dims
	return NewIndexManifest(identity, len(idx.chunks))
}

// Manifest returns a compatibility manifest for the binary index, merging
// caller-owned embedding/chunking identity with index-owned dimensions, row
// count, and median thresholds hash.
func (idx *BinaryIndex) Manifest(identity IndexManifestIdentity) IndexManifest {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	identity.Kind = IndexKindBinary
	identity.Dims = idx.dims
	identity.MediansHash = HashFloat32Slice(idx.medians)
	return NewIndexManifest(identity, len(idx.entries))
}

// Validate returns an actionable error when a manifest is incompatible with the
// expected identity. A fully zero expected identity is valid for backward-compatible
// callers that have not opted into manifest enforcement.
func (m IndexManifest) Validate(expected IndexManifestIdentity) error {
	var mismatches []string
	if !expected.zero() && m.Version != IndexManifestVersion {
		mismatches = append(mismatches, fmt.Sprintf("version=%d expected=%d", m.Version, IndexManifestVersion))
	}
	mismatches = appendStringMismatch(mismatches, "kind", string(m.Kind), string(expected.Kind))
	mismatches = appendIntMismatch(mismatches, "dims", m.Dims, expected.Dims)
	mismatches = appendStringMismatch(mismatches, "embedding_model_id", m.EmbeddingModelID, expected.EmbeddingModelID)
	mismatches = appendStringMismatch(mismatches, "embedding_model_hash", m.EmbeddingModelHash, expected.EmbeddingModelHash)
	mismatches = appendStringMismatch(mismatches, "medians_hash", m.MediansHash, expected.MediansHash)
	mismatches = appendStringMismatch(mismatches, "index_profile_hash", m.IndexProfileHash, expected.IndexProfileHash)
	mismatches = appendStringMismatch(mismatches, "chunk_strategy", m.ChunkStrategy, expected.ChunkStrategy)
	mismatches = appendIntMismatch(mismatches, "chunk_size", m.ChunkSize, expected.ChunkSize)
	mismatches = appendIntMismatch(mismatches, "chunk_overlap", m.ChunkOverlap, expected.ChunkOverlap)
	if len(mismatches) > 0 {
		return fmt.Errorf("vectors: index manifest mismatch: %s", strings.Join(mismatches, "; "))
	}
	return nil
}

// appendStringMismatch records got vs want when want is non-empty (zero value
// means the caller is not enforcing that identity dimension).
func appendStringMismatch(mismatches []string, name, got, want string) []string {
	if want != "" && got != want {
		mismatches = append(mismatches, fmt.Sprintf("%s=%q expected=%q", name, got, want))
	}
	return mismatches
}

// appendIntMismatch records got vs want when want is non-zero (zero value
// means the caller is not enforcing that identity dimension).
func appendIntMismatch(mismatches []string, name string, got, want int) []string {
	if want != 0 && got != want {
		mismatches = append(mismatches, fmt.Sprintf("%s=%d expected=%d", name, got, want))
	}
	return mismatches
}

// HashFloat32Slice returns a stable SHA-256 hash for float32 identity data such
// as binary-index median thresholds.
func HashFloat32Slice(values []float32) string {
	if len(values) == 0 {
		return ""
	}
	buf := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	sum := sha256.Sum256(buf)
	return hex.EncodeToString(sum[:])
}

// HashIndexProfile returns a stable hash for caller-owned index profile fields
// such as candidate limits, quantization settings, or external ANN parameters.
func HashIndexProfile(fields map[string]string) string {
	if len(fields) == 0 {
		return ""
	}
	keys := make([]string, 0, len(fields))
	for key := range fields {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	h := sha256.New()
	for _, key := range keys {
		h.Write([]byte(key))
		h.Write([]byte{0})
		h.Write([]byte(fields[key]))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// HashIndexProfileIdentity returns a transform identity digest for caller-owned
// index profile fields. HashIndexProfile is retained for compatibility with
// existing persisted profile hashes.
func HashIndexProfileIdentity(fields map[string]string) supporthash.Digest {
	if len(fields) == 0 {
		return supporthash.Digest{}
	}
	keys := make([]string, 0, len(fields))
	for key := range fields {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	parts := make([]supporthash.IdentityPart, 0, len(keys)+1)
	parts = append(parts, supporthash.IdentityPart{Kind: "index_profile", Version: "v1"})
	for _, key := range keys {
		parts = append(parts, supporthash.IdentityPart{Kind: "field", ID: key, Value: fields[key]})
	}
	return supporthash.HashIdentity(parts...)
}

func (identity IndexManifestIdentity) zero() bool {
	return identity.Kind == "" &&
		identity.Dims == 0 &&
		identity.EmbeddingModelID == "" &&
		identity.EmbeddingModelHash == "" &&
		identity.MediansHash == "" &&
		identity.IndexProfileHash == "" &&
		identity.ChunkStrategy == "" &&
		identity.ChunkSize == 0 &&
		identity.ChunkOverlap == 0
}
