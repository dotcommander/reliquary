package vectors

import (
	"strings"
	"testing"
)

func TestIndexManifestValidateMatchesExpectedIdentity(t *testing.T) {
	t.Parallel()

	identity := IndexManifestIdentity{
		Kind:               IndexKindExact,
		Dims:               3,
		EmbeddingModelID:   "text-embedding-test",
		EmbeddingModelHash: "model-sha",
		IndexProfileHash:   "profile-sha",
		ChunkStrategy:      "semantic",
		ChunkSize:          800,
		ChunkOverlap:       120,
	}
	manifest := NewIndexManifest(identity, 42)

	if err := manifest.Validate(identity); err != nil {
		t.Fatalf("Validate() error = %v, want nil", err)
	}
	if manifest.Version != IndexManifestVersion {
		t.Fatalf("Version = %d, want %d", manifest.Version, IndexManifestVersion)
	}
	if manifest.IndexedRows != 42 {
		t.Fatalf("IndexedRows = %d, want 42", manifest.IndexedRows)
	}
	if manifest.CreatedAt.IsZero() {
		t.Fatal("CreatedAt is zero")
	}
}

func TestIndexManifestValidateReportsMismatches(t *testing.T) {
	t.Parallel()

	manifest := IndexManifest{
		Version:            IndexManifestVersion,
		Kind:               IndexKindExact,
		Dims:               768,
		EmbeddingModelID:   "embed-v1",
		EmbeddingModelHash: "hash-v1",
		MediansHash:        "medians-v1",
		IndexProfileHash:   "profile-v1",
		ChunkStrategy:      "sliding",
		ChunkSize:          512,
		ChunkOverlap:       64,
	}
	err := manifest.Validate(IndexManifestIdentity{
		Kind:               IndexKindBinary,
		Dims:               1024,
		EmbeddingModelID:   "embed-v2",
		EmbeddingModelHash: "hash-v2",
		MediansHash:        "medians-v2",
		IndexProfileHash:   "profile-v2",
		ChunkStrategy:      "semantic",
		ChunkSize:          768,
		ChunkOverlap:       96,
	})
	if err == nil {
		t.Fatal("Validate() error = nil, want mismatch")
	}
	msg := err.Error()
	for _, want := range []string{
		`kind="exact" expected="binary"`,
		"dims=768 expected=1024",
		`embedding_model_id="embed-v1" expected="embed-v2"`,
		`embedding_model_hash="hash-v1" expected="hash-v2"`,
		`medians_hash="medians-v1" expected="medians-v2"`,
		`index_profile_hash="profile-v1" expected="profile-v2"`,
		`chunk_strategy="sliding" expected="semantic"`,
		"chunk_size=512 expected=768",
		"chunk_overlap=64 expected=96",
	} {
		if !strings.Contains(msg, want) {
			t.Fatalf("Validate() error = %q, want substring %q", msg, want)
		}
	}
}

func TestIndexManifestValidateZeroExpectedIsBackwardCompatible(t *testing.T) {
	t.Parallel()

	if err := (IndexManifest{}).Validate(IndexManifestIdentity{}); err != nil {
		t.Fatalf("zero manifest Validate(zero identity) error = %v, want nil", err)
	}
}

func TestExactIndexManifestUsesIndexOwnedFields(t *testing.T) {
	t.Parallel()

	arena := EncodeFloat32Vec([]float32{1, 0})
	idx := NewExactIndex(2, []IndexChunk{{Group: "doc", ChunkIndex: 0, Offset: 0, Length: len(arena)}}, arena)
	manifest := idx.Manifest(IndexManifestIdentity{
		Kind:               IndexKindBinary,
		Dims:               99,
		EmbeddingModelID:   "embed-v1",
		EmbeddingModelHash: "hash-v1",
		ChunkStrategy:      "semantic",
		ChunkSize:          400,
		ChunkOverlap:       40,
	})

	if manifest.Kind != IndexKindExact {
		t.Fatalf("Kind = %q, want %q", manifest.Kind, IndexKindExact)
	}
	if manifest.Dims != 2 {
		t.Fatalf("Dims = %d, want 2", manifest.Dims)
	}
	if manifest.IndexedRows != 1 {
		t.Fatalf("IndexedRows = %d, want 1", manifest.IndexedRows)
	}
	if manifest.MediansHash != "" {
		t.Fatalf("MediansHash = %q, want empty", manifest.MediansHash)
	}
	if err := manifest.Validate(IndexManifestIdentity{Kind: IndexKindExact, Dims: 2, EmbeddingModelID: "embed-v1"}); err != nil {
		t.Fatalf("Validate() error = %v, want nil", err)
	}
}

func TestBinaryIndexManifestIncludesStableMediansHash(t *testing.T) {
	t.Parallel()

	blobs := [][]byte{
		EncodeFloat32Vec([]float32{1, 0}),
		EncodeFloat32Vec([]float32{0, 1}),
		EncodeFloat32Vec([]float32{1, 1}),
	}
	idx := NewBinaryIndex(blobs, []string{"a", "b", "c"}, []int{0, 0, 0}, 2)
	manifest := idx.Manifest(IndexManifestIdentity{EmbeddingModelID: "embed-v1"})
	wantHash := HashFloat32Slice([]float32{1, 1})

	if manifest.Kind != IndexKindBinary {
		t.Fatalf("Kind = %q, want %q", manifest.Kind, IndexKindBinary)
	}
	if manifest.Dims != 2 {
		t.Fatalf("Dims = %d, want 2", manifest.Dims)
	}
	if manifest.IndexedRows != 3 {
		t.Fatalf("IndexedRows = %d, want 3", manifest.IndexedRows)
	}
	if manifest.MediansHash != wantHash {
		t.Fatalf("MediansHash = %q, want %q", manifest.MediansHash, wantHash)
	}
	if err := manifest.Validate(IndexManifestIdentity{Kind: IndexKindBinary, Dims: 2, MediansHash: wantHash}); err != nil {
		t.Fatalf("Validate() error = %v, want nil", err)
	}
	if err := manifest.Validate(IndexManifestIdentity{MediansHash: HashFloat32Slice([]float32{0, 0})}); err == nil {
		t.Fatal("Validate() medians mismatch error = nil, want error")
	}
}

func TestHashFloat32Slice(t *testing.T) {
	t.Parallel()

	a := HashFloat32Slice([]float32{1, 2, 3})
	b := HashFloat32Slice([]float32{1, 2, 3})
	c := HashFloat32Slice([]float32{1, 2, 4})
	if a == "" {
		t.Fatal("HashFloat32Slice(non-empty) = empty, want hash")
	}
	if a != b {
		t.Fatalf("HashFloat32Slice stable mismatch: %q != %q", a, b)
	}
	if a == c {
		t.Fatalf("HashFloat32Slice collision for changed input: %q", a)
	}
	if got := HashFloat32Slice(nil); got != "" {
		t.Fatalf("HashFloat32Slice(nil) = %q, want empty", got)
	}
}

func TestHashIndexProfile(t *testing.T) {
	t.Parallel()

	a := HashIndexProfile(map[string]string{
		"kind":              "binary",
		"candidate_limit":   "100",
		"quantization":      "median_binary",
		"rerank_exact_topk": "5",
	})
	b := HashIndexProfile(map[string]string{
		"rerank_exact_topk": "5",
		"quantization":      "median_binary",
		"candidate_limit":   "100",
		"kind":              "binary",
	})
	c := HashIndexProfile(map[string]string{
		"kind":              "binary",
		"candidate_limit":   "200",
		"quantization":      "median_binary",
		"rerank_exact_topk": "5",
	})

	if a == "" {
		t.Fatal("HashIndexProfile(non-empty) = empty, want hash")
	}
	if a != b {
		t.Fatalf("HashIndexProfile order mismatch: %q != %q", a, b)
	}
	if a == c {
		t.Fatalf("HashIndexProfile collision for changed profile: %q", a)
	}
	if got := HashIndexProfile(nil); got != "" {
		t.Fatalf("HashIndexProfile(nil) = %q, want empty", got)
	}
}

func TestHashIndexProfileIdentityCanFeedManifestProfileHash(t *testing.T) {
	t.Parallel()

	fields := map[string]string{
		"kind":            "binary",
		"candidate_limit": "100",
		"quantization":    "median_binary",
	}
	legacy := HashIndexProfile(fields)
	digest := HashIndexProfileIdentity(fields)
	if legacy == "" || digest.String() == "" {
		t.Fatalf("profile hashes = legacy %q digest %q", legacy, digest.String())
	}
	if legacy == digest.Hex {
		t.Fatal("identity helper unexpectedly changed legacy HashIndexProfile compatibility contract")
	}

	identity := IndexManifestIdentity{
		Kind:             IndexKindBinary,
		IndexProfileHash: digest.String(),
	}
	manifest := NewIndexManifest(identity, 12)
	if err := manifest.Validate(identity); err != nil {
		t.Fatalf("Validate() with identity profile digest error = %v", err)
	}
}
