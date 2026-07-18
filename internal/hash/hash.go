// Package hash provides stable content hash helpers for reliquary primitives.
package hash

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"strconv"
)

const (
	// AlgorithmSHA256 is the canonical algorithm name used by reliquary metadata.
	AlgorithmSHA256 = "sha256"
)

// Digest identifies content by algorithm and hex-encoded value.
type Digest struct {
	Algorithm string
	Hex       string
}

// IdentityPart names one ordered input to a derived artifact identity. Empty
// fields are omitted from the hash encoding, so callers can fill only the
// dimensions they own.
type IdentityPart struct {
	Kind       string
	ID         string
	Version    string
	ConfigHash string
	InputHash  string
	Value      string
	Digest     Digest
}

// Identity is an ordered list of transform identity parts.
type Identity []IdentityPart

// String returns the portable digest identity, for example sha256:abcd.
func (d Digest) String() string {
	if d.Algorithm == "" || d.Hex == "" {
		return ""
	}
	return d.Algorithm + ":" + d.Hex
}

// SHA256Bytes returns a SHA-256 digest for b.
func SHA256Bytes(b []byte) Digest {
	sum := sha256.Sum256(b)
	return Digest{Algorithm: AlgorithmSHA256, Hex: hex.EncodeToString(sum[:])}
}

// SHA256String returns a SHA-256 digest for s.
func SHA256String(s string) Digest {
	return SHA256Bytes([]byte(s))
}

// SHA256Reader streams r into a SHA-256 digest.
func SHA256Reader(r io.Reader) (Digest, error) {
	h := sha256.New()
	if _, err := io.Copy(h, r); err != nil {
		return Digest{}, fmt.Errorf("hash reader: %w", err)
	}
	return Digest{Algorithm: AlgorithmSHA256, Hex: hex.EncodeToString(h.Sum(nil))}, nil
}

// HashIdentity returns a stable digest for ordered transform identity parts.
// Reordering parts changes the digest. Zero-value parts and empty fields are
// omitted deterministically.
func HashIdentity(parts ...IdentityPart) Digest {
	h := sha256.New()
	for _, part := range parts {
		if !part.hasFields() {
			continue
		}
		writeIdentityField(h, "kind", part.Kind)
		writeIdentityField(h, "id", part.ID)
		writeIdentityField(h, "version", part.Version)
		writeIdentityField(h, "config_hash", part.ConfigHash)
		writeIdentityField(h, "input_hash", part.InputHash)
		writeIdentityField(h, "value", part.Value)
		if part.Digest.Algorithm != "" || part.Digest.Hex != "" {
			writeIdentityField(h, "digest_algorithm", part.Digest.Algorithm)
			writeIdentityField(h, "digest_hex", part.Digest.Hex)
		}
		h.Write([]byte{0xff})
	}
	return Digest{Algorithm: AlgorithmSHA256, Hex: hex.EncodeToString(h.Sum(nil))}
}

// Digest returns a stable digest for id.
func (id Identity) Digest() Digest {
	return HashIdentity(id...)
}

func writeIdentityField(w io.Writer, name, value string) {
	if value == "" {
		return
	}
	_, _ = io.WriteString(w, name)
	_, _ = w.Write([]byte{0})
	_, _ = io.WriteString(w, strconv.Itoa(len(value)))
	_, _ = w.Write([]byte{0})
	_, _ = io.WriteString(w, value)
	_, _ = w.Write([]byte{0})
}

func (part IdentityPart) hasFields() bool {
	return part.Kind != "" ||
		part.ID != "" ||
		part.Version != "" ||
		part.ConfigHash != "" ||
		part.InputHash != "" ||
		part.Value != "" ||
		part.Digest.Algorithm != "" ||
		part.Digest.Hex != ""
}
