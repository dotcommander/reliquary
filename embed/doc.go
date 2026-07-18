// Package embed provides deterministic, dependency-free embedding helpers for
// demos, tests, and reliquary.Quickstart.
//
// The hashing embedder maps text to normalized vectors with the signed feature
// hashing trick. It is useful for examples and local tests, but it is not a
// replacement for a production embedding model.
package embed
