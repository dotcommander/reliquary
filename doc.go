// Package reliquary is the high-level facade for local retrieval pipelines.
//
// It wires document chunking, caller-owned embedding, in-memory storage, hybrid
// reranking, and optional MMR diversification behind a small App. Provider SDKs,
// storage drivers, model runtimes, prompts, and product policy stay outside the
// root package.
package reliquary
