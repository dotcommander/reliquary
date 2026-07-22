package document

import (
	"errors"
	"fmt"
	"io"
	"math"
	"unicode/utf8"
)

const defaultMaxReaderBytes int64 = 16 << 20

var (
	// ErrInputTooLarge indicates that reader input exceeded the configured byte limit.
	ErrInputTooLarge = errors.New("document input too large")
	// ErrInvalidUTF8 indicates that reader input was not valid UTF-8 text.
	ErrInvalidUTF8 = errors.New("document input is not valid UTF-8")
)

// ReaderOption configures a document created by FromReader.
type ReaderOption func(*readerConfig)

type readerConfig struct {
	filename string
	format   Format
	metadata Metadata
	maxBytes int64
}

// WithFilename sets the document title. It does not select a parser or infer a
// format from the filename.
func WithFilename(filename string) ReaderOption {
	return func(config *readerConfig) {
		config.filename = filename
	}
}

// WithFormat sets the document format.
func WithFormat(format Format) ReaderOption {
	return func(config *readerConfig) {
		config.format = format
	}
}

// WithMetadata snapshots metadata for the constructed document.
func WithMetadata(metadata map[string]string) ReaderOption {
	snapshot := cloneMetadata(metadata)
	return func(config *readerConfig) {
		config.metadata = cloneMetadata(snapshot)
	}
}

// WithMaxBytes sets the maximum number of input bytes accepted by FromReader.
// Nonpositive limits are invalid.
func WithMaxBytes(maxBytes int64) ReaderOption {
	return func(config *readerConfig) {
		config.maxBytes = maxBytes
	}
}

// FromReader reads bounded UTF-8 text and constructs a normalized document.
// It does not infer parsing behavior from the filename or format.
func FromReader(id string, reader io.Reader, opts ...ReaderOption) (Document, error) {
	if reader == nil {
		return Document{}, errors.New("document reader is nil")
	}

	config := readerConfig{
		format:   FormatText,
		maxBytes: defaultMaxReaderBytes,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&config)
		}
	}
	if config.maxBytes <= 0 {
		return Document{}, fmt.Errorf("document max bytes must be positive: %d", config.maxBytes)
	}

	text, err := readBoundedText(reader, config.maxBytes)
	if err != nil {
		return Document{}, err
	}

	return Document{
		ID:       id,
		Title:    config.filename,
		Format:   config.format,
		Text:     NormalizeText(text),
		Metadata: config.metadata,
	}, nil
}

func readBoundedText(reader io.Reader, maxBytes int64) (string, error) {
	readLimit := maxBytes
	if maxBytes < math.MaxInt64 {
		readLimit++
	}

	data, err := io.ReadAll(io.LimitReader(reader, readLimit))
	if int64(len(data)) > maxBytes {
		return "", fmt.Errorf("%w: limit is %d bytes", ErrInputTooLarge, maxBytes)
	}
	if err != nil {
		return "", fmt.Errorf("read document: %w", err)
	}
	if !utf8.Valid(data) {
		return "", ErrInvalidUTF8
	}
	return string(data), nil
}

func cloneMetadata(metadata map[string]string) Metadata {
	if metadata == nil {
		return nil
	}
	clone := make(Metadata, len(metadata))
	for key, value := range metadata {
		clone[key] = value
	}
	return clone
}
