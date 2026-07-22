// Package ingestfs reads deterministic, resumable batches from local directory trees.
package ingestfs

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/dotcommander/reliquary/pipeline/ingest"
)

const (
	defaultBatchSize    = 64
	defaultMaxFileBytes = 16 << 20
)

// ErrFileTooLarge indicates that a file exceeds the configured byte limit.
var ErrFileTooLarge = errors.New("ingest filesystem file too large")

// Config configures a local filesystem reader.
type Config struct {
	Root   string
	Source string
	// BatchSize defaults to 64. Negative values are invalid.
	BatchSize int
	// MaxFileBytes defaults to 16 MiB. Negative values are invalid.
	MaxFileBytes int64
	// Include may reject files or prune directories. Paths are slash-normalized
	// and relative to Root.
	Include func(path string, entry fs.DirEntry) bool
}

// Reader snapshots and reads one local directory tree.
type Reader struct {
	root         string
	source       string
	batchSize    int
	maxFileBytes int64
	include      func(string, fs.DirEntry) bool
	open         func(string) (sourceFile, error)

	snapshot []snapshotFile
	ready    bool
}

type sourceFile interface {
	io.Reader
	Stat() (fs.FileInfo, error)
	Close() error
}

type snapshotFile struct {
	path    string
	size    int64
	mode    fs.FileMode
	modTime time.Time
	info    fs.FileInfo
}

// New validates cfg and returns a reader. Relative roots are resolved to
// absolute paths without shell expansion.
func New(cfg Config) (*Reader, error) {
	if strings.TrimSpace(cfg.Root) == "" {
		return nil, errors.New("ingest filesystem root is blank")
	}
	if strings.TrimSpace(cfg.Source) == "" {
		return nil, errors.New("ingest filesystem source is blank")
	}
	if cfg.BatchSize < 0 {
		return nil, fmt.Errorf("ingest filesystem batch size must not be negative: %d", cfg.BatchSize)
	}
	if cfg.MaxFileBytes < 0 {
		return nil, fmt.Errorf("ingest filesystem max file bytes must not be negative: %d", cfg.MaxFileBytes)
	}

	root, err := filepath.Abs(cfg.Root)
	if err != nil {
		return nil, fmt.Errorf("resolve ingest filesystem root: %w", err)
	}
	info, err := os.Lstat(root)
	if err != nil {
		return nil, fmt.Errorf("inspect ingest filesystem root: %w", err)
	}
	if info.Mode()&fs.ModeSymlink != 0 {
		return nil, errors.New("ingest filesystem root must not be a symlink")
	}
	if !info.IsDir() {
		return nil, errors.New("ingest filesystem root must be a directory")
	}

	batchSize := cfg.BatchSize
	if batchSize == 0 {
		batchSize = defaultBatchSize
	}
	maxFileBytes := cfg.MaxFileBytes
	if maxFileBytes == 0 {
		maxFileBytes = defaultMaxFileBytes
	}

	return &Reader{
		root:         root,
		source:       cfg.Source,
		batchSize:    batchSize,
		maxFileBytes: maxFileBytes,
		include:      cfg.Include,
		open: func(path string) (sourceFile, error) {
			return os.Open(path)
		},
	}, nil
}

// Read returns the next lexical batch. The first call snapshots the tree for
// this reader; later additions are not observed.
func (r *Reader) Read(ctx context.Context, cursor ingest.Cursor) (ingest.Batch[[]byte], error) {
	if err := r.validateCursor(cursor); err != nil {
		return ingest.Batch[[]byte]{}, err
	}
	if !r.ready {
		if err := r.takeSnapshot(ctx); err != nil {
			return ingest.Batch[[]byte]{}, err
		}
	}

	start := 0
	if cursor.Token != "" {
		start = sort.Search(len(r.snapshot), func(i int) bool {
			return r.snapshot[i].path > cursor.Token
		})
	}
	if start == len(r.snapshot) {
		return ingest.Batch[[]byte]{Cursor: ingest.Cursor{Source: r.source}}, nil
	}

	remaining := len(r.snapshot) - start
	end := start + min(r.batchSize, remaining)
	records := make([]ingest.Record[[]byte], 0, end-start)
	for _, file := range r.snapshot[start:end] {
		payload, err := r.readFile(ctx, file)
		if err != nil {
			return ingest.Batch[[]byte]{}, fmt.Errorf("%s: %w", file.path, err)
		}
		records = append(records, ingest.Record[[]byte]{
			ID:      file.path,
			Payload: payload,
			Metadata: map[string]string{
				"source":   r.source,
				"path":     file.path,
				"filename": filepath.Base(filepath.FromSlash(file.path)),
			},
		})
	}

	token := ""
	if end < len(r.snapshot) {
		token = r.snapshot[end-1].path
	}
	return ingest.Batch[[]byte]{
		Records: records,
		Cursor:  ingest.Cursor{Source: r.source, Token: token},
	}, nil
}

func (r *Reader) validateCursor(cursor ingest.Cursor) error {
	if cursor.Source == "" && cursor.Token == "" {
		return nil
	}
	if cursor.Source != r.source {
		return fmt.Errorf("ingest filesystem cursor source %q does not match %q", cursor.Source, r.source)
	}
	return nil
}

func (r *Reader) takeSnapshot(ctx context.Context) error {
	files := make([]snapshotFile, 0)
	err := filepath.WalkDir(r.root, func(path string, entry fs.DirEntry, walkErr error) error {
		rel, err := filepath.Rel(r.root, path)
		if err != nil {
			return fmt.Errorf("resolve relative path: %w", err)
		}
		rel = filepath.ToSlash(rel)
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("%s: %w", rel, err)
		}
		if walkErr != nil {
			return fmt.Errorf("%s: %w", rel, walkErr)
		}
		if rel == "." {
			return nil
		}
		if entry.Type()&fs.ModeSymlink != 0 {
			if entry.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if r.include != nil && !r.include(rel, entry) {
			if entry.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if entry.IsDir() {
			return nil
		}

		info, err := entry.Info()
		if err != nil {
			return fmt.Errorf("%s: inspect: %w", rel, err)
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		files = append(files, snapshotFile{
			path:    rel,
			size:    info.Size(),
			mode:    info.Mode(),
			modTime: info.ModTime(),
			info:    info,
		})
		return nil
	})
	if err != nil {
		return fmt.Errorf("snapshot ingest filesystem tree: %w", err)
	}
	sort.Slice(files, func(i, j int) bool { return files[i].path < files[j].path })
	r.snapshot = files
	r.ready = true
	return nil
}

func (r *Reader) readFile(ctx context.Context, snapshot snapshotFile) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	path := filepath.Join(r.root, filepath.FromSlash(snapshot.path))
	pathInfo, err := os.Lstat(path)
	if err != nil {
		return nil, fmt.Errorf("inspect before open: %w", err)
	}
	if !sameSnapshot(snapshot, pathInfo) {
		return nil, errors.New("file changed after snapshot")
	}
	file, err := r.open(path)
	if err != nil {
		return nil, fmt.Errorf("open: %w", err)
	}
	defer file.Close()

	before, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("inspect before read: %w", err)
	}
	if !sameSnapshot(snapshot, before) {
		return nil, errors.New("file changed after snapshot")
	}
	if before.Size() > r.maxFileBytes {
		return nil, fmt.Errorf("%w: limit is %d bytes", ErrFileTooLarge, r.maxFileBytes)
	}

	data, err := readBounded(ctx, file, r.maxFileBytes)
	if err != nil {
		return nil, err
	}
	after, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("inspect after read: %w", err)
	}
	if !sameSnapshot(snapshot, after) {
		return nil, errors.New("file changed after snapshot")
	}
	pathInfo, err = os.Lstat(path)
	if err != nil {
		return nil, fmt.Errorf("inspect after read: %w", err)
	}
	if !sameSnapshot(snapshot, pathInfo) {
		return nil, errors.New("file changed after snapshot")
	}
	return data, nil
}

func sameSnapshot(snapshot snapshotFile, current fs.FileInfo) bool {
	return current.Mode().IsRegular() &&
		current.Size() == snapshot.size &&
		current.Mode() == snapshot.mode &&
		current.ModTime().Equal(snapshot.modTime) &&
		os.SameFile(snapshot.info, current)
}

func readBounded(ctx context.Context, reader io.Reader, maxBytes int64) ([]byte, error) {
	readLimit := maxBytes
	if maxBytes < math.MaxInt64 {
		readLimit++
	}
	limited := io.LimitReader(reader, readLimit)
	capacity := maxBytes
	if capacity > 64<<10 {
		capacity = 64 << 10
	}
	data := make([]byte, 0, int(capacity))
	buf := make([]byte, 64<<10)
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n, err := limited.Read(buf)
		if n > 0 {
			data = append(data, buf[:n]...)
			if int64(len(data)) > maxBytes {
				return nil, fmt.Errorf("%w: limit is %d bytes", ErrFileTooLarge, maxBytes)
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return data, nil
			}
			return nil, fmt.Errorf("read: %w", err)
		}
	}
}
