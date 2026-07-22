package ingestfs

import (
	"bytes"
	"context"
	"errors"
	"io/fs"
	"math"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/dotcommander/reliquary/pipeline/ingest"
)

func TestNewValidationAndDefaults(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	file := writeTestFile(t, dir, "file.txt", "data")
	symlink := filepath.Join(dir, "root-link")
	if err := os.Symlink(filepath.Base(file), symlink); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		cfg  Config
	}{
		{name: "blank root", cfg: Config{Source: "source"}},
		{name: "blank source", cfg: Config{Root: dir}},
		{name: "negative batch size", cfg: Config{Root: dir, Source: "source", BatchSize: -1}},
		{name: "negative max bytes", cfg: Config{Root: dir, Source: "source", MaxFileBytes: -1}},
		{name: "missing root", cfg: Config{Root: filepath.Join(dir, "missing"), Source: "source"}},
		{name: "file root", cfg: Config{Root: file, Source: "source"}},
		{name: "symlink root", cfg: Config{Root: symlink, Source: "source"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := New(tt.cfg); err == nil {
				t.Fatal("New() error = nil, want validation error")
			}
		})
	}

	reader, err := New(Config{Root: dir, Source: "source"})
	if err != nil {
		t.Fatal(err)
	}
	if reader.batchSize != defaultBatchSize {
		t.Errorf("batchSize = %d, want %d", reader.batchSize, defaultBatchSize)
	}
	if reader.maxFileBytes != defaultMaxFileBytes {
		t.Errorf("maxFileBytes = %d, want %d", reader.maxFileBytes, defaultMaxFileBytes)
	}
}

func TestNewResolvesRelativeRootWithoutExpandingTilde(t *testing.T) {
	oldWorkingDirectory, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldWorkingDirectory) })

	if err := os.Mkdir("tree", 0o755); err != nil {
		t.Fatal(err)
	}
	reader, err := New(Config{Root: "tree", Source: "source"})
	if err != nil {
		t.Fatal(err)
	}
	wantRoot, err := filepath.Abs("tree")
	if err != nil {
		t.Fatal(err)
	}
	if reader.root != wantRoot {
		t.Errorf("root = %q, want %q", reader.root, wantRoot)
	}

	if _, err := New(Config{Root: "~", Source: "source"}); err == nil {
		t.Fatal("New(~) error = nil, want literal missing-path error")
	}
}

func TestReadOrdersFilesAndPreservesProvenance(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	writeTestFile(t, dir, "z.txt", "z")
	writeTestFile(t, dir, ".hidden", "hidden")
	writeTestFile(t, dir, filepath.Join("nested", "b.txt"), "b")
	writeTestFile(t, dir, filepath.Join("nested", "a.txt"), "a")

	reader := newTestReader(t, Config{Root: dir, Source: "disk"})
	batch, err := reader.Read(context.Background(), ingest.Cursor{})
	if err != nil {
		t.Fatal(err)
	}
	wantPaths := []string{".hidden", "nested/a.txt", "nested/b.txt", "z.txt"}
	if got := recordIDs(batch.Records); !reflect.DeepEqual(got, wantPaths) {
		t.Fatalf("record IDs = %v, want %v", got, wantPaths)
	}
	if batch.Cursor != (ingest.Cursor{Source: "disk"}) {
		t.Errorf("cursor = %#v, want final empty token", batch.Cursor)
	}
	for i, record := range batch.Records {
		path := wantPaths[i]
		wantMetadata := map[string]string{
			"source":   "disk",
			"path":     path,
			"filename": filepath.Base(filepath.FromSlash(path)),
		}
		if !reflect.DeepEqual(record.Metadata, wantMetadata) {
			t.Errorf("metadata for %q = %#v, want %#v", path, record.Metadata, wantMetadata)
		}
		for key, value := range record.Metadata {
			if strings.Contains(value, dir) {
				t.Errorf("metadata[%q] leaks absolute root: %q", key, value)
			}
		}
		if strings.Contains(record.ID, dir) || strings.Contains(record.ID, `\\`) {
			t.Errorf("record ID is not slash-normalized relative path: %q", record.ID)
		}
	}
}

func TestReadPrunesDirectoriesFiltersFilesAndSkipsNonRegularEntries(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	writeTestFile(t, dir, "keep.txt", "keep")
	writeTestFile(t, dir, "skip.log", "skip")
	writeTestFile(t, dir, filepath.Join("pruned", "hidden.txt"), "pruned")
	writeTestFile(t, dir, filepath.Join("kept", "nested.txt"), "nested")
	if err := os.Symlink("keep.txt", filepath.Join(dir, "file-link")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("kept", filepath.Join(dir, "dir-link")); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "special-dir"), 0o755); err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("unix", filepath.Join(dir, "special-dir", "socket"))
	if err == nil {
		t.Cleanup(func() { _ = listener.Close() })
	}

	var visited []string
	reader := newTestReader(t, Config{
		Root:   dir,
		Source: "disk",
		Include: func(path string, entry fs.DirEntry) bool {
			visited = append(visited, path)
			return path != "pruned" && (entry.IsDir() || filepath.Ext(path) == ".txt")
		},
	})
	batch, err := reader.Read(context.Background(), ingest.Cursor{})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := recordIDs(batch.Records), []string{"keep.txt", "kept/nested.txt"}; !reflect.DeepEqual(got, want) {
		t.Errorf("record IDs = %v, want %v", got, want)
	}
	for _, path := range visited {
		if strings.HasPrefix(path, "pruned/") || strings.HasPrefix(path, "dir-link/") {
			t.Errorf("visited entry under pruned or symlinked directory: %q", path)
		}
	}
}

func TestReadBatchBoundariesAndCursors(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	for _, path := range []string{"a", "b", "c", "d", "e"} {
		writeTestFile(t, dir, path, path)
	}
	reader := newTestReader(t, Config{Root: dir, Source: "disk", BatchSize: 2})

	tests := []struct {
		cursor ingest.Cursor
		ids    []string
		token  string
	}{
		{cursor: ingest.Cursor{}, ids: []string{"a", "b"}, token: "b"},
		{cursor: ingest.Cursor{Source: "disk", Token: "b"}, ids: []string{"c", "d"}, token: "d"},
		{cursor: ingest.Cursor{Source: "disk", Token: "d"}, ids: []string{"e"}, token: ""},
		{cursor: ingest.Cursor{Source: "disk", Token: "e"}, ids: nil, token: ""},
	}
	for i, tt := range tests {
		batch, err := reader.Read(context.Background(), tt.cursor)
		if err != nil {
			t.Fatalf("read %d: %v", i, err)
		}
		if got := recordIDs(batch.Records); !reflect.DeepEqual(got, tt.ids) {
			t.Errorf("read %d IDs = %v, want %v", i, got, tt.ids)
		}
		if batch.Cursor != (ingest.Cursor{Source: "disk", Token: tt.token}) {
			t.Errorf("read %d cursor = %#v, want token %q", i, batch.Cursor, tt.token)
		}
	}
}

func TestReadMaximumBatchSizeAfterCursorDoesNotOverflow(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	for _, path := range []string{"a", "b", "c"} {
		writeTestFile(t, dir, path, path)
	}
	reader := newTestReader(t, Config{Root: dir, Source: "disk", BatchSize: math.MaxInt})
	batch, err := reader.Read(context.Background(), ingest.Cursor{Source: "disk", Token: "a"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := recordIDs(batch.Records), []string{"b", "c"}; !reflect.DeepEqual(got, want) {
		t.Errorf("record IDs = %v, want %v", got, want)
	}
	if batch.Cursor != (ingest.Cursor{Source: "disk"}) {
		t.Errorf("cursor = %#v, want final empty token", batch.Cursor)
	}
}

func TestReadEmptyAndFullyFilteredTrees(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name    string
		prepare func(*testing.T, string)
		include func(string, fs.DirEntry) bool
	}{
		{name: "empty"},
		{name: "filtered", prepare: func(t *testing.T, dir string) { writeTestFile(t, dir, "file", "data") }, include: func(string, fs.DirEntry) bool { return false }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			dir := t.TempDir()
			if tc.prepare != nil {
				tc.prepare(t, dir)
			}
			reader := newTestReader(t, Config{Root: dir, Source: "disk", Include: tc.include})
			batch, err := reader.Read(context.Background(), ingest.Cursor{})
			if err != nil {
				t.Fatal(err)
			}
			if len(batch.Records) != 0 || batch.Cursor != (ingest.Cursor{Source: "disk"}) {
				t.Errorf("batch = %#v, want empty records and final cursor", batch)
			}
		})
	}
}

func TestReadRejectsMismatchedCursorSource(t *testing.T) {
	t.Parallel()

	reader := newTestReader(t, Config{Root: t.TempDir(), Source: "disk"})
	for _, cursor := range []ingest.Cursor{
		{Source: "other"},
		{Token: "file"},
	} {
		if _, err := reader.Read(context.Background(), cursor); err == nil {
			t.Errorf("Read(%#v) error = nil, want source mismatch", cursor)
		}
	}
}

func TestReadResumesAfterDeletedToken(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	writeTestFile(t, dir, "a", "a")
	tokenPath := writeTestFile(t, dir, "b", "b")
	writeTestFile(t, dir, "c", "c")
	if err := os.Remove(tokenPath); err != nil {
		t.Fatal(err)
	}
	reader := newTestReader(t, Config{Root: dir, Source: "disk", BatchSize: 1})
	batch, err := reader.Read(context.Background(), ingest.Cursor{Source: "disk", Token: "b"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := recordIDs(batch.Records), []string{"c"}; !reflect.DeepEqual(got, want) {
		t.Errorf("record IDs = %v, want %v", got, want)
	}
}

func TestReadSnapshotIgnoresAddedFiles(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	writeTestFile(t, dir, "a", "a")
	writeTestFile(t, dir, "c", "c")
	reader := newTestReader(t, Config{Root: dir, Source: "disk", BatchSize: 1})
	first, err := reader.Read(context.Background(), ingest.Cursor{})
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, dir, "b", "b")
	second, err := reader.Read(context.Background(), first.Cursor)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := recordIDs(second.Records), []string{"c"}; !reflect.DeepEqual(got, want) {
		t.Errorf("record IDs = %v, want snapshot-only %v", got, want)
	}

	fresh := newTestReader(t, Config{Root: dir, Source: "disk"})
	batch, err := fresh.Read(context.Background(), ingest.Cursor{})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := recordIDs(batch.Records), []string{"a", "b", "c"}; !reflect.DeepEqual(got, want) {
		t.Errorf("fresh reader IDs = %v, want %v", got, want)
	}
}

func TestReadFailsWhenSnapshottedFileIsDeletedOrChanged(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		name   string
		mutate func(*testing.T, string)
	}{
		{name: "deleted", mutate: func(t *testing.T, path string) {
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
		}},
		{name: "changed", mutate: func(t *testing.T, path string) {
			if err := os.WriteFile(path, []byte("changed-size"), 0o644); err != nil {
				t.Fatal(err)
			}
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			dir := t.TempDir()
			writeTestFile(t, dir, "a", "a")
			path := writeTestFile(t, dir, "b", "b")
			reader := newTestReader(t, Config{Root: dir, Source: "disk", BatchSize: 1})
			first, err := reader.Read(context.Background(), ingest.Cursor{})
			if err != nil {
				t.Fatal(err)
			}
			tt.mutate(t, path)
			batch, err := reader.Read(context.Background(), first.Cursor)
			if err == nil {
				t.Fatal("Read() error = nil, want snapshot mutation error")
			}
			if len(batch.Records) != 0 {
				t.Errorf("Read() returned partial records: %#v", batch.Records)
			}
			if !strings.Contains(err.Error(), "b") {
				t.Errorf("error %q does not identify relative path", err)
			}
		})
	}
}

func TestReadSizeLimitAndNoPartialBatch(t *testing.T) {
	t.Parallel()

	t.Run("exact limit", func(t *testing.T) {
		t.Parallel()
		dir := t.TempDir()
		writeTestFile(t, dir, "file", "1234")
		reader := newTestReader(t, Config{Root: dir, Source: "disk", MaxFileBytes: 4})
		batch, err := reader.Read(context.Background(), ingest.Cursor{})
		if err != nil {
			t.Fatal(err)
		}
		if got := string(batch.Records[0].Payload); got != "1234" {
			t.Errorf("payload = %q, want exact-limit contents", got)
		}
	})

	t.Run("one byte overflow", func(t *testing.T) {
		t.Parallel()
		dir := t.TempDir()
		writeTestFile(t, dir, "a-small", "ok")
		writeTestFile(t, dir, "b-large", "12345")
		reader := newTestReader(t, Config{Root: dir, Source: "disk", MaxFileBytes: 4})
		batch, err := reader.Read(context.Background(), ingest.Cursor{})
		if !errors.Is(err, ErrFileTooLarge) {
			t.Fatalf("error = %v, want errors.Is(ErrFileTooLarge)", err)
		}
		if len(batch.Records) != 0 {
			t.Errorf("Read() returned partial records: %#v", batch.Records)
		}
		if !strings.Contains(err.Error(), "b-large") {
			t.Errorf("error %q does not identify relative path", err)
		}
	})
}

func TestReadOpenAndReadFailuresReturnNoPartialBatch(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		name string
		open func(original func(string) (sourceFile, error), target string, wantErr error) func(string) (sourceFile, error)
	}{
		{
			name: "open",
			open: func(original func(string) (sourceFile, error), target string, wantErr error) func(string) (sourceFile, error) {
				return func(path string) (sourceFile, error) {
					if path == target {
						return nil, wantErr
					}
					return original(path)
				}
			},
		},
		{
			name: "read",
			open: func(original func(string) (sourceFile, error), target string, wantErr error) func(string) (sourceFile, error) {
				return func(path string) (sourceFile, error) {
					file, err := original(path)
					if err != nil || path != target {
						return file, err
					}
					return &errorSourceFile{sourceFile: file, err: wantErr}, nil
				}
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			dir := t.TempDir()
			writeTestFile(t, dir, "a", "a")
			target := writeTestFile(t, dir, "b", "b")
			reader := newTestReader(t, Config{Root: dir, Source: "disk"})
			if err := reader.takeSnapshot(context.Background()); err != nil {
				t.Fatal(err)
			}
			wantErr := errors.New(tt.name + " failure")
			reader.open = tt.open(reader.open, target, wantErr)

			batch, err := reader.Read(context.Background(), ingest.Cursor{})
			if !errors.Is(err, wantErr) {
				t.Fatalf("error = %v, want errors.Is(%v)", err, wantErr)
			}
			if len(batch.Records) != 0 {
				t.Errorf("Read() returned partial records: %#v", batch.Records)
			}
			if !strings.Contains(err.Error(), "b") {
				t.Errorf("error %q does not identify relative path", err)
			}
		})
	}
}

func TestReadCancellationDuringTraversal(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	writeTestFile(t, dir, "a", "a")
	writeTestFile(t, dir, "b", "b")
	ctx, cancel := context.WithCancel(context.Background())
	reader := newTestReader(t, Config{
		Root:   dir,
		Source: "disk",
		Include: func(path string, _ fs.DirEntry) bool {
			if path == "a" {
				cancel()
			}
			return true
		},
	})
	batch, err := reader.Read(ctx, ingest.Cursor{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if len(batch.Records) != 0 {
		t.Errorf("Read() returned partial records: %#v", batch.Records)
	}
}

func TestReadCancellationDuringFileReadReturnsNoPartialBatch(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	writeTestFile(t, dir, "a", "a")
	target := writeTestFile(t, dir, "b", strings.Repeat("b", 128<<10))
	reader := newTestReader(t, Config{Root: dir, Source: "disk", MaxFileBytes: 256 << 10})
	if err := reader.takeSnapshot(context.Background()); err != nil {
		t.Fatal(err)
	}
	originalOpen := reader.open
	ctx, cancel := context.WithCancel(context.Background())
	reader.open = func(path string) (sourceFile, error) {
		file, err := originalOpen(path)
		if err != nil || path != target {
			return file, err
		}
		return &cancelingSourceFile{sourceFile: file, cancel: cancel}, nil
	}

	batch, err := reader.Read(ctx, ingest.Cursor{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if len(batch.Records) != 0 {
		t.Errorf("Read() returned partial records: %#v", batch.Records)
	}
}

func TestReadBoundedCancellationAndOverflow(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	reader := &cancelingReader{cancel: cancel}
	data, err := readBounded(ctx, reader, 1024)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if data != nil {
		t.Errorf("data = %q, want nil on cancellation", data)
	}

	data, err = readBounded(context.Background(), bytes.NewBufferString("12345"), 4)
	if !errors.Is(err, ErrFileTooLarge) {
		t.Fatalf("overflow error = %v, want ErrFileTooLarge", err)
	}
	if data != nil {
		t.Errorf("overflow data = %q, want nil", data)
	}
}

type cancelingReader struct {
	cancel context.CancelFunc
	read   bool
}

type errorSourceFile struct {
	sourceFile
	err error
}

func (f *errorSourceFile) Read([]byte) (int, error) {
	return 0, f.err
}

type cancelingSourceFile struct {
	sourceFile
	cancel context.CancelFunc
	done   bool
}

func (f *cancelingSourceFile) Read(buffer []byte) (int, error) {
	n, err := f.sourceFile.Read(buffer)
	if !f.done {
		f.done = true
		f.cancel()
	}
	return n, err
}

func (r *cancelingReader) Read(p []byte) (int, error) {
	if !r.read {
		r.read = true
		p[0] = 'x'
		r.cancel()
		return 1, nil
	}
	return 0, errors.New("unexpected second read")
}

func newTestReader(t *testing.T, cfg Config) *Reader {
	t.Helper()
	reader, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return reader
}

func writeTestFile(t *testing.T, root, relativePath, contents string) string {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(relativePath))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func recordIDs[T any](records []ingest.Record[T]) []string {
	if len(records) == 0 {
		return nil
	}
	ids := make([]string, len(records))
	for i, record := range records {
		ids[i] = record.ID
	}
	return ids
}
