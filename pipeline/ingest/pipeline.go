package ingest

import "context"

// Pipeline wires a reader, decoder, mapper, and sink into a resumable ingest run.
//
// It keeps source policy at the caller boundary: readers own pagination, decoders
// own source formats, mappers own semantics, and sinks own persistence.
type Pipeline[Decoded, Out any] struct {
	reader Reader[[]byte]
	decode func(context.Context, Record[[]byte]) ([]Record[Decoded], error)
	mapper Mapper[Decoded, Out]
	sink   Sink[Out]
}

// NewPipeline constructs a pipeline from the existing ingest contracts.
func NewPipeline[Decoded, Out any](
	reader Reader[[]byte],
	decoder Decoder[Decoded],
	mapper Mapper[Decoded, Out],
	sink Sink[Out],
) *Pipeline[Decoded, Out] {
	return &Pipeline[Decoded, Out]{
		reader: reader,
		decode: func(ctx context.Context, record Record[[]byte]) ([]Record[Decoded], error) {
			return decoder.Decode(ctx, record.Payload)
		},
		mapper: mapper,
		sink:   sink,
	}
}

// NewRecordPipeline constructs a pipeline whose decoder receives each raw
// record's ID, metadata, and payload. The decoder owns propagation of that
// envelope to its output records.
func NewRecordPipeline[Decoded, Out any](
	reader Reader[[]byte],
	decoder RecordDecoder[Decoded],
	mapper Mapper[Decoded, Out],
	sink Sink[Out],
) *Pipeline[Decoded, Out] {
	return &Pipeline[Decoded, Out]{
		reader: reader,
		decode: func(ctx context.Context, record Record[[]byte]) ([]Record[Decoded], error) {
			return decoder.DecodeRecord(ctx, record)
		},
		mapper: mapper,
		sink:   sink,
	}
}

// Run reads, decodes, maps, and writes batches until it successfully writes a
// batch with an empty cursor token. Decode/map errors are counted and skipped;
// read, write, and non-advancing cursor errors abort the run because the cursor
// or destination state is ambiguous.
func (p *Pipeline[Decoded, Out]) Run(ctx context.Context, cursor Cursor) (Report, error) {
	report := Report{Cursor: cursor}
	if p == nil {
		return report, nil
	}
	for {
		batch, err := p.reader.Read(ctx, cursor)
		if err != nil {
			return report, err
		}
		report.Read += len(batch.Records)
		if batch.Cursor.Token != "" && batch.Cursor == cursor {
			return report, ErrCursorNotAdvanced
		}

		mapped := make([]Record[Out], 0, len(batch.Records))
		for _, raw := range batch.Records {
			decoded, err := p.decode(ctx, raw)
			if err != nil {
				report.Errors++
				continue
			}
			for _, rec := range decoded {
				out, err := p.mapper.Map(ctx, rec)
				if err != nil {
					report.Errors++
					continue
				}
				mapped = append(mapped, out)
			}
		}

		rep, err := p.sink.Write(ctx, Batch[Out]{Records: mapped, Cursor: batch.Cursor})
		report.Written += rep.Written
		report.Skipped += rep.Skipped
		report.Errors += rep.Errors
		if err != nil {
			return report, err
		}
		report.Cursor = batch.Cursor

		cursor = batch.Cursor
		if batch.Cursor.Token == "" {
			break
		}
	}
	return report, nil
}
