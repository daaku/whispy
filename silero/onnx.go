package silero

import (
	"encoding/binary"
	"math"
	"os"

	"github.com/daaku/serr"
)

// This file walks the protobuf wire format of an ONNX model just far enough to
// pull its float tensors out, so the pure Go model reads the same file OpenVINO
// does instead of shipping its own copy of the weights. Only the fields needed
// for that are understood; everything else is skipped.

// errTruncated is set on the wire reader and checked once at the end.
var errTruncated = serr.Errorf("truncated protobuf")

// wire reads the protobuf wire format. A failed read sets err and the caller
// checks it once, which keeps the parsing code free of error plumbing.
type wire struct {
	b   []byte
	i   int
	err error
}

func (w *wire) more() bool { return w.i < len(w.b) && w.err == nil }

// uvarint reads a base 128 varint.
func (w *wire) uvarint() uint64 {
	var v uint64
	var s uint
	for {
		if w.i >= len(w.b) {
			w.err = errTruncated
			return 0
		}
		c := w.b[w.i]
		w.i++
		v |= uint64(c&0x7f) << s
		if c < 0x80 {
			return v
		}
		s += 7
	}
}

// bytes reads a length delimited field.
func (w *wire) bytes() []byte {
	n := int(w.uvarint())
	if w.err != nil {
		return nil
	}
	if n < 0 || w.i+n > len(w.b) {
		w.err = errTruncated
		return nil
	}
	b := w.b[w.i : w.i+n]
	w.i += n
	return b
}

// skip moves past a field of wire type wt.
func (w *wire) skip(wt int) {
	switch wt {
	case 0:
		w.uvarint()
	case 1:
		w.i += 8
	case 2:
		w.bytes()
	case 5:
		w.i += 4
	default:
		w.err = serr.Errorf("unsupported protobuf wire type %d", wt)
	}
}

// each calls f with every field in the message.
func (w *wire) each(f func(num, wt int)) {
	for w.more() {
		key := w.uvarint()
		if w.err != nil {
			return
		}
		f(int(key>>3), int(key&7))
	}
}

// readInitializers reads the float tensors an ONNX model carries. Tensors that
// are not plain float32 or live outside the file are left out.
func readInitializers(path string) (map[string][]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, serr.Wrap(err)
	}
	model := &wire{b: data}
	var graph []byte
	model.each(func(num, wt int) {
		if num == 7 { // ModelProto.graph
			graph = model.bytes()
			return
		}
		model.skip(wt)
	})
	if model.err != nil {
		return nil, serr.Errorf("silero: %s: %w", path, model.err)
	}
	if graph == nil {
		return nil, serr.Errorf("silero: %s: no graph", path)
	}
	inits := map[string][]float32{}
	g := &wire{b: graph}
	g.each(func(num, wt int) {
		if num != 5 { // GraphProto.initializer
			g.skip(wt)
			return
		}
		name, values, ok := floatTensor(g.bytes())
		if ok {
			inits[name] = values
		}
	})
	if g.err != nil {
		return nil, serr.Errorf("silero: %s: %w", path, g.err)
	}
	return inits, nil
}

// floatTensor reads one TensorProto. Only plain float32 tensors that live in
// the file are returned.
func floatTensor(b []byte) (name string, values []float32, ok bool) {
	t := &wire{b: b}
	var (
		elemType int
		raw      []byte
		outside  bool
	)
	t.each(func(num, wt int) {
		switch num {
		case 1: // dims
			if wt == 2 {
				p := &wire{b: t.bytes()}
				for p.more() {
					p.uvarint()
				}
				return
			}
			t.uvarint()
		case 2: // data_type
			elemType = int(t.uvarint())
		case 4: // float_data
			if wt == 2 {
				p := &wire{b: t.bytes()}
				for p.more() {
					values = append(values, float32(math.Float32frombits(uint32(p.uvarint()))))
				}
				return
			}
			values = append(values, float32(math.Float32frombits(uint32(t.uvarint()))))
		case 8: // name
			name = string(t.bytes())
		case 9: // raw_data
			raw = t.bytes()
		case 13: // external_data
			outside = true
			t.skip(wt)
		default:
			t.skip(wt)
		}
	})
	if t.err != nil || elemType != 1 || outside {
		return name, nil, false
	}
	if len(raw) > 0 {
		if len(raw)%4 != 0 {
			return name, nil, false
		}
		values = make([]float32, len(raw)/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	}
	if len(values) == 0 {
		return name, nil, false
	}
	return name, values, true
}
