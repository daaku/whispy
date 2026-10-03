package parakeet

import (
	"bytes"
	"strconv"
)

// normalizeLogSoftmaxAxis rewrites a negative axis on every LogSoftmax layer of
// an OpenVINO IR XML to the positive axis that means the same thing. The NPU
// driver compiler rejects a negative axis ("Got negative index -1 for Dim" from
// its AlignDimensionsForDPU pass), which the joint network's final LogSoftmax
// carries; the CPU and the other devices accept it, and the positive axis
// computes the same logits (TestJointSoftmaxAxis). It returns the input slice
// unchanged when there is nothing to rewrite.
//
// The rewrite is text surgery because the OpenVINO C API has no way to set an
// op attribute. The only route to a patched graph is to read a patched IR, so
// compileWithNormalizedAxis writes this out and compiles that.
func normalizeLogSoftmaxAxis(xml []byte) []byte {
	const marker = `type="LogSoftmax"`
	out := xml
	from := 0
	for {
		rel := bytes.Index(out[from:], []byte(marker))
		if rel < 0 {
			return out
		}
		at := from + rel

		// The marker names the layer; its data, inputs and outputs sit between
		// <layer and </layer>.
		start := bytes.LastIndex(out[:at], []byte("<layer"))
		if start < 0 {
			from = at + len(marker)
			continue
		}
		relEnd := bytes.Index(out[at:], []byte("</layer>"))
		if relEnd < 0 {
			return out
		}
		end := at + relEnd

		attrAt, repl, ok := positiveAxis(out[start:end])
		if !ok {
			from = end
			continue
		}
		attrStart := start + attrAt
		attrEnd := attrStart + axisAttrLen(out[attrStart:])
		if attrEnd <= attrStart {
			from = end
			continue
		}
		next := make([]byte, 0, len(out)-(attrEnd-attrStart)+len(repl))
		next = append(next, out[:attrStart]...)
		next = append(next, repl...)
		next = append(next, out[attrEnd:]...)
		out = next
		from = attrStart + len(repl)
	}
}

// positiveAxis finds the axis attribute in a layer block and returns the offset
// of its first byte within the block and the attribute rewritten to a positive
// value. It reports false when the layer has no negative axis or the axis
// cannot be made positive, which is when the rank is unknown or smaller than
// the axis is deep.
func positiveAxis(layer []byte) (offset int, repl []byte, ok bool) {
	// The axis lives in the layer's <data> element; anything before it is not
	// the op's axis.
	data := bytes.Index(layer, []byte("<data"))
	if data < 0 {
		return 0, nil, false
	}
	rel := bytes.Index(layer[data:], []byte(`axis="-`))
	if rel < 0 {
		return 0, nil, false
	}
	offset = data + rel

	numAt := offset + len(`axis="`)
	numEnd := bytes.IndexByte(layer[numAt:], '"')
	if numEnd < 0 {
		return 0, nil, false
	}
	axis, err := strconv.Atoi(string(layer[numAt : numAt+numEnd]))
	if err != nil || axis >= 0 {
		return 0, nil, false
	}
	rank := firstInputRank(layer)
	if rank <= 0 || axis+rank < 0 {
		return 0, nil, false
	}
	return offset, []byte(`axis="` + strconv.Itoa(axis+rank) + `"`), true
}

// axisAttrLen is the length of the axis="..." attribute that starts s.
func axisAttrLen(s []byte) int {
	if !bytes.HasPrefix(s, []byte(`axis="`)) {
		return 0
	}
	end := bytes.IndexByte(s[len(`axis="`):], '"')
	if end < 0 {
		return 0
	}
	return len(`axis="`) + end + 1
}

// firstInputRank counts the dimensions of the first input port of a layer
// block. A LogSoftmax has exactly one input, so its rank is the axis to write:
// axis -1 on a rank 4 tensor is axis 3.
func firstInputRank(layer []byte) int {
	open := bytes.Index(layer, []byte("<input>"))
	if open < 0 {
		return 0
	}
	rel := bytes.Index(layer[open:], []byte("</input>"))
	if rel < 0 {
		return 0
	}
	return bytes.Count(layer[open:open+rel], []byte("<dim>"))
}
