package parakeet

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// axisLayer builds one OpenVINO IR layer whose data element carries axis, with
// an input and an output port of rank dimensions, the way the exporter writes
// them.
func axisLayer(typ, axis string, rank int) string {
	var dims strings.Builder
	for range rank {
		dims.WriteString("<dim>1</dim>")
	}
	return `<layer id="26" name="logits" type="` + typ + `" version="opset5">
			<data axis="` + axis + `" />
			<input>
				<port id="0" precision="FP32">
					` + dims.String() + `
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					` + dims.String() + `
				</port>
			</output>
		</layer>`
}

func TestNormalizeLogSoftmaxAxis(t *testing.T) {
	cases := []struct {
		name string
		xml  string
		want string
	}{
		{
			// The joint network's actual shape: rank 4, axis -1, which is 3.
			"joint",
			axisLayer("LogSoftmax", "-1", 4),
			`axis="3"`,
		},
		{
			"rank three",
			axisLayer("LogSoftmax", "-1", 3),
			`axis="2"`,
		},
		{
			"deeper than one",
			axisLayer("LogSoftmax", "-2", 4),
			`axis="2"`,
		},
		{
			// The rank does not reach the axis, so there is no positive
			// spelling of it.
			"out of range",
			axisLayer("LogSoftmax", "-3", 2),
			`axis="-3"`,
		},
		{
			"already positive",
			axisLayer("LogSoftmax", "3", 4),
			`axis="3"`,
		},
		{
			// Another op's negative axis means something else and is none of
			// this function's business.
			"another op",
			axisLayer("Softmax", "-1", 4),
			`axis="-1"`,
		},
		{
			"no log softmax",
			axisLayer("Add", "0", 4),
			`axis="0"`,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := string(normalizeLogSoftmaxAxis([]byte(c.xml)))
			if !strings.Contains(got, c.want) {
				t.Fatalf("normalized to %q, want it to contain %q", got, c.want)
			}
			if c.want != `axis="-1"` && strings.Contains(got, `axis="-1"`) {
				t.Fatalf("normalized to %q, a negative axis survived", got)
			}
		})
	}
}

func TestNormalizeLogSoftmaxAxisMultipleLayers(t *testing.T) {
	xml := axisLayer("LogSoftmax", "-1", 4) + axisLayer("LogSoftmax", "-1", 3)
	got := string(normalizeLogSoftmaxAxis([]byte(xml)))
	if strings.Contains(got, `axis="-1"`) {
		t.Fatalf("normalized to %q, a negative axis survived", got)
	}
	if n := strings.Count(got, `axis="3"`); n != 1 {
		t.Fatalf("rank 4 layer rewrote %d times, want once: %q", n, got)
	}
	if n := strings.Count(got, `axis="2"`); n != 1 {
		t.Fatalf("rank 3 layer rewrote %d times, want once: %q", n, got)
	}
}

// The layer with no input has no rank, so there is nothing to normalize even
// if it carries a negative axis.
func TestNormalizeLogSoftmaxAxisNoInput(t *testing.T) {
	xml := `<layer id="1" name="c" type="Constant" version="opset1">` +
		`<data shape="1,4" element_type="f32" />` +
		`<output><port id="0"><dim>1</dim><dim>4</dim></port></output></layer>` +
		axisLayer("LogSoftmax", "-1", 4)
	got := string(normalizeLogSoftmaxAxis([]byte(xml)))
	if !strings.Contains(got, `axis="3"`) {
		t.Fatalf("normalized to %q, want the LogSoftmax rewritten", got)
	}
}

// Only the NPU compiler rejects the negative axis; the routing has to catch the
// spellings of the device, not just the exact string.
func TestNeedsPositiveAxis(t *testing.T) {
	for _, device := range []string{"NPU", "npu", "NPU.0", "Npu.1"} {
		if !needsPositiveAxis(device) {
			t.Errorf("needsPositiveAxis(%q) = false, want true", device)
		}
	}
	for _, device := range []string{"CPU", "GPU", "AUTO", ""} {
		if needsPositiveAxis(device) {
			t.Errorf("needsPositiveAxis(%q) = true, want false", device)
		}
	}
}

// The shipped joint network ends in a LogSoftmax with axis="-1"; that is the
// model the NPU refuses, and the one this function exists for.
func TestNormalizeLogSoftmaxAxisRealModel(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(modelDir(t), "parakeet_joint.xml"))
	if err != nil {
		t.Fatal(err)
	}
	patched := normalizeLogSoftmaxAxis(data)
	if bytes.Equal(patched, data) {
		t.Fatal("the joint network has no negative LogSoftmax axis; update this test with the model")
	}
	if bytes.Contains(patched, []byte(`axis="-1"`)) {
		t.Fatal("a negative axis survived")
	}
	if !bytes.Contains(patched, []byte(`axis="3"`)) {
		t.Fatalf("want the joint's rank 4 axis written as 3")
	}
}
