package openvino

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTensorF32(t *testing.T) {
	tensor, data, err := NewF32Tensor([]int64{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()
	if len(data) != 6 {
		t.Fatalf("len(data) = %d, want 6", len(data))
	}
	for i, v := range data {
		if v != 0 {
			t.Fatalf("data[%d] = %v, want zeroed", i, v)
		}
	}
	if size, err := tensor.Size(); err != nil || size != 6 {
		t.Fatalf("size = %d, %v, want 6", size, err)
	}
	shape, err := tensor.Shape()
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("shape = %v, want [2 3]", shape)
	}
	et, err := tensor.ElementType()
	if err != nil {
		t.Fatal(err)
	}
	if et != F32 {
		t.Fatalf("element type = %d, want f32", int(et))
	}
}

func TestTensorSetInt(t *testing.T) {
	for _, et := range []ElementType{I32, I64} {
		tensor, err := NewTensor(et, []int64{1})
		if err != nil {
			t.Fatal(err)
		}
		if err := tensor.SetInt(42); err != nil {
			t.Fatal(err)
		}
		got, err := tensor.Int64()
		if err != nil {
			t.Fatal(err)
		}
		if got != 42 {
			t.Fatalf("type %d: Int64 = %d, want 42", int(et), got)
		}
		tensor.Close()
	}
}

func TestTensorBadShapes(t *testing.T) {
	// Zero rank would be a scalar, which ov_shape_create refuses.
	if _, err := NewTensor(F32, nil); err == nil {
		t.Fatal("zero rank: expected an error")
	}
	_, err := NewTensor(F32, []int64{-1})
	if err == nil {
		t.Fatal("negative dimension: expected an error")
	}
	if !strings.Contains(err.Error(), "create") {
		t.Fatalf("error %q does not name the failing call", err)
	}
}

const dynamicAddIR = `<?xml version="1.0"?>
<net name="add" version="11">
	<layers>
		<layer id="0" name="a" type="Parameter" version="opset1">
			<data shape="1,?" element_type="f32" />
			<output>
				<port id="0" precision="FP32" names="a">
					<dim>1</dim>
					<dim>-1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="b" type="Parameter" version="opset1">
			<data shape="1,?" element_type="f32" />
			<output>
				<port id="0" precision="FP32" names="b">
					<dim>1</dim>
					<dim>-1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="add" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>-1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>-1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="sum">
					<dim>1</dim>
					<dim>-1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="sum" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>-1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0" />
	</edges>
</net>
`

func TestDynamicModel(t *testing.T) {
	path := filepath.Join(t.TempDir(), "add.xml")
	if err := os.WriteFile(path, []byte(dynamicAddIR), 0o644); err != nil {
		t.Fatal(err)
	}
	core, err := SharedCore()
	if err != nil {
		t.Fatal(err)
	}
	cm, err := core.Compile(path, "CPU")
	if err != nil {
		t.Fatal(err)
	}
	defer cm.Close()

	in, err := cm.InputByIndex(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("input a: dims=%v dynamicRank=%v type=%d", in.Dims, in.DynamicRank, int(in.Type))
	if len(in.Dims) != 2 || in.Dims[0] != 1 || in.Dims[1] != -1 || in.Dim(1) != 0 {
		t.Fatalf("unexpected dynamic shape %v", in.Dims)
	}
	out, err := cm.Output("sum")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("output sum: dims=%v dynamicRank=%v", out.Dims, out.DynamicRank)

	req, err := cm.Request()
	if err != nil {
		t.Fatal(err)
	}
	defer req.Close()
	for _, width := range []int64{4, 7} {
		a, ad, err := NewF32Tensor([]int64{1, width})
		if err != nil {
			t.Fatal(err)
		}
		b, bd, err := NewF32Tensor([]int64{1, width})
		if err != nil {
			t.Fatal(err)
		}
		for i := range ad {
			ad[i] = float32(i)
			bd[i] = float32(2 * i)
		}
		if err := req.Set("a", a); err != nil {
			t.Fatal(err)
		}
		if err := req.Set("b", b); err != nil {
			t.Fatal(err)
		}
		if err := req.Infer(); err != nil {
			t.Fatal(err)
		}
		sum, err := req.Get("sum")
		if err != nil {
			t.Fatal(err)
		}
		data, err := sum.F32()
		if err != nil {
			t.Fatal(err)
		}
		shape, _ := sum.Shape()
		t.Logf("width %d: shape=%v sum=%v", width, shape, data)
		if len(data) != int(width) {
			t.Fatalf("width %d: got %d values", width, len(data))
		}
		for i, v := range data {
			if v != float32(3*i) {
				t.Fatalf("width %d: sum[%d] = %v, want %v", width, i, v, float32(3*i))
			}
		}
		sum.Close()
		a.Close()
		b.Close()
	}
}

// TestCompileProperties proves properties reach the plugin by asking for a
// compiled model cache and checking that it gets written.
func TestCompileProperties(t *testing.T) {
	path := filepath.Join(t.TempDir(), "add.xml")
	if err := os.WriteFile(path, []byte(dynamicAddIR), 0o644); err != nil {
		t.Fatal(err)
	}
	core, err := SharedCore()
	if err != nil {
		t.Fatal(err)
	}
	cache := t.TempDir()
	cm, err := core.CompileWith(path, "CPU", map[string]string{"CACHE_DIR": cache})
	if err != nil {
		t.Fatal(err)
	}
	cm.Close()
	entries, err := os.ReadDir(cache)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) == 0 {
		t.Fatal("CACHE_DIR is still empty, the property did not reach the plugin")
	}
	t.Logf("cache entries: %d", len(entries))
}

func TestCompileTooManyProperties(t *testing.T) {
	core, err := SharedCore()
	if err != nil {
		t.Fatal(err)
	}
	props := map[string]string{"a": "1", "b": "2", "c": "3", "d": "4"}
	if _, err := core.CompileWith("missing.xml", "CPU", props); err == nil {
		t.Fatal("expected an error for too many properties")
	} else if !strings.Contains(err.Error(), "at most") {
		t.Fatalf("unexpected error %v", err)
	}
}
