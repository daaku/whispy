// Package openvino is a minimal cgo binding over the OpenVINO C API, wrapping
// only the pieces needed to run inference. Tensor data is copied in and out of
// OpenVINO owned memory, so no Go pointers are ever retained by C.
//
// https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__c__api.html
package openvino

/*
#cgo LDFLAGS: -lopenvino_c
#include <openvino/c/openvino.h>
#include <stdlib.h>

// cgo cannot call the variadic OpenVINO entry points directly, so wrap the
// ones needed with the property list passed explicitly.
static ov_status_e ov_compile_model_from_file(const ov_core_t* core,
                                              const char* model_path,
                                              const char* device_name,
                                              ov_compiled_model_t** cm) {
	return ov_core_compile_model_from_file(core, model_path, device_name, 0, cm);
}

static ov_status_e ov_compile_model_from_file_1(const ov_core_t* core,
                                                const char* model_path,
                                                const char* device_name,
                                                const char* k1, const char* v1,
                                                ov_compiled_model_t** cm) {
	return ov_core_compile_model_from_file(core, model_path, device_name, 2, cm, k1, v1);
}

static ov_status_e ov_compile_model_from_file_2(const ov_core_t* core,
                                                const char* model_path,
                                                const char* device_name,
                                                const char* k1, const char* v1,
                                                const char* k2, const char* v2,
                                                ov_compiled_model_t** cm) {
	return ov_core_compile_model_from_file(core, model_path, device_name, 4, cm, k1, v1, k2, v2);
}

static ov_status_e ov_compile_model_from_file_3(const ov_core_t* core,
                                                const char* model_path,
                                                const char* device_name,
                                                const char* k1, const char* v1,
                                                const char* k2, const char* v2,
                                                const char* k3, const char* v3,
                                                ov_compiled_model_t** cm) {
	return ov_core_compile_model_from_file(core, model_path, device_name, 6, cm, k1, v1, k2, v2, k3, v3);
}
*/
import "C"

import (
	"sync"
	"unsafe"

	"github.com/daaku/serr"
)

const (
	F32 = C.F32
	I32 = C.I32
	I64 = C.I64
)

// ElementType is the OpenVINO element type enum, aliased so the rest of the
// module does not need to import C.
type ElementType = C.ov_element_type_e

func status(op string, st C.ov_status_e) error {
	if st == C.OK {
		return nil
	}
	if msg := C.GoString(C.ov_get_last_err_msg()); msg != "" {
		return serr.Errorf("%s: %s", op, msg)
	}
	return serr.Errorf("%s: openvino status %d", op, int(st))
}

// withShape builds a temporary ov_shape_t for the duration of f. OpenVINO's
// ov_shape_create rejects a zero rank shape, so scalar tensors cannot be
// allocated directly; use the tensor an infer request already owns instead.
func withShape(shape []int64, f func(C.ov_shape_t) error) error {
	var dims *C.int64_t
	if len(shape) > 0 {
		dims = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	var s C.ov_shape_t
	if err := status("shape create", C.ov_shape_create(C.int64_t(len(shape)), dims, &s)); err != nil {
		return err
	}
	defer C.ov_shape_free(&s)
	return f(s)
}

func shapeDims(s C.ov_shape_t) []int64 {
	dims := make([]int64, int(s.rank))
	if s.dims == nil {
		return dims
	}
	copy(dims, unsafe.Slice((*int64)(unsafe.Pointer(s.dims)), int(s.rank)))
	return dims
}

// The process keeps a single OpenVINO core, created on first use. Freeing a
// core after running inference is not safe: it unloads the plugins while
// OpenVINO's static state and worker threads are still around, which
// segfaults. ov_shutdown() only moves that crash to the next create/teardown
// cycle. Compiled models and infer requests are released normally.
var (
	coreMu     sync.Mutex
	coreShared *Core
)

// SharedCore returns the process wide OpenVINO core, creating it if needed.
func SharedCore() (*Core, error) {
	coreMu.Lock()
	defer coreMu.Unlock()
	if coreShared != nil {
		return coreShared, nil
	}
	core, err := newCore()
	if err != nil {
		return nil, err
	}
	coreShared = core
	return core, nil
}

// Core is an OpenVINO runtime instance. Use SharedCore instead of creating
// one directly.
type Core struct {
	p *C.ov_core_t
}

func newCore() (*Core, error) {
	var p *C.ov_core_t
	if err := status("core create", C.ov_core_create(&p)); err != nil {
		return nil, err
	}
	return &Core{p: p}, nil
}

// MaxCompileProperties is how many properties a single compile can take.
const MaxCompileProperties = 3

// Compile loads an IR, ONNX or PDPD model and compiles it for device.
func (c *Core) Compile(path, device string) (*CompiledModel, error) {
	return c.CompileWith(path, device, nil)
}

// CompileWith is Compile with extra OpenVINO properties for the plugin, such
// as NPU_COMPILER_TYPE (PLUGIN or DRIVER) or CACHE_DIR for compiled model
// caching. Property names are the C API aliases, for example "NPU_TILES".
func (c *Core) CompileWith(
	path, device string, props map[string]string,
) (*CompiledModel, error) {
	if len(props) > MaxCompileProperties {
		return nil, serr.Errorf(
			"compile %s: %d properties, at most %d are supported",
			path, len(props), MaxCompileProperties)
	}
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	cDevice := C.CString(device)
	defer C.free(unsafe.Pointer(cDevice))

	var cProps []*C.char
	for k, v := range props {
		ck := C.CString(k)
		cv := C.CString(v)
		defer C.free(unsafe.Pointer(ck))
		defer C.free(unsafe.Pointer(cv))
		cProps = append(cProps, ck, cv)
	}

	var p *C.ov_compiled_model_t
	var st C.ov_status_e
	switch len(cProps) / 2 {
	case 0:
		st = C.ov_compile_model_from_file(c.p, cPath, cDevice, &p)
	case 1:
		st = C.ov_compile_model_from_file_1(
			c.p, cPath, cDevice, cProps[0], cProps[1], &p)
	case 2:
		st = C.ov_compile_model_from_file_2(
			c.p, cPath, cDevice, cProps[0], cProps[1], cProps[2], cProps[3], &p)
	case 3:
		st = C.ov_compile_model_from_file_3(
			c.p, cPath, cDevice,
			cProps[0], cProps[1], cProps[2], cProps[3], cProps[4], cProps[5], &p)
	}
	if err := status("compile "+path, st); err != nil {
		return nil, err
	}
	return &CompiledModel{p: p}, nil
}

// CompiledModel is a model compiled for a device.
type CompiledModel struct {
	p *C.ov_compiled_model_t
}

// Close releases the compiled model. Any infer requests made from it must be
// closed first.
func (m *CompiledModel) Close() {
	C.ov_compiled_model_free(m.p)
}

// Request creates an inference request. It is not safe to run the same
// request concurrently.
func (m *CompiledModel) Request() (*Request, error) {
	var p *C.ov_infer_request_t
	if err := status("infer request create", C.ov_compiled_model_create_infer_request(m.p, &p)); err != nil {
		return nil, err
	}
	return &Request{p: p}, nil
}

type port struct {
	p *C.ov_output_const_port_t
}

func (m *CompiledModel) portByName(name string, output bool) (*port, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	var p *C.ov_output_const_port_t
	var st C.ov_status_e
	kind := "input"
	if output {
		st = C.ov_compiled_model_output_by_name(m.p, cName, &p)
		kind = "output"
	} else {
		st = C.ov_compiled_model_input_by_name(m.p, cName, &p)
	}
	if err := status(kind+" "+name, st); err != nil {
		return nil, err
	}
	return &port{p: p}, nil
}

func (m *CompiledModel) portByIndex(i int, output bool) (*port, error) {
	var p *C.ov_output_const_port_t
	var st C.ov_status_e
	if output {
		st = C.ov_compiled_model_output_by_index(m.p, C.size_t(i), &p)
	} else {
		st = C.ov_compiled_model_input_by_index(m.p, C.size_t(i), &p)
	}
	if err := status("port index", st); err != nil {
		return nil, err
	}
	return &port{p: p}, nil
}

func (m *CompiledModel) inputByName(name string) (*port, error) {
	return m.portByName(name, false)
}

func (m *CompiledModel) outputByName(name string) (*port, error) {
	return m.portByName(name, true)
}

func (m *CompiledModel) inputByIndex(i int) (*port, error) {
	return m.portByIndex(i, false)
}

func (m *CompiledModel) outputByIndex(i int) (*port, error) {
	return m.portByIndex(i, true)
}

func (p *port) close() {
	C.ov_output_const_port_free(p.p)
}

// PortShape is the shape and element type of a model input or output. A
// dynamic dimension is -1; DynamicRank means the rank itself is unknown.
type PortShape struct {
	Dims        []int64
	DynamicRank bool
	Type        ElementType
}

// Dim returns dimension i, or zero when it is dynamic or missing.
func (s PortShape) Dim(i int) int {
	if i < 0 || i >= len(s.Dims) || s.Dims[i] < 0 {
		return 0
	}
	return int(s.Dims[i])
}

func (p *port) portShape() (PortShape, error) {
	var shape PortShape
	var et C.ov_element_type_e
	if err := status("port type", C.ov_port_get_element_type(p.p, &et)); err != nil {
		return shape, err
	}
	shape.Type = et

	var ps C.ov_partial_shape_t
	if err := status("port shape", C.ov_port_get_partial_shape(p.p, &ps)); err != nil {
		return shape, err
	}
	defer C.ov_partial_shape_free(&ps)
	if bool(C.ov_rank_is_dynamic(ps.rank)) {
		shape.DynamicRank = true
		return shape, nil
	}
	rank := int(ps.rank.min)
	shape.Dims = make([]int64, rank)
	if rank == 0 || ps.dims == nil {
		return shape, nil
	}
	dims := unsafe.Slice((*C.ov_dimension_t)(unsafe.Pointer(ps.dims)), rank)
	for i, d := range dims {
		if bool(C.ov_dimension_is_dynamic(d)) {
			shape.Dims[i] = -1
		} else {
			shape.Dims[i] = int64(d.min)
		}
	}
	return shape, nil
}

// Input describes a named model input.
func (m *CompiledModel) Input(name string) (PortShape, error) {
	p, err := m.inputByName(name)
	if err != nil {
		return PortShape{}, err
	}
	defer p.close()
	return p.portShape()
}

// Output describes a named model output.
func (m *CompiledModel) Output(name string) (PortShape, error) {
	p, err := m.outputByName(name)
	if err != nil {
		return PortShape{}, err
	}
	defer p.close()
	return p.portShape()
}

// InputByIndex describes an input by index, for models whose ports are
// unnamed.
func (m *CompiledModel) InputByIndex(i int) (PortShape, error) {
	p, err := m.inputByIndex(i)
	if err != nil {
		return PortShape{}, err
	}
	defer p.close()
	return p.portShape()
}

// OutputByIndex describes an output by index, for models whose ports are
// unnamed.
func (m *CompiledModel) OutputByIndex(i int) (PortShape, error) {
	p, err := m.outputByIndex(i)
	if err != nil {
		return PortShape{}, err
	}
	defer p.close()
	return p.portShape()
}

// Tensor is an OpenVINO tensor over host memory.
type Tensor struct {
	p *C.ov_tensor_t
}

// NewTensor allocates a tensor of the given element type and shape.
func NewTensor(et ElementType, shape []int64) (*Tensor, error) {
	var p *C.ov_tensor_t
	err := withShape(shape, func(s C.ov_shape_t) error {
		return status("tensor create", C.ov_tensor_create(et, s, &p))
	})
	if err != nil {
		return nil, err
	}
	return &Tensor{p: p}, nil
}

// NewF32Tensor allocates a zeroed f32 tensor and returns it with a slice
// aliasing its memory. The slice is only valid while the tensor is alive.
func NewF32Tensor(shape []int64) (*Tensor, []float32, error) {
	t, err := NewTensor(F32, shape)
	if err != nil {
		return nil, nil, err
	}
	data, err := t.F32()
	if err != nil {
		t.Close()
		return nil, nil, err
	}
	// ov_tensor_create allocates uninitialized memory.
	clear(data)
	return t, data, nil
}

// Close releases the tensor.
func (t *Tensor) Close() {
	C.ov_tensor_free(t.p)
}

// Shape returns the tensor shape.
func (t *Tensor) Shape() ([]int64, error) {
	var s C.ov_shape_t
	if err := status("tensor shape", C.ov_tensor_get_shape(t.p, &s)); err != nil {
		return nil, err
	}
	defer C.ov_shape_free(&s)
	return shapeDims(s), nil
}

// ElementType returns the tensor element type.
func (t *Tensor) ElementType() (ElementType, error) {
	var et C.ov_element_type_e
	if err := status("tensor type", C.ov_tensor_get_element_type(t.p, &et)); err != nil {
		return 0, err
	}
	return et, nil
}

// Size returns the number of elements in the tensor.
func (t *Tensor) Size() (int, error) {
	var n C.size_t
	if err := status("tensor size", C.ov_tensor_get_size(t.p, &n)); err != nil {
		return 0, err
	}
	return int(n), nil
}

func (t *Tensor) ptr() (unsafe.Pointer, error) {
	var p unsafe.Pointer
	if err := status("tensor data", C.ov_tensor_data(t.p, &p)); err != nil {
		return nil, err
	}
	return p, nil
}

// F32 returns a Go slice aliasing the tensor memory, valid only while the
// tensor is alive.
func (t *Tensor) F32() ([]float32, error) {
	n, err := t.Size()
	if err != nil {
		return nil, err
	}
	p, err := t.ptr()
	if err != nil {
		return nil, err
	}
	return unsafe.Slice((*float32)(p), n), nil
}

// Int64 reads a single element scalar tensor as int64, handling both i32 and
// i64 element types.
func (t *Tensor) Int64() (int64, error) {
	et, err := t.ElementType()
	if err != nil {
		return 0, err
	}
	p, err := t.ptr()
	if err != nil {
		return 0, err
	}
	switch et {
	case I64:
		return *(*int64)(p), nil
	case I32:
		return int64(*(*int32)(p)), nil
	}
	return 0, serr.Errorf("scalar tensor is not i32 or i64")
}

// SetInt writes a single element scalar tensor, handling both i32 and i64
// element types.
func (t *Tensor) SetInt(v int64) error {
	et, err := t.ElementType()
	if err != nil {
		return err
	}
	p, err := t.ptr()
	if err != nil {
		return err
	}
	switch et {
	case I64:
		*(*int64)(p) = v
		return nil
	case I32:
		*(*int32)(p) = int32(v)
		return nil
	}
	return serr.Errorf("scalar tensor is not i32 or i64")
}

// Request is an inference request.
type Request struct {
	p *C.ov_infer_request_t
}

// Close releases the request.
func (r *Request) Close() {
	C.ov_infer_request_free(r.p)
}

// Set binds a tensor to a named input.
func (r *Request) Set(name string, t *Tensor) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return status("set "+name, C.ov_infer_request_set_tensor(r.p, cName, t.p))
}

// SetInputIndex binds a tensor to an input by index.
func (r *Request) SetInputIndex(i int, t *Tensor) error {
	st := C.ov_infer_request_set_input_tensor_by_index(r.p, C.size_t(i), t.p)
	return status("set input", st)
}

// Get returns a new handle to the named input or output tensor. The caller
// owns it and must close it.
func (r *Request) Get(name string) (*Tensor, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var p *C.ov_tensor_t
	if err := status("get "+name, C.ov_infer_request_get_tensor(r.p, cName, &p)); err != nil {
		return nil, err
	}
	return &Tensor{p: p}, nil
}

// GetOutputIndex returns a new handle to an output tensor by index.
func (r *Request) GetOutputIndex(i int) (*Tensor, error) {
	var p *C.ov_tensor_t
	st := C.ov_infer_request_get_output_tensor_by_index(r.p, C.size_t(i), &p)
	if err := status("get output", st); err != nil {
		return nil, err
	}
	return &Tensor{p: p}, nil
}

// Infer runs the request synchronously.
func (r *Request) Infer() error {
	return status("infer", C.ov_infer_request_infer(r.p))
}
