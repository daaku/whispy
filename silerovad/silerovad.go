// Package silerovad runs the Silero VAD model through OpenVINO. It expects the
// 16 kHz model (v5/v6), whose graph takes 512 sample windows with 64 samples of
// context and carries a state tensor between calls.
package silerovad

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/daaku/serr"
	"github.com/daaku/whispy/openvino"
)

const (
	// SampleRate is the only sample rate the 16 kHz model supports.
	SampleRate = 16000
	// WindowSamples is the number of samples the model consumes per step.
	WindowSamples = 512
	// Threshold is the speech probability the Silero authors classify with.
	Threshold = 0.5

	// contextSamples are prepended to every window, as the Silero wrapper does.
	contextSamples = 64
	// cpuDevice is the fallback for devices that cannot run the model.
	cpuDevice = "CPU"
)

// Config configures a VAD instance.
type Config struct {
	// Model is the path to silero_vad.onnx or its OpenVINO IR .xml file.
	Model string
	// Device is the OpenVINO device, CPU by default. The model has dynamic
	// shapes, so only devices that accept them can be used; anything else
	// falls back to the CPU with a message.
	Device string
}

// Vad is a streaming voice activity detector. It is safe for concurrent use,
// though a single instance processes one stream at a time.
type Vad struct {
	mu sync.Mutex

	model *openvino.CompiledModel
	req   *openvino.Request

	input     *openvino.Tensor
	inputData []float32
	state     *openvino.Tensor
	stateData []float32

	stateValues []float32
	context     []float32
	pending     []float32
}

// New loads the Silero VAD model.
func New(cfg Config) (*Vad, error) {
	if cfg.Model == "" {
		return nil, serr.Errorf("silerovad: model path is required")
	}
	device := cfg.Device
	if device == "" {
		device = "CPU"
	}
	core, err := openvino.SharedCore()
	if err != nil {
		return nil, err
	}
	compiled, err := core.Compile(cfg.Model, device)
	if err != nil {
		// The Silero model has dynamic shapes, which the NPU rejects. Fall
		// back rather than refusing to start.
		if device == cpuDevice {
			return nil, err
		}
		fallback, cpuErr := core.Compile(cfg.Model, cpuDevice)
		if cpuErr != nil {
			return nil, err
		}
		fmt.Fprintf(os.Stderr,
			"silerovad: %s does not run on %s (%v), using CPU\n",
			filepath.Base(cfg.Model), device, err)
		compiled = fallback
	}
	v := &Vad{model: compiled}
	if err := v.init(); err != nil {
		v.Close()
		return nil, err
	}
	return v, nil
}

func (v *Vad) init() error {
	req, err := v.model.Request()
	if err != nil {
		return err
	}
	v.req = req

	// The model declares dynamic shapes, so the request's own tensors are the
	// source of truth for everything but the batch and time axes.
	stateTensor, err := req.Get("state")
	if err != nil {
		return serr.Wrap(err)
	}
	defer stateTensor.Close()
	stateShape, err := stateTensor.Shape()
	if err != nil {
		return serr.Wrap(err)
	}
	if len(stateShape) != 3 || stateShape[0] != 2 || stateShape[2] <= 0 {
		return serr.Errorf("silerovad: unexpected state shape %v", stateShape)
	}

	// The sample rate is a scalar input; ov_shape_create cannot allocate a
	// scalar tensor, so write straight into the request's own tensor.
	sr, err := req.Get("sr")
	if err != nil {
		return serr.Wrap(err)
	}
	if err := sr.SetInt(SampleRate); err != nil {
		return serr.Wrap(err)
	}
	sr.Close()

	if v.input, v.inputData, err = openvino.NewF32Tensor(
		[]int64{1, WindowSamples + contextSamples}); err != nil {
		return serr.Wrap(err)
	}
	if v.state, v.stateData, err = openvino.NewF32Tensor(
		[]int64{stateShape[0], 1, stateShape[2]}); err != nil {
		return serr.Wrap(err)
	}
	v.stateValues = make([]float32, len(v.stateData))
	v.context = make([]float32, contextSamples)
	return nil
}

// Close releases the model. The OpenVINO core is process wide and stays alive.
func (v *Vad) Close() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.model == nil {
		return nil
	}
	if v.input != nil {
		v.input.Close()
		v.input = nil
	}
	if v.state != nil {
		v.state.Close()
		v.state = nil
	}
	if v.req != nil {
		v.req.Close()
		v.req = nil
	}
	v.model.Close()
	v.model = nil
	return nil
}

// Reset drops the streaming state and any buffered samples, as if the model
// had just been loaded.
func (v *Vad) Reset() {
	v.mu.Lock()
	defer v.mu.Unlock()
	clear(v.stateValues)
	clear(v.stateData)
	clear(v.context)
	v.pending = v.pending[:0]
}

// SpeechProb returns the speech probability of every complete window in the
// stream. Samples that do not fill a window are buffered until the next call,
// and window context crosses call boundaries.
func (v *Vad) SpeechProb(samples []float32) ([]float32, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.model == nil {
		return nil, serr.Errorf("silerovad: closed")
	}
	v.pending = append(v.pending, samples...)
	if len(v.pending) < WindowSamples {
		return nil, nil
	}
	var probs []float32
	off := 0
	for ; off+WindowSamples <= len(v.pending); off += WindowSamples {
		prob, err := v.runWindow(v.pending[off : off+WindowSamples])
		if err != nil {
			return nil, err
		}
		probs = append(probs, prob)
	}
	v.pending = append(v.pending[:0], v.pending[off:]...)
	return probs, nil
}

func (v *Vad) runWindow(window []float32) (float32, error) {
	copy(v.inputData, v.context)
	copy(v.inputData[contextSamples:], window)
	copy(v.stateData, v.stateValues)

	if err := v.req.Set("input", v.input); err != nil {
		return 0, serr.Wrap(err)
	}
	if err := v.req.Set("state", v.state); err != nil {
		return 0, serr.Wrap(err)
	}
	if err := v.req.Infer(); err != nil {
		return 0, serr.Wrap(err)
	}

	prob, err := readScalar(v.req, "output")
	if err != nil {
		return 0, err
	}
	next, err := v.req.Get("stateN")
	if err != nil {
		return 0, serr.Wrap(err)
	}
	defer next.Close()
	nextData, err := next.F32()
	if err != nil {
		return 0, serr.Wrap(err)
	}
	copy(v.stateValues, nextData)
	copy(v.context, window[WindowSamples-contextSamples:])
	return prob, nil
}

// HasSpeech reports whether any window reached the speech threshold.
func HasSpeech(probs []float32) bool {
	for _, p := range probs {
		if p >= Threshold {
			return true
		}
	}
	return false
}

func readScalar(req *openvino.Request, name string) (float32, error) {
	t, err := req.Get(name)
	if err != nil {
		return 0, serr.Wrap(err)
	}
	defer t.Close()
	data, err := t.F32()
	if err != nil {
		return 0, serr.Wrap(err)
	}
	if len(data) == 0 {
		return 0, serr.Errorf("silerovad: %s is empty", name)
	}
	return data[0], nil
}
