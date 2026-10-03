// Package parakeet implements the Parakeet TDT speech to text pipeline on top
// of the OpenVINO C API.
package parakeet

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"

	"github.com/daaku/serr"
	"github.com/daaku/whispy/openvino"
)

const (
	melBins              = 128
	defaultEncoderFrames = 1250
	defaultWindowSamples = 160000
	preprocRoundTo       = 1000
	defaultMaxTokens     = 256
	defaultMaxSymbols    = 10

	// defaultMelPerEncoderFrame is the Conformer's subsampling factor, used
	// when the encoder output shape does not say what it is.
	defaultMelPerEncoderFrame = 8
	cpuDevice                 = "CPU"

	// finalize steps flush trailing tokens after the last encoder frame.
	finalizeSteps  = 8
	finalizeBlanks = 1

	// Chunk boundary deduplication tuning. The frame counts here are mel
	// frames, the time base token timings and chunk offsets share: 160 of
	// them is 1.6 seconds, about the size of the chunk overlap.
	dedupPrevTokens     = 15
	dedupBoundaryFrames = 160
	dedupMaxOverlap     = 30
)

var defaultDurationBins = []int{0, 1, 2, 3, 4}

// Config configures a Parakeet model.
type Config struct {
	// Dir holds the OpenVINO IR files: parakeet_melspectogram,
	// parakeet_encoder, parakeet_decoder, parakeet_joint and
	// parakeet_vocab.json.
	Dir string
	// Device is the OpenVINO device to use for the encoder, decoder and
	// joint network: CPU, GPU, NPU or AUTO. Defaults to AUTO.
	Device string
	// PreprocDevice is the OpenVINO device for the mel spectrogram model.
	// Defaults to CPU: it is fastest there, and the v2 export has a dynamic
	// input, which the NPU does not accept.
	PreprocDevice string
	// DecoderDevice is the OpenVINO device for the decoder and joint
	// networks, which run once per token and once per frame. Defaults to
	// Device. They are much smaller than the encoder and latency bound, so an
	// accelerator is often the wrong home for them even when it is right for
	// the encoder.
	DecoderDevice string
	// BlankID overrides the blank token id. Zero reads it from the vocabulary
	// and falls back to the Parakeet v2 default of 1024.
	BlankID int
	// DurationsFirst says the joint network's logits put the duration bins
	// ahead of the token head. The Parakeet exports put the tokens first,
	// which is the default; a model exported the other way would have its
	// duration bins decoded as tokens, and this is the switch for it.
	DurationsFirst bool
	// DurationBins are the TDT duration bin values. Defaults to 0, 1, 2, 3, 4.
	DurationBins []int
	// MaxTokens caps the tokens produced per chunk. Defaults to 256.
	MaxTokens int
	// MaxSymbolsPerStep caps the tokens one encoder frame may emit before the
	// loop forces a blank and moves on, which is what the TDT decoders NeMo
	// ships do. Defaults to 10.
	MaxSymbolsPerStep int
	// Properties are extra OpenVINO compile properties for the encoder,
	// decoder and joint network, named the way the C API names them: for
	// example NPU_COMPILER_TYPE=PLUGIN, or CACHE_DIR to cache compiled
	// models. They are not passed to the preprocessor, which runs on the
	// CPU, nor to a CPU fallback, since a property can be specific to a
	// device.
	Properties map[string]string
}

// Result is a transcription.
type Result struct {
	Text   string
	Tokens []int
}

type component struct {
	model *openvino.CompiledModel
	req   *openvino.Request
	// name, device and usedDevice describe where this component ended up
	// running, so a fallback from an accelerator to the CPU can be reported.
	// fallback holds the error from the requested device, if any.
	name       string
	path       string
	device     string
	usedDevice string
	fallback   error
}

func (c *component) close() {
	if c == nil {
		return
	}
	if c.req != nil {
		c.req.Close()
		c.req = nil
	}
	if c.model != nil {
		c.model.Close()
		c.model = nil
	}
}

// Model is a loaded Parakeet model.
type Model struct {
	mu sync.Mutex

	core      *openvino.Core
	preproc   *component
	encoder   *component
	decoder   *component
	joint     *component
	tokenizer *tokenizer

	preprocWindow  int
	preprocLenType openvino.ElementType
	melIndex       int
	lengthIndex    int

	encoderFrames  int
	encoderLenType openvino.ElementType

	// melPerEncoderFrame converts the encoder frames the decoder reports a
	// token at into the mel frames the chunker offsets by.
	melPerEncoderFrame int
	encoderHidden      int

	targetsType   openvino.ElementType
	decoderHidden int

	jointOut int

	blankID         int
	durationBins    []int
	durationsFirst  bool
	maxTokens       int
	maxSymbols      int
	requestedDevice string
}

// New loads and compiles the Parakeet IR models from cfg.Dir.
func New(cfg Config) (*Model, error) {
	if cfg.Dir == "" {
		return nil, serr.Errorf("parakeet: model directory is required")
	}
	device := cfg.Device
	if device == "" {
		device = "AUTO"
	}
	preprocDevice := cfg.PreprocDevice
	if preprocDevice == "" {
		preprocDevice = cpuDevice
	}
	decoderDevice := cfg.DecoderDevice
	if decoderDevice == "" {
		decoderDevice = device
	}
	bins := cfg.DurationBins
	if len(bins) == 0 {
		bins = defaultDurationBins
	}
	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	maxSymbols := cfg.MaxSymbolsPerStep
	if maxSymbols <= 0 {
		maxSymbols = defaultMaxSymbols
	}

	tok, err := loadTokenizer(
		filepath.Join(cfg.Dir, "parakeet_vocab.json"), cfg.BlankID,
	)
	if err != nil {
		return nil, err
	}
	core, err := openvino.SharedCore()
	if err != nil {
		return nil, err
	}

	m := &Model{
		core:            core,
		tokenizer:       tok,
		blankID:         tok.blankID,
		durationBins:    bins,
		durationsFirst:  cfg.DurationsFirst,
		maxTokens:       maxTokens,
		maxSymbols:      maxSymbols,
		requestedDevice: device,
	}
	// The GPU's default execution mode changes the decoder and joint output
	// enough to lose the transcript, so they ask for accuracy. The encoder
	// keeps the default, which is where the speed is.
	decoderProps := accuracyProps(decoderDevice, cfg.Properties)
	for _, c := range []struct {
		file   string
		device string
		props  map[string]string
		dst    **component
	}{
		{"parakeet_melspectogram.xml", preprocDevice, nil, &m.preproc},
		{"parakeet_encoder.xml", device, cfg.Properties, &m.encoder},
		{"parakeet_decoder.xml", decoderDevice, decoderProps, &m.decoder},
		{"parakeet_joint.xml", decoderDevice, decoderProps, &m.joint},
	} {
		loaded, err := loadComponent(
			core, filepath.Join(cfg.Dir, c.file), c.device, c.props,
		)
		if err != nil {
			m.Close()
			return nil, err
		}
		*c.dst = loaded
	}
	if err := m.resolve(); err != nil {
		m.Close()
		return nil, err
	}
	m.report()
	return m, nil
}

// report prints where each model ended up running. It is only interesting when
// an accelerator was requested, since a device can refuse a model and leave it
// on the CPU.
func (m *Model) report() {
	components := []*component{m.preproc, m.encoder, m.decoder, m.joint}
	for _, c := range components {
		if c.fallback != nil {
			fmt.Fprintf(os.Stderr, "parakeet: %s: %s, using %s (%s)\n",
				c.name, c.device, c.usedDevice, cleanError(c.fallback))
		}
	}
	if m.requestedDevice != cpuDevice {
		parts := make([]string, 0, len(components))
		for _, c := range components {
			parts = append(parts, fmt.Sprintf("%s=%s", c.name, c.usedDevice))
		}
		fmt.Fprintf(os.Stderr, "parakeet: %s\n", strings.Join(parts, " "))
	}
}

// exceptionRe matches the C++ re-throw preamble OpenVINO puts in front of the
// message that says what actually went wrong.
var exceptionRe = regexp.MustCompile(`Exception from \S+:\s*`)

// cleanError flattens an OpenVINO error onto one line and drops that preamble.
func cleanError(err error) string {
	msg := strings.Join(strings.Fields(exceptionRe.ReplaceAllString(err.Error(), "")), " ")
	if len(msg) > 300 {
		msg = msg[:300] + "..."
	}
	return msg
}

// loadComponent compiles one model for device, falling back to the CPU when
// the device cannot run it. Accelerators are pickier than the CPU: the NPU
// requires static shapes, and devices can lack support for individual
// operations of the encoder.
func loadComponent(
	core *openvino.Core,
	path, device string,
	props map[string]string,
) (*component, error) {
	name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	name = strings.TrimPrefix(name, "parakeet_")
	c := &component{name: name, path: path, device: device, usedDevice: device}
	cm, err := compileForDevice(core, path, device, props)
	if err != nil {
		if device == cpuDevice {
			return nil, err
		}
		// Properties can be specific to the requested device, so the CPU
		// fallback goes without them.
		fallback, cpuErr := core.Compile(path, cpuDevice)
		if cpuErr != nil {
			return nil, err
		}
		cm = fallback
		c.usedDevice = cpuDevice
		c.fallback = err
	}
	req, err := cm.Request()
	if err != nil {
		cm.Close()
		return nil, err
	}
	c.model = cm
	c.req = req
	return c, nil
}

// accuracyProps adds the GPU plugin's accuracy execution mode when the device
// is an Intel GPU, or AUTO, which may resolve to one. In its default
// performance mode the plugin runs a model in a precision and with a dynamic
// quantization that change the output of the small decoder and joint networks:
// on an Xe iGPU an 11 second clip comes back as one word. The encoder is not
// given the hint because it is where the speed is and its output survives the
// precision. The hint is a core level one that every device accepts, so on a
// CPU or NPU AUTO it is at worst ignored. A precision or execution hint the
// caller set is left alone.
func accuracyProps(device string, props map[string]string) map[string]string {
	upper := strings.ToUpper(device)
	if !strings.HasPrefix(upper, "GPU") && upper != "AUTO" {
		return props
	}
	for _, key := range []string{"EXECUTION_MODE_HINT", "INFERENCE_PRECISION_HINT"} {
		if _, ok := props[key]; ok {
			return props
		}
	}
	out := make(map[string]string, len(props)+1)
	for k, v := range props {
		out[k] = v
	}
	out["EXECUTION_MODE_HINT"] = "ACCURACY"
	return out
}

// compileForDevice compiles one IR for device. The NPU driver compiler rejects
// a negative LogSoftmax axis, which the stock joint network carries, so an NPU
// compile goes through the axis rewrite first. Every other device takes the
// file as shipped.
func compileForDevice(
	core *openvino.Core, path, device string, props map[string]string,
) (*openvino.CompiledModel, error) {
	if !needsPositiveAxis(device) {
		return core.CompileWith(path, device, props)
	}
	return compileWithNormalizedAxis(core, path, device, props)
}

// needsPositiveAxis reports whether a device wants the LogSoftmax axis written
// positively. Only the NPU compiler does; AUTO is left alone because the model
// it picks is not known until it has picked one, and a model that fails to
// compile still falls back to the CPU.
func needsPositiveAxis(device string) bool {
	return strings.HasPrefix(strings.ToUpper(device), "NPU")
}

// compileWithNormalizedAxis reads an IR, writes a negative LogSoftmax axis as
// the positive one that means the same thing, and compiles the result. The
// patched XML goes to a temporary file with the weights named explicitly,
// because the OpenVINO C API can only compile what it has read from a file.
// Nothing is written beside the model.
func compileWithNormalizedAxis(
	core *openvino.Core, path, device string, props map[string]string,
) (*openvino.CompiledModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	patched := normalizeLogSoftmaxAxis(data)
	if bytes.Equal(patched, data) {
		return core.CompileWith(path, device, props)
	}
	dir, err := os.MkdirTemp("", "whispy-ir-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	tmp := filepath.Join(dir, filepath.Base(path))
	if err := os.WriteFile(tmp, patched, 0o600); err != nil {
		return nil, err
	}
	bin := strings.TrimSuffix(path, filepath.Ext(path)) + ".bin"
	model, err := core.ReadModel(tmp, bin)
	if err != nil {
		return nil, err
	}
	defer model.Close()
	return core.CompileModel(model, device, props)
}

// resolve discovers tensor shapes and element types from the compiled models
// so the pipeline is not hardcoded to one export.
//
// Dimensions that the model leaves dynamic fall back to the values the
// Parakeet exports use, since the pipeline decides the actual shapes: the
// preprocessor window and the encoder frame count are chosen here, and the
// hidden sizes are also readable from the joint network's inputs.
func (m *Model) resolve() error {
	// The preprocessor is accessed by index: some exports leave its ports
	// unnamed. Its input is dynamic in the v2 export, which means the window
	// is sized to the audio instead.
	pre, err := m.preproc.model.InputByIndex(0)
	if err != nil {
		return err
	}
	m.preprocWindow = pre.Dim(1)
	preLen, err := m.preproc.model.InputByIndex(1)
	if err != nil {
		return err
	}
	m.preprocLenType = preLen.Type
	mel, err := m.preproc.model.OutputByIndex(0)
	if err != nil {
		return err
	}
	// The mel spectrogram is the rank 3 output, the other one is its length.
	m.melIndex, m.lengthIndex = 0, 1
	if !mel.DynamicRank && len(mel.Dims) != 3 {
		m.melIndex, m.lengthIndex = 1, 0
	}

	melIn, err := m.encoder.model.Input("melspectogram")
	if err != nil {
		return err
	}
	m.encoderFrames = melIn.Dim(2)
	if m.encoderFrames <= 0 {
		m.encoderFrames = defaultEncoderFrames
	}
	melLen, err := m.encoder.model.Input("melspectogram_length")
	if err != nil {
		return err
	}
	m.encoderLenType = melLen.Type

	// Hidden sizes come from the encoder and decoder, or from the joint
	// network when those models have dynamic outputs.
	jointEnc, err := m.joint.model.Input("encoder_outputs")
	if err != nil {
		return err
	}
	jointDec, err := m.joint.model.Input("decoder_outputs")
	if err != nil {
		return err
	}
	encOut, err := m.encoder.model.Output("encoder_output")
	if err != nil {
		return err
	}
	if m.encoderHidden = encOut.Dim(1); m.encoderHidden <= 0 {
		m.encoderHidden = jointEnc.Dim(2)
	}
	if m.encoderHidden <= 0 {
		return serr.Errorf("parakeet: unknown encoder hidden size %v", encOut.Dims)
	}
	if _, err := m.encoder.model.Output("encoder_output_length"); err != nil {
		return err
	}
	// The encoder answers a mel frame input with fewer frames back; the
	// decoder stamps tokens with the smaller index, and chunk stitching needs
	// both on the same ruler.
	if out := encOut.Dim(2); out > 0 && m.encoderFrames > out {
		m.melPerEncoderFrame = max(1, (m.encoderFrames+out/2)/out)
	} else {
		m.melPerEncoderFrame = defaultMelPerEncoderFrame
	}

	targets, err := m.decoder.model.Input("targets")
	if err != nil {
		return err
	}
	m.targetsType = targets.Type
	if _, err := m.decoder.model.Input("c_in"); err != nil {
		return err
	}
	decOut, err := m.decoder.model.Output("decoder_output")
	if err != nil {
		return err
	}
	if m.decoderHidden = decOut.Dim(2); m.decoderHidden <= 0 {
		m.decoderHidden = jointDec.Dim(2)
	}
	if m.decoderHidden <= 0 {
		return serr.Errorf("parakeet: unknown decoder hidden size %v", decOut.Dims)
	}
	for _, name := range []string{"h_out", "c_out"} {
		if _, err := m.decoder.model.Output(name); err != nil {
			return err
		}
	}

	// The joint output size is only used to validate the head layout, so a
	// dynamic logits tensor is tolerated.
	if logits, err := m.joint.model.Output("logits"); err == nil && logits.Dim(len(logits.Dims)-1) > 0 {
		m.jointOut = logits.Dim(len(logits.Dims) - 1)
	}
	if m.jointOut > 0 {
		if m.blankID+1+len(m.durationBins) > m.jointOut {
			return serr.Errorf(
				"parakeet: joint output %d smaller than %d token and %d duration heads",
				m.jointOut, m.blankID+1, len(m.durationBins),
			)
		}
	}
	return nil
}

// Close releases the compiled models and infer requests held by the model.
// The OpenVINO core is process wide and stays alive, see openvino.SharedCore.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.core == nil {
		return nil
	}
	m.preproc.close()
	m.encoder.close()
	m.decoder.close()
	m.joint.close()
	m.core = nil
	return nil
}

// melFeatures holds a mel spectrogram in [melBins][timeSteps] layout.
// frames is the number of frames the preprocessor reported as valid, which
// can be smaller than timeSteps for padded input.
type melFeatures struct {
	data      []float32
	frames    int
	timeSteps int
}

// preprocess runs the mel spectrogram model over 16 kHz mono samples.
func (m *Model) preprocess(pcm []float32) (*melFeatures, error) {
	if len(pcm) == 0 {
		return nil, serr.Errorf("parakeet: no audio samples")
	}
	// The v3 preprocessor misbehaves on sample counts that are not a
	// multiple of 1000, so pad with silence up to the next multiple.
	samples := pcm
	if rounded := roundUp(len(pcm), preprocRoundTo); rounded != len(pcm) {
		samples = make([]float32, rounded)
		copy(samples, pcm)
	}
	window := m.preprocWindow
	if window == 0 && len(samples) > defaultWindowSamples {
		window = defaultWindowSamples
	}

	if window == 0 || len(samples) <= window {
		mel, frames, timeSteps, err := m.runPreprocWindow(samples, len(samples), window)
		if err != nil {
			return nil, err
		}
		if frames <= 0 {
			return nil, serr.Errorf("parakeet: preprocessor returned no frames")
		}
		return &melFeatures{data: mel, frames: frames, timeSteps: timeSteps}, nil
	}

	// Long audio: process fixed size windows and concatenate the frames.
	bins := make([][]float32, melBins)
	total := 0
	for offset := 0; offset < len(samples); {
		count := min(window, len(samples)-offset)
		mel, frames, timeSteps, err := m.runPreprocWindow(
			samples[offset:offset+count], count, window,
		)
		if err != nil {
			return nil, err
		}
		if frames > 0 {
			n := min(frames, timeSteps)
			for b := range melBins {
				bins[b] = append(bins[b], mel[b*timeSteps:b*timeSteps+n]...)
			}
			total += n
		}
		offset += count
	}
	if total == 0 {
		return nil, serr.Errorf("parakeet: preprocessor returned no frames")
	}
	data := make([]float32, melBins*total)
	for b := range melBins {
		copy(data[b*total:(b+1)*total], bins[b])
	}
	return &melFeatures{data: data, frames: total, timeSteps: total}, nil
}

// runPreprocWindow runs one preprocessor window. window is the fixed input size
// of the compiled model, or zero when the model has a dynamic input shape.
func (m *Model) runPreprocWindow(
	samples []float32, valid, window int,
) (mel []float32, frames, timeSteps int, err error) {
	req := window
	if req == 0 {
		req = valid
	}
	audio, data, err := openvino.NewF32Tensor([]int64{1, int64(req)})
	if err != nil {
		return nil, 0, 0, err
	}
	defer audio.Close()
	copy(data, samples[:min(req, len(samples))])

	length, err := openvino.NewTensor(m.preprocLenType, []int64{1})
	if err != nil {
		return nil, 0, 0, err
	}
	defer length.Close()
	if err := length.SetInt(int64(valid)); err != nil {
		return nil, 0, 0, err
	}

	if err := m.preproc.req.SetInputIndex(0, audio); err != nil {
		return nil, 0, 0, err
	}
	if err := m.preproc.req.SetInputIndex(1, length); err != nil {
		return nil, 0, 0, err
	}
	if err := m.preproc.req.Infer(); err != nil {
		return nil, 0, 0, err
	}

	melTensor, err := m.preproc.req.GetOutputIndex(m.melIndex)
	if err != nil {
		return nil, 0, 0, err
	}
	defer melTensor.Close()
	shape, err := melTensor.Shape()
	if err != nil {
		return nil, 0, 0, err
	}
	if len(shape) != 3 || shape[1] != melBins {
		return nil, 0, 0, serr.Errorf("parakeet: unexpected mel shape %v", shape)
	}
	timeSteps = int(shape[2])
	melData, err := melTensor.F32()
	if err != nil {
		return nil, 0, 0, err
	}
	mel = slices.Clone(melData)

	lengthTensor, err := m.preproc.req.GetOutputIndex(m.lengthIndex)
	if err != nil {
		return nil, 0, 0, err
	}
	defer lengthTensor.Close()
	valid64, err := lengthTensor.Int64()
	if err != nil {
		return nil, 0, 0, err
	}
	return mel, int(valid64), timeSteps, nil
}

// encoderOutput holds encoder activations in [hiddenSize][timeSteps] layout.
type encoderOutput struct {
	data        []float32
	hiddenSize  int
	timeSteps   int
	validFrames int
}

// encode runs the encoder over one mel chunk.
func (m *Model) encode(mel *melFeatures) (*encoderOutput, error) {
	actual := min(m.encoderFrames, mel.frames)
	input, data, err := openvino.NewF32Tensor([]int64{1, melBins, int64(m.encoderFrames)})
	if err != nil {
		return nil, err
	}
	defer input.Close()

	if mel.timeSteps == m.encoderFrames {
		// The preprocessor tensor already has the encoder's frame count, so
		// keep it as is, including the padding it computed.
		copy(data, mel.data)
	} else {
		for b := range melBins {
			src := mel.data[b*mel.timeSteps : b*mel.timeSteps+actual]
			copy(data[b*m.encoderFrames:b*m.encoderFrames+actual], src)
		}
	}

	length, err := openvino.NewTensor(m.encoderLenType, []int64{1})
	if err != nil {
		return nil, err
	}
	defer length.Close()
	if err := length.SetInt(int64(actual)); err != nil {
		return nil, err
	}

	if err := m.encoder.req.Set("melspectogram", input); err != nil {
		return nil, err
	}
	if err := m.encoder.req.Set("melspectogram_length", length); err != nil {
		return nil, err
	}
	if err := m.encoder.req.Infer(); err != nil {
		return nil, err
	}

	out, err := m.encoder.req.Get("encoder_output")
	if err != nil {
		return nil, err
	}
	defer out.Close()
	shape, err := out.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) != 3 || shape[1] != int64(m.encoderHidden) {
		return nil, serr.Errorf("parakeet: unexpected encoder output shape %v", shape)
	}
	outData, err := out.F32()
	if err != nil {
		return nil, err
	}

	lenTensor, err := m.encoder.req.Get("encoder_output_length")
	if err != nil {
		return nil, err
	}
	defer lenTensor.Close()
	valid, err := lenTensor.Int64()
	if err != nil {
		return nil, err
	}

	timeSteps := int(shape[2])
	return &encoderOutput{
		data:        slices.Clone(outData),
		hiddenSize:  m.encoderHidden,
		timeSteps:   timeSteps,
		validFrames: min(int(valid), timeSteps),
	}, nil
}

// tokenTiming is one emitted token and where it happened. frame is a mel frame
// index, so it shares its time base with the chunk offsets that stitch long
// audio together; the encoder reports 8 times fewer frames than it takes in.
type tokenTiming struct {
	token int
	frame int
}

// decoderState carries the LSTM state, the last token and the cached decoder
// output across chunks.
type decoderState struct {
	hidden    []float32
	cell      []float32
	lstm      bool
	lastToken int
	token     bool
	cache     []float32
	hasCache  bool

	// scratch for the decoder outputs of the current step
	nextHidden []float32
	nextCell   []float32
}

// runDecoder runs TDT greedy decoding over one chunk of encoder activations.
func (m *Model) runDecoder(
	enc *encoderOutput, state *decoderState, isLast bool,
) ([]int, []tokenTiming, error) {
	validFrames := enc.validFrames
	if validFrames == 0 {
		return nil, nil, nil
	}

	// Token head first, then the duration bins, taken from the joint
	// network logits. A model that puts the durations first says so in the
	// config.
	tokensOffset, durationsOffset := headOffsets(
		m.blankID, len(m.durationBins), m.durationsFirst)

	encIn, encData, err := openvino.NewF32Tensor([]int64{1, 1, int64(m.encoderHidden)})
	if err != nil {
		return nil, nil, err
	}
	defer encIn.Close()
	decIn, decData, err := openvino.NewF32Tensor([]int64{1, 1, int64(m.decoderHidden)})
	if err != nil {
		return nil, nil, err
	}
	defer decIn.Close()
	hiddenIn, hiddenData, err := openvino.NewF32Tensor([]int64{2, 1, int64(m.decoderHidden)})
	if err != nil {
		return nil, nil, err
	}
	defer hiddenIn.Close()
	cellIn, cellData, err := openvino.NewF32Tensor([]int64{2, 1, int64(m.decoderHidden)})
	if err != nil {
		return nil, nil, err
	}
	defer cellIn.Close()
	target, err := openvino.NewTensor(m.targetsType, []int64{1, 1})
	if err != nil {
		return nil, nil, err
	}
	defer target.Close()

	if state.lstm {
		copy(hiddenData, state.hidden)
		copy(cellData, state.cell)
	}
	startingToken := m.blankID
	if state.token {
		startingToken = state.lastToken
	}
	lastToken := startingToken

	var tokens []int
	var timings []tokenTiming

	// runDecoderStep runs the predictor LSTM for lastToken, reusing the
	// cached decoder output when possible.
	runDecoderStep := func() error {
		if state.hasCache && state.token && lastToken == state.lastToken {
			copy(decData, state.cache)
			state.nextHidden = append(state.nextHidden[:0], hiddenData...)
			state.nextCell = append(state.nextCell[:0], cellData...)
			return nil
		}
		if err := target.SetInt(int64(lastToken)); err != nil {
			return err
		}
		if err := m.decoder.req.Set("targets", target); err != nil {
			return err
		}
		if err := m.decoder.req.Set("h_in", hiddenIn); err != nil {
			return err
		}
		if err := m.decoder.req.Set("c_in", cellIn); err != nil {
			return err
		}
		if err := m.decoder.req.Infer(); err != nil {
			return err
		}
		var err error
		if err = copyOutput(m.decoder.req, "decoder_output", decData); err != nil {
			return err
		}
		if state.nextHidden, err = readOutput(m.decoder.req, "h_out"); err != nil {
			return err
		}
		if state.nextCell, err = readOutput(m.decoder.req, "c_out"); err != nil {
			return err
		}
		state.cache = append(state.cache[:0], decData...)
		state.hasCache = true
		return nil
	}

	// runJoint extracts the encoder frame and runs the joint network.
	runJoint := func(frame int) (int, int, error) {
		if (m.encoderHidden-1)*enc.timeSteps+frame >= len(enc.data) {
			return 0, 0, serr.Errorf("parakeet: encoder frame %d out of range", frame)
		}
		for c := range m.encoderHidden {
			encData[c] = enc.data[c*enc.timeSteps+frame]
		}
		if err := m.joint.req.Set("encoder_outputs", encIn); err != nil {
			return 0, 0, err
		}
		if err := m.joint.req.Set("decoder_outputs", decIn); err != nil {
			return 0, 0, err
		}
		if err := m.joint.req.Infer(); err != nil {
			return 0, 0, err
		}
		logits, err := readOutput(m.joint.req, "logits")
		if err != nil {
			return 0, 0, err
		}
		if len(logits) < m.blankID+1+len(m.durationBins) {
			return 0, 0, serr.Errorf("parakeet: joint logits too small: %d", len(logits))
		}
		token := argmax(logits[tokensOffset : tokensOffset+m.blankID+1])
		durationBin := argmax(logits[durationsOffset : durationsOffset+len(m.durationBins)])
		return token, m.durationBins[durationBin], nil
	}

	frame := 0
	symbols := 0
	for frame < validFrames && len(tokens) < m.maxTokens {
		if err := runDecoderStep(); err != nil {
			return nil, nil, err
		}
		// Inner loop: blank tokens keep the same decoder output, they only
		// advance the encoder frame.
		for advance := true; advance && frame < validFrames && len(tokens) < m.maxTokens; {
			token, duration, err := runJoint(frame)
			if err != nil {
				return nil, nil, err
			}
			emitted := token != m.blankID && !m.tokenizer.isControl(token)
			if emitted {
				tokens = append(tokens, token)
				timings = append(timings, tokenTiming{
					token: token, frame: frame * m.melPerEncoderFrame,
				})
				lastToken = token
				copy(hiddenData, state.nextHidden)
				copy(cellData, state.nextCell)
				state.hasCache = false
				advance = false
			}
			var step int
			step, symbols = tdtStep(emitted, duration, symbols, m.maxSymbols)
			frame = min(frame+step, validFrames)
		}
	}
	// The loop stops at the token cap with encoder frames still to read, which
	// means this window's text is short. Say so rather than leave a transcript
	// that quietly stops.
	if len(tokens) >= m.maxTokens {
		fmt.Fprintf(os.Stderr,
			"parakeet: window hit the %d token cap, its text may be short\n",
			m.maxTokens)
	}

	// On the last chunk keep decoding at the final frame to flush trailing
	// tokens that need a few extra predictor steps. Control tokens count as
	// blanks here, so a device that keeps picking one stops the loop.
	if isLast {
		lastFrame := validFrames - 1
		steps, blanks := 0, 0
		for steps < finalizeSteps && blanks < finalizeBlanks && len(tokens) < m.maxTokens {
			if err := runDecoderStep(); err != nil {
				return nil, nil, err
			}
			token, _, err := runJoint(lastFrame)
			if err != nil {
				return nil, nil, err
			}
			if token != m.blankID && !m.tokenizer.isControl(token) {
				tokens = append(tokens, token)
				timings = append(timings, tokenTiming{
					token: token, frame: lastFrame * m.melPerEncoderFrame,
				})
				lastToken = token
				copy(hiddenData, state.nextHidden)
				copy(cellData, state.nextCell)
				state.hasCache = false
				blanks = 0
			} else {
				blanks++
			}
			steps++
		}
	}

	state.hidden = append(state.hidden[:0], hiddenData...)
	state.cell = append(state.cell[:0], cellData...)
	state.lstm = true
	state.token = true
	if len(tokens) > 0 {
		state.lastToken = tokens[len(tokens)-1]
		if m.tokenizer.isPunctuation(state.lastToken) {
			state.hasCache = false
		}
	} else {
		state.lastToken = startingToken
	}
	return tokens, timings, nil
}

// Transcribe turns 16 kHz mono samples into text.
func (m *Model) Transcribe(samples []float32) (*Result, error) {
	if len(samples) == 0 {
		return nil, serr.Errorf("parakeet: no audio samples")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.core == nil {
		return nil, serr.Errorf("parakeet: model is closed")
	}
	mel, err := m.preprocess(samples)
	if err != nil {
		return nil, err
	}
	tokens, err := m.processMel(mel)
	if err != nil {
		return nil, err
	}
	return &Result{Text: m.tokenizer.decode(tokens), Tokens: tokens}, nil
}

// windowDecode decodes one window of mel frames into tokens and their timings,
// timed in mel frames of the window. processMel takes the decoder as an argument
// so the arithmetic around it can be tested against a decoder of known manner,
// which is the only way to hold that arithmetic shut: it decides which window
// hears which audio, and a mistake in it is a stretch of the capture nobody
// hears. The model is no witness, it stops speaking early, in a different place
// every run, and says nothing about having done so.
type windowDecode func(
	chunk *melFeatures, isLast bool,
) ([]int, []tokenTiming, error)

// decodeWindow runs the encoder and the decoder over one window. The decoder
// state belongs to the window: what joins windows is their overlap and the
// guards in processMel, not the state of a decoder carried between them.
func (m *Model) decodeWindow(
	mel *melFeatures, isLast bool,
) ([]int, []tokenTiming, error) {
	enc, err := m.encode(mel)
	if err != nil {
		return nil, nil, err
	}
	return m.runDecoder(enc, &decoderState{}, isLast)
}

// processMel decodes a mel spectrogram, splitting long audio into overlapping
// windows of encoder frames so each one stays within the encoder's fixed input
// size, and stitching the texts back together.
func (m *Model) processMel(mel *melFeatures) ([]int, error) {
	return m.stitch(mel, m.decodeWindow)
}

// chunkRewind is what a window re-hears from where the previous decode stopped
// speaking, about a ninth of a window: enough to cover a decoder that starts a
// window late, short enough for the guards between windows to match the re-heard
// text (see dedupPrevTokens).
func chunkRewind(maxFrames int) int { return maxFrames / 9 }

// chunkAdvance is the least a window moves the march forward, so a window that
// comes back with nothing cannot keep the rest of the audio from being heard.
func chunkAdvance(maxFrames int) int { return maxFrames / 4 }

// stitch decodes mel with decode, one window at a time, and joins the text.
//
// Two things about the decoder shape this loop. It stops speaking before the end
// of a window, up to five seconds into fifteen, at a place that changes between
// runs; and a window that starts in the middle of a sentence says nothing for the
// first seconds of what it is given. Between them they can leave a stretch of the
// capture that every window which heard it passed over, which is the failure this
// loop exists to prevent: a capture typed out with a sentence missing and no
// sign of it anywhere. So the windows march from where the words stopped, the
// audio a window holds back for the next one is given back when that next window
// leaves it unsaid, and the windows can be pulled back to a sentence end so a
// fresh decoder starts at the beginning of a sentence.
func (m *Model) stitch(mel *melFeatures, decode windowDecode) ([]int, error) {
	maxFrames := m.encoderFrames
	if mel.frames <= maxFrames {
		tokens, _, err := decode(mel, true)
		return tokens, err
	}

	rewind, advance, context := chunkRewind(maxFrames), chunkAdvance(maxFrames),
		chunkContext(maxFrames)
	windows := 2 + (mel.frames-maxFrames)/advance
	var (
		tokens  []int
		timings []tokenTiming
		// The text of the window before this one that it was left to speak, and
		// where the text emitted so far ended a sentence.
		heldTokens  []int
		heldTimings []tokenTiming
		sentences   []int
	)
	lastEmitted, emitted := 0, 0
	offset := 0
	for n := 0; ; n++ {
		if n >= windows {
			// The march moves at least an advance per window, so it runs out
			// before this can; if it ever does, say so instead of handing back a
			// transcript that stops short.
			return nil, serr.Errorf(
				"parakeet: %d mel frames took %d windows of %d frames",
				mel.frames, n, maxFrames)
		}
		isLast := offset+maxFrames >= mel.frames
		if isLast {
			offset = mel.frames - maxFrames
		}

		// Each window decodes with a decoder of its own. Carrying the predictor
		// state over makes the model think the overlap it re-hears is already
		// said, and then it stays quiet through new speech: on a 35 second
		// capture it skipped a whole sentence, and a token cut in half at the
		// boundary ("for" for "forty") had the next window carry on from the
		// fragment and invent "the same".
		chunkTokens, chunkTimings, err := decode(
			extractMelChunk(mel, offset, maxFrames), isLast,
		)
		if err != nil {
			return nil, err
		}

		skip, end := 0, len(chunkTokens)
		if len(tokens) > 0 {
			// What this window re-hears of the audio before it.
			skip = dedupChunk(m, tokens, chunkTokens, chunkTimings, offset,
				lastEmitted, true)
		}
		if !isLast {
			// What this window hears of the audio the next one covers better.
			end = m.holdbackEnd(chunkTokens, chunkTimings, offset, maxFrames,
				rewind)
		}

		// Give back what the window before this one was left to say, to the
		// extent this one is not going to say it. A decoder that starts a window
		// late leaves its opening words unsaid, and they were the last thing the
		// previous window had.
		// The audio a window goes silent over, which the window before it spoke
		// of, is filled in from that window's reading. A decoder started in the
		// middle of a sentence says nothing for the first seconds of what it is
		// given, and the words in that silence are gone unless somebody has them
		// and the previous window had them and was told to hold them back for
		// nothing. Reading of the audio, once said, is worth more than the chance
		// the next window says the same thing better: a word said twice is a
		// blemish, a word nobody said is missing text.
		if len(heldTokens) > 0 {
			// Held tokens the current window says again at the head of what it is
			// about to emit are the same audio read twice, not a gap to fill.
			drop := repeatedHeld(heldTokens, chunkTokens, skip)
			held, heldTimes := heldTokens[drop:], heldTimings[drop:]
			if end > skip {
				gapTo := chunkTimings[skip].frame - 1
				for i := range held {
					if heldTimes[i].frame > gapTo {
						break
					}
					tokens = append(tokens, held[i])
					timings = append(timings, heldTimes[i])
				}
			} else if len(held) > 0 {
				// It said nothing at all over the audio it was left.
				tokens = append(tokens, held...)
				timings = append(timings, heldTimes...)
			}
		}

		before := len(timings)
		if skip < end {
			tokens = append(tokens, chunkTokens[skip:end]...)
			timings = append(timings, chunkTimings[skip:end]...)
		}

		// What this window said of the audio the next one will hear, and the
		// sentence ends of the text so far.
		heldTokens, heldTimings = nil, nil
		if !isLast {
			heldTokens, heldTimings = chunkTokens[end:], chunkTimings[end:]
		}
		for _, t := range timings[before:] {
			if m.tokenizer.isPunctuation(t.token) {
				sentences = append(sentences, t.frame)
			}
		}
		if len(timings) > emitted {
			lastEmitted = timings[len(timings)-1].frame
		}
		if isLast {
			break
		}

		next := offset + advance
		if lastEmitted+1-rewind > next {
			next = lastEmitted + 1 - rewind
		}
		// Start the next window where a sentence starts, when a sentence started
		// not far before there: a decoder that begins in the middle of one spends
		// the beginning of the audio it is given silent, and the words there are
		// heard by nobody else. The pull back is bounded by a sentence of left
		// context, and by the least advance either way.
		if p := lastSentence(sentences, next-1); p >= 0 && p+1 < next {
			next = max(p+1, offset+advance, next-context)
		}
		offset = min(next, mel.frames-maxFrames)
		emitted = len(timings)
	}
	return tokens, nil
}

// chunkContext is how far back a window may be pulled to reach the start of a
// sentence, about a sentence of left context: ten words is four to five seconds,
// which is a fifth of a window. Further than that buys little, and the audio it
// re-hears is audio the encoder runs again: over four long audiobook captures the
// pull back cost 1.8 times the compute and cut the words nobody spoke from 5.3% of
// the reference to 0.9%.
func chunkContext(maxFrames int) int { return maxFrames / 3 }

// lastSentence is where the text before a frame last ended a sentence, or -1.
func lastSentence(sentences []int, before int) int {
	last := -1
	for _, p := range sentences {
		if p <= before {
			last = p
		}
	}
	return last
}

// repeatedHeld is how many of the tokens a window held back the next window
// says again at the head of what it is about to emit. The two windows read the
// same audio and stamp the same word a few frames apart, which is enough for
// the fill to see a gap in front of the current reading and add a copy of a
// word that is already there; those leading copies are the ones to drop.
func repeatedHeld(held, chunk []int, skip int) int {
	n := 0
	for n < len(held) && skip+n < len(chunk) && held[n] == chunk[skip+n] {
		n++
	}
	return n
}

// holdbackEnd returns how many tokens of a window to emit now that another
// window follows it. Tokens in the window's overlap region belong to audio the
// next window hears with right context in front of it, so they are left to it,
// unless a sentence ends in that region: cutting in the middle of a sentence
// loses words when the next window's reading of the audio comes out shorter,
// which it does, so the cut goes after the last sentence end instead. timings
// and the offsets are mel frames, timings already global.
func (m *Model) holdbackEnd(
	tokens []int, timings []tokenTiming, offset, size, overlap int,
) int {
	threshold := 0
	if size > overlap {
		threshold = size - overlap
	}
	end := len(timings)
	for i, t := range timings {
		if t.frame-offset >= threshold {
			end = i
			break
		}
	}
	for i := end; i < len(tokens); i++ {
		if m.tokenizer.isPunctuation(tokens[i]) {
			end = i + 1
			break
		}
	}
	return end
}

// extractMelChunk copies a slice of frames out of a mel spectrogram. The bins
// are rows of timeSteps floats in one flat slice, so a window that ran past the
// end of the audio would read the next bin's frames rather than fail: the frames
// past the end are left as the silence the encoder expects instead.
func extractMelChunk(mel *melFeatures, offset, size int) *melFeatures {
	data := make([]float32, melBins*size)
	for b := range melBins {
		src := mel.data[b*mel.timeSteps+offset:]
		src = src[:min(size, len(src))]
		copy(data[b*size:b*size+len(src)], src)
	}
	return &melFeatures{data: data, frames: size, timeSteps: size}
}

// dedupChunk returns how many tokens at the front of a window were already
// emitted by the windows before it. The windows overlap, so a window re-hears
// audio that has a transcript already: first the tokens that sit at or before
// the last position emitted, then a run of tokens whose text repeats the tail
// of what is emitted, which catches a re-hearing stamped at another position.
// As a side effect it moves timings from mel frames inside the window to mel
// frames inside the whole audio.
func dedupChunk(
	m *Model,
	prev, curr []int,
	timings []tokenTiming,
	offset, lastEmitted int,
	haveLast bool,
) (skip int) {
	for i := range timings {
		timings[i].frame += offset
	}

	// A token stamped at the position already emitted is the same word a window
	// re-hears: two windows reading the same audio put the same word at the same
	// mel frame. Anything after it is new.
	if haveLast && len(timings) > 0 {
		gate := 0
		for gate < len(timings) && timings[gate].frame <= lastEmitted {
			gate++
		}
		skip = max(skip, gate)
	}

	// Drop a duplicated leading punctuation token.
	punctuation := 0
	if len(prev) > 0 && len(curr) > 0 && curr[0] == prev[len(prev)-1] &&
		m.tokenizer.isPunctuation(curr[0]) {
		punctuation = 1
	}

	working := max(0, len(curr)-punctuation)
	prevTail := min(dedupPrevTokens, len(prev))

	// Longest suffix of prev matching a prefix of curr.
	exact := 0
	for l := min(prevTail, dedupMaxOverlap, working); l > 1; l-- {
		if slices.Equal(prev[len(prev)-l:], curr[punctuation:punctuation+l]) {
			exact = l
			break
		}
	}

	dedup := punctuation
	if exact == 0 {
		// Search for an overlapping run near the start of curr, limited to
		// the right context window of the chunk.
		searchLimit := 0
		if len(timings) > 0 {
			for i, t := range timings {
				if t.frame-offset >= dedupBoundaryFrames {
					break
				}
				if i >= punctuation {
					searchLimit = i - punctuation + 1
				}
			}
		} else {
			searchLimit = min(working, dedupBoundaryFrames)
		}
	outer:
		for l := min(prevTail, dedupMaxOverlap, working); l > 1; l-- {
			prevStartMin := max(0, len(prev)-prevTail)
			prevEnd := 0
			if len(prev) >= l {
				prevEnd = len(prev) - l + 1
			}
			if prevEnd <= prevStartMin {
				continue
			}
			for ps := prevStartMin; ps < prevEnd; ps++ {
				currEndLimit := 0
				if working >= l {
					currEndLimit = working - l + 1
				}
				for co := 0; co < min(searchLimit, currEndLimit); co++ {
					if slices.Equal(prev[ps:ps+l], curr[punctuation+co:punctuation+co+l]) {
						dedup = max(dedup, punctuation+co+l)
						break outer
					}
				}
			}
		}
	} else {
		dedup += exact
	}
	skip = max(skip, dedup)

	// The matches above look at the head of the window. A window that re-hears
	// audio already emitted can stamp the same words a little later, past the
	// position gate, so the head of what the gate leaves can be a duplicate too.
	// The last window is shifted back to end with the audio, which makes it
	// re-read a long stretch, and this is where its reading of the tail lands.
	// Only look when that head is still near the boundary, so a real repeat deep
	// in new audio is not merged.
	if haveLast && skip < len(curr) && skip < len(timings) &&
		timings[skip].frame-lastEmitted <= dedupBoundaryFrames {
		tail := min(dedupPrevTokens, len(prev))
		for l := min(tail, dedupMaxOverlap, len(curr)-skip); l > 1; l-- {
			if slices.Equal(prev[len(prev)-l:], curr[skip:skip+l]) {
				skip += l
				break
			}
		}
	}
	return skip
}

// readOutput copies a named output tensor into a new Go slice.
func readOutput(req *openvino.Request, name string) ([]float32, error) {
	t, err := req.Get(name)
	if err != nil {
		return nil, err
	}
	defer t.Close()
	data, err := t.F32()
	if err != nil {
		return nil, err
	}
	return slices.Clone(data), nil
}

// copyOutput copies a named output tensor into dst.
func copyOutput(req *openvino.Request, name string, dst []float32) error {
	t, err := req.Get(name)
	if err != nil {
		return err
	}
	defer t.Close()
	data, err := t.F32()
	if err != nil {
		return err
	}
	if len(data) != len(dst) {
		return serr.Errorf("parakeet: tensor %s has %d values, want %d",
			name, len(data), len(dst))
	}
	copy(dst, data)
	return nil
}

// headOffsets is where the token head and the duration head start in the
// joint network's logits. The exports put the tokens first, so the duration
// bins follow the blank; a model exported the other way around needs
// Config.DurationsFirst, or the loop would read a duration bin as a token and
// a token as a duration.
func headOffsets(blankID, durationBins int, durationsFirst bool) (tokens, durations int) {
	if durationsFirst {
		return durationBins, 0
	}
	return 0, blankID + 1
}

// tdtAdvance is how many encoder frames the TDT greedy loop moves past after
// one joint output. A non-blank token may predict a duration of zero, which
// says the next token belongs at the same frame: the loop stays there and
// decodes it, which is how a word split across several pieces keeps its
// pieces. A blank cannot predict zero — nothing was said — so it moves at
// least one frame, or the loop would read the same frame forever.
func tdtAdvance(emitted bool, duration int) int {
	if emitted {
		return max(duration, 0)
	}
	return max(duration, 1)
}

// tdtStep is how far the loop moves after one joint output and how many tokens
// have now been emitted at the current frame. A frame that has emitted
// maxSymbols tokens is forced to move on, which is what the TDT decoders NeMo
// ships do: without it a model that keeps predicting a zero duration at one
// frame would fill the token cap there instead of speaking.
func tdtStep(emitted bool, duration, symbols, maxSymbols int) (step, nextSymbols int) {
	if emitted {
		symbols++
	}
	step = tdtAdvance(emitted, duration)
	if maxSymbols > 0 && symbols >= maxSymbols {
		step = max(step, 1)
	}
	if step > 0 {
		symbols = 0
	}
	return step, symbols
}

// argmax returns the index of the largest value.
func argmax(values []float32) int {
	best := 0
	for i, v := range values {
		if v > values[best] {
			best = i
		}
	}
	return best
}

// roundUp rounds n up to a multiple of to.
func roundUp(n, to int) int {
	if n <= 0 {
		return to
	}
	return ((n + to - 1) / to) * to
}
