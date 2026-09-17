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
	cpuDevice            = "CPU"

	// finalize steps flush trailing tokens after the last encoder frame.
	finalizeSteps  = 8
	finalizeBlanks = 1

	// Chunk boundary deduplication tuning.
	dedupPrevTokens     = 15
	dedupBoundaryFrames = 20
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
	// DurationBins are the TDT duration bin values. Defaults to 0, 1, 2, 3, 4.
	DurationBins []int
	// MaxTokens caps the tokens produced per chunk. Defaults to 256.
	MaxTokens int
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
	encoderHidden  int

	targetsType   openvino.ElementType
	decoderHidden int

	jointOut int

	blankID         int
	durationBins    []int
	maxTokens       int
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
		maxTokens:       maxTokens,
		requestedDevice: device,
	}
	for _, c := range []struct {
		file   string
		device string
		props  map[string]string
		dst    **component
	}{
		{"parakeet_melspectogram.xml", preprocDevice, nil, &m.preproc},
		{"parakeet_encoder.xml", device, cfg.Properties, &m.encoder},
		{"parakeet_decoder.xml", decoderDevice, cfg.Properties, &m.decoder},
		{"parakeet_joint.xml", decoderDevice, cfg.Properties, &m.joint},
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
			fmt.Fprintf(os.Stderr, "parakeet: %s: %s, using %s (%s)%s\n",
				c.name, c.device, c.usedDevice,
				cleanError(c.fallback), negativeAxisHint(c.path))
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

// negativeAxisHint points at the known workaround when a model still carries a
// negative axis attribute. The NPU compiler's AlignDimensionsForDPU pass
// rejects those ("Got negative index -1 for Dim"), and for the joint network's
// LogSoftmax the axis can be written positively instead.
func negativeAxisHint(path string) string {
	data, err := os.ReadFile(path)
	if err != nil || !bytes.Contains(data, []byte(`axis="-1"`)) {
		return ""
	}
	return "; this model has a negative axis attribute which the NPU compiler" +
		" rejects, see the NPU notes in the readme"
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
	cm, err := core.CompileWith(path, device, props)
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

	// Token head first, then the duration bins, both taken from the joint
	// network logits.
	tokensOffset, durationsOffset := 0, m.blankID+1

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
		if len(logits) < tokensOffset+m.blankID+1+len(m.durationBins) {
			return 0, 0, serr.Errorf("parakeet: joint logits too small: %d", len(logits))
		}
		token := argmax(logits[tokensOffset : tokensOffset+m.blankID+1])
		durationBin := argmax(logits[durationsOffset : durationsOffset+len(m.durationBins)])
		return token, m.durationBins[durationBin], nil
	}

	frame := 0
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
			if duration <= 0 {
				duration = 1
			}
			if token != m.blankID && !m.tokenizer.isControl(token) {
				tokens = append(tokens, token)
				timings = append(timings, tokenTiming{token: token, frame: frame})
				lastToken = token
				copy(hiddenData, state.nextHidden)
				copy(cellData, state.nextCell)
				state.hasCache = false
				advance = false
			}
			frame = min(frame+duration, validFrames)
		}
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
				timings = append(timings, tokenTiming{token: token, frame: lastFrame})
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

// decodeChunk runs the encoder and decoder over one mel chunk.
func (m *Model) decodeChunk(
	mel *melFeatures, state *decoderState, isLast bool,
) ([]int, []tokenTiming, error) {
	enc, err := m.encode(mel)
	if err != nil {
		return nil, nil, err
	}
	return m.runDecoder(enc, state, isLast)
}

// processMel decodes a mel spectrogram, splitting long audio into overlapping
// chunks so the encoder input stays within its fixed frame count.
func (m *Model) processMel(mel *melFeatures) ([]int, error) {
	maxFrames := m.encoderFrames
	state := &decoderState{}
	if mel.frames <= maxFrames {
		tokens, _, err := m.decodeChunk(mel, state, true)
		return tokens, err
	}

	overlap := min(maxFrames-1, max(4, maxFrames/9))
	stride := maxFrames - overlap
	var tokens []int
	var timings []tokenTiming
	lastEmitted, haveLast := 0, false
	first := true
	for offset := 0; offset < mel.frames; {
		size := min(maxFrames, mel.frames-offset)
		isLast := offset+size >= mel.frames
		chunkTokens, chunkTimings, err := m.decodeChunk(
			extractMelChunk(mel, offset, size), state, isLast,
		)
		if err != nil {
			return nil, err
		}
		if first {
			// The first chunk keeps all of its tokens, including the
			// overlap region, so the next chunk has something to match.
			for i := range chunkTimings {
				chunkTimings[i].frame += offset
			}
			tokens = chunkTokens
			timings = chunkTimings
			first = false
		} else {
			skip, end := dedupChunk(m, tokens, chunkTokens, chunkTimings,
				offset, size, overlap, isLast, lastEmitted, haveLast)
			if skip < end {
				tokens = append(tokens, chunkTokens[skip:end]...)
				timings = append(timings, chunkTimings[skip:end]...)
			}
		}
		if len(timings) > 0 {
			lastEmitted = timings[len(timings)-1].frame
			haveLast = true
		}
		if offset+size >= mel.frames {
			break
		}
		offset += stride
	}
	return tokens, nil
}

// extractMelChunk copies a slice of frames out of a mel spectrogram.
func extractMelChunk(mel *melFeatures, offset, size int) *melFeatures {
	data := make([]float32, melBins*size)
	for b := range melBins {
		src := mel.data[b*mel.timeSteps+offset : b*mel.timeSteps+offset+size]
		copy(data[b*size:(b+1)*size], src)
	}
	return &melFeatures{data: data, frames: size, timeSteps: size}
}

// dedupChunk finds the token prefix of a chunk that repeats the previous
// chunk and the token suffix that should be held back for the next one. It
// adjusts timings to global frame indices as a side effect.
func dedupChunk(
	m *Model,
	prev, curr []int,
	timings []tokenTiming,
	offset, size, overlap int,
	isLast bool,
	lastEmitted int,
	haveLast bool,
) (skip, emitEnd int) {
	for i := range timings {
		timings[i].frame += offset
	}
	emitEnd = len(curr)

	// Enforce monotonically increasing global frame indices.
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
	if exact > 0 {
		dedup += exact
	} else {
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
	}
	skip = max(skip, dedup)

	// Hold back tokens in the overlap region of a non final chunk so the next
	// chunk can decode them with more right context.
	if !isLast && len(timings) > 0 {
		if right := min(overlap, size); right > 0 {
			threshold := 0
			if size > right {
				threshold = size - right
			}
			holdbackStart := len(curr)
			for i, t := range timings {
				if t.frame-offset >= threshold {
					holdbackStart = i
					break
				}
			}
			if holdbackStart < len(curr) {
				emitEnd = min(emitEnd, holdbackStart)
			}
		}
	}
	return skip, emitEnd
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
