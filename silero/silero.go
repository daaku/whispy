// Package silero runs the 16 kHz Silero VAD model (v5/v6) in pure Go. It reads
// the weights straight out of the ONNX file the OpenVINO implementation uses,
// so both can run the same model, on the same audio, and be compared against
// each other.
//
// The graph is the one OpenVINO runs: a reflect padded STFT magnitude stack,
// four convolutions, one LSTM step and a final convolution with a sigmoid. On
// amd64 with GOEXPERIMENT=simd the matrix kernels use simd/archsimd, everywhere
// else they are plain scalar loops.
package silero

import (
	"math"
	"math/bits"
	"sync"

	"github.com/daaku/serr"
)

const (
	// SampleRate is the only sample rate the 16 kHz model supports.
	SampleRate = 16000
	// WindowSamples is the number of samples the model consumes per step.
	WindowSamples = 512
	// Threshold is the speech probability the Silero authors classify with.
	Threshold = 0.5

	// contextSamples is prepended to every window, as the Silero wrapper does.
	contextSamples = 64
	// chunkSamples is what the model takes in one call: context plus window.
	chunkSamples = WindowSamples + contextSamples
	// reflectPad is added to the chunk before the STFT, as the ONNX graph does.
	reflectPad = 64

	fftSize    = 256
	hopSamples = 128
	binCount   = fftSize/2 + 1
	// stftFrames is how many STFT frames a chunk produces.
	stftFrames = (chunkSamples+reflectPad-fftSize)/hopSamples + 1

	hiddenSize = 128
	gateSize   = 4 * hiddenSize
)

// The convolution stack, in order: input and output channels. Every layer has
// a kernel of 3 and a padding of 1; conv1 and conv4 have a stride of 1 and
// conv2 and conv3 halve the frame count with a stride of 2.
const (
	conv1In, conv1Out = binCount, 128
	conv2In, conv2Out = 128, 64
	conv3In, conv3Out = 64, 64
	conv4In, conv4Out = 64, 128
)

// Frame counts through the stack. conv1 keeps all the STFT frames, conv2 and
// conv3 halve them with a stride 2 and a kernel 3 whose padding cancels the
// loss, and conv4 keeps the one frame that is left.
const (
	conv2Frames = (stftFrames + 1) / 2
	conv3Frames = (conv2Frames + 1) / 2
	conv4Frames = conv3Frames
)

// Config configures a VAD instance.
type Config struct {
	// Model is the path to silero_vad.onnx.
	Model string
}

// Vad is a streaming voice activity detector. It is safe for concurrent use,
// though a single instance processes one stream at a time.
type Vad struct {
	mu      sync.Mutex
	model   *model
	chunk   []float32
	context []float32
	pending []float32
}

// New loads the Silero VAD weights from an ONNX model file.
func New(cfg Config) (*Vad, error) {
	if cfg.Model == "" {
		return nil, serr.Errorf("silero: model path is required")
	}
	inits, err := readInitializers(cfg.Model)
	if err != nil {
		return nil, err
	}
	m, err := newModel(inits)
	if err != nil {
		return nil, err
	}
	return &Vad{
		model:   m,
		chunk:   make([]float32, chunkSamples),
		context: make([]float32, contextSamples),
	}, nil
}

// Close drops the model. The Vad cannot be used after this.
func (v *Vad) Close() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.model = nil
	return nil
}

// Reset drops the streaming state and any buffered samples, as if the model
// had just been loaded.
func (v *Vad) Reset() {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.model == nil {
		return
	}
	v.model.reset()
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
		return nil, serr.Errorf("silero: closed")
	}
	v.pending = append(v.pending, samples...)
	if len(v.pending) < WindowSamples {
		return nil, nil
	}
	var probs []float32
	off := 0
	for ; off+WindowSamples <= len(v.pending); off += WindowSamples {
		probs = append(probs, v.runWindow(v.pending[off:off+WindowSamples]))
	}
	v.pending = append(v.pending[:0], v.pending[off:]...)
	return probs, nil
}

func (v *Vad) runWindow(window []float32) float32 {
	copy(v.chunk, v.context)
	copy(v.chunk[contextSamples:], window)
	prob := v.model.forward(v.chunk)
	copy(v.context, window[WindowSamples-contextSamples:])
	return prob
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

// model holds the packed weights and every scratch buffer the forward pass
// needs, so a window costs no allocations.
type model struct {
	conv1W, conv1B []float32
	conv2W, conv2B []float32
	conv3W, conv3B []float32
	conv4W, conv4B []float32
	rnnWih, rnnWhh []float32
	rnnB           []float32
	finalW, finalB []float32

	fft    *fft256
	padded []float32 // chunkSamples + reflectPad
	mag    []float32 // stftFrames * binCount
	conv1  []float32 // stftFrames * conv1Out
	conv2  []float32 // conv2Frames * conv2Out
	conv3  []float32 // conv3Frames * conv3Out
	conv4  []float32 // conv4Frames * conv4Out
	patch  []float32 // the largest convolution input patch
	gates  []float32 // gateSize
	hidden []float32 // hiddenSize, the LSTM state carried between windows
	cell   []float32 // hiddenSize, the LSTM state carried between windows
	relu   []float32 // hiddenSize, the LSTM output the final convolution sees
}

func newModel(inits map[string][]float32) (*model, error) {
	l := &loader{inits: inits}
	m := &model{
		conv1W: packConv(l.get("encoder.0.reparam_conv.weight", conv1Out*conv1In*3), conv1In, conv1Out, 3),
		conv1B: l.get("encoder.0.reparam_conv.bias", conv1Out),
		conv2W: packConv(l.get("encoder.1.reparam_conv.weight", conv2Out*conv2In*3), conv2In, conv2Out, 3),
		conv2B: l.get("encoder.1.reparam_conv.bias", conv2Out),
		conv3W: packConv(l.get("encoder.2.reparam_conv.weight", conv3Out*conv3In*3), conv3In, conv3Out, 3),
		conv3B: l.get("encoder.2.reparam_conv.bias", conv3Out),
		conv4W: packConv(l.get("encoder.3.reparam_conv.weight", conv4Out*conv4In*3), conv4In, conv4Out, 3),
		conv4B: l.get("encoder.3.reparam_conv.bias", conv4Out),
		rnnWih: packRNN(l.get("decoder.rnn.weight_ih", gateSize*hiddenSize)),
		rnnWhh: packRNN(l.get("decoder.rnn.weight_hh", gateSize*hiddenSize)),
		finalW: l.get("decoder.decoder.2.weight", hiddenSize),
		finalB: l.get("decoder.decoder.2.bias", 1),
		fft:    newFFT256(),
		padded: make([]float32, chunkSamples+reflectPad),
		mag:    make([]float32, stftFrames*binCount),
		conv1:  make([]float32, stftFrames*conv1Out),
		conv2:  make([]float32, conv2Frames*conv2Out),
		conv3:  make([]float32, conv3Frames*conv3Out),
		conv4:  make([]float32, conv4Frames*conv4Out),
		patch:  make([]float32, conv1In*3),
		gates:  make([]float32, gateSize),
		hidden: make([]float32, hiddenSize),
		cell:   make([]float32, hiddenSize),
		relu:   make([]float32, hiddenSize),
	}
	bih := l.get("decoder.rnn.bias_ih", gateSize)
	bhh := l.get("decoder.rnn.bias_hh", gateSize)
	m.rnnB = make([]float32, gateSize)
	for i := range m.rnnB {
		m.rnnB[i] = bih[i] + bhh[i]
	}
	if l.err != nil {
		return nil, l.err
	}
	return m, nil
}

// reset clears the state carried between windows.
func (m *model) reset() {
	clear(m.hidden)
	clear(m.cell)
}

// forward runs one chunk of chunkSamples and returns the speech probability.
func (m *model) forward(chunk []float32) float32 {
	n := len(chunk)
	copy(m.padded, chunk)
	for i := 0; i < reflectPad; i++ {
		m.padded[n+i] = chunk[2*(n-1)-(n+i)]
	}

	for f := 0; f < stftFrames; f++ {
		m.fft.magnitude(m.padded, f*hopSamples, m.mag[f*binCount:])
	}

	m.conv(m.conv1W, m.conv1B, m.mag, m.conv1, conv1In, conv1Out, stftFrames, stftFrames, 1, 1)
	relu(m.conv1)
	m.conv(m.conv2W, m.conv2B, m.conv1, m.conv2, conv2In, conv2Out, stftFrames, conv2Frames, 2, 1)
	relu(m.conv2)
	m.conv(m.conv3W, m.conv3B, m.conv2, m.conv3, conv3In, conv3Out, conv2Frames, conv3Frames, 2, 1)
	relu(m.conv3)
	m.conv(m.conv4W, m.conv4B, m.conv3, m.conv4, conv4In, conv4Out, conv3Frames, conv4Frames, 1, 1)
	relu(m.conv4)

	m.lstmStep(m.conv4)
	copy(m.relu, m.hidden)
	relu(m.relu)

	logit := m.finalB[0]
	for i, w := range m.finalW {
		logit += w * m.relu[i]
	}
	return sigmoid(logit)
}

// conv runs one convolution layer. The input and output are frame major: frame
// f of channel c is at f*channels+c. The kernel walks time with zero padding at
// the edges, and m.patch is the transposed input window the matrix kernel
// wants.
func (m *model) conv(
	w, b, in, out []float32,
	inCh, outCh, inFrames, outFrames, stride, pad int,
) {
	patch := m.patch[:inCh*3]
	for of := 0; of < outFrames; of++ {
		for tap := 0; tap < 3; tap++ {
			col := patch[tap*inCh : (tap+1)*inCh]
			at := of*stride + tap - pad
			if at < 0 || at >= inFrames {
				clear(col)
				continue
			}
			copy(col, in[at*inCh:(at+1)*inCh])
		}
		dst := out[of*outCh : (of+1)*outCh]
		copy(dst, b)
		matvecAdd(dst, w, patch, inCh*3, outCh)
	}
}

// lstmStep runs one LSTMCell step with x as the input, keeping the hidden and
// cell state in the model.
func (m *model) lstmStep(x []float32) {
	copy(m.gates, m.rnnB)
	matvecAdd(m.gates, m.rnnWih, x, hiddenSize, gateSize)
	matvecAdd(m.gates, m.rnnWhh, m.hidden, hiddenSize, gateSize)
	for i := 0; i < hiddenSize; i++ {
		in := sigmoid(m.gates[i])
		forget := sigmoid(m.gates[hiddenSize+i])
		cell := forget*m.cell[i] + in*tanh32(m.gates[2*hiddenSize+i])
		m.cell[i] = cell
		out := sigmoid(m.gates[3*hiddenSize+i])
		m.hidden[i] = out * tanh32(cell)
	}
}

// matvecAddScalar adds packedᵀ x to dst, where packed holds nIn rows of nOut.
// It is the fallback for builds without a vector kernel.
func matvecAddScalar(dst, packed, x []float32, nIn, nOut int) {
	for i := 0; i < nIn; i++ {
		xv := x[i]
		row := packed[i*nOut : (i+1)*nOut]
		for o, w := range row {
			dst[o] += w * xv
		}
	}
}

// packConv rewrites a convolution weight [outCh][inCh][3] into rows of outCh,
// ordered by (tap, inCh), so matvecAdd can broadcast one input at a time.
func packConv(w []float32, inCh, outCh, k int) []float32 {
	packed := make([]float32, inCh*k*outCh)
	for oc := 0; oc < outCh; oc++ {
		for ic := 0; ic < inCh; ic++ {
			for tap := 0; tap < k; tap++ {
				packed[(tap*inCh+ic)*outCh+oc] = w[oc*inCh*k+ic*k+tap]
			}
		}
	}
	return packed
}

// packRNN rewrites an LSTMCell weight [gateSize][hiddenSize] into rows of
// gateSize, so matvecAdd can broadcast one input at a time.
func packRNN(w []float32) []float32 {
	packed := make([]float32, hiddenSize*gateSize)
	for g := 0; g < gateSize; g++ {
		for i := 0; i < hiddenSize; i++ {
			packed[i*gateSize+g] = w[g*hiddenSize+i]
		}
	}
	return packed
}

// loader pulls the weights out of the ONNX initializers. Both the names the
// 16 kHz graph uses directly and the ones the single model export prefixes with
// "model." are accepted.
type loader struct {
	inits map[string][]float32
	err   error
}

func (l *loader) get(name string, n int) []float32 {
	if l.err != nil {
		return make([]float32, n)
	}
	v, ok := l.inits[name]
	if !ok {
		v, ok = l.inits["model."+name]
	}
	if !ok {
		l.err = serr.Errorf("silero: model has no %s", name)
		return make([]float32, n)
	}
	if len(v) != n {
		l.err = serr.Errorf("silero: %s has %d values, want %d", name, len(v), n)
		return make([]float32, n)
	}
	return v
}

// fft256 is a 256 point complex FFT for the STFT. The real input is windowed
// and loaded in bit reversed order, then combined with the usual radix 2
// butterflies over the precomputed twiddles.
type fft256 struct {
	win [fftSize]float32
	cos [fftSize / 2]float32
	sin [fftSize / 2]float32
	rev [fftSize]uint8
	re  [fftSize]float32
	im  [fftSize]float32
}

func newFFT256() *fft256 {
	f := &fft256{}
	for i := 0; i < fftSize; i++ {
		angle := 2 * math.Pi * float64(i) / fftSize
		// The periodic Hann window torch.stft uses, and the twiddles for the
		// forward transform.
		f.win[i] = float32(0.5 - 0.5*math.Cos(angle))
		if i < fftSize/2 {
			f.cos[i] = float32(math.Cos(angle))
			f.sin[i] = float32(math.Sin(angle))
		}
		f.rev[i] = bits.Reverse8(uint8(i))
	}
	return f
}

// magnitude fills dst with the magnitude of the binCount bins of the frame at
// off in src.
func (f *fft256) magnitude(src []float32, off int, dst []float32) {
	for i := 0; i < fftSize; i++ {
		j := f.rev[i]
		f.re[j] = src[off+i] * f.win[i]
		f.im[j] = 0
	}
	for size := 2; size <= fftSize; size <<= 1 {
		half := size >> 1
		step := fftSize / size
		for start := 0; start < fftSize; start += size {
			for j := 0; j < half; j++ {
				k := j * step
				wr, wi := f.cos[k], -f.sin[k]
				a, b := start+j, start+j+half
				br, bi := f.re[b], f.im[b]
				tr := br*wr - bi*wi
				ti := br*wi + bi*wr
				ar, ai := f.re[a], f.im[a]
				f.re[a], f.im[a] = ar+tr, ai+ti
				f.re[b], f.im[b] = ar-tr, ai-ti
			}
		}
	}
	for i := 0; i < binCount; i++ {
		re, im := f.re[i], f.im[i]
		dst[i] = float32(math.Sqrt(float64(re*re + im*im)))
	}
}

func relu(s []float32) {
	for i, x := range s {
		if x < 0 {
			s[i] = 0
		}
	}
}

func sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
}

func tanh32(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}
