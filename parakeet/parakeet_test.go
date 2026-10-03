package parakeet

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/daaku/serr"
	"github.com/daaku/whispy/audio"
	"github.com/daaku/whispy/openvino"
)

// modelDir locates the Parakeet v3 OpenVINO IR files, skipping the test when
// they are not available on the machine.
func modelDir(t *testing.T) string {
	t.Helper()
	home, _ := os.UserHomeDir()
	dirs := []string{
		os.Getenv("PARAKEET_MODEL_DIR"),
		filepath.Join(home, ".cache/whispy/parakeet-v3"),
	}
	for _, dir := range dirs {
		if dir == "" {
			continue
		}
		if _, err := os.Stat(filepath.Join(dir, "parakeet_encoder.xml")); err == nil {
			return dir
		}
	}
	t.Skip("parakeet model files not found, set PARAKEET_MODEL_DIR")
	return ""
}

// testDevice is the CPU unless PARAKEET_DEVICE names another OpenVINO device,
// which is how the same expectations get checked on a GPU or an NPU:
//
//	PARAKEET_DEVICE=GPU go test -count=1 -v ./parakeet/
//
// The models report where they ended up on stderr, so a device that quietly
// falls back to the CPU is visible in the output.
func testDevice() string {
	if device := os.Getenv("PARAKEET_DEVICE"); device != "" {
		return device
	}
	return "CPU"
}

// testDecoderDevice is PARAKEET_DECODER_DEVICE, empty by default so the
// decoder and joint networks follow the encoder:
//
//	PARAKEET_DEVICE=GPU PARAKEET_DECODER_DEVICE=CPU go test ./parakeet/
func testDecoderDevice() string {
	return os.Getenv("PARAKEET_DECODER_DEVICE")
}

func newTestModel(t *testing.T) *Model {
	t.Helper()
	m, err := New(Config{
		Dir:           modelDir(t),
		Device:        testDevice(),
		DecoderDevice: testDecoderDevice(),
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func TestTokenizerRealVocab(t *testing.T) {
	tok, err := loadTokenizer(
		filepath.Join(modelDir(t), "parakeet_vocab.json"), 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if tok.blankID != 8192 {
		t.Fatalf("blankID = %d, want 8192", tok.blankID)
	}
	if got, want := tok.decode([]int{1976, 547, 7877}), "And so,"; got != want {
		t.Fatalf("decode = %q, want %q", got, want)
	}
	// The first ids are <unk> and the control tags, not text.
	if !tok.isControl(0) || !tok.isControl(1) || tok.isControl(1976) {
		t.Fatal("control tokens are not distinguished from text")
	}
	if got, want := tok.decode([]int{1976, 0, 547}), "And so"; got != want {
		t.Fatalf("decode with <unk> = %q, want %q", got, want)
	}
	// "I want forty two and thirty five" written with digits: ▁I ▁want 4 2 ▁and
	// 3 5. There is no piece that could carry the space in front of a digit, so
	// the decoder has to supply it, and digits still join each other.
	if got, want := tok.decode([]int{380, 4648, 238, 236, 575, 237, 239}),
		"I want 42 and 35"; got != want {
		t.Fatalf("decode with digits = %q, want %q", got, want)
	}
}

// The decoder and joint networks can be placed separately from the encoder,
// which matters when an accelerator is good at the wide encoder inference and
// bad at the per token ones. A device that does not exist proves the wiring:
// the two per token models fall back, the encoder does not.
func TestDecoderDevice(t *testing.T) {
	m, err := New(Config{
		Dir:           modelDir(t),
		Device:        "CPU",
		DecoderDevice: "GHOST",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	if m.encoder.fallback != nil {
		t.Fatalf("encoder fell back: %v", m.encoder.fallback)
	}
	if m.decoder.fallback == nil || m.joint.fallback == nil {
		t.Fatal("decoder and joint did not fall back to the CPU")
	}
	if m.decoder.usedDevice != "CPU" || m.joint.usedDevice != "CPU" {
		t.Fatal("decoder and joint did not end up on the CPU")
	}
}

func TestTranscribeJFK(t *testing.T) {
	m := newTestModel(t)
	samples, err := audio.Read(filepath.Join("testdata", "jfk.wav"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := m.Transcribe(samples)
	if err != nil {
		t.Fatal(err)
	}
	const want = "And so, my fellow Americans, ask not what your country can do " +
		"for you, ask what you can do for your country."
	if result.Text != want {
		t.Fatalf("text = %q, want %q", result.Text, want)
	}

	// Decoding state must not leak from one call to the next.
	again, err := m.Transcribe(samples)
	if err != nil {
		t.Fatal(err)
	}
	if again.Text != want {
		t.Fatalf("second call text = %q, want %q", again.Text, want)
	}

	// A very short recording must not fail or panic.
	short, err := m.Transcribe(make([]float32, 1000))
	if err != nil {
		t.Fatal(err)
	}
	if short.Text != "" {
		t.Fatalf("short audio text = %q, want empty", short.Text)
	}
}

// Transcribing audio longer than the encoder frame count exercises chunking
// and boundary deduplication.
// Inference on the CPU device is not reproducible bit for bit across processes,
// and greedy decoding turns a nudge into a different word, so the tests over
// audio longer than one encoder window assert that the whole capture came back
// rather than an exact sentence. The bug they hold shut is a window of audio
// whose words went missing; exact text is pinned by the tokenizer tests and by
// the single window transcriptions above.
func TestTranscribeLongAudio(t *testing.T) {
	m := newTestModel(t)
	samples, err := audio.Read(filepath.Join("testdata", "first_15s.wav"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := m.Transcribe(samples)
	if err != nil {
		t.Fatal(err)
	}
	if words := len(strings.Fields(result.Text)); words < 20 {
		t.Errorf("%d words from 15 seconds of speech, want most of 25: %q",
			words, result.Text)
	}
	// The model writes "aged" + "2" + "3" with no token in between, so this also
	// pins the spaces the decoder puts in front of a number.
	if !strings.Contains(result.Text, "aged 23 to 33") {
		t.Errorf("text = %q, want the numbers spaced out", result.Text)
	}
}

// Repeating the same audio exercises boundary deduplication over many chunks.
func TestTranscribeRepeatedAudio(t *testing.T) {
	m := newTestModel(t)
	samples, err := audio.Read(filepath.Join("testdata", "first_15s.wav"))
	if err != nil {
		t.Fatal(err)
	}
	long := make([]float32, 0, len(samples)*4)
	for range 4 {
		long = append(long, samples...)
	}
	result, err := m.Transcribe(long)
	if err != nil {
		t.Fatal(err)
	}
	// Four repetitions of the same 25 words. Stitching is allowed to merge a
	// repeat it cannot tell apart from a duplicate, which is what the boundary
	// deduplication is for, but it may not go quiet over a window of audio.
	const opener = "Previously on Bearbrock"
	repeats := strings.Count(result.Text, opener)
	if repeats < 3 {
		t.Errorf("%q opens with %q %d times, want at least 3 of 4",
			result.Text, opener, repeats)
	}
	if words := len(strings.Fields(result.Text)); words < 60 {
		t.Errorf("%d words from 60 seconds of speech, want most of 100: %q",
			words, result.Text)
	}
}

// patchedModelDir links the model files into a temp dir, replacing one of
// them with patched contents. The patch returns nil when the file does not
// have the expected contents, which skips the test.
func patchedModelDir(
	t *testing.T, replace string, patch func([]byte) []byte,
) string {
	t.Helper()
	src := modelDir(t)
	dir := t.TempDir()
	for _, name := range []string{
		"parakeet_melspectogram.xml", "parakeet_melspectogram.bin",
		"parakeet_encoder.xml", "parakeet_encoder.bin",
		"parakeet_decoder.xml", "parakeet_decoder.bin",
		"parakeet_joint.xml", "parakeet_joint.bin",
		"parakeet_vocab.json",
	} {
		if name == replace {
			data, err := os.ReadFile(filepath.Join(src, name))
			if err != nil {
				t.Fatal(err)
			}
			if data = patch(data); data == nil {
				t.Skipf("%s does not have the expected contents", name)
			}
			err = os.WriteFile(filepath.Join(dir, name), data, 0o644)
			if err != nil {
				t.Fatal(err)
			}
			continue
		}
		err := os.Symlink(filepath.Join(src, name), filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// TestDynamicWindow makes the mel spectrogram input dynamic, the way the v2
// export declares it, and checks the pipeline handles it. Dynamic models
// cannot be compiled for the NPU, which requires static shapes, so this path
// is CPU only.
func TestDynamicWindow(t *testing.T) {
	dir := patchedModelDir(t, "parakeet_melspectogram.xml", func(data []byte) []byte {
		patched := bytes.Replace(data, []byte(`shape="1,240000"`), []byte(`shape="?,?"`), 1)
		if bytes.Equal(patched, data) {
			return nil
		}
		return patched
	})
	m, err := New(Config{Dir: dir, Device: "CPU"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.preprocWindow != 0 {
		t.Fatalf("preprocWindow = %d, want a dynamic window", m.preprocWindow)
	}
	samples, err := audio.Read(filepath.Join("testdata", "jfk.wav"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := m.Transcribe(samples)
	if err != nil {
		t.Fatal(err)
	}
	const want = "And so, my fellow Americans, ask not what your country can do " +
		"for you, ask what you can do for your country."
	if result.Text != want {
		t.Fatalf("text = %q, want %q", result.Text, want)
	}
}

// jointLogits runs the model's own joint network on fixed inputs.
func jointLogits(t *testing.T, m *Model, enc, dec []float32) []float32 {
	t.Helper()
	return jointLogitsReq(t, m, m.joint.req, enc, dec)
}

// jointLogitsReq runs an arbitrary joint inference request, which is how the
// logits of a model compiled from a patched IR are compared with the shipped
// one.
func jointLogitsReq(
	t *testing.T, m *Model, req *openvino.Request, enc, dec []float32,
) []float32 {
	t.Helper()
	encT, encD, err := openvino.NewF32Tensor([]int64{1, 1, int64(m.encoderHidden)})
	if err != nil {
		t.Fatal(err)
	}
	defer encT.Close()
	decT, decD, err := openvino.NewF32Tensor([]int64{1, 1, int64(m.decoderHidden)})
	if err != nil {
		t.Fatal(err)
	}
	defer decT.Close()
	copy(encD, enc)
	copy(decD, dec)
	if err := req.Set("encoder_outputs", encT); err != nil {
		t.Fatal(err)
	}
	if err := req.Set("decoder_outputs", decT); err != nil {
		t.Fatal(err)
	}
	if err := req.Infer(); err != nil {
		t.Fatal(err)
	}
	out, err := req.Get("logits")
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	data, err := out.F32()
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), data...)
}

// TestJointSoftmaxAxis checks that the joint network's LogSoftmax axis can be
// written as a positive number and still compute the same logits, which is the
// rewrite the NPU needs. It goes through compileWithNormalizedAxis, the path an
// NPU compile takes, so the patched IR, the explicit weights path and the
// model-object compile are all exercised on a CPU that can run both.
func TestJointSoftmaxAxis(t *testing.T) {
	base := newTestModel(t)
	data, err := os.ReadFile(base.joint.path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(data, []byte(`axis="-1"`)) {
		t.Skip("the joint network does not carry a negative axis")
	}
	core, err := openvino.SharedCore()
	if err != nil {
		t.Fatal(err)
	}
	patched, err := compileWithNormalizedAxis(core, base.joint.path, "CPU", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer patched.Close()
	req, err := patched.Request()
	if err != nil {
		t.Fatal(err)
	}
	defer req.Close()

	enc := make([]float32, base.encoderHidden)
	dec := make([]float32, base.decoderHidden)
	for i := range enc {
		enc[i] = float32(math.Sin(float64(i))) * 0.5
	}
	for i := range dec {
		dec[i] = float32(math.Cos(float64(i))) * 0.5
	}
	want := jointLogits(t, base, enc, dec)
	got := jointLogitsReq(t, base, req, enc, dec)
	if len(got) != len(want) {
		t.Fatalf("got %d logits, want %d", len(got), len(want))
	}
	if argmax(got) != argmax(want) {
		t.Fatalf("argmax %d, want %d", argmax(got), argmax(want))
	}
	var maxDiff float32
	for i := range want {
		if d := float32(math.Abs(float64(got[i] - want[i]))); d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-5 {
		t.Fatalf("max absolute logit difference %g", maxDiff)
	}
	t.Logf("%d logits, max absolute difference %g", len(got), maxDiff)
}

// TestProperties checks that compile properties reach the plugin, and that a
// property the resolved device rejects does not stop the model from loading:
// the CPU fallback drops them.
func TestProperties(t *testing.T) {
	dir := modelDir(t)
	samples, err := audio.Read(filepath.Join("testdata", "jfk.wav"))
	if err != nil {
		t.Fatal(err)
	}
	const want = "And so, my fellow Americans, ask not what your country can do " +
		"for you, ask what you can do for your country."

	cache := t.TempDir()
	m, err := New(Config{
		Dir:        dir,
		Device:     "CPU",
		Properties: map[string]string{"CACHE_DIR": cache},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	result, err := m.Transcribe(samples)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != want {
		t.Fatalf("text = %q, want %q", result.Text, want)
	}
	entries, err := os.ReadDir(cache)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) == 0 {
		t.Fatal("CACHE_DIR is still empty, the property did not reach the plugin")
	}
	t.Logf("cache entries: %d", len(entries))

	// This combination only works for some devices, so it must fall back.
	m2, err := New(Config{
		Dir:        dir,
		Device:     "AUTO",
		Properties: map[string]string{"NPU_COMPILER_TYPE": "PLUGIN"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer m2.Close()
	result, err = m2.Transcribe(samples)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != want {
		t.Fatalf("fallback text = %q, want %q", result.Text, want)
	}
}

func TestCleanError(t *testing.T) {
	err := serr.Errorf("compile x.xml: Exception from src/a.cpp:1:\n" +
		"Exception from src/b/c.cpp:22:\nNotFound: no such property\n")
	want := "compile x.xml: NotFound: no such property"
	if got := cleanError(err); got != want {
		t.Fatalf("cleanError = %q, want %q", got, want)
	}
	plain := serr.Errorf("something went wrong")
	if got := cleanError(plain); got != "something went wrong" {
		t.Fatalf("cleanError = %q", got)
	}
}

// A token the loop emitted may predict a duration of zero, which says the next
// token belongs at the same encoder frame: forcing it forward skipped a token
// the model meant to place there, and the model's own decoder does not do it.
// A blank or a control token is not emitted and may not predict zero, or the
// frame pointer would never move.
func TestTDTAdvance(t *testing.T) {
	cases := []struct {
		emitted  bool
		duration int
		want     int
	}{
		{true, 0, 0},
		{true, 1, 1},
		{true, 4, 4},
		{false, 0, 1},
		{false, 1, 1},
		{false, 4, 4},
	}
	for _, c := range cases {
		if got := tdtAdvance(c.emitted, c.duration); got != c.want {
			t.Errorf("tdtAdvance(emitted=%v, duration=%d) = %d, want %d",
				c.emitted, c.duration, got, c.want)
		}
	}
}

// The frame pointer, not the token count, is what a zero duration would let
// stall; the per-step cap forces the loop on once a frame has said its fill,
// which is what the TDT decoders NeMo ships do.
func TestTDTStep(t *testing.T) {
	cases := []struct {
		name              string
		emitted           bool
		duration, symbols int
		maxSymbols        int
		wantStep          int
		wantSymbols       int
	}{
		{"first zero duration stays", true, 0, 0, 10, 0, 1},
		{"under the cap stays", true, 0, 8, 10, 0, 9},
		{"at the cap moves on", true, 0, 9, 10, 1, 0},
		{"a duration under the cap is kept", true, 1, 9, 10, 1, 0},
		{"a blank always moves", false, 0, 3, 10, 1, 0},
		{"a longer duration is kept", false, 4, 0, 10, 4, 0},
		{"a cap of zero disables it", true, 0, 99, 0, 0, 100},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			step, symbols := tdtStep(c.emitted, c.duration, c.symbols, c.maxSymbols)
			if step != c.wantStep || symbols != c.wantSymbols {
				t.Errorf("tdtStep = %d, %d, want %d, %d",
					step, symbols, c.wantStep, c.wantSymbols)
			}
		})
	}
}

// The GPU encoder is fine at the plugin's default precision, but the small
// decoder and joint networks are not: they get the accuracy execution mode,
// unless the caller already chose a precision or an execution mode.
func TestAccuracyProps(t *testing.T) {
	for _, device := range []string{"GPU", "gpu", "GPU.0", "AUTO", "auto"} {
		got := accuracyProps(device, nil)
		if got["EXECUTION_MODE_HINT"] != "ACCURACY" {
			t.Errorf("%s: props %v, want the accuracy hint", device, got)
		}
	}
	for _, device := range []string{"CPU", "NPU", ""} {
		if got := accuracyProps(device, nil); got != nil {
			t.Errorf("%s: props %v, want them unchanged", device, got)
		}
	}
	for _, key := range []string{"EXECUTION_MODE_HINT", "INFERENCE_PRECISION_HINT"} {
		got := accuracyProps("GPU", map[string]string{key: "given"})
		if len(got) != 1 || got[key] != "given" {
			t.Errorf("%s set: props %v, want the caller's value kept", key, got)
		}
	}
	got := accuracyProps("GPU", map[string]string{"CACHE_DIR": "/tmp/x"})
	if got["CACHE_DIR"] != "/tmp/x" || got["EXECUTION_MODE_HINT"] != "ACCURACY" {
		t.Errorf("props %v, want the cache dir kept and the accuracy hint added", got)
	}
}

// The joint network lays its token head and its duration head out one way
// around; a model exported the other way needs DurationsFirst, or the loop
// reads a duration bin as a token and a token as a duration.
func TestHeadOffsets(t *testing.T) {
	tokens, durations := headOffsets(8192, 5, false)
	if tokens != 0 || durations != 8193 {
		t.Fatalf("tokens first: offsets %d, %d, want 0, 8193", tokens, durations)
	}
	tokens, durations = headOffsets(8192, 5, true)
	if tokens != 5 || durations != 0 {
		t.Fatalf("durations first: offsets %d, %d, want 5, 0", tokens, durations)
	}
}
