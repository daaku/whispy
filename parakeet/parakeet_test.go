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
	const want = "Previously on Bearbrock. Here lies the mortal remains known " +
		"only to God of a woman aged23 to33 and a girl trying to be ask you."
	if result.Text != want {
		t.Fatalf("text = %q, want %q", result.Text, want)
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
	const want = "Previously on Bearbrock. Here lies the mortal remains known " +
		"only to God of a woman aged23 to33 and a girl child. Here lies the " +
		"mortal remains known only to God of a woman aged23 to33."
	if result.Text != want {
		t.Fatalf("text = %q, want %q", result.Text, want)
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

// jointLogits runs the joint network on fixed inputs.
func jointLogits(t *testing.T, m *Model, enc, dec []float32) []float32 {
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
	if err := m.joint.req.Set("encoder_outputs", encT); err != nil {
		t.Fatal(err)
	}
	if err := m.joint.req.Set("decoder_outputs", decT); err != nil {
		t.Fatal(err)
	}
	if err := m.joint.req.Infer(); err != nil {
		t.Fatal(err)
	}
	out, err := m.joint.req.Get("logits")
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
// written as a positive number. The vpux compiler used by the NPU rejects
// axis="-1" ("Got negative index -1 for Dim"), and rewriting it to the
// equivalent axis="3" must not change the logits.
func TestJointSoftmaxAxis(t *testing.T) {
	dir := patchedModelDir(t, "parakeet_joint.xml", func(data []byte) []byte {
		patched := bytes.Replace(data, []byte(`axis="-1"`), []byte(`axis="3"`), 1)
		if bytes.Equal(patched, data) {
			return nil
		}
		return patched
	})
	base := newTestModel(t)
	patched, err := New(Config{Dir: dir, Device: "CPU"})
	if err != nil {
		t.Fatal(err)
	}
	defer patched.Close()

	enc := make([]float32, base.encoderHidden)
	dec := make([]float32, base.decoderHidden)
	for i := range enc {
		enc[i] = float32(math.Sin(float64(i))) * 0.5
	}
	for i := range dec {
		dec[i] = float32(math.Cos(float64(i))) * 0.5
	}
	want := jointLogits(t, base, enc, dec)
	got := jointLogits(t, patched, enc, dec)
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

func TestNegativeAxisHint(t *testing.T) {
	dir := t.TempDir()
	withAxis := filepath.Join(dir, "with.xml")
	if err := os.WriteFile(withAxis, []byte(`<data axis="-1" />`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := negativeAxisHint(withAxis); !strings.Contains(got, "readme") {
		t.Fatalf("hint = %q, want a pointer at the readme", got)
	}
	without := filepath.Join(dir, "without.xml")
	if err := os.WriteFile(without, []byte(`<data axis="3" />`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := negativeAxisHint(without); got != "" {
		t.Fatalf("hint = %q, want none", got)
	}
	if got := negativeAxisHint(filepath.Join(dir, "missing.xml")); got != "" {
		t.Fatalf("hint = %q, want none for a missing file", got)
	}
}
