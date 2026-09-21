package silero

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/daaku/whispy/audio"
)

// modelPath locates the Silero VAD model, skipping the test when the machine
// does not have it.
func modelPath(tb testing.TB) string {
	tb.Helper()
	home, _ := os.UserHomeDir()
	for _, path := range []string{
		os.Getenv("SILERO_VAD_MODEL"),
		filepath.Join(home, ".cache/whispy/silero_vad.onnx"),
	} {
		if path == "" {
			continue
		}
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	tb.Skip("silero vad model not found, set SILERO_VAD_MODEL")
	return ""
}

func newVad(tb testing.TB) *Vad {
	tb.Helper()
	v, err := New(Config{Model: modelPath(tb)})
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { v.Close() })
	return v
}

func speechSamples(tb testing.TB) []float32 {
	tb.Helper()
	samples, err := audio.Read(filepath.Join("testdata", "speech.wav"))
	if err != nil {
		tb.Fatal(err)
	}
	return samples
}

// goldenProbs is what OpenVINO produced for the test clip, recorded when this
// implementation was checked against it. The two agreed to about 1e-6, which
// is the guard the vector kernel and any later refactor of the forward pass
// are held to now that the OpenVINO engine is gone.
func goldenProbs(tb testing.TB) []float32 {
	tb.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "speech.txt"))
	if err != nil {
		tb.Fatal(err)
	}
	var probs []float32
	for _, line := range strings.Fields(string(data)) {
		v, err := strconv.ParseFloat(line, 32)
		if err != nil {
			tb.Fatal(err)
		}
		probs = append(probs, float32(v))
	}
	return probs
}

func TestSpeechProb(t *testing.T) {
	v := newVad(t)
	probs, err := v.SpeechProb(speechSamples(t))
	if err != nil {
		t.Fatal(err)
	}
	want := goldenProbs(t)
	if len(probs) != len(want) {
		t.Fatalf("got %d probabilities, want %d", len(probs), len(want))
	}
	if !HasSpeech(probs) {
		t.Fatal("no speech detected in speech")
	}
	for i := range want {
		if diff := math.Abs(float64(probs[i] - want[i])); diff > 1e-4 {
			t.Fatalf("window %d = %v, want %v (diff %v)", i, probs[i], want[i], diff)
		}
	}
}

// TestSilence checks the other side of the threshold.
func TestSilence(t *testing.T) {
	v := newVad(t)
	probs, err := v.SpeechProb(make([]float32, SampleRate))
	if err != nil {
		t.Fatal(err)
	}
	if len(probs) != SampleRate/WindowSamples {
		t.Fatalf("got %d probabilities, want %d", len(probs), SampleRate/WindowSamples)
	}
	if HasSpeech(probs) {
		t.Fatalf("speech detected in silence, probs = %v", probs)
	}
}

// TestStreaming checks that the buffering and carried state do not change the
// result compared to feeding the whole stream at once.
func TestStreaming(t *testing.T) {
	speech := speechSamples(t)
	want, err := newVad(t).SpeechProb(speech)
	if err != nil {
		t.Fatal(err)
	}

	v := newVad(t)
	var got []float32
	// Odd sized chunks exercise the leftover sample buffering.
	for off := 0; off < len(speech); off += SampleRate/3 + 7 {
		probs, err := v.SpeechProb(speech[off:min(off+SampleRate/3+7, len(speech))])
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, probs...)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d probabilities, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("probability %d = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestReset(t *testing.T) {
	v := newVad(t)
	speech := speechSamples(t)
	if probs, err := v.SpeechProb(speech); err != nil {
		t.Fatal(err)
	} else if !HasSpeech(probs) {
		t.Fatal("no speech detected in speech")
	}
	v.Reset()
	// After a reset the buffered samples are gone too, so a partial window
	// produces no probabilities at all.
	probs, err := v.SpeechProb(speech[:WindowSamples-1])
	if err != nil {
		t.Fatal(err)
	}
	if len(probs) != 0 {
		t.Fatalf("got %d probabilities for a partial window, want 0", len(probs))
	}
}

func TestClosed(t *testing.T) {
	v := newVad(t)
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := v.SpeechProb(make([]float32, SampleRate)); err == nil {
		t.Fatal("expected an error from a closed VAD")
	}
	if err := v.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
}

// TestFFTMagnitude checks the FFT against a naive DFT of the same windowed
// frames, which is the part of the model the golden probabilities would
// otherwise be the only guard for.
func TestFFTMagnitude(t *testing.T) {
	f := newFFT256()
	frames := 3
	src := make([]float32, (frames-1)*hopSamples+fftSize)
	for i := range src {
		src[i] = float32(math.Sin(float64(i)*0.37) + 0.3*math.Cos(float64(i)*0.11))
	}
	got := make([]float32, binCount)
	for frame := 0; frame < frames; frame++ {
		off := frame * hopSamples
		f.magnitude(src, off, got)
		for bin := 0; bin < binCount; bin++ {
			var re, im float64
			for i := 0; i < fftSize; i++ {
				angle := 2 * math.Pi * float64(bin*i) / fftSize
				x := float64(src[off+i]) * float64(f.win[i])
				re += x * math.Cos(angle)
				im -= x * math.Sin(angle)
			}
			want := math.Hypot(re, im)
			if diff := math.Abs(float64(got[bin]) - want); diff > 1e-3 {
				t.Fatalf("frame %d bin %d = %v, want %v", frame, bin, got[bin], want)
			}
		}
	}
}

// BenchmarkSpeechProb runs the whole test clip through the model. The
// sub-benchmark name says which matrix kernel the build picked up: run it once
// with and once without GOEXPERIMENT=simd.
func BenchmarkSpeechProb(b *testing.B) {
	v, err := New(Config{Model: modelPath(b)})
	if err != nil {
		b.Fatal(err)
	}
	defer v.Close()
	speech := speechSamples(b)
	b.Run(Implementation(), func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(speech) * 4))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := v.SpeechProb(speech); err != nil {
				b.Fatal(err)
			}
			v.Reset()
		}
		audioSeconds := float64(len(speech)) / SampleRate * float64(b.N)
		b.ReportMetric(audioSeconds/b.Elapsed().Seconds(), "x-realtime")
	})
}
