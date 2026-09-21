package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/daaku/whispy/audio"
	"github.com/daaku/whispy/silero"
	"github.com/daaku/whispy/silerovad"
)

// sileroModelPath locates the Silero VAD model, skipping the test when the
// machine does not have it.
func sileroModelPath(tb testing.TB) string {
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

func speechSamples(tb testing.TB) []float32 {
	tb.Helper()
	samples, err := audio.Read(filepath.Join("silerovad", "testdata", "speech.wav"))
	if err != nil {
		tb.Fatal(err)
	}
	return samples
}

// TestGoParityWithOpenVINO checks that the pure Go model produces the same
// probabilities as the OpenVINO one on real audio. They are different engines
// for the same graph, so the numbers should only differ by float noise.
func TestGoParityWithOpenVINO(t *testing.T) {
	model := sileroModelPath(t)
	speech := speechSamples(t)

	ov, err := silerovad.New(silerovad.Config{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	defer ov.Close()
	goVad, err := silero.New(silero.Config{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	defer goVad.Close()

	want, err := ov.SpeechProb(speech)
	if err != nil {
		t.Fatal(err)
	}
	got, err := goVad.SpeechProb(speech)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d probabilities, want %d", len(got), len(want))
	}
	var maxDiff, at float64
	for i := range want {
		diff := float64(got[i] - want[i])
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff, at = diff, float64(want[i])
		}
	}
	if maxDiff > 1e-3 {
		t.Fatalf("max probability difference %v (at openvino %v), want < 1e-3", maxDiff, at)
	}
	t.Logf("max probability difference %v", maxDiff)
}

// TestGoFindsSpeech is the model sanity check that needs no OpenVINO.
func TestGoFindsSpeech(t *testing.T) {
	v, err := silero.New(silero.Config{Model: sileroModelPath(t)})
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	probs, err := v.SpeechProb(speechSamples(t))
	if err != nil {
		t.Fatal(err)
	}
	if !silero.HasSpeech(probs) {
		t.Fatalf("no speech detected, probs = %v", probs)
	}
}

func benchmarkVAD(b *testing.B, prob func([]float32) ([]float32, error), reset func()) {
	speech := speechSamples(b)
	b.ReportAllocs()
	b.SetBytes(int64(len(speech) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := prob(speech); err != nil {
			b.Fatal(err)
		}
		reset()
	}
	audioSeconds := float64(len(speech)) / silero.SampleRate * float64(b.N)
	b.ReportMetric(audioSeconds/b.Elapsed().Seconds(), "x-realtime")
}

// BenchmarkVADOpenVINO is the current engine: the model through OpenVINO.
func BenchmarkVADOpenVINO(b *testing.B) {
	model := sileroModelPath(b)
	v, err := silerovad.New(silerovad.Config{Model: model})
	if err != nil {
		b.Fatal(err)
	}
	defer v.Close()
	benchmarkVAD(b, v.SpeechProb, v.Reset)
}

// BenchmarkVADGo is the same graph in pure Go. Run it twice, once with
// GOEXPERIMENT=simd and once without, to see the vector kernel's share.
func BenchmarkVADGo(b *testing.B) {
	v, err := silero.New(silero.Config{Model: sileroModelPath(b)})
	if err != nil {
		b.Fatal(err)
	}
	defer v.Close()
	b.Run("impl="+silero.Implementation(), func(b *testing.B) {
		benchmarkVAD(b, v.SpeechProb, v.Reset)
	})
}
