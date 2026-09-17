package silerovad

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/daaku/whispy/audio"
)

// modelPath locates the Silero VAD model, skipping the test when the machine
// does not have it.
func modelPath(t *testing.T) string {
	t.Helper()
	home, _ := os.UserHomeDir()
	for _, path := range []string{
		os.Getenv("SILERO_VAD_MODEL"),
		filepath.Join(home, ".cache/whispy/silero_vad.onnx"),
		filepath.Join(home, ".cache/whispy/silero_vad.xml"),
	} {
		if path == "" {
			continue
		}
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	t.Skip("silero vad model not found, set SILERO_VAD_MODEL")
	return ""
}

func newTestVad(t *testing.T) *Vad {
	t.Helper()
	v, err := New(Config{Model: modelPath(t)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { v.Close() })
	return v
}

func TestSpeechProb(t *testing.T) {
	v := newTestVad(t)
	speech, err := audio.Read(filepath.Join("testdata", "speech.wav"))
	if err != nil {
		t.Fatal(err)
	}

	probs, err := v.SpeechProb(speech)
	if err != nil {
		t.Fatal(err)
	}
	if want := len(speech) / WindowSamples; len(probs) != want {
		t.Fatalf("got %d probabilities, want %d", len(probs), want)
	}
	if !HasSpeech(probs) {
		t.Fatalf("no speech detected in speech, probs = %v", probs)
	}

	v.Reset()
	silence := make([]float32, SampleRate)
	probs, err = v.SpeechProb(silence)
	if err != nil {
		t.Fatal(err)
	}
	if HasSpeech(probs) {
		t.Fatalf("speech detected in silence, probs = %v", probs)
	}
}

// TestStreaming checks that the buffering and carried state do not change the
// result compared to feeding the whole stream at once.
func TestStreaming(t *testing.T) {
	speech, err := audio.Read(filepath.Join("testdata", "speech.wav"))
	if err != nil {
		t.Fatal(err)
	}

	whole := newTestVad(t)
	want, err := whole.SpeechProb(speech)
	if err != nil {
		t.Fatal(err)
	}

	chunked := newTestVad(t)
	var got []float32
	// Odd sized chunks exercise the leftover sample buffering.
	for off := 0; off < len(speech); off += SampleRate/3 + 7 {
		probs, err := chunked.SpeechProb(speech[off:min(off+SampleRate/3+7, len(speech))])
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
	v := newTestVad(t)
	speech, err := audio.Read(filepath.Join("testdata", "speech.wav"))
	if err != nil {
		t.Fatal(err)
	}
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
	v := newTestVad(t)
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

// TestDeviceFallback asks for a device that usually cannot run the model. The
// VAD must load and work either way, falling back to the CPU when needed.
func TestDeviceFallback(t *testing.T) {
	v, err := New(Config{Model: modelPath(t), Device: "NPU"})
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	speech, err := audio.Read(filepath.Join("testdata", "speech.wav"))
	if err != nil {
		t.Fatal(err)
	}
	probs, err := v.SpeechProb(speech)
	if err != nil {
		t.Fatal(err)
	}
	if !HasSpeech(probs) {
		t.Fatalf("no speech detected, probs = %v", probs)
	}
}
