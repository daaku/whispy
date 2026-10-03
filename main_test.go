package main

import (
	"encoding/binary"
	"math"
	"syscall"
	"testing"

	"github.com/daaku/whispy/audio"
)

func TestEventOf(t *testing.T) {
	cases := []struct {
		sig  syscall.Signal
		want captureEvent
	}{
		{syscall.SIGUSR1, eventCommand},
		{syscall.SIGUSR2, eventToggle},
	}
	for _, c := range cases {
		if got := eventOf(c.sig); got != c.want {
			t.Errorf("eventOf(%v) = %d, want %d", c.sig, got, c.want)
		}
	}
}

func TestCaptureNext(t *testing.T) {
	cases := []struct {
		name        string
		capturing   bool
		event       captureEvent
		start, stop bool
	}{
		// Idle: either key starts a capture of its own kind.
		{"idle command", false, eventCommand, true, false},
		{"idle toggle", false, eventToggle, true, false},
		// A silence of its own is not a reason to record, which is what a
		// leftover from a capture that already stopped would be.
		{"idle silence", false, eventSilence, false, false},
		// Running: the key that started a command capture does nothing, since
		// silence ends it. The toggle and silence do end it.
		{"capturing command", true, eventCommand, false, false},
		{"capturing toggle", true, eventToggle, false, true},
		{"capturing silence", true, eventSilence, false, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			start, stop := captureNext(c.capturing, c.event)
			if start != c.start || stop != c.stop {
				t.Errorf("captureNext(capturing=%v, event=%d) = (%v, %v), want (%v, %v)",
					c.capturing, c.event, start, stop, c.start, c.stop)
			}
		})
	}
}

// TestBytesIntoF32 checks the audio is decoded little endian, and that a read
// which stops part way through a sample is dropped rather than panicking. A
// capture can be stopped between one read and the next, which is exactly what
// used to take the process down.
func TestBytesIntoF32(t *testing.T) {
	samples := []float32{1, -0.5, 0, 32768}
	bytes := make([]byte, 4*len(samples))
	for i, f := range samples {
		binary.LittleEndian.PutUint32(bytes[i*4:], math.Float32bits(f))
	}
	cases := []struct {
		name  string
		n     int
		count int
	}{
		{"all of it", len(bytes), len(samples)},
		{"one sample", 4, 1},
		{"a sample and a half", 6, 1},
		{"three bytes of one sample", 3, 0},
		{"nothing", 0, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := bytesIntoF32(bytes[:c.n], make([]float32, 10))
			if len(got) != c.count {
				t.Fatalf("bytesIntoF32(%d bytes) returned %d samples, want %d",
					c.n, len(got), c.count)
			}
			for i := range got {
				if got[i] != samples[i] {
					t.Errorf("sample %d = %v, want %v", i, got[i], samples[i])
				}
			}
		})
	}
}

// TestSignalEnd checks the end of a capture travels on the channel and cannot
// block, since the capture loop reads it once and a reader goroutine that
// stopped holding the pipe would hang the daemon.
func TestSignalEnd(t *testing.T) {
	end := make(chan struct{}, 1)
	signalEnd(end)
	select {
	case <-end:
	default:
		t.Fatal("signalEnd did not send")
	}
	// Full, and then empty with nobody reading: both have to return.
	signalEnd(end)
	signalEnd(end)
}

// TestOverCapture is the bound on one capture: a microphone left open cannot
// grow the buffer for ever, and the audio in hand up to the bound is still
// worth transcribing.
func TestOverCapture(t *testing.T) {
	cases := []struct {
		name    string
		seconds float64
		want    bool
	}{
		{"nothing", 0, false},
		{"a sentence", 3, false},
		{"the endpointer's own bound", 30, false},
		{"exactly the bound", maxCaptureSeconds, false},
		{"a sample past it", maxCaptureSeconds + 0.0001, true},
		{"five minutes", 300, true},
	}
	for _, c := range cases {
		samples := int(c.seconds * audio.SampleRate)
		if got := overCapture(samples); got != c.want {
			t.Errorf("overCapture(%d samples, %.1fs) = %v, want %v",
				samples, c.seconds, got, c.want)
		}
	}
	if maxCaptureSeconds < 30 {
		t.Errorf("a capture bound of %d seconds is shorter than a sentence", maxCaptureSeconds)
	}
}
