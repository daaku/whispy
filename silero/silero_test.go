package silero

import (
	"math"
	"testing"
)

// TestFFTMagnitude checks the FFT against a naive DFT of the same windowed
// frames, which is the part of the model the parity test with OpenVINO would
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
