package audio

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeFile writes bytes to a temp file and returns its path.
func writeFile(t *testing.T, name string, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// wav builds a WAV file with a PCM 16 bit or float 32 payload.
func wav(t *testing.T, format, bits uint16, rate uint32, channels uint16, body []byte) []byte {
	t.Helper()
	var buf []byte
	buf = append(buf, "RIFF"...)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(36+len(body)))
	buf = append(buf, "WAVEfmt "...)
	buf = binary.LittleEndian.AppendUint32(buf, 16)
	buf = binary.LittleEndian.AppendUint16(buf, format)
	buf = binary.LittleEndian.AppendUint16(buf, channels)
	buf = binary.LittleEndian.AppendUint32(buf, rate)
	buf = binary.LittleEndian.AppendUint32(buf, rate*uint32(channels)*uint32(bits)/8)
	buf = binary.LittleEndian.AppendUint16(buf, channels*bits/8)
	buf = binary.LittleEndian.AppendUint16(buf, bits)
	buf = append(buf, "data"...)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(body)))
	return append(buf, body...)
}

// put32 appends v in the byte order of the AU header.
func put32(order binary.ByteOrder, buf []byte, v uint32) []byte {
	if order == binary.LittleEndian {
		return binary.LittleEndian.AppendUint32(buf, v)
	}
	return binary.BigEndian.AppendUint32(buf, v)
}

func pcm16(samples ...int16) []byte {
	var buf []byte
	for _, s := range samples {
		buf = binary.LittleEndian.AppendUint16(buf, uint16(s))
	}
	return buf
}

func TestReadWAVPCM16(t *testing.T) {
	data := wav(t, 1, 16, SampleRate, 1, pcm16(0, 16384, -32768, 32767))
	path := writeFile(t, "test.wav", data)
	samples, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 0.5, -1, 32767.0 / 32768.0}
	if len(samples) != len(want) {
		t.Fatalf("got %d samples, want %d", len(samples), len(want))
	}
	for i := range want {
		if samples[i] != want[i] {
			t.Fatalf("sample %d = %v, want %v", i, samples[i], want[i])
		}
	}
}

func TestReadWAVFloat32(t *testing.T) {
	var body []byte
	for _, v := range []float32{0, 0.25, -0.75} {
		body = binary.LittleEndian.AppendUint32(body, math.Float32bits(v))
	}
	path := writeFile(t, "test.wav", wav(t, 3, 32, SampleRate, 1, body))
	samples, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 3 || samples[1] != 0.25 || samples[2] != -0.75 {
		t.Fatalf("got %v", samples)
	}
}

func TestReadAU(t *testing.T) {
	for _, tc := range []struct {
		name     string
		order    binary.ByteOrder
		magic    string
		encoding uint32
		body     []byte
		want     []float32
	}{
		{
			name: "pw-record little endian float", order: binary.LittleEndian,
			magic: "dns.", encoding: 6,
			body: func() []byte {
				var b []byte
				for _, v := range []float32{0, 0.5, -0.5} {
					b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
				}
				return b
			}(),
			want: []float32{0, 0.5, -0.5},
		},
		{
			name: "big endian 16 bit", order: binary.BigEndian,
			magic: ".snd", encoding: 3,
			body: func() []byte {
				var b []byte
				for _, v := range []int16{0, 16384, -16384} {
					b = binary.BigEndian.AppendUint16(b, uint16(v))
				}
				return b
			}(),
			want: []float32{0, 0.5, -0.5},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var header []byte
			header = append(header, tc.magic...)
			header = put32(tc.order, header, 24)
			header = put32(tc.order, header, uint32(len(tc.body)))
			header = put32(tc.order, header, tc.encoding)
			header = put32(tc.order, header, SampleRate)
			header = put32(tc.order, header, 1)
			path := writeFile(t, "test.au", append(header, tc.body...))
			samples, err := Read(path)
			if err != nil {
				t.Fatal(err)
			}
			if len(samples) != len(tc.want) {
				t.Fatalf("got %d samples, want %d", len(samples), len(tc.want))
			}
			for i := range tc.want {
				if samples[i] != tc.want[i] {
					t.Fatalf("sample %d = %v, want %v", i, samples[i], tc.want[i])
				}
			}
		})
	}
}

func TestReadRealWAV(t *testing.T) {
	// The fixture the parakeet tests use, a real 16 kHz mono recording.
	path := filepath.Join("..", "parakeet", "testdata", "jfk.wav")
	if _, err := os.Stat(path); err != nil {
		t.Skip("no fixture available")
	}
	samples, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 176000 {
		t.Fatalf("got %d samples, want 176000", len(samples))
	}
	var peak float32
	for _, v := range samples {
		peak = max(peak, float32(math.Abs(float64(v))))
	}
	if peak < 0.05 || peak > 1 {
		t.Fatalf("peak %v is not a plausible recording", peak)
	}
}

func TestReadErrors(t *testing.T) {
	short := writeFile(t, "short.wav", []byte("RIFF"))
	if _, err := Read(short); err == nil {
		t.Fatal("expected an error for a truncated file")
	}
	stereo := writeFile(t, "stereo.wav", wav(t, 1, 16, SampleRate, 2, pcm16(0, 0)))
	if _, err := Read(stereo); err == nil {
		t.Fatal("expected an error for stereo")
	} else if !strings.Contains(err.Error(), "ffmpeg") {
		t.Fatalf("error %q does not say how to convert", err)
	}
	rate := writeFile(t, "rate.wav", wav(t, 1, 16, 24000, 1, pcm16(0, 0)))
	if _, err := Read(rate); err == nil {
		t.Fatal("expected an error for the wrong sample rate")
	}
	depth := writeFile(t, "depth.wav", wav(t, 1, 8, SampleRate, 1, []byte{0, 1}))
	if _, err := Read(depth); err == nil {
		t.Fatal("expected an error for 8 bit samples")
	}
	if _, err := Read(filepath.Join(t.TempDir(), "missing.wav")); err == nil {
		t.Fatal("expected an error for a missing file")
	}
}
