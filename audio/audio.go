// Package audio reads the 16 kHz mono audio that whispy and its models work
// with: WAV files and the AU files written by pw-record. It also writes WAV
// files, for the audio it needs to keep.
package audio

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"

	"github.com/daaku/serr"
)

// SampleRate is the only sample rate the models accept.
const SampleRate = 16000

// Write writes samples to a 16 bit mono WAV file, which is what Write produces
// and what every audio tool reads. It is the inverse of Read for that format:
// the same scale of 32768 is used both ways, so a file written and read back
// comes within one step of the samples it started with. The file belongs to
// whoever wrote it alone: it is speech that was recorded at a key press, which
// includes whatever happened to be said near it.
func Write(path string, samples []float32) error {
	n := uint32(len(samples))
	out := make([]byte, 0, 44+2*int(n))
	out = append(out, "RIFF"...)
	out = binary.LittleEndian.AppendUint32(out, 36+2*n)
	out = append(out, "WAVEfmt "...)
	out = binary.LittleEndian.AppendUint32(out, 16) // the fmt chunk that follows
	out = binary.LittleEndian.AppendUint16(out, 1)  // uncompressed
	out = binary.LittleEndian.AppendUint16(out, 1)  // mono
	out = binary.LittleEndian.AppendUint32(out, SampleRate)
	out = binary.LittleEndian.AppendUint32(out, SampleRate*2)
	out = binary.LittleEndian.AppendUint16(out, 2) // one sample per frame
	out = binary.LittleEndian.AppendUint16(out, 16)
	out = append(out, "data"...)
	out = binary.LittleEndian.AppendUint32(out, 2*n)
	for _, v := range samples {
		out = binary.LittleEndian.AppendUint16(out, uint16(scale16(v)))
	}
	if err := os.WriteFile(path, out, 0o600); err != nil {
		return serr.Errorf("audio: %w", err)
	}
	return nil
}

// scale16 scales a sample to the integer Read divides by 32768. Audio that
// comes from the microphone can go past full scale, which is clamped rather
// than wrapped around into the other channel.
func scale16(v float32) int16 {
	v = float32(math.Round(float64(v) * 32768))
	switch {
	case v <= -32768:
		return -32768
	case v >= 32767:
		return 32767
	}
	return int16(v)
}

// Read reads a 16 kHz mono audio file into samples in [-1, 1).
func Read(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, serr.Errorf("audio: %w", err)
	}
	switch {
	case len(data) >= 12 && bytes.Equal(data[0:4], []byte("RIFF")):
		return decodeWAV(path, data)
	case len(data) >= 24 &&
		(bytes.Equal(data[0:4], []byte(".snd")) || bytes.Equal(data[0:4], []byte("dns."))):
		return decodeAU(path, data)
	}
	return nil, serr.Errorf("audio: %s: not a WAV or AU file", path)
}

// checkFormat rejects anything the models cannot take.
func checkFormat(path string, rate, channels int) error {
	if rate != SampleRate || channels != 1 {
		return serr.Errorf(
			"audio: %s: %d Hz %d channel, want %d Hz mono "+
				"(convert with: ffmpeg -i %s -ar 16000 -ac 1 out.wav)",
			path, rate, channels, SampleRate, path)
	}
	return nil
}

func decodeWAV(path string, data []byte) ([]float32, error) {
	if !bytes.Equal(data[8:12], []byte("WAVE")) {
		return nil, serr.Errorf("audio: %s: RIFF file is not WAVE", path)
	}
	var (
		format    uint16
		channels  uint16
		rate      uint32
		bits      uint16
		haveFmt   bool
		samples   []float32
		haveAudio bool
	)
	for off := 12; off+8 <= len(data); {
		id := string(data[off : off+4])
		size := int(binary.LittleEndian.Uint32(data[off+4 : off+8]))
		body := data[off+8:]
		if size > len(body) {
			size = len(body)
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, serr.Errorf("audio: %s: truncated fmt chunk", path)
			}
			format = binary.LittleEndian.Uint16(body[0:2])
			channels = binary.LittleEndian.Uint16(body[2:4])
			rate = binary.LittleEndian.Uint32(body[4:8])
			bits = binary.LittleEndian.Uint16(body[14:16])
			// WAVE_FORMAT_EXTENSIBLE keeps the real format in the first two
			// bytes of the subformat GUID.
			if format == 0xFFFE && size >= 26 {
				format = binary.LittleEndian.Uint16(body[24:26])
			}
			haveFmt = true
		case "data":
			if !haveFmt {
				return nil, serr.Errorf("audio: %s: data chunk before fmt chunk", path)
			}
			if err := checkFormat(path, int(rate), int(channels)); err != nil {
				return nil, err
			}
			var err error
			samples, err = wavSamples(path, body[:size], format, bits)
			if err != nil {
				return nil, err
			}
			haveAudio = true
		}
		off += 8 + size + (size & 1)
	}
	if !haveFmt {
		return nil, serr.Errorf("audio: %s: no fmt chunk", path)
	}
	if !haveAudio {
		return nil, serr.Errorf("audio: %s: no data chunk", path)
	}
	return samples, nil
}

func wavSamples(path string, body []byte, format, bits uint16) ([]float32, error) {
	switch {
	case format == 1 && bits == 16:
		samples := make([]float32, 0, len(body)/2)
		for i := 0; i+1 < len(body); i += 2 {
			v := int16(binary.LittleEndian.Uint16(body[i : i+2]))
			samples = append(samples, float32(v)/32768)
		}
		return samples, nil
	case format == 3 && bits == 32:
		samples := make([]float32, 0, len(body)/4)
		for i := 0; i+3 < len(body); i += 4 {
			bits := binary.LittleEndian.Uint32(body[i : i+4])
			samples = append(samples, math.Float32frombits(bits))
		}
		return samples, nil
	}
	return nil, serr.Errorf(
		"audio: %s: unsupported WAV format %d with %d bits", path, format, bits)
}

// decodeAU handles both the big endian AU header of the format and the little
// endian variant pw-record writes, where even the magic is reversed.
func decodeAU(path string, data []byte) ([]float32, error) {
	order := binary.ByteOrder(binary.BigEndian)
	if bytes.Equal(data[0:4], []byte("dns.")) {
		order = binary.LittleEndian
	}
	offset := order.Uint32(data[4:8])
	size := order.Uint32(data[8:12])
	encoding := order.Uint32(data[12:16])
	rate := order.Uint32(data[16:20])
	channels := order.Uint32(data[20:24])
	if err := checkFormat(path, int(rate), int(channels)); err != nil {
		return nil, err
	}
	if int64(offset) > int64(len(data)) {
		return nil, serr.Errorf("audio: %s: data offset %d past the end", path, offset)
	}
	body := data[offset:]
	if size > 0 && int64(size) < int64(len(body)) {
		body = body[:size]
	}
	switch encoding {
	case 6: // 32 bit float
		samples := make([]float32, 0, len(body)/4)
		for i := 0; i+3 < len(body); i += 4 {
			samples = append(samples, math.Float32frombits(order.Uint32(body[i:i+4])))
		}
		return samples, nil
	case 3: // 16 bit linear
		samples := make([]float32, 0, len(body)/2)
		for i := 0; i+1 < len(body); i += 2 {
			v := int16(order.Uint16(body[i : i+2]))
			samples = append(samples, float32(v)/32768)
		}
		return samples, nil
	}
	return nil, serr.Errorf("audio: %s: unsupported AU encoding %d", path, encoding)
}
