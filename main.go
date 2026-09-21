package main

import (
	"context"
	"encoding/binary"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/daaku/serr"
	"github.com/daaku/whispy/audio"
	"github.com/daaku/whispy/parakeet"
	"github.com/daaku/whispy/silerovad"
	"github.com/daaku/whispy/timetext"
	"github.com/daaku/words2num"
	"github.com/joshuarubin/go-sway"
)

// compileProperties parses KEY=VALUE pairs separated by commas. A leading ~ is
// expanded, because sway runs its exec lines with a shell that leaves it alone.
func compileProperties(spec string) map[string]string {
	home, _ := os.UserHomeDir()
	props := map[string]string{}
	for _, pair := range strings.Split(spec, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		key, value, ok := strings.Cut(pair, "=")
		if !ok {
			println("ignoring property without a value:", pair)
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if value == "~" || strings.HasPrefix(value, "~/") {
			value = home + strings.TrimPrefix(value, "~")
		}
		props[key] = value
	}
	return props
}

func bytesIntoF32(b []byte, floats []float32) []float32 {
	if len(b)%4 != 0 {
		panic("length not multiple of 4")
	}
	floats = floats[0 : len(b)/4]
	for i := range floats {
		bits := binary.LittleEndian.Uint32(b[i*4 : (i+1)*4])
		floats[i] = math.Float32frombits(bits)
	}
	return floats
}

// textReplacer is what the transcript rewriters have in common: strings.Replacer
// and words2num.Words2Num both replace text in place and leave it alone when
// there is nothing to do.
type textReplacer interface {
	Replace(string) string
}

// applyReplacers rewrites text with each replacer in order.
func applyReplacers(text string, replacers []textReplacer) string {
	for _, r := range replacers {
		text = r.Replace(text)
	}
	return text
}

// transcribeFile transcribes one audio file and prints the text.
func transcribeFile(
	m *parakeet.Model,
	replacers []textReplacer,
	path string,
	printTime bool,
) error {
	samples, err := audio.Read(path)
	if err != nil {
		return serr.Wrap(err)
	}
	// The first call pays for lazily initialized kernels, buffers and threads,
	// so run it once to warm up and time the second one.
	if _, err := m.Transcribe(samples); err != nil {
		return serr.Wrap(err)
	}
	start := time.Now()
	result, err := m.Transcribe(samples)
	if err != nil {
		return serr.Wrap(err)
	}
	if printTime {
		println("Took", time.Since(start).Truncate(time.Millisecond).String(),
			"for", len(samples)/audio.SampleRate, "seconds of audio")
	}
	println(applyReplacers(strings.TrimSpace(result.Text), replacers))
	return nil
}

func run(ctx context.Context) error {
	home, _ := os.UserHomeDir()
	printText := flag.Bool("print-text", false, "print the transcribed text")
	printTime := flag.Bool("print-time", false, "print the transcription duration")
	keepAudio := flag.Bool("keep-audio", false, "save the captured audio to /tmp/a.au")
	replacerPath := flag.String("replacer", filepath.Join(home, ".config/whispy/replacer.csv"), "path to a replacements CSV")
	modelDir := flag.String("model-dir", filepath.Join(home, ".cache/whispy/parakeet-v3"), "directory with the parakeet OpenVINO IR files")
	device := flag.String("device", "CPU", "OpenVINO device for the parakeet encoder, decoder and joint network (CPU, GPU, NPU or AUTO)")
	properties := flag.String("properties", "", "extra OpenVINO compile properties as KEY=VALUE pairs, e.g. CACHE_DIR=~/.cache/whispy/openvino")
	preprocDevice := flag.String("preproc-device", "CPU", "OpenVINO device for the mel spectrogram model, on the CPU by default")
	decoderDevice := flag.String("decoder-device", "", "OpenVINO device for the decoder and joint networks, which run per token (defaults to -device)")
	vadPath := flag.String("vad", filepath.Join(home, ".cache/whispy/silero_vad.onnx"), "path to the silero vad model (onnx or openvino ir)")
	transcribePath := flag.String("transcribe", "", "transcribe a 16 kHz mono WAV or AU file and exit")
	flag.Parse()

	replacer, err := loadReplacer(*replacerPath)
	if err != nil {
		return err
	}
	replacers := []textReplacer{replacer, words2num.Words2Num{}, timetext.Time{}}

	parakeetModel, err := parakeet.New(parakeet.Config{
		Dir:           *modelDir,
		Device:        *device,
		PreprocDevice: *preprocDevice,
		DecoderDevice: *decoderDevice,
		Properties:    compileProperties(*properties),
	})
	if err != nil {
		return serr.Wrap(err)
	}
	defer parakeetModel.Close()

	// Transcribing a file needs no VAD and no sway session, which makes it
	// the way to compare devices or check a model installation.
	if *transcribePath != "" {
		return transcribeFile(parakeetModel, replacers, *transcribePath, *printTime)
	}

	vad, err := silerovad.New(silerovad.Config{Model: *vadPath})
	if err != nil {
		return serr.Wrap(err)
	}
	defer vad.Close()

	sigs := make(chan os.Signal, 10)
	signal.Notify(sigs, syscall.SIGUSR1, syscall.SIGUSR2)

	swayClient, err := sway.New(ctx)
	if err != nil {
		return serr.Wrap(err)
	}

	const tmpFile = "/tmp/a.au"
	var pwRecordCmd *exec.Cmd
	auHeader := make([]byte, 24) // AU header in case we keep audio
	var rawPCM []float32
	var pipeR *io.PipeReader
	var pipeW *io.PipeWriter
	var pipeWG sync.WaitGroup
	for sig := range sigs {
		searchMode := sig == syscall.SIGUSR1

		if pwRecordCmd == nil {
			pwRecordCmd = exec.Command("pw-record", "--format=f32", "--rate=16000", "--channels=1", "-")
			pipeR, pipeW = io.Pipe()
			pwRecordCmd.Stdout = pipeW
			rawPCM = rawPCM[0:0]
			vad.Reset()
			pipeWG.Go(func() {
				if _, err := io.ReadFull(pipeR, auHeader[0:]); err != nil {
					panic(err)
				}

				const chunkSize = 4 * 16000 * 1 // f32 sized, 16000 rate, 1 second
				var bytesChunk [chunkSize]byte
				floatChunk := make([]float32, chunkSize/4)
				speechStarted := false
				sigSent := false
				for {
					n, err := io.ReadFull(pipeR, bytesChunk[:])
					floatChunk = floatChunk[0:0]
					switch err {
					case nil:
						floatChunk = bytesIntoF32(bytesChunk[:], floatChunk)
					case io.EOF, io.ErrUnexpectedEOF:
						if n > 0 {
							floatChunk = bytesIntoF32(bytesChunk[:n], floatChunk)
						}
					default:
						panic(err.Error())
					}

					rawPCM = append(rawPCM, floatChunk...)

					// detect silence if in searchMode to automatically end
					if searchMode {
						probs, err := vad.SpeechProb(floatChunk)
						if err != nil {
							panic(err)
						}
						if !silerovad.HasSpeech(probs) {
							// speech had started, and has now ended
							if speechStarted && !sigSent {
								sigs <- syscall.SIGUSR1
								sigSent = true
							}
						} else {
							speechStarted = true
						}
					}

					if err != nil {
						return
					}
				}
			})
			if err := pwRecordCmd.Start(); err != nil {
				return serr.Wrap(err)
			}
		} else {
			if err := pwRecordCmd.Process.Signal(syscall.SIGTERM); err != nil {
				return serr.Wrap(err)
			}
			pwRecordCmd.Wait()
			pwRecordCmd = nil
			pipeW.Close()
			pipeWG.Wait()

			start := time.Now()
			var text string
			if len(rawPCM) > 0 {
				result, err := parakeetModel.Transcribe(rawPCM)
				if err != nil {
					return serr.Wrap(err)
				}
				text = strings.TrimSpace(result.Text)
			}
			text = applyReplacers(text, replacers)

			if *printTime {
				println("Took", time.Since(start).Truncate(time.Millisecond).String())
			}
			if *printText {
				println(text)
			}
			if *keepAudio {
				f, err := os.Create(tmpFile)
				if err != nil {
					return serr.Wrap(err)
				}
				// write the AU header to make it a proper AU file, this is what we discarded above
				if _, err := f.Write(auHeader[:]); err != nil {
					return serr.Wrap(err)
				}
				if err := binary.Write(f, binary.LittleEndian, rawPCM); err != nil {
					return serr.Wrap(err)
				}
			}

			if searchMode {
				u := "https://duckduckgo.com/?q=" + url.QueryEscape(casualText(text))
				if err := exec.Command("xdg-open", u).Run(); err != nil {
					return serr.Wrap(err)
				}
			} else {
				tree, err := swayClient.GetTree(ctx)
				if err != nil {
					return serr.Wrap(err)
				}
				focusedNode := tree.FocusedNode()
				if strings.Contains(focusedNode.Name, "WhatsApp") {
					text = casualText(text)
				}

				if pasteMode(focusedNode) {
					wlCopyCmd := exec.Command("wl-copy", "--foreground", text)
					if err := wlCopyCmd.Start(); err != nil {
						return serr.Wrap(err)
					}
					if err := exec.Command("wtype", "-M", "ctrl", "-s", "20", "v", "-s", "20", "-m", "ctrl").Run(); err != nil {
						return serr.Wrap(err)
					}
					wlCopyCmd.Process.Kill()
					wlCopyCmd.Wait()
				} else {
					if err := exec.Command("wtype", text).Run(); err != nil {
						return serr.Wrap(err)
					}
				}
			}
		}
	}
	return nil
}

func pasteMode(n *sway.Node) bool {
	appID := *n.AppID
	if !strings.HasPrefix(appID, "firefox") &&
		!strings.HasPrefix(appID, "chromium") &&
		!strings.HasPrefix(appID, "brave") {
		return false
	}
	return true
}

func loadReplacer(path string) (*strings.Replacer, error) {
	if path == "" {
		return strings.NewReplacer(), nil
	}
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		// The replacements file is optional.
		return strings.NewReplacer(), nil
	}
	if err != nil {
		return nil, serr.Errorf("open replacements csv %q: %w", path, err)
	}
	defer f.Close()

	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, serr.Errorf("parsing replacements csv %q: %w", path, err)
	}

	var pairs []string
	for i, r := range rows {
		if len(r) != 2 {
			return nil, serr.Errorf("invalid row %d with %v in csv %q", i, r, path)
		}
		pairs = append(pairs, r[0], r[1])
	}
	return strings.NewReplacer(pairs...), nil
}

func casualText(text string) string {
	return strings.TrimSuffix(text, ".")
}

func main() {
	if err := run(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "%+v\n", err)
		os.Exit(1)
	}
}
