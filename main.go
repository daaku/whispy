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
	"github.com/daaku/whispy/command"
	"github.com/daaku/whispy/debuglog"
	"github.com/daaku/whispy/multiplier"
	"github.com/daaku/whispy/parakeet"
	"github.com/daaku/whispy/silero"
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
	// A capture can be stopped between one read and the next, so the last read
	// can hand over a partial sample. Three spare bytes are not worth losing a
	// daemon over, and not worth keeping either: they are a quarter of one
	// sample of audio, at most.
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

// captureEvent is what the capture loop reacts to.
type captureEvent int

const (
	// eventCommand is the command mode key, SIGUSR1.
	eventCommand captureEvent = iota
	// eventToggle is the dictation toggle, SIGUSR2.
	eventToggle
	// eventSilence is the running command capture hearing speech stop.
	eventSilence
)

// eventOf maps one of the two signals whispy listens for to its event.
func eventOf(sig os.Signal) captureEvent {
	if sig == syscall.SIGUSR1 {
		return eventCommand
	}
	return eventToggle
}

// captureNext says what a capture loop does with an event: start a capture,
// stop the one that is running, or neither. A command capture is ended by
// silence rather than by the key that started it, so pressing that key again
// while one is running does nothing, and a silence left over from a capture
// that already stopped cannot start a new one.
func captureNext(capturing bool, e captureEvent) (start, stop bool) {
	switch {
	case !capturing:
		return e != eventSilence, false
	case e == eventCommand:
		return false, false
	default:
		return false, true
	}
}

// signalEnd tells a running command capture that it is over. The loop reads
// the channel only once per capture, so this never blocks: a reader goroutine
// that stopped could hold the pipe open and hang the daemon on its way out.
func signalEnd(end chan struct{}) {
	select {
	case end <- struct{}{}:
	default:
	}
}

// maxCaptureSeconds bounds one capture. The buffer for a capture grows without
// limit while the microphone is open, and a capture that runs for minutes costs
// that much audio in memory and then that much transcription time, so the
// capture ends instead. Thirty seconds is the endpointer's bound on a speaker
// who never stops; this is the bound on a capture that keeps going in pieces,
// and on a dictation capture, which the VAD does not watch at all.
const maxCaptureSeconds = 60

// maxCaptureSamples is maxCaptureSeconds of audio at the capture rate.
const maxCaptureSamples = maxCaptureSeconds * audio.SampleRate

// keepAudioPath is where -keep-audio writes the last capture. It sits with
// everything else whispy keeps in the cache directory, because what it holds is
// the audio of one capture: speech, including anything said near the key press.
// /tmp was the wrong home for it - world writable, world readable, and at a
// fixed path anyone else could have made first.
func keepAudioPath() (string, error) {
	cache, err := os.UserCacheDir()
	if err != nil {
		return "", serr.Wrap(err)
	}
	return filepath.Join(cache, "whispy", "last-capture.au"), nil
}

// keepAudio writes the audio of the capture that just ended, header and all,
// over the previous one. The file is for the person who ran whispy, so it is
// theirs alone, and so is the directory under it.
func keepAudio(path string, header []byte, samples []float32) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return serr.Wrap(err)
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return serr.Wrap(err)
	}
	defer f.Close()
	// The AU header makes it a proper AU file, which is what pw-record gave us
	// and what we discarded above to get at the samples.
	if _, err := f.Write(header); err != nil {
		return serr.Wrap(err)
	}
	if err := binary.Write(f, binary.LittleEndian, samples); err != nil {
		return serr.Wrap(err)
	}
	return f.Close()
}

// overCapture reports whether a capture has run past its bound.
func overCapture(samples int) bool { return samples > maxCaptureSamples }

func run(ctx context.Context) error {
	home, _ := os.UserHomeDir()
	printText := flag.Bool("print-text", false, "print the transcribed text")
	printTime := flag.Bool("print-time", false, "print the transcription duration")
	keepAudioFlag := flag.Bool("keep-audio", false,
		"save the captured audio to the cache directory, overwriting the last one")
	debugLog := flag.Bool("debug-log", false, "log each capture's audio and text to the cache directory, for debugging")
	replacerPath := flag.String("replacer", filepath.Join(home, ".config/whispy/replacer.csv"), "path to a replacements CSV")
	modelDir := flag.String("model-dir", filepath.Join(home, ".cache/whispy/parakeet-v3"), "directory with the parakeet OpenVINO IR files")
	device := flag.String("device", "CPU", "OpenVINO device for the parakeet encoder, decoder and joint network (CPU, GPU, NPU or AUTO)")
	properties := flag.String("properties", "", "extra OpenVINO compile properties as KEY=VALUE pairs, e.g. CACHE_DIR=~/.cache/whispy/openvino")
	preprocDevice := flag.String("preproc-device", "CPU", "OpenVINO device for the mel spectrogram model, on the CPU by default")
	decoderDevice := flag.String("decoder-device", "", "OpenVINO device for the decoder and joint networks, which run per token (defaults to -device)")
	vadPath := flag.String("vad", filepath.Join(home, ".cache/whispy/silero_vad.onnx"), "path to the silero vad onnx model")
	transcribePath := flag.String("transcribe", "", "transcribe a 16 kHz mono WAV or AU file and exit")
	flag.Parse()

	replacer, err := loadReplacer(*replacerPath)
	if err != nil {
		return err
	}
	replacers := []textReplacer{replacer, words2num.Words2Num{}, multiplier.Multiplier{}, timetext.Time{}}

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

	// The debug log is off unless it was asked for. A cache directory that
	// cannot be written is worth hearing about once here rather than after
	// every capture.
	var captures *debuglog.Log
	if *debugLog {
		dir, err := debuglog.Dir()
		if err != nil {
			return serr.Wrap(err)
		}
		if captures, err = debuglog.Open(dir); err != nil {
			return serr.Wrap(err)
		}
		fmt.Fprintf(os.Stderr, "whispy: logging captures to %s\n", dir)
	}

	vad, err := silero.New(silero.Config{Model: *vadPath})
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

	keptAudio := ""
	if *keepAudioFlag {
		path, err := keepAudioPath()
		if err != nil {
			return err
		}
		keptAudio = path
		fmt.Fprintf(os.Stderr, "whispy: keeping the last capture in %s\n", path)
	}

	var pwRecordCmd *exec.Cmd
	// autoEnd belongs to the running capture and is nil while idle, so the
	// silence that ends one capture cannot end the next one.
	var autoEnd chan struct{}
	captureCommand := false
	auHeader := make([]byte, 24) // AU header in case we keep audio
	var rawPCM []float32
	var pipeR *io.PipeReader
	var pipeW *io.PipeWriter
	var pipeWG sync.WaitGroup
	for {
		var (
			event              captureEvent
			starting, stopping bool
		)
		select {
		case sig := <-sigs:
			event = eventOf(sig)
			starting, stopping = captureNext(pwRecordCmd != nil, event)
		case <-autoEnd:
			event = eventSilence
			stopping = true
		}
		if !starting && !stopping {
			continue
		}

		if starting {
			captureCommand = event == eventCommand
			pwRecordCmd = exec.Command("pw-record", "--format=f32", "--rate=16000", "--channels=1", "-")
			pipeR, pipeW = io.Pipe()
			pwRecordCmd.Stdout = pipeW
			autoEnd = make(chan struct{}, 1)
			end := autoEnd
			rawPCM = rawPCM[0:0]
			vad.Reset()
			// The endpointer belongs to this capture, so it starts knowing
			// nothing about speech.
			endpoint := silero.NewEndpoint(silero.DefaultEndpointConfig)
			pipeWG.Go(func() {
				// A capture can be stopped before the header arrives, which just
				// means there is no audio to read.
				if _, err := io.ReadFull(pipeR, auHeader[0:]); err != nil {
					return
				}

				const chunkSize = 4 * 16000 * 1 // f32 sized, 16000 rate, 1 second
				var bytesChunk [chunkSize]byte
				floatChunk := make([]float32, chunkSize/4)
				sigSent := false
				// dropped is set once the capture is over for a reason of our
				// own, the bound or a VAD that stopped working. The recorder
				// is still writing into the pipe, and a pipe nobody reads
				// blocks the recorder, which blocks the copy into the pipe,
				// which blocks the Wait that ends the capture - so the audio
				// keeps being read here and is thrown away.
				dropped := false
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
						// Anything else is a pipe that went wrong under us, most
						// likely because the recorder died. What was read still
						// belongs to the capture, and the audio already in hand
						// is still worth transcribing, so this says so and ends
						// the reading rather than taking the daemon down.
						if n > 0 {
							floatChunk = bytesIntoF32(bytesChunk[:n], floatChunk)
						} else {
							return
						}
					}

					if dropped {
						// Nothing is being collected any more, but the
						// recorder still has somewhere to put its audio.
						if err != nil {
							return
						}
						continue
					}

					if len(floatChunk) > 0 {
						rawPCM = append(rawPCM, floatChunk...)
					}

					// The bound is on the audio, not on the speaker, so it
					// applies to dictation too. The audio in hand up to here
					// is still transcribed.
					if overCapture(len(rawPCM)) {
						fmt.Fprintf(os.Stderr,
							"whispy: capture reached %d seconds, ending it\n", maxCaptureSeconds)
						signalEnd(end)
						dropped = true
						continue
					}

					// A command capture ends itself once speech stops.
					if captureCommand && len(floatChunk) > 0 {
						probs, err := vad.SpeechProb(floatChunk)
						if err != nil {
							// The capture cannot tell when speech stops without
							// the VAD. End it and transcribe what there is; the
							// daemon has no business dying over one model
							// refusing one chunk.
							fmt.Fprintf(os.Stderr, "whispy: vad: %v\n", err)
							signalEnd(end)
							dropped = true
							continue
						}
						// The endpointer decides over the individual windows,
						// so a breath in the middle of a sentence is not the
						// end of one. Once it has ended the capture it stays
						// ended, which is what lets the signal go out once:
						// nobody reads the channel again until this capture is
						// over, so a second send would block this goroutine.
						if _, done := endpoint.Add(probs); done && !sigSent {
							signalEnd(end)
							sigSent = true
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
			continue
		}

		if err := pwRecordCmd.Process.Signal(syscall.SIGTERM); err != nil {
			// It may have exited on its own, which should not take the daemon
			// down with it: the cleanup below is the same either way.
			fmt.Fprintf(os.Stderr, "whispy: stopping pw-record: %v\n", err)
		}
		pwRecordCmd.Wait()
		pwRecordCmd = nil
		autoEnd = nil
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
		if keptAudio != "" {
			if err := keepAudio(keptAudio, auHeader[:], rawPCM); err != nil {
				return serr.Wrap(err)
			}
		}

		// A debug log is a debugging aid, so a write that fails is reported and
		// the dictation goes on: only the transcript itself can take the daemon
		// down.
		if captures != nil {
			if err := captures.Write(time.Now(), rawPCM, captureCommand, text); err != nil {
				fmt.Fprintf(os.Stderr, "whispy: %v\n", err)
			}
		}

		if text == "" {
			// Nothing was said, so there is nothing to run or type.
			continue
		}

		if captureCommand {
			action, err := command.Run(ctx, text)
			if err != nil {
				return serr.Wrap(err)
			}
			// An empty action is the one command mode runs when no rule
			// matched: nothing was asked for, so there is nothing to say.
			if *printText && action.Program != "" {
				println(action.String())
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
