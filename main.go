package main

import (
	"context"
	"encoding/binary"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/daaku/serr"
	"github.com/joshuarubin/go-sway"
)

/*
#cgo CFLAGS: -I${SRCDIR}/whisper.cpp/include -I${SRCDIR}/whisper.cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/whisper.cpp/build/bin -Wl,-rpath,${SRCDIR}/whisper.cpp/build/bin
#cgo LDFLAGS: -lwhisper -lparakeet -lggml -lggml-base -lggml-cpu -lggml-vulkan
#include <whisper.h>
#include <parakeet.h>
#include <stdlib.h>
*/
import "C"

func parakeetInit(path string) *C.struct_parakeet_context {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	params := C.parakeet_context_default_params()
	return C.parakeet_init_from_file_with_params(cPath, params)
}

func vadInit(path string) *C.struct_whisper_vad_context {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	params := C.whisper_vad_default_context_params()
	return C.whisper_vad_init_from_file_with_params(cPath, params)
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

func run(ctx context.Context) error {
	home, _ := os.UserHomeDir()
	printText := flag.Bool("print-text", false, "print the transcribed text")
	printTime := flag.Bool("print-time", false, "print the transcription duration")
	keepAudio := flag.Bool("keep-audio", false, "save the captured audio to /tmp/a.au")
	replacerPath := flag.String("replacer", filepath.Join(home, ".config/whispy/replacer.csv"), "path to a replacements CSV")
	modelPath := flag.String("model", filepath.Join(home, ".cache/whispy/ggml-parakeet-tdt-0.6b-v3-q8_0.bin"), "path to model")
	vadPath := flag.String("vad", filepath.Join(home, ".cache/whispy/ggml-silero-v6.2.0.bin"), "path to vad model")
	flag.Parse()

	replacer, err := loadReplacer(*replacerPath)
	if err != nil {
		return err
	}

	C.ggml_backend_load_all()

	parakeetCtx := parakeetInit(*modelPath)
	if parakeetCtx == nil {
		panic("unable to initialize parakeet context")
	}
	vadCtx := vadInit(*vadPath)
	if vadCtx == nil {
		panic("unable to initialize vad context")
	}

	params := C.parakeet_full_default_params(C.PARAKEET_SAMPLING_GREEDY)
	params.n_threads = C.int(runtime.NumCPU())
	params.no_context = true

	vadParams := C.whisper_vad_default_params()

	sigs := make(chan os.Signal, 10)
	signal.Notify(sigs, syscall.SIGUSR1, syscall.SIGUSR2)

	swayClient, err := sway.New(ctx)
	if err != nil {
		return serr.Wrap(err)
	}

	const tmpFile = "/tmp/a.au"
	var sb strings.Builder
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
						success := C.whisper_vad_detect_speech(vadCtx, (*C.float)(&floatChunk[0]), C.int(len(floatChunk)))
						if !success {
							panic("failed to vad detect speech")
						}
						segments := C.whisper_vad_segments_from_probs(vadCtx, vadParams)
						nSegments := C.whisper_vad_segments_n_segments(segments)
						C.whisper_vad_free_segments(segments)
						if nSegments == 0 {
							// speech had started, and has now ended
							if speechStarted {
								if !sigSent {
									sigs <- syscall.SIGUSR1
									sigSent = true
								}
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
			r := C.parakeet_full(parakeetCtx, params, (*C.float)(&rawPCM[0]), C.int(len(rawPCM)))
			if r != 0 {
				panic("whisper full fail")
			}

			sb.Reset()
			numSegments := C.parakeet_full_n_segments(parakeetCtx)
			for i := range numSegments {
				text := C.parakeet_full_get_segment_text(parakeetCtx, i)
				sb.WriteString(C.GoString(text))
			}
			text := strings.TrimSpace(sb.String())
			text = replacer.Replace(text)

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

				appID := *focusedNode.AppID
				pasteMode := strings.HasPrefix(appID, "firefox") ||
					strings.HasPrefix(appID, "chromium") ||
					strings.HasPrefix(appID, "brave")
				if pasteMode {
					wlCopyCmd := exec.Command("wl-copy", "--foreground", text)
					if err := wlCopyCmd.Start(); err != nil {
						return serr.Wrap(err)
					}
					if err := exec.Command("wtype", "-M", "ctrl", "v", "-m", "ctrl").Run(); err != nil {
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

func loadReplacer(path string) (*strings.Replacer, error) {
	if path == "" {
		return strings.NewReplacer(), nil
	}
	f, err := os.Open(path)
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
