package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/joshuarubin/go-sway"
	"github.com/pkg/errors"
)

/*
#cgo CFLAGS: -I${SRCDIR}/whisper.cpp/include -I${SRCDIR}/whisper.cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/whisper.cpp/build/src -L${SRCDIR}/whisper.cpp/build/ggml/src -L${SRCDIR}/whisper.cpp/build/ggml/src/ggml-sycl -L${SRCDIR}/whisper.cpp/build/ggml/src/ggml-blas
#cgo LDFLAGS: -lwhisper -lggml -lggml-base -lggml-cpu -lggml-sycl -lggml-blas
#cgo LDFLAGS: -lOpenCL -larcher -ldnnl -lgomp -limf -lintlc -liomp5 -lirng -lm -lmkl_core -lmkl_intel_ilp64 -lmkl_sycl_blas -lmkl_tbb_thread -lstdc++ -lsvml -lsycl -ltbb -lur_loader
#include <whisper.h>
#include <stdlib.h>
*/
import "C"

func whisperInit(path string) *C.struct_whisper_context {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	params := C.whisper_context_default_params()
	return C.whisper_init_from_file_with_params(cPath, params)
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
	printText := os.Getenv("PRINT_TEXT") == "1"
	printTime := os.Getenv("PRINT_TIME") == "1"
	keepAudio := os.Getenv("KEEP_AUDIO") == "1"

	whisperCtx := whisperInit(os.Args[1])
	if whisperCtx == nil {
		panic("unable to initialize whisper context")
	}
	vadCtx := vadInit(os.Args[2])
	if vadCtx == nil {
		panic("unable to initialize vad context")
	}

	params := C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)
	params.n_threads = C.int(runtime.NumCPU())
	params.no_context = true
	params.no_timestamps = true
	params.print_progress = false
	params.print_timestamps = false
	params.single_segment = true
	params.suppress_blank = true
	params.suppress_nst = true

	vadParams := C.whisper_vad_default_params()

	sigs := make(chan os.Signal, 10)
	signal.Notify(sigs, syscall.SIGUSR2)

	swayClient, err := sway.New(ctx)
	if err != nil {
		return errors.WithStack(err)
	}

	const tmpFile = "/tmp/a.au"
	var sb strings.Builder
	var pwRecordCmd *exec.Cmd
	auHeader := make([]byte, 24) // AU header in case we keep audio
	var rawPCM []float32
	var pipeR *io.PipeReader
	var pipeW *io.PipeWriter
	var pipeWG sync.WaitGroup
	for range sigs {
		if pwRecordCmd == nil {
			pwRecordCmd = exec.Command("pw-record", "--format=f32", "--rate=16000", "--channels=1", "-")
			pipeR, pipeW = io.Pipe()
			pwRecordCmd.Stdout = pipeW
			rawPCM = rawPCM[0:0]
			pipeWG.Go(func() {
				if _, err := io.ReadFull(pipeR, auHeader[0:]); err != nil {
					panic(err)
				}

				const chunkSize = 4 * 16000 * 2 // f32 sized, 16000 rate, 1 second
				var bytesChunk [chunkSize]byte
				floatChunk := make([]float32, chunkSize/4)
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
					success := C.whisper_vad_detect_speech(vadCtx, (*C.float)(&floatChunk[0]), C.int(len(floatChunk)))
					if !success {
						panic("failed to vad detect speech")
					}
					segments := C.whisper_vad_segments_from_probs(vadCtx, vadParams)
					nSegments := C.whisper_vad_segments_n_segments(segments)
					C.whisper_vad_free_segments(segments)
					if nSegments == 0 {
						println("no speech")
					} else {
						println("has speech")
					}

					if err != nil {
						return
					}
				}
			})
			if err := pwRecordCmd.Start(); err != nil {
				return errors.WithStack(err)
			}
		} else {
			if err := pwRecordCmd.Process.Signal(syscall.SIGTERM); err != nil {
				return errors.WithStack(err)
			}
			pwRecordCmd.Wait()
			pwRecordCmd = nil
			pipeW.Close()
			pipeWG.Wait()

			start := time.Now()
			r := C.whisper_full(whisperCtx, params, (*C.float)(&rawPCM[0]), C.int(len(rawPCM)))
			if r != 0 {
				panic("whisper full fail")
			}

			sb.Reset()
			numSegments := C.whisper_full_n_segments(whisperCtx)
			for i := range numSegments {
				text := C.whisper_full_get_segment_text(whisperCtx, i)
				sb.WriteString(C.GoString(text))
			}
			text := strings.TrimSpace(sb.String())

			tree, err := swayClient.GetTree(ctx)
			if err != nil {
				return errors.WithStack(err)
			}
			focusedNode := tree.FocusedNode()

			if printTime {
				println("Took", time.Since(start).Truncate(time.Millisecond).String())
			}
			if printText {
				println(text)
			}
			if keepAudio {
				f, err := os.Create(tmpFile)
				if err != nil {
					return errors.WithStack(err)
				}
				// write the AU header to make it a proper AU file, this is what we discarded above
				if _, err := f.Write(auHeader[:]); err != nil {
					return errors.WithStack(err)
				}
				if err := binary.Write(f, binary.LittleEndian, rawPCM); err != nil {
					return errors.WithStack(err)
				}
			}
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
					return errors.WithStack(err)
				}
				if err := exec.Command("ydotool", "key", "29:1", "47:1", "47:0", "29:0").Run(); err != nil {
					return errors.WithStack(err)
				}
				wlCopyCmd.Process.Kill()
				wlCopyCmd.Wait()
			} else {
				if err := exec.Command("ydotool", "type", "-d=8", "-H=6", text).Run(); err != nil {
					return errors.WithStack(err)
				}
			}
		}
	}
	return nil
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
