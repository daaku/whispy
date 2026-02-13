package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strings"
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

func bytesToFloat32s(b []byte) ([]float32, error) {
	if len(b)%4 != 0 {
		return nil, errors.New("length not multiple of 4")
	}
	floats := make([]float32, len(b)/4)
	for i := range floats {
		bits := binary.LittleEndian.Uint32(b[i*4 : (i+1)*4])
		floats[i] = math.Float32frombits(bits)
	}
	return floats, nil
}

func run(ctx context.Context) error {
	printText := os.Getenv("PRINT_TEXT") == "1"
	printTime := os.Getenv("PRINT_TIME") == "1"
	keepAudio := os.Getenv("KEEP_AUDIO") == "1"

	whisperCtx := whisperInit(os.Args[1])
	if whisperCtx == nil {
		panic("unable to initialize whisper context")
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

	sigs := make(chan os.Signal, 10)
	signal.Notify(sigs, syscall.SIGUSR2)

	swayClient, err := sway.New(ctx)
	if err != nil {
		return errors.WithStack(err)
	}

	const tmpFile = "/tmp/a.au"
	var sb strings.Builder
	var pwRecordCmd *exec.Cmd
	var rawPCM bytes.Buffer
	for range sigs {
		if pwRecordCmd == nil {
			pwRecordCmd = exec.Command("pw-record", "--format=f32", "--rate=16000", "--channels=1", "-")
			rawPCM.Reset()
			pwRecordCmd.Stdout = &rawPCM
			if err := pwRecordCmd.Start(); err != nil {
				return errors.WithStack(err)
			}
		} else {
			if err := pwRecordCmd.Process.Signal(syscall.SIGTERM); err != nil {
				return errors.WithStack(err)
			}
			pwRecordCmd.Wait()
			pwRecordCmd = nil
			// skip the 24 byte header, then we have the data in the expected format
			samples, err := bytesToFloat32s(rawPCM.Bytes()[24:])
			if err != nil {
				return err
			}

			start := time.Now()
			r := C.whisper_full(whisperCtx, params, (*C.float)(&samples[0]), C.int(len(samples)))
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
				// NOTE: we're writing the file including the 24 byte header for the AU format
				if err := os.WriteFile(tmpFile, rawPCM.Bytes(), 0o600); err != nil {
					return errors.WithStack(err)
				}
			}
			if strings.Contains(focusedNode.Name, "WhatsApp") {
				text = casualText(text)
			}

			appID := *focusedNode.AppID
			pasteMode := strings.HasPrefix(appID, "firefox") || strings.HasPrefix(appID, "chromium")
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
