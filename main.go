package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"
	"unsafe"

	"github.com/pkg/errors"
)

/*
#cgo CFLAGS: -I${SRCDIR}/whisper.cpp/include -I${SRCDIR}/whisper.cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/whisper.cpp/build/src -L${SRCDIR}/whisper.cpp/build/ggml/src -L${SRCDIR}/whisper.cpp/build/ggml/src/ggml-vulkan -L${SRCDIR}/whisper.cpp/build/ggml/src/ggml-blas
#cgo LDFLAGS: -lwhisper -lggml -lggml-base -lggml-cpu -lggml-blas -lggml-vulkan -lvulkan -lopenblas -lgomp -lm -lstdc++
#include <whisper.h>
#include <stdlib.h>
*/
import "C"

func whisperInit(path string) *C.struct_whisper_context {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	params := C.whisper_context_default_params()
	params.flash_attn = true
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

func fileAsF32(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	return bytesToFloat32s(data)
}

func run(ctx context.Context) error {
	printText := os.Getenv("PRINT_TEXT") == "1"
	keepAudio := os.Getenv("KEEP_AUDIO") == "1"

	whisperCtx := whisperInit(os.Args[1])
	if whisperCtx == nil {
		panic("unable to initialize whisper context")
	}

	params := C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)
	params.n_threads = 8
	params.no_context = true
	params.no_timestamps = true
	params.print_progress = false
	params.print_timestamps = false
	params.single_segment = true
	params.suppress_blank = true
	params.suppress_nst = true

	sigs := make(chan os.Signal, 10)
	signal.Notify(sigs, syscall.SIGUSR2)

	backend, err := NewBackend(ctx)
	if err != nil {
		return errors.WithStack(err)
	}

	const tmpFile = "/tmp/a.au"
	var sb strings.Builder
	var pwRecordCmd *exec.Cmd
	for range sigs {
		if pwRecordCmd == nil {
			pwRecordCmd = exec.Command("pw-record", "--format=f32", "--rate=16000", "--channels=1", tmpFile)
			if err := pwRecordCmd.Start(); err != nil {
				return errors.WithStack(err)
			}
		} else {
			if err := pwRecordCmd.Process.Signal(syscall.SIGTERM); err != nil {
				return errors.WithStack(err)
			}
			pwRecordCmd.Wait()
			pwRecordCmd = nil
			samples, err := fileAsF32(tmpFile)
			if err != nil {
				return err
			}

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

			if printText {
				println(text)
			}
			if !keepAudio {
				if err := os.Remove(tmpFile); err != nil {
					return errors.WithStack(err)
				}
			}

			windowClass, err := backend.GetWindowClass(ctx)
			if err != nil {
				return errors.WithStack(err)
			}

			if ShouldPaste(windowClass) {
				if err := backend.PasteText(text); err != nil {
					return errors.WithStack(err)
				}
			} else {
				if err := backend.TypeText(text); err != nil {
					return errors.WithStack(err)
				}
			}
		}
	}
	return nil
}

func main() {
	if err := run(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "%+v\n", err)
		os.Exit(1)
	}
}
