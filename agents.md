# whispy

Dictation daemon for Wayland: PipeWire captures audio, Parakeet TDT
transcribes it through OpenVINO, and the text is injected into the focused
window with `wtype` or `wl-copy`.

## Layout

- `main.go`: the daemon. Signal driven capture loop, VAD for search mode,
  text injection, replacements CSV.
- `openvino/`: cgo bindings for the OpenVINO C API.
- `parakeet/`: Parakeet TDT v2/v3 speech to text.
- `silero/`: Silero VAD v5/v6 for 16 kHz audio, in pure Go. The matrix kernels
  use `simd/archsimd` on amd64, which `GOEXPERIMENT=simd` turns on.
- `audio/`: reads the 16 kHz mono WAV and AU files the daemon and the tests
  use. Tests take their fixtures through it instead of parsing audio
  themselves.
- `timetext/`: rewrites clock times written as two numbers (`11 30 pm`) into
  `11:30pm`.

There is no C or C++ in this repository. The OpenVINO C API is the only native
dependency, and only parakeet goes through it; the VAD is pure Go.

## Text

Transcript text runs through a pipeline of `textReplacer` values, in order:
the `-replacer` CSV, `words2num`, then `timetext`. `-transcribe` uses the same
pipeline as the daemon, so a file comes out the way a dictation would be typed.
`casualText` stays outside the pipeline and applies only in search mode and to
WhatsApp.

`timetext` deliberately runs after `words2num`: that is what turns "eleven
thirty pm" into "11 30 pm" for `timetext` to put the colon in. It matches an
hour and minutes separated by a space, with an optional am/pm marker (spaces,
`a.m.` and `p.m.` included) that it attaches in lower case. One digit of
minutes only counts next to a marker, so "chapter 9 5" is left alone, and a
letter glued to the minutes rejects the match, so "11 30amsterdam" is not a
time. `hasTime` prescans with the same matcher so `Replace` costs no
allocations when there is no time; keep the two in step.

## openvino

`openvino/ov.go` is the only cgo file in the module. It wraps just enough of
the C API to load models, run inference and move tensor data. Tensor data is
copied in and out of OpenVINO owned memory: never hand C a pointer to Go
memory that outlives the call, and prefer Go over C for anything that is not
an API call.

- OpenVINO's core must never be freed: `ov_core_free` unloads the plugins
  while OpenVINO's static state and worker threads are still around, which
  segfaults. `ov_shutdown()` only moves that crash to the next create/teardown
  cycle. `openvino.SharedCore()` keeps one core per process, and each
  library's `Close` releases only its compiled models and infer requests.
- `CompileWith` passes plugin properties, which need care: the C API wants
  `property_args_size` to be the number of _arguments_ (two per key/value
  pair), not the number of properties. Passing the property count fails with
  `INVALID_C_PARAM` and no message. Property names are the C API aliases
  (`NPU_COMPILER_TYPE`, `CACHE_DIR`), not `ov::` names.
- `ov_shape_create` rejects a zero rank shape, so scalar tensors cannot be
  allocated. Use the tensor an infer request already owns instead.
- `PortShape` reports dynamic dimensions as -1 and `Dim` as 0, so callers pick
  their own fallback instead of tripping over OpenVINO's "to_shape was called
  on a dynamic shape" error.

## parakeet

- Model files: `parakeet_{melspectogram,encoder,decoder,joint}.{xml,bin}` and
  `parakeet_vocab.json` from `FluidInference/parakeet-tdt-0.6b-v3-ov`, in
  `~/.cache/whispy/parakeet-v3` by default. The v3 vocabulary carries
  `blank_id: 8192`, which selects the v3 head layout.
- Everything the models expose is read from the compiled model ports at load
  time (shapes, element types, head sizes), so another export with the same
  tensor names should work without code changes. Dynamic dimensions fall back
  to the values the exports use: the preprocessor window is sized to the audio
  instead (the v2 export is dynamic), and a dynamic encoder frame count uses
  1250. Hidden sizes come from the encoder and decoder outputs, or from the
  joint network's inputs when those are dynamic.
- The v3 shapes, for reference: the mel preprocessor is `1x240000` in and
  `1x128x1501` out, the encoder is `1x128x1501` in and `1x1024x188` out, the
  decoder is `1x1` (i64 targets) plus two `2x1x640` states, and the joint
  network is `1x1x1024` and `1x1x640` in and `1x1x1x8198` out. All static.
- The vocabularies reserve the low ids for control tokens (`<unk>`,
  `<|nospeech|>`, language and speaker tags); `tokenizer.isControl` keeps them
  out of the token list and the transcript, since a device that rounds
  differently from the CPU can make one of them win the argmax. Only pieces
  wrapped in angle brackets count: the digits and a bare `▁` are ordinary
  tokens the transcript needs.
- The preprocessor runs on the CPU by default (`Config.PreprocDevice`): the
  work is small relative to the data volume, offloading it would add transfers
  for no compute win, and the v2 export has a dynamic input there, which the
  NPU rejects. The encoder, decoder and joint network follow `Config.Device`,
  falling back to the CPU per model when a device cannot compile it (the NPU
  needs static shapes and may not support every operation). `Model.report`
  prints a line per fallback and a summary of the device each model ended up on.
- NPU notes: the driver compiler rejects the joint network's
  `LogSoftmax axis="-1"` (vpux `AlignDimensionsForDPU`: "Got negative index -1
  for Dim"). The axis can be rewritten to `3` without changing the logits, see
  `TestJointSoftmaxAxis`. `NPU_COMPILER_TYPE=PLUGIN` selects the compiler
  inside the plugin instead of the driver one, but on Panther Lake with driver
  1.38 it refused all three models, so the default driver compiler is the one
  that works there. With the axis rewritten, all three compile for the NPU on
  that machine. `negativeAxisHint` adds a pointer to the readme when a refused
  model still carries `axis="-1"`.
- Fallback messages go through `cleanError`, which strips OpenVINO's
  `Exception from <file>:<line>:` re-throw preamble so the line says what the
  plugin actually complained about. Keep new diagnostics in that style.
- `-transcribe FILE` transcribes one file and exits. It needs no VAD and no
  sway session, so it is the way to check a model install or compare devices
  on identical audio (`-device CPU` versus `-device NPU`).
- Tests need the model files and skip when they are missing. Point
  `PARAKEET_MODEL_DIR` at another directory to override. `TestDynamicWindow`
  rewrites the preprocessor input to be dynamic in a temp dir, which is how
  the dynamic path is covered without downloading the v2 models.

## silero

The 16 kHz Silero VAD (v5/v6), in pure Go with no OpenVINO. `New` reads the
weights straight out of the ONNX file with the small protobuf reader in
`onnx.go`, so the model file is the only thing to install. It comes from
`istupakov/silero-vad-onnx`, either the `silero_vad_op18_ifless.onnx` export or
the one that also carries the 8 kHz branch, which is ignored. `-vad` points at
it and defaults to `~/.cache/whispy/silero_vad.onnx`.

The model consumes `WindowSamples` plus 64 samples of context per step and
carries hidden and cell state between steps, so `SpeechProb` streams: it
buffers partial windows and carries context across calls, the way the OpenVINO
implementation it replaced did.

- The graph is fixed to the 16 kHz path: reflect pad the 576 sample chunk by
  64, four STFT frames of 256 with a hop of 128 and a periodic Hann window,
  magnitude over 129 bins, `conv(129,128)`, `conv(128,64)`, `conv(64,64)`,
  `conv(64,128)` with a kernel of 3 and a padding of 1 (the middle two with a
  stride of 2, which leaves one frame), a ReLU after each, one LSTMCell step, a
  ReLU on its output, a `128,1` convolution and a sigmoid.
- The STFT is a plain radix 2 FFT with a precomputed window and twiddles rather
  than the model's 258x256 basis convolution: the same math for a fraction of
  the work.
- `testdata/speech.txt` holds the probabilities OpenVINO produced for the test
  clip back when both engines were in the tree. The two agreed to about 1e-6,
  and `TestSpeechProb` holds this implementation to 1e-4 of them. That is the
  numerical guard now that there is no second engine to compare against.
- Weights are packed so `matvecAdd` can broadcast one input and keep a block of
  outputs in a vector register: `packConv` orders rows by (tap, input channel)
  and `packRNN` by input, both with output channels contiguous. `matvecAdd` is
  the only kernel; `silero.go` holds the scalar version and `kernels_simd.go`
  the amd64 + `goexperiment.simd` build, which falls back to scalar when AVX is
  missing. Without the experiment the package still builds and passes its
  tests, it is just slower.
- `archsimd.ClearAVXUpperBits()` at the end of the vector kernel is load
  bearing. The LSTM activations are scalar math right after it, and without the
  `VZEROUPPER` every `sigmoid` and `tanh` following a kernel paid a false
  dependency penalty: the vector build was slower than the scalar one (17.5ms
  against 13.0ms for the test clip) until it was added, and 5.4ms after.

Benchmarks live in `silero/silero_test.go` and take the model and the fixture
from `silero/testdata`:

```
go test -run '^$' -bench BenchmarkSpeechProb -benchtime 3s ./silero/
GOEXPERIMENT=simd go test -run '^$' -bench BenchmarkSpeechProb -benchtime 3s ./silero/
```

The sub-benchmark name is the kernel the build picked up. `GOEXPERIMENT=simd`
also makes the compiler target AVX2 for everything else, so the simd run is
faster than the scalar run by more than the kernel alone.

## Build and test

- `./build` installs whispy and runs it. Whatever builds it has to set
  `GOEXPERIMENT=simd` (the PKGBUILD does) or it gets the scalar kernel.
- `go build ./...` works either way.
- `go test ./...` covers the parakeet pipeline and the VAD. The audio fixtures
  under `parakeet/testdata` and `silero/testdata` are 16 kHz mono WAV.
