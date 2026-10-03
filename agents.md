# whispy

Dictation daemon for Wayland: PipeWire captures audio, Parakeet TDT
transcribes it through OpenVINO, and the text is injected into the focused
window with `wtype` or `wl-copy`.

## Layout

- `main.go`: the daemon. Signal driven capture loop, VAD for command mode,
  text injection, replacements CSV.
- `command/`: matches a transcript against command patterns and runs the
  action. A transcript that matches no pattern does nothing.
- `openvino/`: cgo bindings for the OpenVINO C API.
- `parakeet/`: Parakeet TDT v2/v3 speech to text.
- `silero/`: Silero VAD v5/v6 for 16 kHz audio, in pure Go. The matrix kernels
  use `simd/archsimd` on amd64, which `GOEXPERIMENT=simd` turns on.
- `audio/`: reads the 16 kHz mono WAV and AU files the daemon and the tests
  use, and writes the 16 bit WAV files the debug log keeps. Tests take their
  fixtures through it instead of parsing audio themselves.
- `debuglog/`: keeps the audio and the text of each capture in
  `$XDG_CACHE_HOME/whispy/debug` for `-debug-log`, oldest first out. Its own
  package so the retention rule has a test and the capture loop stays short.
- `timetext/`: rewrites clock times written as two numbers (`11 30 pm`) into
  `11:30pm`.
- `multiplier/`: rewrites a spoken multiplier (`hundred x`) as `100x`.
- `wer/`: scores a transcript against a transcript of record, word error rate
  and the three kinds of difference. Measurement only, nothing in the daemon
  imports it; it exists so `eval/` has a rate it can be tested on.

There is no C or C++ in this repository. The OpenVINO C API is the only native
dependency, and only parakeet goes through it; the VAD is pure Go.

## Text

Transcript text runs through a pipeline of `textReplacer` values, in order:
the `-replacer` CSV, `words2num`, `multiplier`, then `timetext`. `-transcribe`
uses the same pipeline as the daemon, so a file comes out the way a dictation
would be typed. `casualText` stays outside the pipeline and applies only to
WhatsApp.

`multiplier` deliberately runs after `words2num` too: a scale word that was
part of a number is digits by then, so it only sees the bare ones. That is what
makes `hundred x` into `100x` while `one hundred x` is `100x` and not
`one 100x`. It writes the x lower case, and leaves `3 x 4` a multiplication by
refusing an x that starts a number.

`timetext` deliberately runs after `words2num`: that is what turns "eleven
thirty pm" into "11 30 pm" for `timetext` to put the colon in. It matches an
hour and minutes separated by a space, with an optional am/pm marker (spaces,
`a.m.` and `p.m.` included) that it attaches in lower case. One digit of
minutes only counts next to a marker, so "chapter 9 5" is left alone, and a
letter glued to the minutes rejects the match, so "11 30amsterdam" is not a
time. `hasTime` prescans with the same matcher so `Replace` costs no
allocations when there is no time; keep the two in step.

## command

`command` turns a transcript into an action and runs it: `Parse` walks the
`rules` table and `Run` runs the match. A transcript that matches no rule runs
nothing and returns the zero `Action`, which is the whole privacy story of
command mode: speech that was only speech must not be `xdg-open`ed as a search
query, so `search for <query>` is the only rule that opens a browser. The rules
are one line each, so adding a volume command or a URL is a one line change.

- Exact rules match the whole transcript ignoring case and apostrophes, since
  speech to text is not consistent about them (`what's` and `whats` are the
  same command).
- Pattern rules take a variable at the end: the volume rules take a
  percentage written `20%`, `20 percent` or `20 per cent` and pass the digits
  to noctalia after clamping them to 0..100, so a mistranscribed "set volume to
  200 percent" cannot hand noctalia a number off the end of the dial.
  `search for <query>` searches only the query rather than the whole command.
- The alarm rule matches a trigger (`set alarm`, `set an alarm`, `set timer`,
  `remind me`), then `in <duration>` or `at <clock>`, then an optional
  `to <label>`, and hands snoozer one flag per part. Durations are minutes or
  hours. `oneAsDigit` rewrites a leading "one" as "1" in those two slots:
  words2num leaves a lonesome "one" as a word, but a slot that asks for a
  number can only mean the count, so "set a timer in one minute" still works.
  A label that opens with a dash is refused rather than passed on: actions are
  built as argv so a label can never reach a shell, but `--label=pwn` said out
  loud would be `--label=--label=pwn`, which snoozer reads as its own flag.
- A clock reading that does not say am or pm is resolved by `reading.resolve`
  to the next time the clock shows it, because snoozer reads a bare `3:20` as
  the 24 hour 03:20: at 1pm it is `3:20pm`, at 1am `3:20am`. That is why `parse`
  takes the time and `Parse` hands it `time.Now`, and why the clock tests pin
  the time.
- Polite framing is dropped before matching, from `prefixes` and `suffixes` in
  `phrases.go`: 188 phrases that open a command ("could you", "hey whispy", "i
  was wondering if you could") and 127 that close one ("please", "thanks",
  "right now"). Mining them is in the `whispy-phrases` working notes rather
  than the repo: MASSIVE, the Snips/Sonos NLU benchmark and the Stanford
  Politeness Corpus, with counts per phrase.
- `candidates` tries the transcript as it stands, then with up to three phrases
  off the front and three off the back, and a candidate is only used when a
  rule matches it. That is what keeps a suffix out of a real part of a command:
  `remind me in 15 minutes to pick up the kids for me` keeps its "for me", and
  the "thanks" in `remind me at 3:20 to leave for school thanks` lands in the
  label because the alarm rule matches as it stands. Loosening that means
  sorting the suffixes into those that can never be command content and those
  that can.
- Query text is treated differently: `trimPoliteness` takes the trailing
  politeness off what `search for` pulled out, but never the leading words,
  since a query can open with a content word that reads as framing ("right
  whale").
- A phrase needs a word boundary and something left behind, so "so" does not
  bite into "solve the puzzle" and "thanks" on its own is not a command.
  Apostrophes are ignored on both sides, so `if you dont mind` matches "if you
  don't mind".
- `trim` also drops the punctuation speech to text puts around a command: a
  trailing full stop, quotes, and the comma in "mute speakers, please".
- `trim` drops outer space and one trailing full stop before matching, which
  speech to text adds. Pattern variables are taken from the trimmed transcript
  as written, so a query keeps its case.
- Order matters: the exact rules come first so "set volume to max" is not read
  as a percentage.
- `Action.Run` captures stderr, because the programs it calls say what went
  wrong there, `xdg-open` especially, and stdout for actions that ask for it.
- `Action.Notify` shows what the program printed as a desktop notification
  titled with the program, which is how setting an alarm says so: snoozer
  prints `Alarm set for Wed 03:04pm: go for a walk` and that line is the
  notification body. `notify-send` failing only warns on stderr, since the
  alarm is set either way.
- `Run` runs the match and returns it, and `-print-text` prints it after. An
  `Action` with no program is the no match answer: the daemon checks for it so
  an unmatched transcript prints nothing and runs nothing. When a command
  misbehaves, that printed line is what says whether a rule matched at all.

## capture

`main.go` runs one capture at a time, and `captureNext` is the whole rule for
what an event does to it. It is a small function so it can be tested on its own
(`main_test.go`), because getting it wrong takes the daemon down:

- Idle: `SIGUSR1` (the command key) starts a command capture and `SIGUSR2` (the
  dictation key) starts a dictation capture. The mode is remembered for the
  whole capture in `captureCommand`, so whatever ends it does not change what
  happens to the audio.
- A command capture ends when the VAD hears speech stop, which is why pressing
  the command key again while one is running does nothing.
- The end of speech travels on the capture's own `autoEnd` channel instead of
  arriving as `SIGUSR1`, so a signal left over from a capture that already
  stopped cannot start or end the next one. Treating it as an activation is
  what made a second press stop the recorder before its header arrived, which
  panicked the reader goroutine and killed the process.
- `SIGUSR2` ends either kind of capture. A transcript with nothing in it is
  dropped rather than searched for or typed.
- Both ways of keeping audio happen after transcription and see the same two
  things: the `rawPCM` of the capture and the finished text. `-keep-audio`
  writes the last capture to one file; `-debug-log` appends a file per capture
  through `debuglog`, which also writes the text log and drops the oldest
  captures. A debug log that cannot be written is warned about and the
  dictation continues, because only the transcript is worth taking the daemon
  down for. A capture of no audio at all is skipped, so the log fills with
  things that were actually recorded, and a capture of audio with no text is
  kept, since that is one of the things being debugged.

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
- The only number pieces in the vocabulary are the bare `"0"`..`"9"` and none of
  them carries a word boundary marker, so when the model writes a number with
  digits there is no token that could hold the space in front of it. That space
  comes from `tokenizer.decode`, which opens a word when a digit follows a
  letter; without it a transcript reads `want42` instead of `want 42`. Marks
  that live inside a number (`,` `.` `:`) are not letters, so `1,000`, `3.5`
  and `10:30` stay in one piece. A name spelled out letter by letter, `MP3`, is
  split by the same rule, and the token stream has nothing to tell the two
  apart. Numbers the model spells as words are unaffected: `words2num` keeps
  whatever space was already there.
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
- Captures longer than one encoder window (1501 mel frames, 15.01 s) are decoded
  in overlapping windows and stitched together; `stitch` owns that (`processMel`
  hands it the model's own decoder), and the things it depends on are worth
  keeping in mind before touching it:
  - Token timings are **mel frames**, the same ruler the window offsets use.
    The encoder answers 188 frames for the 1501 it is given and stamps its
    tokens with the smaller index; `melPerEncoderFrame` converts, read from the
    shapes at load time. Comparing the two time bases is silently wrong: the
    guards below go dead and audio goes missing, which is how a sentence
    vanished from a 35 second capture.
  - Every window gets a **fresh decoder state**. Carrying the predictor state
    over makes the model treat the audio it re-hears as already said and stay
    quiet through the new speech after it; it also lets a word cut in half at a
    boundary ("for" for "forty") talk the next window into inventing "the same".
  - The windows **march**: a window starts where the previous one stopped
    speaking, rewound by `chunkRewind`, rather than a fixed distance after it.
    The model stops decoding before the end of a long window (a 15 second window
    came back with 10 seconds of words), so a fixed overlap is not the overlap
    you get and the audio past the stop point is heard by nobody. `chunkAdvance`
    is the floor on progress per window, so a window that returns nothing cannot
    stall the march.
  - Every window is the encoder's **full frame count** and the last one ends
    with the audio. The encoder needs a fixed number of frames, so a window that
    stops short is padded with silence, and a short padded window decodes to
    almost nothing: a 15 second capture left 1.7 seconds for its last window and
    it returned half a sentence.
  - Between windows the text is cut at the first **sentence end** inside the
    overlap region, when there is one. Cutting in the middle of a sentence loses
    the words there whenever the next window's reading comes out shorter than
    this one's, which it does.
  - Two guards remove the re-heard text: the position gate (tokens at or before
    the last emitted frame — `<=`, since two windows reading the same audio stamp
    the same word at the same frame and one of them has to go), then a run of
    tokens matching the tail of what is already emitted (at most
    `dedupPrevTokens` tokens, at most `dedupMaxOverlap`, and only within
    `dedupBoundaryFrames` of the boundary). Neither can tell a repeat in the
    audio from a duplicate of the transcript, so speech that really does repeat
    itself gets merged. That is by design.
  - The march has a bound: `windows` counts what the least advance per window
    allows, and going past it is an error rather than a transcript that stops
    short. A window that reads as the audio before it rather than failing is a
    quiet bug — the mel bins are rows in one flat slice, so `extractMelChunk`
    leaves frames past the end of the capture as the silence the encoder
    expects.
- The mel comes back mean-normalized: a tail of digital silence measures -0.13
  against -0.01 for the speech on the same capture, both near zero. So the mel
  cannot tell a capture that ended from a decoder that went quiet over live
  audio, and a loudness check has to be made on the samples.
- `stitch` takes the window decoder as a `windowDecode` argument, and
  `stitch_test.go` runs the march against a decoder written in the test over a
  spectrogram whose frames hold their own number. That is how the geometry is
  tested without model files and without the run to run wander of inference: what
  it has to guarantee is that every frame of the capture reaches a window and no
  window comes back short, which is arithmetic. The scripted decoder speaks in
  the ways the real one does, stopping early, starting late and going silent over
  the middle of a sentence, because those are what the march has to survive. It
  also pins what stitching cannot do: audio still sounding after the last window
  fell silent belongs to no other window. Keep the scripted decoder honest when
  the march changes — a mutation of the stitching (a gate comparison, a holdback,
  the fill, the pull back) is meant to fail a test here, and one that does not is
  a hole in the tests rather than a harmless change.
- The windows march from where the words stopped speaking, and a window is
  **pulled back to a sentence start** when one is within `chunkContext` (a third
  of a window, about a sentence) behind where it would have started. A fresh
  decoder dropped into the middle of a sentence goes silent over the first seconds
  of the audio it is given — measured up to four seconds on a dense audiobook
  reading — and the words in that silence are heard by nobody else. The sentence
  ends come from the text emitted so far, so the pull back is free information.
- What a window holds back for the next one is **given back** when that next
  window leaves it unsaid (`fill` in `stitch`). Loss is worse than a duplicate:
  the previous window's reading of the audio is kept when the window that was
  meant to improve on it says nothing there. On a scripted decoder that takes
  seven seconds to get its bearings this is the difference between 20 words lost
  and 53.
- Both together were measured on the LibriStem corpus (`eval/`): over four long
  captures, 343 seconds of continuous reading, the words that came back as nothing
  went from 5.3% of the reference to 0.9% and the error rate from 7.6% to 5.0%, for
  1.8 times the compute. Widening the plain overlap instead (rewind a third of a
  window) bought a hundredth of the error rate for half again the compute, and was
  left alone. Compare numbers with the corpus test, which folds spelled out numbers
  on both sides: without that, the corpora writing `SEVEN` where the model writes
  `7` counts as a deleted word and an inserted one, which is where the inflated
  figures in earlier notes came from.
- CPU inference is not reproducible bit for bit from one process to the next, and
  greedy decoding turns a nudge into a different word. Long audio tests therefore
  assert coverage (word counts, how often a sentence came back) and exact text is
  pinned by the tokenizer tests and the single window transcriptions.
- `-transcribe FILE` transcribes one file and exits. It needs no VAD and no
  sway session, so it is the way to check a model install or compare devices
  on identical audio (`-device CPU` versus `-device NPU`). It prints through
  `println`, which goes to **stderr**, so `whispy -transcribe x.wav > out.txt`
  leaves `out.txt` empty; the corpus test reads stderr for that reason.
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
- Adding simd has to keep the toolchain's own `GOEXPERIMENT` default, which is
  why the PKGBUILD appends to `go env GOEXPERIMENT` instead of assigning: this
  go build defaults to `nodwarf5`, and dropping it turns DWARF5 back on, which
  makes `debugedit` log `Unsupported .debug_line directory 0 path
  DW_FORM_0x8` while packaging. `GOEXPERIMENT=simd` alone is fine for tests and
  benchmarks.
- `go build ./...` works either way.
- `go test ./...` covers the parakeet pipeline and the VAD. The audio fixtures
  under `parakeet/testdata` and `silero/testdata` are 16 kHz mono WAV.
- Measuring transcripts against real speech is a separate, skipped test: run
  `eval/fetch.sh`, then `WHISPY_CORPUS=eval/data go test -count=1 -run TestCorpus
  ./parakeet/`. Anything that changes what comes back from the audio — the
  stitching, the decoder, the tokenizer — wants its numbers before and after, and
  `eval/readme.md` says what it measures today and what it cannot see. The corpus
  audio is gitignored, and `WHISPY_CORPUS_LIMIT=6` is the quick pass over a few
  files.
- The suite skips what the machine does not have: the parakeet tests skip without
  the model files, and the corpus test without `WHISPY_CORPUS`. Keep a skip saying
  what it needed, so a run that passes says what it actually checked. The stitching
  tests are the ones that run on any machine, which is why the geometry belongs
  there and not in a model test.
