# Measuring transcripts

A speech model has no test oracle of its own. `go test ./parakeet/` can say that
one 15 second capture came back, and the scripted decoder in `stitch_test.go` can
say the window stitching loses no audio it could keep, but neither says how much
of an hour of real speech comes back as nothing. That is what this directory is
for: an open corpus of real speech with transcripts of record, scored with
[`wer`](../wer), so a change to the stitching or the decoder is argued with numbers
instead of with how a capture sounded.

## The corpus

**LibriStem** ([OpenSLR SLR31](https://www.openslr.org/31/), CC BY 4.0) is a subset
of LibriSpeech made for regression testing: readings of public domain audiobooks,
`dev-clean-2` here, 1089 utterances over 27 speakers, with a line of text per
utterance. It is free to download and free to redistribute, but 126 MB of audio has
no business in a repository, so it is fetched on demand into `data/` (gitignored)
and only the numbers come back here.

```sh
eval/fetch.sh                    # every 20th utterance, plus 4 long captures
eval/fetch.sh 5                  # more of them
WHISPY_CORPUS=eval/data go test -count=1 -run TestCorpus ./parakeet/
```

`fetch.sh` verifies the download by md5 (a corpus that changed underneath a
measurement makes the numbers meaningless), converts to the 16 kHz mono WAV a
capture is, and writes `data/manifest.tsv`: kind, file, reference text, tab
separated.

| kind        | what it is                                                   | what it measures                   |
| ----------- | ------------------------------------------------------------ | ---------------------------------- |
| `utterance` | one sentence, 2 to 16 seconds                                | the model, and the single window   |
| `long`      | a chapter read straight on, 75 seconds and about six windows | the stitching between windows      |

`WHISPY_CORPUS_LIMIT=6` runs a few files while iterating. `PARAKEET_DEVICE=GPU`
(or `NPU`, with `PARAKEET_DECODER_DEVICE=CPU`) scores the same audio on another
device.

## What it says today

Ryzen 9 5900X, `CPU` device, `eval/fetch.sh` at its default step, 2026-10-02:

| kind        | files | audio  | reference words | word error rate      | came back as nothing |
| ----------- | ----- | ------ | --------------- | -------------------- | -------------------- |
| `utterance` | 55    | 343 s  | 985             | 2.5% (sub 20 del 4)  | 0.4%                 |
| `long`      | 4     | 343 s  | 909             | 3.2% (sub 17 del 7)  | 0.9%                 |

686 seconds of audio in 43 seconds, sixteen times faster than real time, model
load included. Before the windows were pulled to sentence starts and a window's silence
filled in from the window before it, the long captures measured 7.6% and **5.3%**
came back as nothing, and the test fails that build at its 5% tripwire.

The test trips at 12% error rate over the utterances, 15% over the long captures,
and 5% came-back-as-nothing for both. Those are tripwires, not measurements: they
are a few points over today's numbers so that losing a stretch of somebody's
capture stops a build. Move them by re-running the corpus, not by moving the
number.

## Another implementation

The same OpenVINO models have another port, FluidInference's `eddy` (C++), which
whispy's decoder was first ported from. It is worth running beside whispy because
it shows where an implementation of this model goes wrong: short utterances come
out the same, but its long-audio chunking loses a large part of a capture, which
is the failure the stitching here exists to prevent. Building `eddy`'s
`parakeet_cli` and running both on the four `long` captures, 2026-10-03 (`eddy`'s
counts were the same on two runs; whispy's are from after the boundary fixes):

| capture       | reference words | whispy | eddy |
| ------------- | --------------- | ------ | ---- |
| 0-1272-135031 | 231             | 226    | 181  |
| 1-1272-141231 | 204             | 206    | 56   |
| 2-1462-170142 | 228             | 228    | 74   |
| 3-1462-170145 | 246             | 247    | 157  |

`eddy` reads its models from `$XDG_CACHE_HOME/eddy/models/parakeet-v3/files`, so
link the IR files there and run `parakeet_cli FILE --model parakeet-v3 --device
CPU`. It is a check on the decode loop, not a test, and not a fair reading of
`eddy`: the two detokenize differently (it drops a standalone `▁` and writes a
number with no space in front of it, which whispy fixes), and on audio short
enough for one encoder window the two agree word for word. What it measures is
that the long-audio geometry is the part of a port that goes wrong, which is why
that is where whispy's tests are.

## What it cannot tell you

Both sides go through `words2num` first, because the corpora write `SEVEN` where
the model writes `7`, and counting that as a word deleted and a word inserted
inflates the error rate and, worse, inflates the deletion rate, which is the number
that watches for unheard audio. Every figure quoted in the repository is with that
folding in place; earlier notes of mine that said 8.3% came from counting spelled
out numbers as unheard audio.

What the corpus is not, and no corpus is:

- It is studio speech read aloud by volunteer readers, not a desk microphone with
  a keyboard, a fan and a room behind it. Expect real dictation to be worse, and
  use it for comparing two builds of whispy, not for promising anything to a user.
- The reference is another human's transcript of the audio, so when the model and
  the reader disagree about a word, the model is charged. A rate of zero is not the
  goal and will not be reached.
- Some of these books are not English ("IL POPOLO È UNA BESTIA"), which is a fair
  test of an English model and will cost a few substitutions.
- It scores words, so it cannot see what the daemon does around the transcript:
  the command parsing, the politeness that gets ignored, the `11 30 pm` that becomes
  `11:30pm`. Those have their own tests over text.

Worth adding, when there is a reason to: a set of noise and music to check the
model does not speak over it (ESC-50 is 600 clips, easy to fetch), speech from
more than a few accents, and — the one that would actually be worth its weight —
captures donated from `-debug-log`, which are the audio the daemon really gets.
