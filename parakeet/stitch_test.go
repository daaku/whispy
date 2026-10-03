package parakeet

import (
	"fmt"
	"strings"
	"testing"
)

// stitch cuts a capture longer than one encoder window into windows and joins
// what each one says. Getting that wrong loses a stretch of the capture in
// silence, so it is checked here against a decoder written in the test: the
// model gives no account of itself (it stops speaking early, in a different
// place every run, and says nothing about having done so), and it is not
// available to a machine without the model files.
//
// The synthetic capture is a spectrogram whose every bin of every frame holds
// that frame's own number, so a decoder in a test knows exactly what it is
// being made to hear. Words are spoken every fakeWordFrames mel frames, which
// is how the tests count them.

const (
	fakeWordFrames = 20
	fakeWordID     = 100
	fakePeriod     = 5
	fakeWindow     = 1501 // the v3 encoder's frame count
)

// fakeVocab is a vocabulary of word tokens plus the punctuation stitch cuts on.
func fakeVocab() map[int]string {
	vocab := map[int]string{0: "<blank>", fakePeriod: "."}
	for n := range 2000 {
		vocab[wordID(n)] = "\u2581w" + fmt.Sprint(n)
	}
	return vocab
}

func wordID(n int) int { return fakeWordID + n }

func wordName(n int) string { return "w" + fmt.Sprint(n) }

// newCapture returns a spectrogram of frames mel frames in which word n is
// spoken at mel frame n*fakeWordFrames.
func newCapture(frames int) *melFeatures {
	data := make([]float32, melBins*frames)
	for b := range melBins {
		for f := range frames {
			data[b*frames+f] = float32(f)
		}
	}
	return &melFeatures{data: data, frames: frames, timeSteps: frames}
}

// window is a window stitch handed to the decoder.
type window struct {
	offset int
	frames int
	last   bool
}

func (w window) String() string {
	return fmt.Sprintf("%d..%d", w.offset, w.offset+w.frames)
}

// speech is a decoder with a known manner of speaking, in the ways the real one
// has:
//
//   - stopAt: it stops speaking stopAt mel frames into a window and says nothing
//     more of what it hears there. The real model does this, five seconds into a
//     fifteen second window, at a place that changes from one run to the next.
//   - late: the first late words of a window go unsaid, because a decoder
//     started in the middle of a sentence tends to begin speaking late.
//   - silent: it says nothing anywhere, which stitch must survive.
//   - every: a "." follows every nth word, which is the only thing that tells
//     stitch where a sentence ends.
//   - repeat: the capture itself speaks the same phrase over and over, which is
//     the case stitch cannot tell apart from hearing a window twice.
type speech struct {
	tok    *tokenizer
	stopAt int
	// midSentence is how long a window stays silent before it gets its bearings,
	// unless it begins in a pause. This is what a capture of dense speech does to
	// a fresh decoder started in the middle of a sentence: it says nothing over
	// the first seconds of what it is given, and says nothing about having done
	// so. Measured on an audiobook reading: up to four seconds.
	midSentence int
	// stopLast is the stop point of the window that ends the capture, -1 to
	// stop there as well. The audio past the last window's stop point is out of
	// reach: no other window hears it, and the march cannot open another window
	// over it without running past the end of the audio.
	stopLast int
	late     int
	silent   bool
	every    int
	repeat   int
	// bias stamps every token of a window after the first a few frames later
	// than the audio it heard, the way a window that starts at a different
	// alignment does. It is what makes a re-heard word land just after the copy
	// already emitted instead of on top of it.
	bias int

	windows []window
	heard   map[int]bool // mel frame -> a window was given it
	spoke   []int        // mel frames of the words this decoder said, all of them
}

func newSpeech() *speech {
	return &speech{
		tok:         newTokenizer(fakeVocab(), 0),
		stopAt:      -1,
		stopLast:    -1,
		midSentence: -1,
		heard:       map[int]bool{},
	}
}

// speak is the word spoken at mel frame abs, or -1 for none.
func (s *speech) speak(abs int) int {
	if abs < 0 || abs%fakeWordFrames != 0 {
		return -1
	}
	n := abs / fakeWordFrames
	if s.repeat > 0 {
		return wordID(n % s.repeat)
	}
	return wordID(n)
}

// decode is a windowDecode over the capture.
func (s *speech) decode(chunk *melFeatures, isLast bool) ([]int, []tokenTiming, error) {
	offset := 0
	if len(chunk.data) > 0 {
		offset = int(chunk.data[0])
	}
	s.windows = append(s.windows, window{offset: offset, frames: chunk.frames, last: isLast})
	for f := range chunk.frames {
		s.heard[offset+f] = true
	}

	spoken := s.stops(isLast, chunk.frames)
	if s.silent {
		return nil, nil, nil
	}
	bearing := s.bearings(offset)

	var (
		tokens  []int
		timings []tokenTiming
		words   int
	)
	for f := bearing - offset; f < spoken; f++ {
		word := s.speak(offset + f)
		if word < 0 {
			continue
		}
		if words++; words <= s.late {
			continue // it begins speaking late
		}
		frame := f
		if offset > 0 {
			frame += s.bias
		}
		tokens, timings = append(tokens, word), append(timings,
			tokenTiming{token: word, frame: frame})
		s.spoke = append(s.spoke, offset+f)
		if s.every > 0 && words%s.every == 0 {
			tokens, timings = append(tokens, fakePeriod), append(timings,
				tokenTiming{token: fakePeriod, frame: frame + 1})
		}
	}
	return tokens, timings, nil
}

// words says which words of the capture the decoder heard in each window, as
// spoken: nothing after its stop point, nothing among the first ones it missed.
func (s *speech) said() []int {
	var out []int
	for _, w := range s.windows {
		spoken := s.stops(w.last, w.frames)
		words := 0
		for f := range spoken {
			word := s.speak(w.offset + f)
			if word < 0 {
				continue
			}
			if words++; words <= s.late {
				continue
			}
			out = append(out, word)
		}
	}
	return out
}

// lostWords counts the words of the capture that are not in the text.
func lostWords(s *speech, frames int, got []int) int {
	in := map[string]int{}
	for _, tok := range got {
		if tok != fakePeriod {
			in[s.tok.vocab[tok]]++
		}
	}
	lost := 0
	for f := 0; f < frames; f += fakeWordFrames {
		word := s.speak(f)
		if word >= 0 && in[s.tok.vocab[word]] == 0 {
			lost++
		}
	}
	return lost
}

// fakePause is how far before a word a window has to start to be in the pause in
// front of it.
const fakePause = 8

// bearings is the frame the window begins speaking at: where the words start, or
// seconds later when it was dropped into the middle of a sentence.
func (s *speech) bearings(offset int) int {
	// The capture's own start is a beginning, not the middle of a sentence: the
	// speaker starts there.
	if s.midSentence < 0 || offset == 0 {
		return offset
	}
	for f := offset; f < offset+fakePause; f++ {
		if s.speak(f) >= 0 {
			return offset + s.midSentence // it starts mid-word and goes quiet
		}
	}
	return offset
}

// stops is how far a window speaks, in frames of the window.
func (s *speech) stops(last bool, frames int) int {
	if last && s.stopLast < 0 {
		return frames
	}
	if last && s.stopLast >= 0 {
		return min(frames, s.stopLast)
	}
	if s.stopAt >= 0 {
		return min(frames, s.stopAt)
	}
	return frames
}

// text is the words of a token list, "." left out, so a test can read it.
func (s *speech) text(tokens []int) []string {
	var out []string
	for _, tok := range tokens {
		if tok == fakePeriod {
			continue
		}
		out = append(out, strings.TrimPrefix(s.tok.vocab[tok], "\u2581"))
	}
	return out
}

// checkSpokenFrom compares what stitch emitted against the words of the capture
// from a frame on. Words before it belong to a window that would not speak them,
// and no other window has that audio to say them with.
func checkSpokenFrom(
	t *testing.T, s *speech, got []int, from, frames int,
) []string {
	t.Helper()

	text := s.text(got)
	count := map[string]int{}
	for _, w := range text {
		count[w]++
	}
	var missing []string
	for n := from / fakeWordFrames; n < frames/fakeWordFrames; n++ {
		if count[wordName(n)] == 0 {
			missing = append(missing, wordName(n))
		}
	}
	if len(missing) > 0 {
		t.Errorf("%d words lost from %d: %s", len(missing),
			frames/fakeWordFrames-from/fakeWordFrames, list(missing, 12))
		for _, w := range s.windows {
			t.Logf("window %s", w)
		}
	}
	return text
}

// checkSpoken compares what stitch emitted against the words of the capture.
// Order is checked; a word that is missing was not heard by any window, or was
// thrown away at a boundary. A word that appears twice is a stitch that failed
// to see the overlap it was given twice, which is sloppy but not a loss.
func checkSpoken(t *testing.T, s *speech, got []int, frames int) []string {
	t.Helper()

	text := s.text(got)
	count := map[string]int{}
	for _, w := range text {
		count[w]++
	}
	want := frames / fakeWordFrames
	var missing, doubled []string
	for n := range want {
		name := wordName(n)
		switch count[name] {
		case 0:
			missing = append(missing, name)
		case 1:
		default:
			doubled = append(doubled, fmt.Sprintf("%s x%d", name, count[name]))
		}
	}
	if len(missing) > 0 {
		t.Errorf("%d of %d words lost: %s", len(missing), want,
			list(missing, 12))
		for _, w := range s.windows {
			t.Logf("window %s", w)
		}
	}
	if len(doubled) > 0 {
		t.Errorf("%d words spoken twice: %s", len(doubled), list(doubled, 12))
	}
	return text
}

func list(items []string, max int) string {
	if len(items) > max {
		return strings.Join(items[:max], " ") + fmt.Sprintf(" (+%d more)", len(items)-max)
	}
	return strings.Join(items, " ")
}

// fakeModel is a Model with only what stitch reads from it: the encoder's frame
// count and a tokenizer to tell punctuation from words.
func fakeModel(s *speech, frames int) *Model {
	return &Model{
		tokenizer:          s.tok,
		encoderFrames:      frames,
		melPerEncoderFrame: defaultMelPerEncoderFrame,
	}
}

// A capture longer than one window is cut into windows of the encoder's full
// frame count, they reach the end of the audio, and no frame of it is left for
// nobody to hear.
func TestStitchWindows(t *testing.T) {
	for _, tc := range []struct {
		name   string
		frames int
	}{
		{"one window", fakeWindow - 1},
		{"exactly one", fakeWindow},
		{"one frame over", fakeWindow + 1},
		// The capture that started this: 15 seconds gave a last window of 1.7
		// seconds, padded to full size with silence, and it came back with half
		// a sentence.
		{"runt over", fakeWindow + 170},
		{"two windows", 2 * fakeWindow},
		{"a third of a third", 2*fakeWindow + fakeWindow/3},
		{"thirty five seconds", 3501},
		{"a minute", 6001},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newSpeech()
			m := fakeModel(s, fakeWindow)
			if _, err := m.stitch(newCapture(tc.frames), s.decode); err != nil {
				t.Fatal(err)
			}

			if tc.frames <= fakeWindow {
				if len(s.windows) != 1 || !s.windows[0].last {
					t.Fatalf("%d windows for %d frames, want the one window",
						len(s.windows), tc.frames)
				}
				return
			}

			want := 1 + (tc.frames-fakeWindow+chunkAdvance(fakeWindow)-1)/chunkAdvance(fakeWindow)
			if len(s.windows) > want {
				t.Errorf("%d windows, want at most %d: %v",
					len(s.windows), want, s.windows)
			}
			heard := make([]bool, tc.frames)
			for n, w := range s.windows {
				if w.frames != fakeWindow {
					t.Errorf("window %d is %d frames, want the encoder's full %d",
						n, w.frames, fakeWindow)
				}
				if w.last != (n == len(s.windows)-1) {
					t.Errorf("window %d of %d has isLast %v",
						n, len(s.windows), w.last)
				}
				if end := w.offset + w.frames; end > tc.frames {
					t.Errorf("window %d at %d runs %d frames past the audio",
						n, w.offset, end-tc.frames)
				}
				for f := range w.frames {
					heard[w.offset+f] = true
				}
			}
			left := countUnheard(heard)
			if left > 0 {
				t.Errorf("%d of %d mel frames are in no window", left, tc.frames)
			}
		})
	}
}

// A decoder that stops speaking early, which the real one does five seconds
// into a fifteen second window, must not cost the audio after the stop point:
// the next window starts where the words stopped, not where the window did.
func TestStitchWordsPastWhereTheDecoderStopped(t *testing.T) {
	const frames = 3 * fakeWindow
	s := newSpeech()
	s.stopAt = fakeWindow / 3 // five seconds into a fifteen second window
	m := fakeModel(s, fakeWindow)

	got, err := m.stitch(newCapture(frames), s.decode)
	if err != nil {
		t.Fatal(err)
	}
	checkSpoken(t, s, got, frames)

	// The march keeps moving after the stop point instead of giving up on the
	// rest of the audio.
	if len(s.windows) < 3 {
		t.Errorf("%d windows, want the march to keep going: %v",
			len(s.windows), s.windows)
	}
	if last := s.windows[len(s.windows)-1]; last.offset+last.frames < frames {
		t.Errorf("the march stopped at %v, short of the %d frames", last, frames)
	}
}

// A decoder that begins speaking late, which the real one does when a window
// starts in the middle of a sentence, must not cost those words: the window
// before them was told to hold the end back for the next one, and the next one
// re-hears the audio with a sentence in front of it this time.
func TestStitchWordsTheNextWindowBeganLateOn(t *testing.T) {
	const frames = 3 * fakeWindow
	s := newSpeech()
	s.late = 3
	m := fakeModel(s, fakeWindow)

	got, err := m.stitch(newCapture(frames), s.decode)
	if err != nil {
		t.Fatal(err)
	}
	checkSpokenFrom(t, s, got, fakeWindow, frames)
}

// Both at once, since that is how they arrive: a window that starts late and
// stops early says little, and everything it skipped has to be said by someone.
// A decoder dropped into the middle of a sentence spends seconds saying nothing,
// which is how whole passages go missing from dense speech. Stitch has two
// answers and neither is enough alone: it pulls a window back to where a sentence
// starts, and it takes back what the window before left unsaid.
func TestStitchWordsLostToAWindowThatTookTimeToStart(t *testing.T) {
	const frames = 4*fakeWindow + fakeWindow/2
	s := newSpeech()
	s.midSentence = chunkContext(fakeWindow) // four seconds of silent bearing
	s.every = 8                              // sentences, so there are ends to find
	m := fakeModel(s, fakeWindow)

	got, err := m.stitch(newCapture(frames), s.decode)
	if err != nil {
		t.Fatal(err)
	}
	checkSpoken(t, s, got, frames)

	// It got its bearings somewhere by starting at a pause: that is the anchor
	// doing its work, not luck about where the windows landed.
	clean := 0
	for _, w := range s.windows {
		quiet := true
		for f := w.offset; f < w.offset+fakePause; f++ {
			if s.speak(f) >= 0 {
				quiet = false
			}
		}
		if quiet {
			clean++
		}
	}
	if clean == 0 {
		t.Errorf("no window of %d started in a pause: %v", len(s.windows),
			s.windows)
	}
}

// When there is no sentence end near enough to pull back to, the window starts in
// the middle of one and says nothing over its opening, and the only thing between
// that silence and a missing passage is the previous window's own reading of the
// audio. Sentences here are a dozen seconds apart, further back than a window may
// be pulled.
func TestStitchWordsTheWindowBeforeSpoke(t *testing.T) {
	const frames = 4*fakeWindow + fakeWindow/2
	s := newSpeech()
	s.midSentence = fakeWindow / 2 // seven seconds to get its bearings
	s.every = 60                   // a sentence end every twelve seconds
	m := fakeModel(s, fakeWindow)

	got, err := m.stitch(newCapture(frames), s.decode)
	if err != nil {
		t.Fatal(err)
	}
	// What the windows could not cover is bounded, not zero, and that is the
	// geometry: a window's silent opening can run past the audio the window
	// before it had. Filling in what the previous window spoke takes the loss
	// here from 53 words to 20, and nothing beyond that is available to do
	// better, short of starting every window a third of a window earlier, which
	// was measured: 1.5 times the compute for a hundredth of the error rate.
	if lost := lostWords(s, frames, got); lost > 30 {
		t.Errorf("%d words lost of %d, want at most 30: filling in what the "+
			"window before spoke is what holds this down to 20, from the 53 "+
			"lost without it", lost, frames/fakeWordFrames)
	}
}

// A held word the next window says again must not be filled in on top of it.
// The two windows read the same audio and stamp the same word a few frames
// apart, so the fill sees a gap in front of the current reading and would add a
// copy; this is the duplicate a real capture came back with ("leastast").
func TestStitchHeldWordTheNextWindowSaysAgain(t *testing.T) {
	const frames = 2 * fakeWindow
	s := newSpeech()
	s.bias = 2
	m := fakeModel(s, fakeWindow)

	got, err := m.stitch(newCapture(frames), s.decode)
	if err != nil {
		t.Fatal(err)
	}
	checkSpoken(t, s, got, frames)
}

func TestStitchLateAndEarly(t *testing.T) {
	const frames = 4 * fakeWindow
	s := newSpeech()
	s.late, s.stopAt = 2, fakeWindow-fakeWindow/6
	m := fakeModel(s, fakeWindow)

	got, err := m.stitch(newCapture(frames), s.decode)
	if err != nil {
		t.Fatal(err)
	}
	checkSpokenFrom(t, s, got, s.late*fakeWordFrames, frames)
}

// What stitch cannot do: the capture goes on sounding after the last window
// stopped speaking, and there is nowhere to put another window that hears it
// without running past the end of the audio. The words there are the decoder's,
// and this pins the boundary of what the stitching is responsible for.
func TestStitchReachesTheLastWindowStopOnly(t *testing.T) {
	const frames = 4*fakeWindow + fakeWindow/2
	s := newSpeech()
	s.stopAt, s.stopLast = fakeWindow-fakeWindow/6, fakeWindow/2
	m := fakeModel(s, fakeWindow)

	got, err := m.stitch(newCapture(frames), s.decode)
	if err != nil {
		t.Fatal(err)
	}
	// Everything up to where the last window fell silent is there, and only the
	// audio after that is not.
	reached := (frames - fakeWindow) + s.stopLast
	checkSpokenFrom(t, s, got, 0, reached)
	text := s.text(got)
	count := map[string]int{}
	for _, w := range text {
		count[w]++
	}
	for n := (reached + fakeWordFrames) / fakeWordFrames; n < frames/fakeWordFrames; n++ {
		if count[wordName(n)] > 0 {
			t.Errorf("%s was spoken past the last window's stop point", wordName(n))
		}
	}
	if reached/fakeWordFrames >= frames/fakeWordFrames {
		t.Fatal("the test needs audio after the stop point")
	}
}

// A decoder that says nothing must not stop the march or hang it: the audio
// goes unheard either way, and a hung daemon types nothing at all.
func TestStitchThroughASilentDecoder(t *testing.T) {
	const frames = 4 * fakeWindow
	s := newSpeech()
	s.silent = true
	m := fakeModel(s, fakeWindow)

	got, err := m.stitch(newCapture(frames), s.decode)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("a silent decoder emitted %d tokens", len(got))
	}
	if len(s.windows) < 2 {
		t.Errorf("%d windows, want the march to carry on to the end: %v",
			len(s.windows), s.windows)
	}
	if last := s.windows[len(s.windows)-1]; !last.last {
		t.Error("the march ended before the last window")
	}
	// Nothing was said, so the least a window may advance is the only thing
	// moving the march along, and it has to be enough.
	if want := 1 + (4*fakeWindow-fakeWindow+chunkAdvance(fakeWindow)-1)/
		chunkAdvance(fakeWindow); len(s.windows) > want {
		t.Errorf("%d windows for %d frames, want at most %d: the march is taking "+
			"more than the least advance per window", len(s.windows),
			4*fakeWindow, want)
	}
}

// The overlap is handed to two windows. What comes back the second time is the
// same text, and stitch is what notices; a capture that speaks over itself is
// the one case where that is the wrong call, and it is the trade being made.
func TestStitchOverlapOnce(t *testing.T) {
	const frames = 3 * fakeWindow
	s := newSpeech()
	s.every = 8 // sentences, so the cut between windows lands on one
	m := fakeModel(s, fakeWindow)

	got, err := m.stitch(newCapture(frames), s.decode)
	if err != nil {
		t.Fatal(err)
	}
	text := checkSpoken(t, s, got, frames)
	if len(text) == 0 {
		t.Fatal("nothing emitted")
	}
}

// Words are spoken every fakeWordFrames frames of the capture, so a perfect
// decoder over a long capture has each word said once and only once.
func TestStitchCleanDecoder(t *testing.T) {
	for _, frames := range []int{fakeWindow + 1, fakeWindow + 170, 2 * fakeWindow, 3501, 6001} {
		t.Run(fmt.Sprint(frames), func(t *testing.T) {
			s := newSpeech()
			m := fakeModel(s, fakeWindow)
			got, err := m.stitch(newCapture(frames), s.decode)
			if err != nil {
				t.Fatal(err)
			}
			checkSpoken(t, s, got, frames)
		})
	}
}

// holdbackEnd leaves the end of a window to the next one. It cuts where the
// overlap starts, except when a sentence ends after there: a sentence left
// half-emitted loses its words whenever the next window reads the audio shorter.
func TestHoldbackEnd(t *testing.T) {
	s := newSpeech()
	s.every = 4
	m := fakeModel(s, fakeWindow)

	const (
		offset  = 8000
		size    = fakeWindow
		overlap = fakeWindow / 9
	)
	t.Run("mid sentence", func(t *testing.T) {
		var tokens []int
		var timings []tokenTiming
		for f := 0; f < size; f += fakeWordFrames {
			tokens = append(tokens, wordID(f/fakeWordFrames))
			timings = append(timings, tokenTiming{
				token: wordID(f / fakeWordFrames), frame: offset + f,
			})
		}
		end := m.holdbackEnd(tokens, timings, offset, size, overlap)
		// The first word at or after where the overlap starts.
		want := (size - overlap + fakeWordFrames - 1) / fakeWordFrames
		if end != want {
			t.Errorf("cut at token %d, want %d where the overlap starts", end, want)
		}
		if end >= len(tokens) {
			t.Error("nothing held back for the next window")
		}
	})

	t.Run("after a sentence end", func(t *testing.T) {
		// Words, then a "." in the overlap region, then more words.
		var tokens []int
		var timings []tokenTiming
		add := func(tok, frame int) {
			tokens = append(tokens, tok)
			timings = append(timings, tokenTiming{token: tok, frame: offset + frame})
		}
		// A sentence ends just inside the overlap region.
		ends := (size - overlap + fakeWordFrames - 1) / fakeWordFrames * fakeWordFrames
		for f := 0; f < size; f += fakeWordFrames {
			add(wordID(f/fakeWordFrames), f)
			if f == ends {
				add(fakePeriod, f+1)
			}
		}
		end := m.holdbackEnd(tokens, timings, offset, size, overlap)
		if got := timings[end-1].token; got != fakePeriod {
			t.Errorf("cut leaves %q as the last token, want it to end a sentence",
				s.tok.vocab[got])
		}
		if end <= (size-overlap)/fakeWordFrames {
			t.Errorf("cut at %d leaves the sentence end to the next window", end)
		}
		if end >= len(tokens) {
			t.Error("nothing held back for the next window")
		}
	})
}

// The fill gives back what the window before this one held, but not a word this
// window says again at the head of its own reading: the two windows read the
// same audio and stamp the same word a few frames apart, which looks like a gap
// in front of the current reading and would add a duplicate.
func TestRepeatedHeld(t *testing.T) {
	cases := []struct {
		name  string
		held  []int
		chunk []int
		skip  int
		want  int
	}{
		{"all repeated", []int{1, 2}, []int{1, 2, 3}, 0, 2},
		{"after the gated head", []int{1, 2}, []int{9, 1, 2}, 1, 2},
		{"not aligned", []int{1, 2}, []int{9, 1, 2}, 0, 0},
		{"partial", []int{1, 2}, []int{1, 3}, 0, 1},
		{"nothing to compare", []int{1, 2}, nil, 0, 0},
		{"skip past the end", []int{1, 2}, []int{9}, 1, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := repeatedHeld(c.held, c.chunk, c.skip); got != c.want {
				t.Errorf("repeatedHeld = %d, want %d", got, c.want)
			}
		})
	}
}

// The gate that drops what a window re-hears compares token positions against
// window offsets, so both are mel frames; the encoder reports eight times fewer
// frames than it takes. A window's timings are local, dedupChunk makes them
// global.
func TestDedupChunkPositions(t *testing.T) {
	const offset = 3000
	m := fakeModel(newSpeech(), fakeWindow)

	t.Run("a word at the position already emitted is that word", func(t *testing.T) {
		prev := []int{wordID(1), wordID(2), wordID(3)}
		curr := []int{wordID(3), wordID(4)}
		timings := []tokenTiming{{token: wordID(3)}, {token: wordID(4), frame: fakeWordFrames}}
		// The window before this one ended on word 3, at the very position this
		// one puts it. Two windows reading the same audio agree on that.
		if skip := dedupChunk(m, prev, curr, timings, offset, 3000, true); skip != 1 {
			t.Errorf("skip = %d, want the re-heard word dropped", skip)
		}
		if timings[0].frame != offset {
			t.Errorf("first position = %d, want %d: dedupChunk makes the timings "+
				"positions in the whole audio", timings[0].frame, offset)
		}
	})

	t.Run("a run of re-heard words goes even when the positions differ", func(t *testing.T) {
		prev := []int{wordID(1), wordID(2), wordID(3)}
		curr := []int{wordID(2), wordID(3), wordID(4)}
		timings := []tokenTiming{
			{token: wordID(2)}, {token: wordID(3), frame: fakeWordFrames},
			{token: wordID(4), frame: 2 * fakeWordFrames},
		}
		// The previous window's reading ran ahead of where this one starts.
		if skip := dedupChunk(m, prev, curr, timings, offset, 2900, true); skip != 2 {
			t.Errorf("skip = %d, want the two re-heard words dropped", skip)
		}
	})

	t.Run("one re-heard word after the last position is kept", func(t *testing.T) {
		prev := []int{wordID(1), wordID(2), wordID(3)}
		curr := []int{wordID(3), wordID(4)}
		timings := []tokenTiming{{token: wordID(3)}, {token: wordID(4), frame: fakeWordFrames}}
		// A run of one is not matched against the text on purpose: the words of
		// a sentence are not a fingerprint, and a "the" dropped from the wrong
		// side of a boundary is a lost word. This is why the position above,
		// which is a fingerprint, decides it when it can.
		if skip := dedupChunk(m, prev, curr, timings, offset, 2980, true); skip != 0 {
			t.Errorf("skip = %d, want a single word kept", skip)
		}
	})

	t.Run("a window that said nothing before says nothing to drop", func(t *testing.T) {
		curr := []int{wordID(0), wordID(1)}
		timings := []tokenTiming{{token: wordID(0)}, {token: wordID(1), frame: fakeWordFrames}}
		if skip := dedupChunk(m, nil, curr, timings, 0, 0, false); skip != 0 {
			t.Errorf("skip = %d, want nothing dropped", skip)
		}
	})

	t.Run("a repeat the gate leaves past the boundary goes too", func(t *testing.T) {
		// The last window is shifted back to end with the audio, so it re-reads
		// a long stretch. The gate drops what sits at the positions already
		// emitted, and what is left opens with the same words stamped a little
		// later, past where the window-head search looks.
		prev := []int{wordID(0), wordID(1), wordID(2), wordID(3)}
		curr := []int{wordID(9), wordID(1), wordID(2), wordID(3), wordID(4)}
		timings := []tokenTiming{
			{token: wordID(9), frame: 0},
			{token: wordID(1), frame: dedupBoundaryFrames},
			{token: wordID(2), frame: dedupBoundaryFrames + fakeWordFrames},
			{token: wordID(3), frame: dedupBoundaryFrames + 2*fakeWordFrames},
			{token: wordID(4), frame: dedupBoundaryFrames + 3*fakeWordFrames},
		}
		if skip := dedupChunk(m, prev, curr, timings, offset, offset, true); skip != 4 {
			t.Errorf("skip = %d, want the repeat of words 1..3 dropped", skip)
		}
	})

	t.Run("a real repeat far from the boundary is kept", func(t *testing.T) {
		// The same run, but stamped well past the boundary: that is a speaker
		// repeating themselves, not a window re-hearing what is emitted.
		prev := []int{wordID(0), wordID(1), wordID(2), wordID(3)}
		curr := []int{wordID(9), wordID(1), wordID(2), wordID(3), wordID(4)}
		far := 4 * dedupBoundaryFrames
		timings := []tokenTiming{
			{token: wordID(9), frame: 0},
			{token: wordID(1), frame: far},
			{token: wordID(2), frame: far + fakeWordFrames},
			{token: wordID(3), frame: far + 2*fakeWordFrames},
			{token: wordID(4), frame: far + 3*fakeWordFrames},
		}
		if skip := dedupChunk(m, prev, curr, timings, offset, offset, true); skip != 1 {
			t.Errorf("skip = %d, want only the gated word dropped", skip)
		}
	})
}

// FuzzStitch keeps the march honest over any capture length and any decoder
// manner: it finishes, every frame of the audio reaches a window, and it does
// not emit words that were never spoken.
func FuzzStitch(f *testing.F) {
	f.Add(1502, 0, 0, 0)
	f.Add(3001, 500, 2, 8)
	f.Add(9000, 100, 5, 4)
	f.Add(60000, -1, 0, 0)
	fuzzVocab := fakeVocab()
	f.Fuzz(func(t *testing.T, frames, stopAt, late, every int) {
		// Bound the work: the mel is 128 floats a frame.
		if frames < 1 || frames > 40000 {
			t.Skip()
		}
		s := &speech{
			tok:    newTokenizer(fuzzVocab, 0),
			stopAt: clampRange(stopAt, -1, 0, frames),
			late:   clampRange(late, 0, 0, 50),
			every:  clampRange(every, 0, 0, 64),
			heard:  map[int]bool{},
		}
		m := fakeModel(s, fakeWindow)
		got, err := m.stitch(newCapture(frames), s.decode)
		if err != nil {
			t.Fatal(err)
		}

		for n, w := range s.windows {
			if w.frames != fakeWindow && frames > fakeWindow {
				t.Fatalf("window %d is %d frames, want %d", n, w.frames, fakeWindow)
			}
		}
		if frames > fakeWindow {
			heard := make([]bool, frames)
			for _, w := range s.windows {
				for f := range w.frames {
					heard[w.offset+f] = true
				}
			}
			if left := countUnheard(heard); left > 0 {
				t.Fatalf("%d of %d mel frames in no window, %d windows: %v",
					left, frames, len(s.windows),
					s.windows[:min(len(s.windows), 8)])
			}
		}

		spoken := map[int]bool{}
		for _, tok := range s.said() {
			spoken[tok] = true
		}
		for _, f := range s.spoke {
			spoken[s.speak(f)] = true
		}
		for _, tok := range got {
			if tok == fakePeriod {
				continue // the sentence ends are the decoder's mannerism
			}
			if !spoken[tok] {
				t.Fatalf("emitted a word %d that no window spoke", tok)
			}
		}
	})
}

// countUnheard is how many frames no window was given.
func countUnheard(heard []bool) int {
	left := 0
	for _, ok := range heard {
		if !ok {
			left++
		}
	}
	return left
}

// clampRange puts n in range, keeping def when n is below the low end.
func clampRange(n, def, low, high int) int {
	if n < low {
		return def
	}
	return min(n, high)
}
