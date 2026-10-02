package parakeet

import (
	"bufio"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/daaku/whispy/audio"
	"github.com/daaku/whispy/wer"
	"github.com/daaku/words2num"
)

// TestCorpus measures transcripts against an open corpus of real speech with a
// transcript of record: LibriStem, set up by eval/fetch.sh. The tests in this
// package that use the shipped fixtures ask whether a particular capture came
// back; this one asks how much of hours of speech came back as nothing, which is
// the question a change to the window stitching has to answer. It is skipped
// without the corpus:
//
//	eval/fetch.sh
//	WHISPY_CORPUS=eval/data go test -count=1 -run TestCorpus ./parakeet/
//
// The thresholds are set a few points over what the corpus measures today, which
// makes them a tripwire rather than a measurement: see eval/readme.md for the
// numbers, and re-run the corpus before moving them. What they watch separately is
// the deletion rate, the share of the reference that came back as silence: any
// word the model heard wrong shows up in the error rate, but a stretch of the
// capture nobody spoke shows up there and nowhere else.
func TestCorpus(t *testing.T) {
	dir := os.Getenv("WHISPY_CORPUS")
	if dir == "" {
		t.Skip("no corpus: run eval/fetch.sh and set WHISPY_CORPUS")
	}
	dir = corpusDir(dir)
	lines := readManifest(t, filepath.Join(dir, "manifest.tsv"))
	if limit := envInt("WHISPY_CORPUS_LIMIT"); limit > 0 && len(lines) > limit {
		lines = lines[:limit]
	}

	m := newTestModel(t)
	tallies := map[string]*tally{}
	start := time.Now()
	for _, line := range lines {
		samples, err := audio.Read(filepath.Join(dir, line.file))
		if err != nil {
			t.Fatalf("%s: %v", line.file, err)
		}
		result, err := m.Transcribe(samples)
		if err != nil {
			t.Fatalf("%s: %v", line.file, err)
		}
		// The corpora write their numbers out as words and the model writes them
		// as digits, so both sides go through the module the daemon uses for that
		// before they are compared. Anything else the daemon does to a transcript
		// is not the model's doing and stays out of the measurement.
		c := wer.Compare(spoken(line.text), spoken(result.Text))

		at := tallies[line.kind]
		if at == nil {
			at = &tally{}
			tallies[line.kind] = at
		}
		at.files++
		at.seconds += float64(len(samples)) / audio.SampleRate
		at.count = at.count.Add(c)
		if c.DeletionRate() > at.worstRate {
			at.worstRate = c.DeletionRate()
			at.worstFile = line.file
			at.worstWords = wer.Missing(line.text, result.Text)
		}
	}
	elapsed := time.Since(start)

	kinds := slices.Sorted(maps.Keys(tallies))
	var audio float64
	for _, at := range tallies {
		audio += at.seconds
	}
	for _, kind := range kinds {
		at := tallies[kind]
		t.Logf("%s: %d files, %.0f seconds of audio, error rate %s", kind,
			at.files, at.seconds, at.count)
		if at.worstFile != "" {
			t.Logf("  most missing: %s, %d of %d words: %s", at.worstFile,
				len(at.worstWords), at.count.Words, words(at.worstWords, 14))
		}

		// A rate here is over all the reference words of the kind, so one bad
		// capture moves it a little; the deletion rate is the one that says audio
		// went unheard.
		switch kind {
		case "utterance":
			if at.count.Rate() > 0.12 {
				t.Errorf("%s: error rate %.1f%% over %d utterances, want under 12%%",
					kind, 100*at.count.Rate(), at.files)
			}
		case "long":
			if at.count.Rate() > 0.15 {
				t.Errorf("%s: error rate %.1f%% over %d long captures, want under "+
					"15%%", kind, 100*at.count.Rate(), at.files)
			}
		}
		if at.count.DeletionRate() > 0.05 {
			t.Errorf("%s: %.1f%% of the reference came back as nothing, want under "+
				"5%%: that is audio nobody spoke", kind,
				100*at.count.DeletionRate())
		}
	}
	t.Logf("%.0f seconds of audio in %s, %.1f times faster than real time",
		audio, elapsed.Truncate(time.Millisecond), audio/elapsed.Seconds())
}

// corpusDir is where the corpus actually is. A test runs in its own package
// directory, so a path given relative to the repository is looked for there too,
// which lets WHISPY_CORPUS=eval/data work from either place.
func corpusDir(dir string) string {
	for _, d := range []string{dir, filepath.Join("..", dir)} {
		if _, err := os.Stat(filepath.Join(d, "manifest.tsv")); err == nil {
			return d
		}
	}
	return dir
}

// spoken is a text with its numbers as digits, which is how the model writes them
// and how the corpora do not.
func spoken(text string) string {
	return words2num.Words2Num{}.Replace(text)
}

// line is one file of the corpus: what kind of capture it is, the audio, and the
// text it is supposed to come back as.
type line struct {
	kind, file, text string
}

// tally is the score of a kind of capture.
type tally struct {
	files   int
	seconds float64
	count   wer.Count

	worstFile  string
	worstRate  float64
	worstWords []string
}

func readManifest(t *testing.T, path string) []line {
	t.Helper()

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("corpus manifest: %v (run eval/fetch.sh)", err)
	}
	defer f.Close()

	var lines []line
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 1<<20), 8<<20) // a long capture's text is a long line
	for n := 1; s.Scan(); n++ {
		fields := strings.Split(s.Text(), "\t")
		if len(fields) != 3 || fields[0] == "" || fields[1] == "" {
			t.Fatalf("%s:%d: want kind, file and text, tab separated: %q",
				path, n, s.Text())
		}
		lines = append(lines, line{kind: fields[0], file: fields[1], text: fields[2]})
	}
	if err := s.Err(); err != nil {
		t.Fatal(err)
	}
	if len(lines) == 0 {
		t.Fatalf("%s is empty", path)
	}
	return lines
}

func envInt(name string) int {
	n, _ := strconv.Atoi(os.Getenv(name))
	return n
}

func words(list []string, max int) string {
	if len(list) > max {
		return strings.Join(list[:max], " ") + " …"
	}
	return strings.Join(list, " ")
}
