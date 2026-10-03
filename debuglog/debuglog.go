// Package debuglog keeps the audio and the text of each capture, so a
// transcription that looks wrong can be listened to later. whispy only uses it
// with -debug-log. Everything it writes sits in one directory under the cache
// directory, which is safe to delete at any time.
package debuglog

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/daaku/serr"
	"github.com/daaku/whispy/audio"
)

// LogName is the text log inside the directory, one tab separated line per
// capture: the audio file, whether it was a command or dictation, and the text.
// The text comes last so that a replacement from the CSV holding a tab cannot
// shift the other columns.
const LogName = "captures.tsv"

// Keep is how many captures the directory holds before the oldest are deleted.
// Audio is the only thing here with real size, about 32KB for a second of it,
// so this is what stops a long lived daemon from filling the disk.
const Keep = 200

// stamp is the file name format. It sorts as text, which is how trim finds the
// oldest capture, and it goes down to the millisecond, which is as close
// together as two captures can be.
const stamp = "2006-01-02T15-04-05.000"

// Dir returns where the log belongs: a debug folder inside whispy's folder in
// the cache directory, which is $XDG_CACHE_HOME or ~/.cache when that is not
// set. Honoring the variable is why there is no flag for the path.
func Dir() (string, error) {
	cache, err := os.UserCacheDir()
	if err != nil {
		return "", serr.Errorf("debuglog: %w", err)
	}
	return filepath.Join(cache, "whispy", "debug"), nil
}

// Log writes captures to one directory. Everything in it is private: what was
// said, and the audio that says it.
type Log struct {
	Dir string
}

// Open creates the directory, so a cache directory that cannot be written is
// reported at startup rather than once per capture.
func Open(dir string) (*Log, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, serr.Errorf("debuglog: %w", err)
	}
	return &Log{Dir: dir}, nil
}

// Write saves one capture: its audio as a WAV file named after when the capture
// ended, then its line in the text log, then the oldest audio files beyond Keep
// are deleted. A capture with no audio is skipped, since there is nothing to
// hear; a capture with audio and no text is still written, because that is one
// of the things worth debugging.
func (l *Log) Write(
	at time.Time,
	samples []float32,
	commandMode bool,
	text string,
) error {
	if len(samples) == 0 {
		return nil
	}
	name := at.Format(stamp) + ".wav"
	if err := audio.Write(filepath.Join(l.Dir, name), samples); err != nil {
		return err
	}
	mode := "dictation"
	if commandMode {
		mode = "command"
	}
	log, err := os.OpenFile(
		filepath.Join(l.Dir, LogName),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600,
	)
	if err != nil {
		return serr.Errorf("debuglog: %w", err)
	}
	defer log.Close()
	if _, err := fmt.Fprintf(log, "%s\t%s\t%s\n", name, mode, text); err != nil {
		return serr.Errorf("debuglog: %w", err)
	}
	return l.trim()
}

// trim deletes the oldest audio files once there are more than Keep of them.
// ReadDir returns the directory sorted by name, and the names are timestamps.
func (l *Log) trim() error {
	entries, err := os.ReadDir(l.Dir)
	if err != nil {
		return serr.Errorf("debuglog: %w", err)
	}
	var wavs []string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".wav") {
			wavs = append(wavs, e.Name())
		}
	}
	if len(wavs) <= Keep {
		return nil
	}
	// The names sort oldest first, so the excess comes off the front.
	for _, name := range wavs[:len(wavs)-Keep] {
		if err := os.Remove(filepath.Join(l.Dir, name)); err != nil {
			return serr.Errorf("debuglog: %w", err)
		}
	}
	return nil
}
