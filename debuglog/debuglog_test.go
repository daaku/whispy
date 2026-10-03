package debuglog

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/daaku/whispy/audio"
)

var when = time.Date(2026, 2, 7, 14, 4, 5, 512000000, time.Local)

var samples = []float32{0, 0.5, -0.5, 1, -1, 2}

func TestWrite(t *testing.T) {
	dir := t.TempDir()
	log, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := log.Write(when, samples, false, "buy some milk"); err != nil {
		t.Fatal(err)
	}
	err = log.Write(when.Add(time.Second), samples, true, "set volume to 50 percent")
	if err != nil {
		t.Fatal(err)
	}

	want := "2026-02-07T14-04-05.512.wav\tdictation\tbuy some milk\n" +
		"2026-02-07T14-04-06.512.wav\tcommand\tset volume to 50 percent\n"
	got, err := os.ReadFile(filepath.Join(dir, LogName))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != want {
		t.Fatalf("log = %q, want %q", got, want)
	}

	// The audio lands as a file any tool can read. The round trip through the
	// 16 bit scale is the audio package's own test.
	got2, err := audio.Read(filepath.Join(dir, "2026-02-07T14-04-05.512.wav"))
	if err != nil {
		t.Fatal(err)
	}
	if len(got2) != len(samples) {
		t.Fatalf("%d samples, want %d", len(got2), len(samples))
	}
}

// A capture with audio and no text is worth keeping, and one with neither is
// not.
func TestSkipsCapturesWithNoAudio(t *testing.T) {
	dir := t.TempDir()
	log, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := log.Write(when, nil, false, ""); err != nil {
		t.Fatal(err)
	}
	if err := log.Write(when, []float32{}, false, ""); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("wrote %v, want nothing", entries)
	}
}

// The text log keeps every line, and the audio only for the newest Keep
// captures.
func TestTrim(t *testing.T) {
	dir := t.TempDir()
	log, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	one := []float32{0}
	for i := range Keep + 3 {
		if err := log.Write(when.Add(time.Duration(i)*time.Second), one, false, ""); err != nil {
			t.Fatal(err)
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var wavs []string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".wav") {
			wavs = append(wavs, e.Name())
		}
	}
	if len(wavs) != Keep {
		t.Fatalf("%d captures kept, want %d", len(wavs), Keep)
	}
	if wavs[0] != when.Add(time.Duration(3)*time.Second).Format(stamp)+".wav" {
		t.Fatalf("oldest kept is %q, want the fourth capture", wavs[0])
	}
	if wavs[len(wavs)-1] != when.Add(time.Duration(Keep+2)*time.Second).Format(stamp)+".wav" {
		t.Fatalf("newest kept is %q, want the last capture", wavs[len(wavs)-1])
	}
	b, err := os.ReadFile(filepath.Join(dir, LogName))
	if err != nil {
		t.Fatal(err)
	}
	if lines := strings.Count(string(b), "\n"); lines != Keep+3 {
		t.Fatalf("%d log lines, want %d", lines, Keep+3)
	}
}

// The directory comes from the XDG cache directory, which is why there is no
// flag for the path.
func TestDir(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", filepath.Join(t.TempDir(), "cache"))
	got, err := Dir()
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(os.Getenv("XDG_CACHE_HOME"), "whispy", "debug"); got != want {
		t.Fatalf("Dir = %q, want %q", got, want)
	}
}

// What was said, and the audio that says it, belong to the person who ran
// whispy. The debug log is the one place the daemon keeps private speech on
// disk, so the directory and everything in it is not readable by anyone else
// on the machine.
func TestWritesArePrivate(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "private", "debug")
	log, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := log.Write(when, samples, false, "my password is hunter two"); err != nil {
		t.Fatal(err)
	}
	checks := []struct {
		name string
		path string
	}{
		{"directory", dir},
		{"audio", filepath.Join(dir, when.Format(stamp)+".wav")},
		{"text log", filepath.Join(dir, LogName)},
	}
	for _, c := range checks {
		info, err := os.Stat(c.path)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if mode := info.Mode().Perm(); mode&0o077 != 0 {
			t.Errorf("%s %s is %o, want no access for group or other", c.name, c.path, mode)
		}
	}
	// The parent whispy made for it is private too, since it names the daemon.
	if mode, err := os.Stat(filepath.Dir(dir)); err == nil && mode.Mode().Perm()&0o077 != 0 {
		t.Errorf("the cache directory above it is %o", mode.Mode().Perm())
	}
}
