package command

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParse(t *testing.T) {
	cases := []struct {
		in   string
		want string // the action, or "" when nothing matches
	}{
		{"whats the weather like today", "xdg-open " + weatherURL},
		{"Whats The Weather Like Today.", "xdg-open " + weatherURL},
		{"what's the weather like today", "xdg-open " + weatherURL},
		{"  whats the weather like today  ", "xdg-open " + weatherURL},
		{"set volume to max", "noctalia msg volume-set 100"},
		{"Set Volume To Max.", "noctalia msg volume-set 100"},
		{"mute speakers", "noctalia msg volume-mute"},
		{"mute speakers.", "noctalia msg volume-mute"},
		{"reduce volume by 20%", "noctalia msg volume-down 20"},
		{"reduce volume by 20 percent", "noctalia msg volume-down 20"},
		{"Reduce Volume By 20 Percent.", "noctalia msg volume-down 20"},
		{"increase volume by 5%", "noctalia msg volume-up 5"},
		{"increase volume by 20%", "noctalia msg volume-up 20"},
		{"set volume to 50 percent", "noctalia msg volume-set 50"},
		{"set volume to 50%", "noctalia msg volume-set 50"},
		{"search for marvel movies in chronological order",
			"xdg-open https://duckduckgo.com/?q=marvel+movies+in+chronological+order"},
		{"search for Marvel Movies", "xdg-open https://duckduckgo.com/?q=Marvel+Movies"},
		// Patterns that almost match leave it to the search fallback.
		{"set volume to loud", ""},
		{"reduce volume by a lot", ""},
		{"increase volume by 20", ""},
		{"search for", ""},
		{"turn on the lights", ""},
		{"", ""},
	}
	for _, c := range cases {
		a, ok := Parse(c.in)
		got := ""
		if ok {
			got = a.String()
		}
		if got != c.want {
			t.Errorf("Parse(%q) = %q (matched %v), want %q", c.in, got, ok, c.want)
		}
	}
}

// fakePrograms puts recording stubs for the programs commands run on PATH, so
// Run can be tested without a desktop.
func fakePrograms(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	log := filepath.Join(dir, "log")
	script := "#!/bin/sh\nprintf '%s\\n' \"$*\" >> " + log + "\n"
	for _, name := range []string{"noctalia", "xdg-open"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(script), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", dir)
	return log
}

func TestRun(t *testing.T) {
	cases := []struct {
		in   string
		want string // the action, and the arguments the program was called with
	}{
		{"mute speakers", "noctalia msg volume-mute"},
		{"reduce volume by 20 percent", "noctalia msg volume-down 20"},
		{"whats the weather like today", "xdg-open " + weatherURL},
		// No rule matches, so the whole transcript is searched for.
		{"how tall is mount everest", "xdg-open https://duckduckgo.com/?q=how+tall+is+mount+everest"},
	}
	for _, c := range cases {
		log := fakePrograms(t)
		action, err := Run(context.Background(), c.in)
		if err != nil {
			t.Fatal(err)
		}
		if got := action.String(); got != c.want {
			t.Errorf("Run(%q) ran %q, want %q", c.in, got, c.want)
		}
		data, err := os.ReadFile(log)
		if err != nil {
			t.Fatal(err)
		}
		want := strings.TrimPrefix(c.want, action.Program+" ")
		if got := strings.TrimSpace(string(data)); got != want {
			t.Errorf("Run(%q) called %s %q, want %q", c.in, action.Program, got, want)
		}
	}
}

func TestRunFailure(t *testing.T) {
	dir := t.TempDir()
	script := "#!/bin/sh\necho something went wrong >&2\nexit 3\n"
	if err := os.WriteFile(filepath.Join(dir, "xdg-open"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
	_, err := Run(context.Background(), "how tall is mount everest")
	if err == nil {
		t.Fatal("expected an error from a failing program")
	}
	for _, want := range []string{"xdg-open", "exit status 3", "something went wrong"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}
