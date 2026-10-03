package command

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// testNow is 1pm, so a bare "3:20" is the 3:20pm later today and "11" is
// 11pm tonight.
var testNow = time.Date(2024, 1, 3, 13, 0, 0, 0, time.UTC)

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
		// volume-mute is a toggle, so all three say the same thing.
		{"unmute speakers", "noctalia msg volume-mute"},
		{"Unmute Speakers.", "noctalia msg volume-mute"},
		{"toggle mute", "noctalia msg volume-mute"},
		{"Toggle Mute", "noctalia msg volume-mute"},
		{"reduce volume by 20%", "noctalia msg volume-down 20"},
		{"reduce volume by 20 percent", "noctalia msg volume-down 20"},
		{"Reduce Volume By 20 Percent.", "noctalia msg volume-down 20"},
		{"increase volume by 5%", "noctalia msg volume-up 5"},
		{"increase volume by 20%", "noctalia msg volume-up 20"},
		{"set volume to 50 percent", "noctalia msg volume-set 50"},
		{"set volume to 50%", "noctalia msg volume-set 50"},
		// A percentage out of range lands on the end of the dial rather than
		// being handed to noctalia as it was heard.
		{"set volume to 100 percent", "noctalia msg volume-set 100"},
		{"set volume to 200 percent", "noctalia msg volume-set 100"},
		{"set volume to 1000000 percent", "noctalia msg volume-set 100"},
		{"set volume to 0 percent", "noctalia msg volume-set 0"},
		{"reduce volume by 250%", "noctalia msg volume-down 100"},
		{"increase volume by 500 percent", "noctalia msg volume-up 100"},
		{"search for marvel movies in chronological order",
			"xdg-open https://duckduckgo.com/?q=marvel+movies+in+chronological+order"},
		{"search for Marvel Movies", "xdg-open https://duckduckgo.com/?q=Marvel+Movies"},

		// Alarms. The duration and the clock go to snoozer, and "to" names it.
		{"set alarm in 15 minutes", "snoozer --in=15m"},
		{"set an alarm in 15 minutes", "snoozer --in=15m"},
		{"set timer in 15 minutes", "snoozer --in=15m"},
		{"set a timer in 15 minutes", "snoozer --in=15m"},
		{"Set An Alarm In 15 Minutes.", "snoozer --in=15m"},
		{"set alarm in 1 minute", "snoozer --in=1m"},
		// A lonesome "one" stays a word in the transcript, but these two slots
		// can only mean the count.
		{"set alarm in one minute", "snoozer --in=1m"},
		{"remind me in one hour to stretch",
			"snoozer --in=1h --label=stretch"},
		{"set an alarm at one", "snoozer --at=1am"},
		{"set an alarm at one am", "snoozer --at=1am"},
		{"set alarm in 2 hours", "snoozer --in=2h"},
		{"remind me in 15 minutes to leave for school",
			"snoozer --in=15m --label=leave for school"},
		{"set alarm in 15 minutes to go for a walk",
			"snoozer --in=15m --label=go for a walk"},
		{"set timer in 15 minutes to go for a walk",
			"snoozer --in=15m --label=go for a walk"},
		{"set alarm at 11am", "snoozer --at=11am"},
		{"set alarm at 11 am", "snoozer --at=11am"},
		{"set timer at 11:40am", "snoozer --at=11:40am"},
		{"set alarm at 11am to go for a walk",
			"snoozer --at=11am --label=go for a walk"},
		// A bare clock is the next time it is on the clock, so at 1pm this is
		// 3:20pm. A 24 hour reading says which one it means itself.
		{"remind me at 3:20 to leave for school",
			"snoozer --at=3:20pm --label=leave for school"},
		{"set alarm at 15:20", "snoozer --at=15:20"},
		{"remind me in 15 minutes to walk to school",
			"snoozer --in=15m --label=walk to school"},

		// Patterns that almost match leave it to the search fallback.
		{"set volume to loud", ""},
		{"reduce volume by a lot", ""},
		{"increase volume by 20", ""},
		{"search for", ""},
		{"turn on the lights", ""},
		{"set alarm in a bit", ""},
		{"set alarm at banana", ""},
		{"set alarm at 25", ""},
		{"set alarm at 0", ""},
		{"set alarm", ""},
		{"remind me tomorrow", ""},
		{"", ""},
	}
	for _, c := range cases {
		a, ok := parse(c.in, testNow)
		got := ""
		if ok {
			got = a.String()
		}
		if got != c.want {
			t.Errorf("Parse(%q) = %q (matched %v), want %q", c.in, got, ok, c.want)
		}
	}
}

// TestAlarmClock checks that a clock reading without am or pm lands on the
// next time the clock shows it.
func TestAlarmClock(t *testing.T) {
	cases := []struct {
		now  string
		in   string
		want string
	}{
		{"2024-01-03T13:00:00Z", "remind me at 3:20 to leave for school",
			"snoozer --at=3:20pm --label=leave for school"},
		{"2024-01-03T01:00:00Z", "remind me at 3:20 to leave for school",
			"snoozer --at=3:20am --label=leave for school"},
		{"2024-01-03T23:00:00Z", "set alarm at 3:20", "snoozer --at=3:20am"},
		{"2024-01-03T13:00:00Z", "set alarm at 11", "snoozer --at=11pm"},
		{"2024-01-03T01:00:00Z", "set alarm at 11", "snoozer --at=11am"},
		{"2024-01-03T13:00:00Z", "set alarm at 12", "snoozer --at=12am"},
		{"2024-01-03T13:00:00Z", "set alarm at 12pm", "snoozer --at=12pm"},
		{"2024-01-03T13:00:00Z", "set alarm at 11:59pm", "snoozer --at=11:59pm"},
	}
	for _, c := range cases {
		now, err := time.Parse(time.RFC3339, c.now)
		if err != nil {
			t.Fatal(err)
		}
		a, ok := parse(c.in, now)
		got := ""
		if ok {
			got = a.String()
		}
		if got != c.want {
			t.Errorf("at %s, parse(%q) = %q (matched %v), want %q",
				c.now, c.in, got, ok, c.want)
		}
	}
}

// TestPoliteFraming checks that the framing around a command is ignored, and
// that a suffix only comes off when a rule would not have matched without it.
func TestPoliteFraming(t *testing.T) {
	cases := []struct{ in, want string }{
		// Prefixes.
		{"please mute speakers", "noctalia msg volume-mute"},
		{"Please set volume to max.", "noctalia msg volume-set 100"},
		{"could you mute speakers", "noctalia msg volume-mute"},
		{"could you please mute speakers", "noctalia msg volume-mute"},
		{"can you set volume to 50 percent", "noctalia msg volume-set 50"},
		{"i would like you to set volume to 50 percent",
			"noctalia msg volume-set 50"},
		{"i'd like you to set volume to 50 percent",
			"noctalia msg volume-set 50"},
		{"hey whispy, whats the weather like today", "xdg-open " + weatherURL},
		{"i was wondering if you could mute speakers",
			"noctalia msg volume-mute"},
		{"please can you mute speakers", "noctalia msg volume-mute"},
		{"so now set a timer in 15 minutes", "snoozer --in=15m"},
		{"can you increase volume by 20 percent please",
			"noctalia msg volume-up 20"},

		// Suffixes.
		{"mute speakers please", "noctalia msg volume-mute"},
		{"mute speakers thanks", "noctalia msg volume-mute"},
		{"mute speakers, thanks", "noctalia msg volume-mute"},
		{"mute speakers now", "noctalia msg volume-mute"},
		{"set a timer in 15 minutes please", "snoozer --in=15m"},
		{"set a timer in 15 minutes now", "snoozer --in=15m"},
		{"set an alarm at 11am if you could", "snoozer --at=11am"},
		{"reduce volume by 20 percent please", "noctalia msg volume-down 20"},

		// Both ends, and more than one phrase at an end.
		{"please set a timer in 15 minutes thanks", "snoozer --in=15m"},
		{"could you please mute speakers please", "noctalia msg volume-mute"},

		// A suffix only comes off when a rule would not have matched otherwise,
		// so a label keeps its words.
		{"set a timer in 15 minutes for me", "snoozer --in=15m"},
		{"remind me in 15 minutes to pick up the kids for me",
			"snoozer --in=15m --label=pick up the kids for me"},
		// The alarm matches as it stands here, so the thanks lands in the label.
		// That is the price of not eating words that belong to the command.
		{"remind me at 3:20 to leave for school thanks",
			"snoozer --at=3:20pm --label=leave for school thanks"},

		// Words that only look like framing.
		{"solve the puzzle", ""},
		{"thanks", ""},
		{"sos", ""},
	}
	for _, c := range cases {
		a, ok := parse(c.in, testNow)
		got := ""
		if ok {
			got = a.String()
		}
		if got != c.want {
			t.Errorf("parse(%q) = %q (matched %v), want %q", c.in, got, ok, c.want)
		}
	}
}

// TestSearchQueries checks what a query looks like in the two places a
// transcript becomes one, and that nothing else becomes one at all.
func TestSearchQueries(t *testing.T) {
	// A rule matched, so the query is what it pulled out, without the trailing
	// politeness.
	a, ok := parse("search for cats please", testNow)
	if !ok {
		t.Fatal("search for cats please did not match")
	}
	if want := "xdg-open https://duckduckgo.com/?q=cats"; a.String() != want {
		t.Errorf("got %q, want %q", a.String(), want)
	}

	// "right" is both polite framing and a content word, so it only comes off
	// when a rule then matches.
	if _, ok := parse("right mute speakers", testNow); !ok {
		t.Error("right mute speakers did not match")
	}
	// A query that opens with a word that also reads as framing keeps it, since
	// "right whale" is no kind of request.
	a, ok = parse("search for right whale sounds", testNow)
	if !ok {
		t.Fatal("search for right whale sounds did not match")
	}
	if want := "xdg-open https://duckduckgo.com/?q=right+whale+sounds"; a.String() != want {
		t.Errorf("got %q, want %q", a.String(), want)
	}
}

// TestNoActionUnlessMatched is the privacy rule: a transcript that matches no
// rule runs nothing at all. Command mode hears whatever was said near the
// command key, and private speech must not leave the machine as a search query.
func TestNoActionUnlessMatched(t *testing.T) {
	for _, in := range []string{
		"my password is hunter two",
		"my pin is 1 2 3 4",
		"remind me to call the doctor about the biopsy results",
		"i love you so much",
		"so I was thinking about quitting my job",
		"the meeting is at ten fifteen",
		"how do i fix a door please",
		"right whale sounds",
		"thanks",
		"so",
		"ok",
		"",
	} {
		log := fakePrograms(t)
		action, err := Run(context.Background(), in)
		if err != nil {
			t.Fatalf("Run(%q): %v", in, err)
		}
		if action.Program != "" {
			t.Errorf("Run(%q) ran %q, want no action", in, action.String())
		}
		if data, err := os.ReadFile(log); err == nil {
			t.Errorf("Run(%q) ran something: %q", in, data)
		}
	}
}

// fakePrograms puts recording stubs for the programs commands run on PATH, so
// Run can be tested without a desktop. Every stub appends its arguments to the
// returned log, snoozer also prints the line the notification is built from,
// and notify-send marks its calls so they can be told apart.
func fakePrograms(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	log := filepath.Join(dir, "log")
	record := "printf '%s\\n' \"$*\" >> " + log + "\n"
	stub := func(name, body string) {
		t.Helper()
		script := "#!/bin/sh\n" + body
		if err := os.WriteFile(filepath.Join(dir, name), []byte(script), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// noctalia prints too, so a quiet action can be told from one that shows.
	stub("noctalia", record+"echo 'volume handled'\n")
	stub("xdg-open", record)
	stub("notify-send", "printf 'notify-send %s\\n' \"$*\" >> "+log+"\n")
	stub("snoozer", record+"echo 'Alarm set for Wed 03:04pm: go for a walk'\n")
	t.Setenv("PATH", dir)
	return log
}

// TestRunNotifies checks that an action that announces itself runs the program
// and then shows what it printed.
func TestRunNotifies(t *testing.T) {
	log := fakePrograms(t)
	action, err := Run(context.Background(), "set alarm in 15 minutes to go for a walk")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := action.String(), "snoozer --in=15m --label=go for a walk"; got != want {
		t.Errorf("Run ran %q, want %q", got, want)
	}
	data, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	want := "notify-send snoozer Alarm set for Wed 03:04pm: go for a walk"
	if !strings.Contains(string(data), want) {
		t.Errorf("log %q does not have %q", data, want)
	}
}

// TestRunQuiet checks that an action that does not announce itself shows
// nothing, even though the program prints.
func TestRunQuiet(t *testing.T) {
	log := fakePrograms(t)
	if _, err := Run(context.Background(), "mute speakers"); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "notify-send") {
		t.Errorf("mute speakers showed a notification: %q", data)
	}
}

func TestRun(t *testing.T) {
	cases := []struct {
		in   string
		want string // the action, and the arguments the program was called with
	}{
		{"mute speakers", "noctalia msg volume-mute"},
		{"reduce volume by 20 percent", "noctalia msg volume-down 20"},
		{"whats the weather like today", "xdg-open " + weatherURL},
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
	_, err := Run(context.Background(), "search for how tall is mount everest")
	if err == nil {
		t.Fatal("expected an error from a failing program")
	}
	for _, want := range []string{"xdg-open", "exit status 3", "something went wrong"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}
