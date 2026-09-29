// Package command turns a spoken command into an action and runs it. A
// transcript is matched against a table of rules: exact matches first, then
// patterns with a variable at the end. Anything that matches no rule searches
// the web for the whole transcript, which is the default action.
package command

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/daaku/serr"
)

// Action is the program and arguments a command runs.
type Action struct {
	Program string
	Args    []string
	// Notify shows what the program printed as a desktop notification. The
	// alarm rule sets it, so setting an alarm says so.
	Notify bool
}

// String renders the action as the command line it runs, which is what the
// daemon logs to show which rule matched.
func (a Action) String() string {
	return strings.Join(append([]string{a.Program}, a.Args...), " ")
}

// Run executes the action. The programs whispy calls print to stderr when they
// fail, so that is what the error carries. An action that announces itself has
// its output shown as a notification once it succeeds.
func (a Action) Run(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, a.Program, a.Args...)
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
			return serr.Errorf("%s: %w: %s", a.Program, err, msg)
		}
		return serr.Errorf("%s: %w", a.Program, err)
	}
	if a.Notify {
		if msg := strings.TrimSpace(stdout.String()); msg != "" {
			a.notify(ctx, msg)
		}
	}
	return nil
}

// notify shows msg as a desktop notification titled with the program that
// printed it. A notification that cannot be shown is not worth failing the
// command over: whatever it did is already done, so it only says so on stderr.
func (a Action) notify(ctx context.Context, msg string) {
	if err := exec.CommandContext(ctx, "notify-send", a.Program, msg).Run(); err != nil {
		fmt.Fprintf(os.Stderr, "command: notify-send: %v\n", err)
	}
}

// weatherURL is Dubai, the place this is set up for.
const weatherURL = "https://www.accuweather.com/en/ae/dubai/323091/weather-forecast/323091"

// A rule matches a transcript and builds the action to run. It is handed the
// transcript with outer space and a trailing full stop removed, and the time
// the command was given, which the alarm rules need to resolve a clock reading
// that does not say am or pm.
type rule func(text string, now time.Time) (Action, bool)

// rules are tried in order, so an exact match wins over a pattern that could
// also match it: "set volume to max" over "set volume to <n>%".
var rules = []rule{
	exact("whats the weather like today", open(weatherURL)),
	exact("set volume to max", noctalia("volume-set", "100")),
	exact("mute speakers", noctalia("volume-mute")),
	volume("reduce volume by ", "volume-down"),
	volume("increase volume by ", "volume-up"),
	volume("set volume to ", "volume-set"),
	alarm,
	searchFor("search for "),
}

// Parse returns the action the transcript asks for. It reports false when no
// rule matched, and the caller should fall back to a search.
func Parse(text string) (Action, bool) {
	return parse(text, time.Now())
}

// parse is Parse with the time injected, so tests can pin it.
func parse(text string, now time.Time) (Action, bool) {
	t := trim(text)
	for _, r := range rules {
		if a, ok := r(t, now); ok {
			return a, true
		}
	}
	return Action{}, false
}

// Run runs the action for the transcript, or a search for the whole transcript
// when nothing matched, and returns the action it ran.
func Run(ctx context.Context, text string) (Action, error) {
	a, ok := parse(text, time.Now())
	if !ok {
		a = search(trim(text))
	}
	return a, a.Run(ctx)
}

// exact matches the whole transcript, ignoring case and the apostrophes speech
// to text is not consistent about.
func exact(pattern string, a Action) rule {
	return func(text string, _ time.Time) (Action, bool) {
		if strings.EqualFold(dropApostrophes(text), pattern) {
			return a, true
		}
		return Action{}, false
	}
}

// volume matches a percentage at the end, written "20%" or "20 percent", and
// runs the noctalia volume command with the number.
func volume(prefix, sub string) rule {
	return func(text string, _ time.Time) (Action, bool) {
		rest, ok := cut(text, prefix)
		if !ok {
			return Action{}, false
		}
		amount, ok := percent(rest)
		if !ok {
			return Action{}, false
		}
		return noctalia(sub, amount), true
	}
}

// searchFor matches a query at the end and searches the web for it, so the
// words "search for" are not part of the query.
func searchFor(prefix string) rule {
	return func(text string, _ time.Time) (Action, bool) {
		query, ok := cut(text, prefix)
		if !ok || query == "" {
			return Action{}, false
		}
		return search(query), true
	}
}

// triggers introduce an alarm. They are tried longest first, and the article
// is optional: "set a timer in 5 minutes" and "set timer in 5 minutes" are the
// same command.
var triggers = []string{"set an alarm ", "set a timer ", "set alarm ", "set timer ", "remind me "}

// alarm matches the alarm phrasings: a trigger, then "in <duration>" or
// "at <clock>", then an optional "to <label>". It runs snoozer.
func alarm(text string, now time.Time) (Action, bool) {
	rest, ok := cutAny(text, triggers)
	if !ok {
		return Action{}, false
	}
	spec, label := rest, ""
	if before, after, ok := cutFold(rest, " to "); ok {
		spec, label = before, after
	}
	var when string
	if in, ok := cut(spec, "in "); ok {
		d, ok := duration(in)
		if !ok {
			return Action{}, false
		}
		when = "--in=" + d
	} else if at, ok := cut(spec, "at "); ok {
		r, ok := readClock(at)
		if !ok {
			return Action{}, false
		}
		when = "--at=" + r.resolve(now)
	} else {
		return Action{}, false
	}
	args := []string{when}
	if label != "" {
		args = append(args, "--label="+label)
	}
	// snoozer prints the line to confirm the alarm, which is worth showing.
	return Action{Program: "snoozer", Args: args, Notify: true}, true
}

// cutAny returns what follows the first prefix that matches, ignoring case.
func cutAny(text string, prefixes []string) (string, bool) {
	for _, p := range prefixes {
		if rest, ok := cut(text, p); ok {
			return rest, true
		}
	}
	return "", false
}

// cutFold splits text at the first sep, ignoring case, and returns the two
// sides trimmed. It reports false and leaves text whole when sep is absent.
func cutFold(text, sep string) (before, after string, ok bool) {
	for i := 0; i+len(sep) <= len(text); i++ {
		if strings.EqualFold(text[i:i+len(sep)], sep) {
			return strings.TrimSpace(text[:i]), strings.TrimSpace(text[i+len(sep):]), true
		}
	}
	return text, "", false
}

// duration reads "15 minutes" or "2 hours" and returns snoozer's duration
// string, "15m" or "2h".
func duration(text string) (string, bool) {
	end := 0
	for end < len(text) && isDigit(text[end]) {
		end++
	}
	if end == 0 {
		return "", false
	}
	n := text[:end]
	switch strings.ToLower(strings.TrimSpace(text[end:])) {
	case "minute", "minutes", "min", "mins", "m":
		return n + "m", true
	case "hour", "hours", "hr", "hrs", "h":
		return n + "h", true
	}
	return "", false
}

// reading is a clock time as spoken: the digits, and which half of the day
// they are in. mer is empty when the transcript did not say.
type reading struct {
	digits       string // the hours and minutes as written, "11" or "3:20"
	hour, minute int
	mer          string // "am", "pm" or ""
}

// readClock reads a whole clock time: "11", "11am", "3:20", "11:40am",
// "15:20".
func readClock(text string) (reading, bool) {
	end := 0
	for end < len(text) && isDigit(text[end]) {
		end++
	}
	if end == 0 || end > 2 {
		return reading{}, false
	}
	hourEnd := end
	r := reading{digits: text[:end]}
	if end < len(text) && text[end] == ':' {
		mins := end + 1
		for mins < len(text) && isDigit(text[mins]) {
			mins++
		}
		if mins-end-1 != 2 {
			return reading{}, false
		}
		r.minute = number(text[end+1 : mins])
		r.digits = text[:mins]
		end = mins
	}
	r.hour = number(text[:hourEnd])
	switch strings.ToLower(strings.TrimSpace(text[end:])) {
	case "":
	case "am", "pm":
		r.mer = strings.ToLower(strings.TrimSpace(text[end:]))
	default:
		return reading{}, false
	}
	if r.minute > 59 {
		return reading{}, false
	}
	if r.mer != "" {
		if r.hour < 1 || r.hour > 12 {
			return reading{}, false
		}
	} else if r.hour < 1 || r.hour > 23 {
		return reading{}, false
	}
	return r, true
}

// resolve returns the value to hand snoozer. A reading that says am or pm, or
// a 24 hour one like 15:20, is used as it is. A bare reading is the next time
// the clock shows it: at 1pm "3:20" is 3:20pm, at 1am it is 3:20am.
func (r reading) resolve(now time.Time) string {
	if r.mer != "" || r.hour > 12 {
		return r.digits + r.mer
	}
	first := r.hour % 12
	best := time.Duration(1 << 62)
	mer := "am"
	for _, hour := range []int{first, first + 12} {
		at := time.Date(now.Year(), now.Month(), now.Day(), hour, r.minute, 0, 0, now.Location())
		if !at.After(now) {
			at = at.AddDate(0, 0, 1)
		}
		if d := at.Sub(now); d < best {
			best = d
			if hour < 12 {
				mer = "am"
			} else {
				mer = "pm"
			}
		}
	}
	return r.digits + mer
}

// number reads one or two digits.
func number(text string) int {
	n := 0
	for i := 0; i < len(text); i++ {
		n = n*10 + int(text[i]-'0')
	}
	return n
}

// open builds an action that opens a URL in the browser.
func open(u string) Action {
	return Action{Program: "xdg-open", Args: []string{u}}
}

// noctalia builds one of the volume commands.
func noctalia(args ...string) Action {
	return Action{Program: "noctalia", Args: append([]string{"msg"}, args...)}
}

// search builds a DuckDuckGo search for query.
func search(query string) Action {
	return open("https://duckduckgo.com/?q=" + url.QueryEscape(query))
}

// trim removes outer space and a trailing full stop, which speech to text
// often adds: "mute speakers." is the same command as "mute speakers".
func trim(text string) string {
	return strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(text), "."))
}

// cut returns what follows prefix, comparing the prefix ignoring case and
// leaving the rest as written so a query keeps its own case.
func cut(text, prefix string) (string, bool) {
	if len(text) < len(prefix) || !strings.EqualFold(text[:len(prefix)], prefix) {
		return "", false
	}
	return strings.TrimSpace(text[len(prefix):]), true
}

// dropApostrophes removes the apostrophes speech to text sometimes writes, so
// "what's the weather like today" matches "whats the weather like today".
func dropApostrophes(text string) string {
	if !strings.ContainsAny(text, "'\u2019") {
		return text
	}
	return strings.NewReplacer("'", "", "\u2019", "").Replace(text)
}

// percent reads a percentage at the start of text and returns its digits.
func percent(text string) (string, bool) {
	i := 0
	for i < len(text) && isDigit(text[i]) {
		i++
	}
	if i == 0 {
		return "", false
	}
	switch strings.ToLower(strings.TrimSpace(text[i:])) {
	case "%", "percent", "per cent":
		return text[:i], true
	}
	return "", false
}

func isDigit(c byte) bool { return '0' <= c && c <= '9' }
