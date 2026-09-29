// Package command turns a spoken command into an action and runs it. A
// transcript is matched against a table of rules: exact matches first, then
// patterns with a variable at the end. Anything that matches no rule searches
// the web for the whole transcript, which is the default action.
package command

import (
	"context"
	"net/url"
	"os/exec"
	"strings"

	"github.com/daaku/serr"
)

// Action is the program and arguments a command runs.
type Action struct {
	Program string
	Args    []string
}

// String renders the action as the command line it runs, which is what the
// daemon logs to show which rule matched.
func (a Action) String() string {
	return strings.Join(append([]string{a.Program}, a.Args...), " ")
}

// Run executes the action. The programs whispy calls print to stderr when they
// fail, so that is what the error carries.
func (a Action) Run(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, a.Program, a.Args...)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
			return serr.Errorf("%s: %w: %s", a.Program, err, msg)
		}
		return serr.Errorf("%s: %w", a.Program, err)
	}
	return nil
}

// weatherURL is Dubai, the place this is set up for.
const weatherURL = "https://www.accuweather.com/en/ae/dubai/323091/weather-forecast/323091"

// A rule matches a transcript and builds the action to run. It is handed the
// transcript with outer space and a trailing full stop removed.
type rule func(text string) (Action, bool)

// rules are tried in order, so an exact match wins over a pattern that could
// also match it: "set volume to max" over "set volume to <n>%".
var rules = []rule{
	exact("whats the weather like today", open(weatherURL)),
	exact("set volume to max", noctalia("volume-set", "100")),
	exact("mute speakers", noctalia("volume-mute")),
	volume("reduce volume by ", "volume-down"),
	volume("increase volume by ", "volume-up"),
	volume("set volume to ", "volume-set"),
	searchFor("search for "),
}

// Parse returns the action the transcript asks for. It reports false when no
// rule matched, and the caller should fall back to a search.
func Parse(text string) (Action, bool) {
	t := trim(text)
	for _, r := range rules {
		if a, ok := r(t); ok {
			return a, true
		}
	}
	return Action{}, false
}

// Run runs the action for the transcript, or a search for the whole transcript
// when nothing matched, and returns the action it ran.
func Run(ctx context.Context, text string) (Action, error) {
	a, ok := Parse(text)
	if !ok {
		a = search(trim(text))
	}
	return a, a.Run(ctx)
}

// exact matches the whole transcript, ignoring case and the apostrophes speech
// to text is not consistent about.
func exact(pattern string, a Action) rule {
	return func(text string) (Action, bool) {
		if strings.EqualFold(dropApostrophes(text), pattern) {
			return a, true
		}
		return Action{}, false
	}
}

// volume matches a percentage at the end, written "20%" or "20 percent", and
// runs the noctalia volume command with the number.
func volume(prefix, sub string) rule {
	return func(text string) (Action, bool) {
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
	return func(text string) (Action, bool) {
		query, ok := cut(text, prefix)
		if !ok || query == "" {
			return Action{}, false
		}
		return search(query), true
	}
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
	for i < len(text) && '0' <= text[i] && text[i] <= '9' {
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
