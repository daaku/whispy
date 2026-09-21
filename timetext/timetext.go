// Package timetext rewrites clock times written as separate numbers into a
// colon form. Speech to text writes them out ("eleven thirty pm"), words2num
// then turns the words into "11 30 pm", and this makes it "11:30pm".
package timetext

// Time rewrites "11 30 pm" style clock times. The zero value is ready to use.
type Time struct{}

// Replace rewrites the clock times in s, attaching an am/pm marker when one
// follows. Text without a time comes back untouched, and that costs no
// allocations.
func (Time) Replace(s string) string {
	if !hasTime(s) {
		return s
	}
	out := make([]byte, 0, len(s)+8)
	last := 0
	for i := 0; i < len(s); {
		if !isDigit(s[i]) {
			i++
			continue
		}
		m, ok := matchAt(s, i)
		if !ok {
			// Skip the whole digit run: a run that is not an hour cannot
			// start one part way through either.
			for i < len(s) && isDigit(s[i]) {
				i++
			}
			continue
		}
		out = append(out, s[last:i]...)
		out = append(out, m.hour...)
		out = append(out, ':')
		if len(m.minute) == 1 {
			out = append(out, '0')
		}
		out = append(out, m.minute...)
		out = append(out, m.mer...)
		last = m.end
		i = m.end
	}
	out = append(out, s[last:]...)
	return string(out)
}

// hasTime reports whether s holds a clock time, exactly matching what Replace
// rewrites. It makes no allocations, which is what lets Replace return text
// without a time as it is.
func hasTime(s string) bool {
	for i := 0; i < len(s); {
		if !isDigit(s[i]) {
			i++
			continue
		}
		if _, ok := matchAt(s, i); ok {
			return true
		}
		for i < len(s) && isDigit(s[i]) {
			i++
		}
	}
	return false
}

// clock is one matched time: where it ends and how it is written.
type clock struct {
	end    int    // offset just after the time
	hour   string // hour digits, as written
	minute string // minute digits, one or two of them
	mer    string // "am", "pm" or nothing
}

// matchAt matches a clock time whose hour starts at i. The hour is one or two
// digits, minutes follow after a single space, and an am/pm marker may follow
// the minutes. A single digit of minutes is only a time next to a marker,
// since "chapter 9 5" is otherwise as likely as "9 5 pm".
func matchAt(s string, i int) (clock, bool) {
	h := i
	for h < len(s) && isDigit(s[h]) {
		h++
	}
	if h-i > 2 || h >= len(s) || s[h] != ' ' {
		return clock{}, false
	}
	m := h + 1
	mEnd := m
	for mEnd < len(s) && isDigit(s[mEnd]) {
		mEnd++
	}
	if mEnd == m || mEnd-m > 2 {
		return clock{}, false
	}
	if tinyInt(s[i:h]) > 23 || tinyInt(s[m:mEnd]) > 59 {
		return clock{}, false
	}
	mer, end := meridiem(s, mEnd)
	if end == mEnd && mEnd < len(s) && isLetter(s[mEnd]) {
		return clock{}, false // "11 30amsterdam" is one odd word
	}
	if len(s[m:mEnd]) == 1 && mer == "" {
		return clock{}, false
	}
	return clock{end: end, hour: s[i:h], minute: s[m:mEnd], mer: mer}, true
}

// meridiem reads an am/pm marker at i, allowing spaces before it, a dot or a
// space between its letters, and a dot after them, so "a.m.", "p.m." and the
// "p m" speech to text sometimes writes are markers too. It returns the marker
// lower cased and the offset after it, or i when there is no marker.
func meridiem(s string, i int) (string, int) {
	j := i
	for j < len(s) && s[j] == ' ' {
		j++
	}
	if j >= len(s) {
		return "", i
	}
	c := lower(s[j])
	if c != 'a' && c != 'p' {
		return "", i
	}
	k := j + 1
	dotted := false
	if k < len(s) && (s[k] == '.' || s[k] == ' ') {
		dotted = s[k] == '.'
		k++
	}
	if k >= len(s) || lower(s[k]) != 'm' {
		return "", i
	}
	k++
	if dotted && k < len(s) && s[k] == '.' {
		k++
	}
	// "amsterdam" is not a marker, so a marker has to end the word.
	if k < len(s) && isLetter(s[k]) {
		return "", i
	}
	if c == 'p' {
		return "pm", k
	}
	return "am", k
}

// tinyInt reads one or two digits, which matchAt has already checked.
func tinyInt(s string) int {
	n := 0
	for i := 0; i < len(s); i++ {
		n = n*10 + int(s[i]-'0')
	}
	return n
}

func isDigit(c byte) bool { return '0' <= c && c <= '9' }

func isLetter(c byte) bool {
	return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z')
}

func lower(c byte) byte {
	if 'A' <= c && c <= 'Z' {
		return c + 'a' - 'A'
	}
	return c
}
