// Package multiplier writes a spoken multiplier as the Nx notation people
// type: "hundred x or more" becomes "100x or more" and "two hundred x" becomes
// "200x". It runs after words2num, so the scale words it sees are the bare
// ones that did not belong to a number.
package multiplier

// maxWordLen is the length of the longest scale word, so the lower cased copy
// used for the lookup stays on the stack.
const maxWordLen = 8 // trillion

// values are the scale words that stand on their own before an x, and the
// digits they become.
var values = map[string]string{
	"hundred":  "100",
	"thousand": "1,000",
	"million":  "1,000,000",
	"billion":  "1,000,000,000",
	"trillion": "1,000,000,000,000",
}

// Multiplier rewrites spoken multipliers. The zero value is ready to use.
type Multiplier struct{}

// Replace rewrites every multiplier followed by an x: a bare scale word or a
// run of digits. Text without one comes back untouched, and that costs no
// allocations.
func (Multiplier) Replace(s string) string {
	if !hasMultiplier(s) {
		return s
	}
	var (
		out  = make([]byte, 0, len(s))
		last int
	)
	for i := 0; i < len(s); {
		digits, end := multiplierAt(s, i)
		if end == i {
			i++
			continue
		}
		xEnd, ok := xAfter(s, end)
		if !ok || digits == "" {
			i = end
			continue
		}
		out = append(out, s[last:i]...)
		out = append(out, digits...)
		out = append(out, 'x')
		last, i = xEnd, xEnd
	}
	out = append(out, s[last:]...)
	return string(out)
}

// hasMultiplier reports whether s holds a multiplier followed by an x, which
// is what lets Replace return text without one untouched.
func hasMultiplier(s string) bool {
	for i := 0; i < len(s); {
		digits, end := multiplierAt(s, i)
		if end == i {
			i++
			continue
		}
		if digits != "" {
			if _, ok := xAfter(s, end); ok {
				return true
			}
		}
		i = end
	}
	return false
}

// multiplierAt reads a multiplier at i and returns the digits to write for it,
// or "" when the token there is not one. The returned offset is the end of the
// token either way, and equals i only on a character that is neither a letter
// nor a digit.
func multiplierAt(s string, i int) (digits string, end int) {
	if isDigit(s[i]) {
		j := i
		for j < len(s) {
			if isDigit(s[j]) {
				j++
				continue
			}
			if s[j] == ',' && j+1 < len(s) && isDigit(s[j+1]) {
				j++
				continue
			}
			break
		}
		return s[i:j], j
	}
	tok, end := wordAt(s, i)
	if end == i {
		return "", i
	}
	v, ok := value(tok)
	if !ok {
		return "", end
	}
	return v, end
}

// value returns the digits for a scale word, ignoring case.
func value(tok string) (string, bool) {
	if len(tok) == 0 || len(tok) > maxWordLen {
		return "", false
	}
	var buf [maxWordLen]byte
	for i := 0; i < len(tok); i++ {
		buf[i] = lower(tok[i])
	}
	v, ok := values[string(buf[:len(tok)])]
	return v, ok
}

// xAfter returns the offset after the x that follows at i, skipping spaces. It
// has to be a word of its own, so "hundred xy" is left alone, and it must not
// start a number, which keeps "3 x 4" a multiplication.
func xAfter(s string, i int) (int, bool) {
	for i < len(s) && s[i] == ' ' {
		i++
	}
	if i >= len(s) || lower(s[i]) != 'x' {
		return i, false
	}
	end := i + 1
	if end < len(s) && isLetter(s[end]) {
		return i, false
	}
	j := end
	for j < len(s) && s[j] == ' ' {
		j++
	}
	if j < len(s) && isDigit(s[j]) {
		return i, false
	}
	return end, true
}

// wordAt returns the run of letters at i, and the index after it.
func wordAt(s string, i int) (string, int) {
	if i >= len(s) || !isLetter(s[i]) {
		return "", i
	}
	j := i + 1
	for j < len(s) && isLetter(s[j]) {
		j++
	}
	return s[i:j], j
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
