package multiplier

import (
	"strings"
	"testing"

	"github.com/daaku/words2num"
)

func TestReplace(t *testing.T) {
	cases := []struct{ in, out string }{
		// The example.
		{"Hundred X or more", "100x or more"},

		{"hundred x", "100x"},
		{"this is hundred X faster", "this is 100x faster"},
		{"hundred    x", "100x"},
		{"thousand x", "1,000x"},
		{"million x or more", "1,000,000x or more"},
		{"billion x", "1,000,000,000x"},
		{"trillion x", "1,000,000,000,000x"},
		{"HUNDRED X", "100x"},
		{"a hundred x, another hundred x", "a 100x, another 100x"},
		{"100 x", "100x"},
		{"1,000 x or more", "1,000x or more"},
		{"3 x faster", "3x faster"},
		{"12,345 x", "12,345x"},

		// Left alone.
		{"", ""},
		{"hundred of them", "hundred of them"},
		{"hundred times faster", "hundred times faster"},
		{"hundred xy", "hundred xy"},
		{"x hundred", "x hundred"},
		{"hundred", "hundred"},
		{"thousands x", "thousands x"},
		// A multiplication, not a multiplier.
		{"3 x 4", "3 x 4"},
		{"555 1234 x 99", "555 1234 x 99"},
		{"3 x", "3x"},
	}
	m := Multiplier{}
	for _, c := range cases {
		if got := m.Replace(c.in); got != c.out {
			t.Errorf("Replace(%q) = %q, want %q", c.in, got, c.out)
		}
	}
}

// TestAfterWords2Num checks the pairing the daemon uses. words2num runs first,
// so a scale word that was part of a number is already digits by the time the
// multiplier sees it, and only the bare ones are rewritten.
func TestAfterWords2Num(t *testing.T) {
	var w words2num.Words2Num
	m := Multiplier{}
	cases := []struct{ in, out string }{
		{"hundred x or more", "100x or more"},
		{"one hundred x", "100x"},
		{"two hundred x", "200x"},
		{"one thousand x", "1,000x"},
		{"one million x", "1,000,000x"},
		{"twenty three x", "23x"},
		{"hundred of them", "hundred of them"},
	}
	for _, c := range cases {
		if got := m.Replace(w.Replace(c.in)); got != c.out {
			t.Errorf("after words2num, Replace(%q) = %q, want %q", c.in, got, c.out)
		}
	}
}

func TestReplaceNoAllocations(t *testing.T) {
	inputs := []string{
		"",
		"no multipliers here",
		"hundred of them",
		"hundred times faster",
		"3 x 4",
		strings.Repeat("words that are nothing like digits at all, never here. ", 40),
	}
	m := Multiplier{}
	for _, in := range inputs {
		if got := m.Replace(in); got != in {
			t.Errorf("Replace(%q) = %q", in, got)
		}
		allocs := testing.AllocsPerRun(100, func() {
			m.Replace(in)
		})
		if allocs != 0 {
			t.Errorf("Replace(%q) allocated %g times per run, want 0", in, allocs)
		}
	}
}
