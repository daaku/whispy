package wer

import (
	"math"
	"strings"
	"testing"
)

func TestWords(t *testing.T) {
	for _, tc := range []struct {
		text string
		want []string
	}{
		{text: "Hello, world!", want: []string{"HELLO", "WORLD"}},
		{text: "aged 23 to 33", want: []string{"AGED", "23", "TO", "33"}},
		{text: "don't  touch", want: []string{"DON'T", "TOUCH"}},
		// Punctuation is a word break, and the models and the corpora put their
		// marks in different places.
		{text: "want42 more", want: []string{"WANT42", "MORE"}},
		{text: "  ", want: nil},
		{text: "'quote' end", want: []string{"QUOTE", "END"}},
	} {
		got := Words(tc.text)
		if strings.Join(got, "|") != strings.Join(tc.want, "|") {
			t.Errorf("Words(%q) = %v, want %v", tc.text, got, tc.want)
		}
	}
}

func TestCompare(t *testing.T) {
	for _, tc := range []struct {
		name     string
		ref, hyp string
		want     Count
	}{
		{
			name: "the same",
			ref:  "the cat sat on the mat",
			hyp:  "The cat sat on the mat.",
			want: Count{Words: 6},
		},
		{
			name: "one word heard for another",
			ref:  "the cat sat on the mat",
			hyp:  "the cat sat on the hat",
			want: Count{Words: 6, Substitutions: 1},
		},
		{
			name: "a passage that came back as nothing",
			ref:  "one two three four five six",
			hyp:  "one two five six",
			want: Count{Words: 6, Deletions: 2},
		},
		{
			name: "said twice",
			ref:  "one two three",
			hyp:  "one two two three",
			want: Count{Words: 3, Insertions: 1},
		},
		{
			name: "nothing at all",
			ref:  "one two three",
			hyp:  "",
			want: Count{Words: 3, Deletions: 3},
		},
		{
			name: "said over a silence",
			ref:  "",
			hyp:  "one two",
			want: Count{Insertions: 2},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := Compare(tc.ref, tc.hyp)
			want := tc.want
			if got != want {
				t.Errorf("Compare(%q, %q) = %+v, want %+v", tc.ref, tc.hyp, got,
					want)
			}
		})
	}
}

// The rate is over the reference words, which is what makes a rate of 1 all the
// words missing and a rate above 1 a transcript longer than the audio.
func TestRates(t *testing.T) {
	c := Compare("one two three four", "one two")
	if c.Rate() != 0.5 {
		t.Errorf("rate = %v, want 0.5", c.Rate())
	}
	if c.DeletionRate() != 0.5 {
		t.Errorf("deletion rate = %v, want 0.5", c.DeletionRate())
	}
	if got := c.String(); !strings.Contains(got, "50.0%") {
		t.Errorf("String() = %q", got)
	}

	c = Compare("", "one two")
	if c.Rate() != 0 {
		t.Errorf("empty reference rate = %v, want 0 rather than a divide by zero",
			c.Rate())
	}
	if math.IsNaN(c.Rate()) || math.IsInf(c.Rate(), 0) {
		t.Error("rate is not a number")
	}
}

// A word is counted against the transcript once per time it was said, so a
// passage that repeats in the audio is not reported as missing because the
// transcript merged it.
func TestMissing(t *testing.T) {
	got := Missing("the train arrived and the train left", "the train arrived")
	if strings.Join(got, " ") != "AND THE TRAIN LEFT" {
		t.Errorf("Missing() = %v, want the words from where the transcript stopped",
			got)
	}
	if got := Missing("one two three", "one two three"); len(got) != 0 {
		t.Errorf("Missing() = %v, want nothing", got)
	}
}

// A corpus score is over all its words: a long capture counts for more than a
// short one, rather than both counting as one file.
func TestAdd(t *testing.T) {
	a := Count{Words: 100, Deletions: 8}.Add(Count{Words: 100, Deletions: 2, Insertions: 3})
	if a.Words != 200 || a.Deletions != 10 || a.Insertions != 3 {
		t.Errorf("Add() = %+v", a)
	}
	if a.DeletionRate() != 0.05 {
		t.Errorf("deletion rate = %v, want 0.05 over the whole", a.DeletionRate())
	}
}

func FuzzCompare(f *testing.F) {
	f.Add("one two three", "one two three")
	f.Add("hello world", "world hello world")
	f.Add("the cat sat", "")
	f.Fuzz(func(t *testing.T, ref, hyp string) {
		c := Compare(ref, hyp)
		r, h := Words(ref), Words(hyp)
		if c.Words != len(r) {
			t.Fatalf("%d reference words counted, %d there", c.Words, len(r))
		}
		// Every reference word is matched, substituted or deleted, and every
		// word only the transcript has is an insertion. Words that match are in
		// neither count, which is why these are bounds and not equalities.
		if c.Substitutions+c.Deletions > c.Words {
			t.Fatalf("sub %d + del %d is more than the %d reference words",
				c.Substitutions, c.Deletions, c.Words)
		}
		if c.Substitutions+c.Insertions > len(h) {
			t.Fatalf("sub %d + ins %d is more than the %d hypothesis words",
				c.Substitutions, c.Insertions, len(h))
		}
		// The fewest differences between two texts is the words one of them has
		// and the other has not, and the count cannot depend on which side is
		// called the reference.
		if n := len(r) - len(h); abs(n) > c.Errors() {
			t.Fatalf("%d errors between texts of %d and %d words, want at least %d",
				c.Errors(), len(r), len(h), abs(n))
		}
		if back := Compare(hyp, ref); back.Errors() != c.Errors() {
			t.Fatalf("%d errors one way, %d the other", c.Errors(), back.Errors())
		}
		if same := Compare(ref, ref); same.Errors() != 0 {
			t.Fatalf("%d errors between a text and itself", same.Errors())
		}
	})
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}
