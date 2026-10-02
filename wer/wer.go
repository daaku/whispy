// Package wer scores a transcript against a reference the way speech recognition
// does: line up the two strings word by word and count what had to be changed.
//
// It is for measuring, not for the daemon's own work, so the maths is here on its
// own and can be tested without a model. What the counts are for: substitutions
// are the model hearing a different word, insertions are it saying something that
// was not said, and deletions are words of the audio that came back as nothing.
// Only the last one is a bug in whispy rather than in the model, which is why a
// deletion rate over a corpus gets watched on its own.
package wer

import (
	"fmt"
	"slices"
	"strings"
)

// Words is the words a score counts, in order. Text is upper cased, which is how
// the open corpora write their references, and anything that is not a letter, a
// digit or an apostrophe between letters is a word break: the models put their
// own punctuation and the transcripts of the corpora put theirs, and counting the
// difference as an error would be scoring the punctuation.
func Words(text string) []string {
	var out []string
	for _, field := range strings.Fields(text) {
		r := []rune(field)
		w := strings.Builder{}
		for i, c := range r {
			switch {
			case 'A' <= c && c <= 'Z', '0' <= c && c <= '9':
				w.WriteRune(c)
			case 'a' <= c && c <= 'z':
				w.WriteRune(c - 'a' + 'A')
			case c == '\'' && betweenLetters(r, i):
				w.WriteRune(c) // DON'T, and its rarer cousin WE'RE
			}
			// Anything else is a word break: nothing is written, so the two
			// halves of the field fall apart into two words.
		}
		if word := w.String(); word != "" {
			out = append(out, word)
		}
	}
	return out
}

// betweenLetters reports whether an apostrophe has a letter on each side of it,
// which is the only place one belongs inside a word.
func betweenLetters(r []rune, i int) bool {
	if i == 0 || i+1 >= len(r) {
		return false
	}
	is := func(c rune) bool {
		return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z'
	}
	return is(r[i-1]) && is(r[i+1])
}

// Count is what it took to turn reference words into a hypothesis.
type Count struct {
	// Words is the count of reference words, the denominator of the rate.
	Words int
	// Substitutions are reference words the transcript has in place with another
	// word, deletions are reference words the transcript does not have, and
	// insertions are words the transcript has that were not in the reference.
	Substitutions int
	Deletions     int
	Insertions    int
}

// Errors is the number of words that differ, the numerator of the rate.
func (c Count) Errors() int {
	return c.Substitutions + c.Deletions + c.Insertions
}

// Rate is the error rate over the reference words: the word error rate, which can
// come out above 1 when a transcript says a great deal that was not said.
func (c Count) Rate() float64 {
	if c.Words == 0 {
		return 0
	}
	return float64(c.Errors()) / float64(c.Words)
}

// DeletionRate is the share of the reference that came back as nothing, which is
// the audio nobody spoke.
func (c Count) DeletionRate() float64 {
	if c.Words == 0 {
		return 0
	}
	return float64(c.Deletions) / float64(c.Words)
}

// Add totals two counts, which is how a score over a corpus is built: the rate of
// a corpus is over all its words, not the average of its files' rates.
func (c Count) Add(o Count) Count {
	c.Words += o.Words
	c.Substitutions += o.Substitutions
	c.Deletions += o.Deletions
	c.Insertions += o.Insertions
	return c
}

func (c Count) String() string {
	return fmt.Sprintf("%.1f%% (sub %d del %d ins %d of %d words)",
		100*c.Rate(), c.Substitutions, c.Deletions, c.Insertions, c.Words)
}

// Compare lines up a reference with a hypothesis and counts the differences. Both
// are taken as written, see Words for the normalization.
func Compare(reference, hypothesis string) Count {
	return CompareWords(Words(reference), Words(hypothesis))
}

// CompareWords is Compare over words already split out.
func CompareWords(ref, hyp []string) Count {
	c, _, _ := align(ref, hyp)
	return c
}

// Missing reports the reference words the hypothesis does not have, in the order
// they were said, for a report to show which words of the audio came back as
// nothing. It is the alignment speaking, so a transcript that stops mid sentence
// says so, rather than a word count that could be matched up somewhere else.
func Missing(reference, hypothesis string) []string {
	_, deleted, _ := align(Words(reference), Words(hypothesis))
	return deleted
}

// align is the edit distance between two word lists with each operation costed at
// one, walked back to say which operations it used. The table is filled forward,
// then read backwards from the end.
func align(ref, hyp []string) (c Count, deleted, inserted []string) {
	n, m := len(ref), len(hyp)
	d := make([][]int, n+1)
	for i := range d {
		d[i] = make([]int, m+1)
		d[i][0] = i
	}
	for j := range d[0] {
		d[0][j] = j
	}
	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1,
				d[i-1][j-1]+b2i(ref[i-1] != hyp[j-1]))
		}
	}

	c = Count{Words: n}
	for i, j := n, m; i > 0 || j > 0; {
		switch {
		case i > 0 && j > 0 && d[i][j] == d[i-1][j-1]+b2i(ref[i-1] != hyp[j-1]):
			if ref[i-1] != hyp[j-1] {
				c.Substitutions++
			}
			i, j = i-1, j-1
		case i > 0 && d[i][j] == d[i-1][j]+1:
			c.Deletions++
			deleted = append(deleted, ref[i-1])
			i--
		default:
			c.Insertions++
			inserted = append(inserted, hyp[j-1])
			j--
		}
	}
	slices.Reverse(deleted)
	slices.Reverse(inserted)
	return c, deleted, inserted
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
