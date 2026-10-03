package parakeet

import (
	"os"
	"path/filepath"
	"testing"
)

func writeVocab(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "vocab.json")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestTokenizerV2(t *testing.T) {
	path := writeVocab(t, `{"0":"<unk>","1":"\u2581foo","2":"bar","3":"."}`)
	tok, err := loadTokenizer(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if tok.blankID != defaultBlankID {
		t.Fatalf("blankID = %d, want %d", tok.blankID, defaultBlankID)
	}
	if got, want := tok.decode([]int{1, 2, 3}), "foobar."; got != want {
		t.Fatalf("decode = %q, want %q", got, want)
	}
	if !tok.isPunctuation(3) || tok.isPunctuation(2) {
		t.Fatal("unexpected punctuation detection")
	}
}

func TestTokenizerV3(t *testing.T) {
	path := writeVocab(t,
		`{"type":"sentencepiece","vocab_size":4,"blank_id":4,
		  "id_to_token":["<unk>","\u2581hello","\u2581world","!"]}`)
	tok, err := loadTokenizer(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if tok.blankID != 4 {
		t.Fatalf("blankID = %d, want 4", tok.blankID)
	}
	if got, want := tok.decode([]int{1, 2, 3}), "hello world!"; got != want {
		t.Fatalf("decode = %q, want %q", got, want)
	}
	if got, want := tok.decode([]int{1, 4, 2}), "hello world"; got != want {
		t.Fatalf("decode with blank = %q, want %q", got, want)
	}
	if !tok.isPunctuation(3) {
		t.Fatal("expected ! to be punctuation")
	}
}

// A device that rounds differently from the CPU can pick a control token, and
// those must never reach the text.
func TestTokenizerControlTokens(t *testing.T) {
	path := writeVocab(t,
		`{"blank_id":6,"id_to_token":["<unk>","<|nospeech|>","\u2581hey","<","\u2581"]}`)
	tok, err := loadTokenizer(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := tok.decode([]int{0, 1, 2, 0, 1}), "hey"; got != want {
		t.Fatalf("decode = %q, want %q", got, want)
	}
	// A lone "<" is text and only a piece wrapped in angle brackets is a
	// control token, so a bare word boundary is kept.
	if got, want := tok.decode([]int{2, 3, 4}), "hey<"; got != want {
		t.Fatalf("decode = %q, want %q", got, want)
	}
	if !tok.isControl(0) || !tok.isControl(1) {
		t.Fatal("control tokens not detected")
	}
	for _, id := range []int{2, 3, 4, 5, tok.blankID, -1, 7} {
		if tok.isControl(id) {
			t.Fatalf("id %d must not be a control token", id)
		}
	}
}

// A control token is one wrapped in angle brackets, whichever of the two word
// boundary spellings sits in front of it, and it stays out of the text either
// way.
func TestTokenizerControlAltBoundary(t *testing.T) {
	path := writeVocab(t, `{"blank_id":3,"id_to_token":[
		"\u0120<|nospeech|>","\u2581<|nospeech|>","\u2581hey"]}`)
	tok, err := loadTokenizer(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !tok.isControl(0) || !tok.isControl(1) {
		t.Fatal("a boundary-prefixed control token was not detected")
	}
	if got, want := tok.decode([]int{0, 1, 2}), "hey"; got != want {
		t.Fatalf("decode = %q, want %q", got, want)
	}
}

func TestTokenizerV3Map(t *testing.T) {
	path := writeVocab(t,
		`{"blank_id":2,"id_to_token":{"0":"\u2581one","1":"\u2581two"}}`)
	tok, err := loadTokenizer(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := tok.decode([]int{0, 1}), "one two"; got != want {
		t.Fatalf("decode = %q, want %q", got, want)
	}
}

func TestTokenizerBlankOverride(t *testing.T) {
	path := writeVocab(t, `{"blank_id":2,"id_to_token":["\u2581one","\u2581two"]}`)
	tok, err := loadTokenizer(path, 7)
	if err != nil {
		t.Fatal(err)
	}
	if tok.blankID != 7 {
		t.Fatalf("blankID = %d, want 7", tok.blankID)
	}
	if n := len(tok.vocab); n != 8 {
		t.Fatalf("vocab size = %d, want 8", n)
	}
}

// The vocabulary has no word boundary marker on a digit, so a number the model
// spells with digits gets its space from the decoder alone.
func TestTokenizerDigits(t *testing.T) {
	path := writeVocab(t, `{"blank_id":9,"id_to_token":[
		"<unk>","\u2581want","4","2","1","0",",",".","\u2581bananas",
		"<pad>","\u2581M","P"]}`)
	tok, err := loadTokenizer(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		tokens []int
		want   string
	}{
		// "want" + "42" is the bug this rule fixes.
		{[]int{1, 2, 3, 8}, "want 42 bananas"},
		// A number that opens the text keeps no leading space.
		{[]int{2, 3}, "42"},
		// The marks inside a number stay inside it.
		{[]int{4, 6, 5, 5, 5, 8}, "1,000 bananas"},
		{[]int{2, 7, 3}, "4.2"},
		// The cost of the rule: a name spelled out letter by letter is split
		// too, because the token stream says nothing more about it.
		{[]int{10, 11, 2}, "MP 4"},
	}
	for _, c := range cases {
		if got := tok.decode(c.tokens); got != c.want {
			t.Fatalf("decode(%v) = %q, want %q", c.tokens, got, c.want)
		}
	}
}

// A vocabulary and a model that disagree about the token ids must not take the
// daemon down, and text that is not English is not out of range: the v3
// vocabulary is built for twenty five languages and its pieces hold Cyrillic,
// Greek and accented Latin.
func TestTokenizerDecodeBoundsAndScripts(t *testing.T) {
	path := writeVocab(t, `{"blank_id":9,"id_to_token":[
		"<unk>","\u2581want","4","\u2581\u043f\u043d","3","\u00e9",
		"\u2581","\u0120","2","<pad>","\u0442"]}`)
	tok, err := loadTokenizer(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name   string
		tokens []int
		want   string
	}{
		// An id outside the vocabulary is skipped rather than indexing past
		// the end of it.
		{"out of range", []int{1, 2, -1, 9000, 1 << 20}, "want 4"},
		{"only out of range", []int{-1, 9000}, ""},
		// A digit after a letter from another script needs the same space the
		// rule exists for. Reading the last byte instead of the last rune made
		// the last byte of a UTF 8 sequence, which is never an ASCII letter.
		{"cyrillic", []int{3, 2}, "пн 4"},
		{"greek", []int{10, 4}, "т 3"},
		{"accented latin", []int{5, 4}, "é 3"},
		// A boundary marker that stands on its own says there is a space here,
		// which is all a reader has when the piece after it carries no marker
		// of its own. Both spellings of the marker mean the same thing.
		{"bare boundary before a bare piece", []int{1, 6, 5}, "want é"},
		{"alt boundary before a bare piece", []int{1, 7, 5}, "want é"},
		// The marker does not stack into a second space, and does not open or
		// close the text with one.
		{"bare boundary before a marked piece", []int{1, 6, 1}, "want want"},
		{"leading bare boundary", []int{6, 1, 2}, "want 4"},
		{"trailing bare boundary", []int{1, 6}, "want"},
		{"boundary piece alone", []int{7}, ""},
	}
	for _, c := range cases {
		if got := tok.decode(c.tokens); got != c.want {
			t.Errorf("%s: decode(%v) = %q, want %q", c.name, c.tokens, got, c.want)
		}
	}
}
