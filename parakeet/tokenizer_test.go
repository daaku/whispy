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
