package parakeet

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/daaku/serr"
)

// wordBoundary is the SentencePiece word boundary marker U+2581.
const wordBoundary = "\u2581"

// altBoundary is the other spelling of the same marker. SentencePiece writes
// it as U+2581 and some conversions of the same vocabulary write it as U+0120,
// which is what the v3 vocabulary holds at one id. Both mean a space.
const altBoundary = "\u0120"

// defaultBlankID is the blank token id used by Parakeet v2 models. v3
// vocabularies carry their own blank id.
const defaultBlankID = 1024

// tokenizer decodes token ids into text using a Parakeet vocabulary file.
// Two layouts exist in the wild: a flat map of id to token (v2) and a
// wrapper object with an id_to_token list or map (v3).
type tokenizer struct {
	vocab   []string
	blankID int
}

func loadTokenizer(path string, blankID int) (*tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, serr.Errorf("read vocabulary %q: %w", path, err)
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(data, &top); err != nil {
		return nil, serr.Errorf("parse vocabulary %q: %w", path, err)
	}

	if blankID == 0 {
		if raw, ok := top["blank_id"]; ok {
			var v int
			if err := json.Unmarshal(raw, &v); err != nil {
				return nil, serr.Errorf("parse blank_id in %q: %w", path, err)
			}
			blankID = v
		}
	}
	if blankID == 0 {
		blankID = defaultBlankID
	}

	vocab := map[int]string{}
	if raw, ok := top["id_to_token"]; ok {
		var list []string
		if err := json.Unmarshal(raw, &list); err == nil {
			for id, tok := range list {
				vocab[id] = tok
			}
			return newTokenizer(vocab, blankID), nil
		}
		var obj map[string]string
		if err := json.Unmarshal(raw, &obj); err != nil {
			return nil, serr.Errorf("parse id_to_token in %q: %w", path, err)
		}
		for k, tok := range obj {
			if id, err := strconv.Atoi(k); err == nil {
				vocab[id] = tok
			}
		}
		return newTokenizer(vocab, blankID), nil
	}

	for k, raw := range top {
		id, err := strconv.Atoi(k)
		if err != nil {
			continue
		}
		var tok string
		if err := json.Unmarshal(raw, &tok); err != nil {
			return nil, serr.Errorf("parse token %d in %q: %w", id, path, err)
		}
		vocab[id] = tok
	}
	return newTokenizer(vocab, blankID), nil
}

func newTokenizer(vocab map[int]string, blankID int) *tokenizer {
	size := blankID + 1
	for id := range vocab {
		if id >= size {
			size = id + 1
		}
	}
	t := &tokenizer{vocab: make([]string, size), blankID: blankID}
	for id, tok := range vocab {
		t.vocab[id] = tok
	}
	return t
}

// isControl reports whether a token id is one of the control tokens rather
// than text. The vocabularies put things like <unk>, <|nospeech|> and the
// language and speaker tags in the low ids, none of which belong in a
// transcript. A device that rounds differently from the CPU can make one of
// them the argmax, most easily at the end of a recording.
func (t *tokenizer) isControl(id int) bool {
	if id < 0 || id >= len(t.vocab) {
		return false
	}
	piece := strings.TrimPrefix(t.vocab[id], wordBoundary)
	return strings.HasPrefix(piece, "<") && strings.HasSuffix(piece, ">")
}

// piece returns what a token id is worth as text and whether it opened with a
// word boundary marker. An id that is not in the vocabulary at all is nothing:
// the decoder argmaxes over the tokens it knows, so this only happens if a
// model or a vocabulary file disagrees with the other, and a transcript is not
// worth a panic over it.
func (t *tokenizer) piece(id int) (text string, boundary bool) {
	if id < 0 || id >= len(t.vocab) || id == t.blankID || t.isControl(id) {
		return "", false
	}
	text = t.vocab[id]
	for _, marker := range []string{wordBoundary, altBoundary} {
		if strings.HasPrefix(text, marker) {
			return strings.TrimPrefix(text, marker), true
		}
	}
	return text, false
}

// decode turns token ids into text, skipping the blank and control tokens. A
// leading word boundary marker becomes a space between words, matching
// SentencePiece decoding for Parakeet vocabularies.
//
// Digits need one rule of their own. The only number pieces in the vocabulary
// are the bare "0".."9", with no word boundary marker on any of them, so a
// number the model spells with digits has no token that could carry the space
// in front of it: "want" + "4" + "2" decodes as "want42". A digit that follows
// a letter therefore starts a word, while digits keep joining each other and
// the marks between them, which is what keeps "1,000", "10:30" and "3.5" in one
// piece. Spelled out names like "MP3" pay for that, and nothing in the token
// stream can tell them apart from the number in "want 42".
//
// A letter is any script's letter, not only an ASCII one. The vocabulary is
// built for twenty five languages and holds pieces like "▁п" and "▁é", so
// reading the rule as ASCII only loses the space for the speakers the model is
// not English for.
func (t *tokenizer) decode(tokens []int) string {
	var (
		sb   strings.Builder
		last rune
		// gap is a boundary marker that stood on its own. It is a space
		// between the words around it, and it has to be kept rather than
		// dropped, because a bare marker in front of a token that carries no
		// marker of its own is the only thing saying there is a space there.
		gap bool
	)
	for _, id := range tokens {
		piece, boundary := t.piece(id)
		if piece == "" {
			if boundary {
				gap = true
			}
			continue
		}
		if sb.Len() > 0 && (boundary || gap || startsNumber(piece, last)) {
			sb.WriteRune(' ')
		}
		sb.WriteString(piece)
		gap = false
		// The last rune, not the last byte: the rule below asks whether the
		// text before a digit is a letter, and in another script the last byte
		// of a piece is the middle of a UTF 8 sequence.
		last, _ = utf8.DecodeLastRuneInString(piece)
	}
	return sb.String()
}

// startsNumber reports whether the piece opens a number that cannot belong to
// the text before it, which is a digit right after a letter.
func startsNumber(piece string, last rune) bool {
	return len(piece) > 0 && isDigit(piece[0]) && unicode.IsLetter(last)
}

func isDigit(c byte) bool { return '0' <= c && c <= '9' }

// isPunctuation reports whether the token is one of the sentence ending
// punctuation tokens, used to break decoder output caching at chunk edges.
func (t *tokenizer) isPunctuation(id int) bool {
	if id < 0 || id >= len(t.vocab) {
		return false
	}
	piece := strings.TrimPrefix(t.vocab[id], wordBoundary)
	return piece == "." || piece == "?" || piece == "!"
}
