package parakeet

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"

	"github.com/daaku/serr"
)

// wordBoundary is the SentencePiece word boundary marker U+2581.
const wordBoundary = "\u2581"

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
func (t *tokenizer) decode(tokens []int) string {
	var (
		sb    strings.Builder
		first = true
		last  byte
	)
	for _, id := range tokens {
		if id == t.blankID || t.isControl(id) {
			continue
		}
		piece := t.vocab[id]
		boundary := strings.HasPrefix(piece, wordBoundary)
		if boundary {
			piece = strings.TrimPrefix(piece, wordBoundary)
		}
		if piece == "" {
			continue
		}
		if !first && sb.Len() > 0 && (boundary || startsNumber(piece, last)) {
			sb.WriteByte(' ')
		}
		sb.WriteString(piece)
		last = piece[len(piece)-1]
		first = false
	}
	return sb.String()
}

// startsNumber reports whether the piece opens a number that cannot belong to
// the text before it, which is a digit right after an ASCII letter. Text from
// another script is out of range here: the model is an English one.
func startsNumber(piece string, last byte) bool {
	return len(piece) > 0 && isDigit(piece[0]) && isLetter(last)
}

func isDigit(c byte) bool { return '0' <= c && c <= '9' }

func isLetter(c byte) bool {
	return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z')
}

// isPunctuation reports whether the token is one of the sentence ending
// punctuation tokens, used to break decoder output caching at chunk edges.
func (t *tokenizer) isPunctuation(id int) bool {
	if id < 0 || id >= len(t.vocab) {
		return false
	}
	piece := strings.TrimPrefix(t.vocab[id], wordBoundary)
	return piece == "." || piece == "?" || piece == "!"
}
