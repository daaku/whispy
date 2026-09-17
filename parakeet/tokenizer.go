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
func (t *tokenizer) decode(tokens []int) string {
	var sb strings.Builder
	first := true
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
		if !first && boundary && sb.Len() > 0 {
			sb.WriteByte(' ')
		}
		sb.WriteString(piece)
		first = false
	}
	return sb.String()
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
