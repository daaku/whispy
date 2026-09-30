package command

import (
	"slices"
	"strings"
)

// Package-level data for the polite framing stripped before matching. The
// lists come from mining MASSIVE 1.1, the Snips/Sonos NLU benchmark and the
// Stanford Politeness Corpus; the counts are kept in the whispy-phrases notes.
//
// They are lowercase, without apostrophes (the match ignores them) and hold
// only phrases that open or close a command, never one that sits inside it.
// Adding to them is cheap: a phrase only ever removes words when a rule still
// matches what is left.

var politePrefixes = []string{
	"please",
	"please kindly",
	"kindly",
	"hey",
	"hi",
	"hello",
	"yo",
	"oi",
	"excuse me",
	"pardon me",
	"pardon",
	"sorry",
	"listen",
	"look",
	"so",
	"well",
	"ok",
	"okay",
	"alright",
	"all right",
	"right",
	"now",
	"um",
	"uh",
	"er",
	"erm",
	"hmm",
	"mm",
	"ah",
	"oh",
	"like",
	"you know",
	"i mean",
	"basically",
	"actually",
	"honestly",
	"plainly",
	"simply",
	"just",
	"quickly",
	"quick",
	"real quick",
	"can you",
	"can you please",
	"can you kindly",
	"can you possibly",
	"could you",
	"could you please",
	"could you kindly",
	"could you possibly",
	"would you",
	"would you please",
	"would you kindly",
	"will you",
	"will you please",
	"wont you",
	"would you mind",
	"do you mind",
	"do you think you could",
	"do you think you can",
	"could i get you to",
	"can i get you to",
	"can i have you",
	"id like you to",
	"i would like you to",
	"id like for you to",
	"i would like for you to",
	"id love you to",
	"i would love you to",
	"i want you to",
	"i need you to",
	"id appreciate it if you",
	"i would appreciate it if you",
	"id appreciate if you",
	"i would appreciate if you",
	"im going to need you to",
	"im gonna need you to",
	"i was wondering if you could",
	"i was wondering if you can",
	"i was wondering if you would",
	"i was wondering if youd",
	"i was wondering whether you could",
	"i was wondering whether you can",
	"i wonder if you could",
	"i wonder if you can",
	"i wonder whether you could",
	"i wonder whether you can",
	"i was hoping you could",
	"i was hoping you can",
	"i was hoping you would",
	"i was hoping that you could",
	"was wondering if you could",
	"wondering if you could",
	"if you could",
	"if you can",
	"if you would",
	"if you dont mind",
	"if you wouldnt mind",
	"if you have a moment",
	"if you have a second",
	"if you have a minute",
	"if you have time",
	"if you get a chance",
	"if you get the chance",
	"if possible",
	"if thats possible",
	"if its possible",
	"if its not too much trouble",
	"if it isnt too much trouble",
	"would you be able to",
	"would you be willing to",
	"could you maybe",
	"maybe you could",
	"maybe you can",
	"perhaps you could",
	"perhaps you can",
	"might you",
	"might you be able to",
	"any chance you could",
	"any chance you can",
	"is there any chance you could",
	"i dont suppose you could",
	"i dont suppose you can",
	"go ahead and",
	"please go ahead and",
	"just go ahead and",
	"feel free to",
	"be a dear and",
	"be a sweetheart and",
	"be so kind as to",
	"if youd be so kind",
	"if you would be so kind",
	"do me a favor and",
	"do me a favour and",
	"do us a favor and",
	"do us a favour and",
	"help me",
	"please help me",
	"can you help me",
	"can you help me to",
	"could you help me",
	"could you help me to",
	"would you help me",
	"id like",
	"i would like",
	"id like to",
	"i would like to",
	"i want to",
	"i need to",
	"id love to",
	"i would love to",
	"lets",
	"let us",
	"whispy",
	"hey whispy",
	"ok whispy",
	"okay whispy",
	"hey computer",
	"computer",
	"ok google",
	"okay google",
	"hey google",
	"hey siri",
	"siri",
	"alexa",
	"hey alexa",
	"hey assistant",
	"assistant",
	"jarvis",
	"alfred",
	"dear",
	"my friend",
	"friend",
	"sir",
	"maam",
	"madam",
	"mate",
	"buddy",
	"pal",
	"dude",
	"man",
	"bro",
	"boss",
	"chief",
	"honey",
	"sweetie",
	"love",
	"darling",
}

var politeSuffixes = []string{
	"thanks",
	"thank you",
	"thank you so much",
	"thanks so much",
	"thanks a lot",
	"thanks a bunch",
	"thanks a ton",
	"thanks again",
	"thank you again",
	"many thanks",
	"thanks in advance",
	"thank you in advance",
	"cheers",
	"ta",
	"ta very much",
	"much appreciated",
	"appreciated",
	"i appreciate it",
	"id appreciate it",
	"i would appreciate it",
	"i appreciate that",
	"that would be great",
	"thatd be great",
	"that would be nice",
	"thatd be nice",
	"that would be lovely",
	"thatd be lovely",
	"that would be perfect",
	"thatd be perfect",
	"that would be awesome",
	"thatd be awesome",
	"that would be amazing",
	"thatd be amazing",
	"that helps a lot",
	"youre a star",
	"youre the best",
	"nice one",
	"good man",
	"good girl",
	"please and thank you",
	"please",
	"if you please",
	"if you would",
	"if you could",
	"if you can",
	"if you dont mind",
	"if you wouldnt mind",
	"if possible",
	"if thats possible",
	"if its possible",
	"if thats all right",
	"if thats alright",
	"if thats ok",
	"if thats okay",
	"if its not too much trouble",
	"if it isnt too much trouble",
	"would you",
	"could you",
	"can you",
	"will you",
	"wont you",
	"would you mind",
	"do you mind",
	"for me",
	"for us",
	"for me please",
	"please do",
	"now",
	"right now",
	"right away",
	"immediately",
	"asap",
	"as soon as possible",
	"as soon as you can",
	"at your convenience",
	"at your earliest convenience",
	"when you can",
	"when you get a chance",
	"when you get the chance",
	"when you have a chance",
	"when you have a moment",
	"when you have a minute",
	"when you have a second",
	"when you have time",
	"when youre free",
	"when you are free",
	"soon",
	"quickly",
	"real quick",
	"ok",
	"okay",
	"alright",
	"all right",
	"right",
	"yeah",
	"yep",
	"yup",
	"yes",
	"huh",
	"eh",
	"mm",
	"hmm",
	"so",
	"well",
	"then",
	"though",
	"anyway",
	"anyways",
	"now then",
	"man",
	"dude",
	"bro",
	"sir",
	"maam",
	"madam",
	"buddy",
	"pal",
	"boss",
	"chief",
	"honey",
	"sweetie",
	"love",
	"dear",
	"friend",
	"mate",
	"guys",
	"folks",
}

// prefixes and suffixes are tried longest first, so a phrase that starts with
// another phrase is seen first: "could you please" is not cut down to "could
// you" plus "please".
var (
	prefixes = longestFirst(politePrefixes)
	suffixes = longestFirst(politeSuffixes)
)

// longestFirst orders phrases by word count and then by length, most first.
func longestFirst(phrases []string) []string {
	out := slices.Clone(phrases)
	slices.SortStableFunc(out, func(a, b string) int {
		if d := strings.Count(b, " ") - strings.Count(a, " "); d != 0 {
			return d
		}
		return len(b) - len(a)
	})
	return out
}

// stripPrefix removes one polite prefix from text, the longest that matches.
func stripPrefix(text string) (string, bool) {
	for _, p := range prefixes {
		if rest, ok := cutPhrase(text, p); ok {
			return rest, true
		}
	}
	return "", false
}

// stripSuffix removes one polite suffix from text, the longest that matches.
func stripSuffix(text string) (string, bool) {
	for _, p := range suffixes {
		if rest, ok := trimPhrase(text, p); ok {
			return rest, true
		}
	}
	return "", false
}

// cutPhrase returns what follows phrase, which has to be a word of its own, so
// "so" does not match the start of "solve". Something has to be left, or the
// whole command would be gone.
func cutPhrase(text, phrase string) (string, bool) {
	i := 0
	for j := 0; j < len(phrase); j++ {
		i += skipApostrophes(text[i:])
		if i == len(text) || lower(text[i]) != phrase[j] {
			return "", false
		}
		i++
	}
	rest := text[i:]
	if rest == "" || !boundary(rest[0]) {
		return "", false
	}
	return trim(rest), true
}

// trimPhrase returns what precedes phrase, the same way round.
func trimPhrase(text, phrase string) (string, bool) {
	i := len(text)
	for j := len(phrase) - 1; j >= 0; j-- {
		for {
			n := apostropheLenBack(text[:i])
			if n == 0 {
				break
			}
			i -= n
		}
		if i == 0 || lower(text[i-1]) != phrase[j] {
			return "", false
		}
		i--
	}
	before := text[:i]
	if before == "" || !boundary(before[len(before)-1]) {
		return "", false
	}
	return trim(before), true
}

// skipApostrophes returns the length of any apostrophes at the start of text,
// which speech to text is not consistent about: "don't" and "dont" are the
// same word, and the lists are written without them.
func skipApostrophes(text string) int {
	i := 0
	for i < len(text) {
		n := apostropheLen(text[i:])
		if n == 0 {
			break
		}
		i += n
	}
	return i
}

// apostropheLen returns the length of the apostrophe at the start of text, or 0.
func apostropheLen(text string) int {
	if text[0] == '\'' {
		return 1
	}
	if strings.HasPrefix(text, "\u2019") {
		return len("\u2019")
	}
	return 0
}

// apostropheLenBack is apostropheLen for the end of text.
func apostropheLenBack(text string) int {
	if n := len(text); n > 0 && text[n-1] == '\'' {
		return 1
	}
	if strings.HasSuffix(text, "\u2019") {
		return len("\u2019")
	}
	return 0
}

// boundary reports whether c can sit next to a phrase: anything that is not
// part of a word.
func boundary(c byte) bool { return !isLetter(c) && !isDigit(c) }

func isLetter(c byte) bool { return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' }

func lower(c byte) byte {
	if 'A' <= c && c <= 'Z' {
		return c + 'a' - 'A'
	}
	return c
}
