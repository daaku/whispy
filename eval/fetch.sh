#!/usr/bin/env bash
#
# fetch.sh lays out open speech corpora for measuring whispy's transcripts, in
# eval/data (gitignored): the audio is large and is not ours to ship, so only the
# numbers it produces end up in the repository.
#
#	eval/fetch.sh [every]   download and build; take every Nth utterance of the
#	                      corpus, 20 by default
#
# LibriStem (OpenSLR SLR31, CC BY 4.0) is a subset of LibriSpeech made for
# regression testing: readings of public domain audiobooks, with a line of text per
# utterance. It is clean studio speech, not what a desktop microphone hears, but it
# is hours of real speech with real transcripts, and it is what tells whether a
# change to the window stitching helps or hurts.
#
# Then measure:
#
#	WHISPY_CORPUS=eval/data go test -count=1 -run TestCorpus ./parakeet/
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data="$here/data"
url=https://openslr.trmal.net/resources/31/dev-clean-2.tar.gz
sum=6d7ab67ac6a1d2c993d050e16d61080d
every="${1:-20}"
long_seconds=75        # a capture long enough to need several encoder windows
long_captures=4

for tool in curl tar md5sum ffmpeg ffprobe; do
	command -v "$tool" >/dev/null || {
		echo "fetch.sh needs $tool" >&2
		exit 1
	}
done

# The download is verified: a corpus that changed underneath a measurement makes
# the numbers meaningless.
mkdir -p "$data"
tar="$data/libristem-dev-clean-2.tar.gz"
if [[ ! -s $tar || $(md5sum "$tar" | cut -d' ' -f1) != $sum ]]; then
	echo "downloading LibriStem dev-clean-2 (126M)"
	curl -fL --retry 3 -o "$tar" "$url"
	echo "$sum  $tar" | md5sum -c -
fi

speech="$data/libristem/LibriSpeech/dev-clean-2"
if [[ ! -d $speech ]]; then
	mkdir -p "$data/libristem"
	tar xzf "$tar" -C "$data/libristem"
fi

# seconds FILE -> the length of an audio file, in whole seconds.
seconds() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" |
	awk '{printf "%d", $1}'
}

# text FILE TRANSCRIPT -> the line of a transcript file that belongs to an audio
# file, empty when there is none.
text() {
	local id=${1##*/}
	grep -m1 "^${id%.flac} " "$2" | cut -d' ' -f2- || true
}

# Every corpus file becomes a 16 kHz mono WAV — what a capture is — with its
# reference text beside it in manifest.tsv: kind, file, reference.
manifest="$data/manifest.tsv"
rm -rf "$data/utterance" "$data/long" "$manifest"
mkdir -p "$data/utterance" "$data/long"
: >"$manifest"

picked=0
n=0
for flac in $(find "$speech" -name '*.flac' | sort); do
	if ((n % every != 0)); then
		n=$((n + 1))
		continue
	fi
	n=$((n + 1))
	id=${flac#"$speech/"}
	speaker=${id%%/*}
	chapter=${id#*/}
	chapter=${chapter%%/*}
	ref=$(text "$flac" "$speech/$speaker/$chapter/$speaker-$chapter.trans.txt")
	# Non-English books are in here; the transcripts are fine, but a corpus that
	# is half another language measures something else, so they go in as they are.
	[[ -n $ref ]] || continue
	ffmpeg -v error -y -i "$flac" -ar 16000 -ac 1 "$data/utterance/$picked.wav"
	printf 'utterance\tutterance/%s.wav\t%s\n' "$picked" "$ref" >>"$manifest"
	picked=$((picked + 1))
done

# A chapter read straight on, which is what the stitching between windows is for:
# one utterance ends mid paragraph and the next begins, with no pause between.
long=0
for transcript in $(find "$speech" -name '*.trans.txt' | sort); do
	if ((long >= long_captures)); then
		break
	fi
	key=${transcript##*/}
	key=${key%.trans.txt}
	list="$data/long/$long-$key.list"
	: >"$list"
	total=0
	ref=""
	used=0
	for flac in $(find "$(dirname "$transcript")" -name "$key-*.flac" | sort); do
		line=$(text "$flac" "$transcript")
		[[ -n $line ]] || continue
		printf "file '%s'\n" "$flac" >>"$list"
		ref="$ref $line"
		total=$((total + $(seconds "$flac")))
		used=$((used + 1))
		if ((total >= long_seconds)); then
			break
		fi
	done
	if ((used < 4)); then
		rm -f "$list"
		continue
	fi
	ffmpeg -v error -y -f concat -safe 0 -i "$list" -ar 16000 -ac 1 \
		"$data/long/$long-$key.wav"
	rm -f "$list"
	printf 'long\tlong/%s-%s.wav\t%s\n' "$long" "$key" "${ref# }" >>"$manifest"
	long=$((long + 1))
done

echo "corpus in $data"
cut -f1 "$manifest" | uniq -c
echo
echo "	WHISPY_CORPUS=$data go test -count=1 -run TestCorpus ./parakeet/"
