package silero

import (
	"testing"
	"time"
)

// Windows are 32 ms, so these are the counts the default configuration asks
// for. Spelling them out here is the point: a change to a threshold or a
// duration has to be a deliberate one.
func TestWindowCounts(t *testing.T) {
	cases := []struct {
		d    time.Duration
		want int
	}{
		{0, 1},
		{time.Millisecond, 1},
		{windowDuration, 1},
		{250 * time.Millisecond, 8},
		{500 * time.Millisecond, 16},
		{30 * time.Second, 938},
	}
	for _, c := range cases {
		if got := windows(c.d); got != c.want {
			t.Errorf("windows(%v) = %d, want %d", c.d, got, c.want)
		}
	}
	if windowDuration != 32*time.Millisecond {
		t.Errorf("a window is %v, so the durations below are not what they say", windowDuration)
	}
}

// probs returns n windows at p.
func probs(p float32, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = p
	}
	return out
}

// feed adds each slice in turn and reports the read it ended the capture on,
// or -1, and whether speech ever started.
func feed(e *Endpoint, reads ...[]float32) (endedRead int, started bool) {
	for i, r := range reads {
		s, ended := e.Add(r)
		started = started || s
		if ended {
			return i, started
		}
	}
	return -1, started
}

// TestEndpointSilenceNeverStarts is the idle case: nothing is said, so nothing
// starts and nothing ends.
func TestEndpointSilenceNeverStarts(t *testing.T) {
	e := NewEndpoint(DefaultEndpointConfig)
	if started, ended := e.Add(probs(0.01, 100)); started || ended {
		t.Errorf("silence gave started %v ended %v", started, ended)
	}
	if e.Ended() {
		t.Error("the capture ended before it started")
	}
}

// TestEndpointBriefNoiseDoesNotArm is why there is a minimum speech: one window
// over the threshold is a cough or a key clack, and it must not leave the
// capture armed so that the silence after it ends things.
func TestEndpointBriefNoiseDoesNotArm(t *testing.T) {
	e := NewEndpoint(DefaultEndpointConfig)
	// A single loud window, then a long silence.
	_, started := e.Add(append(probs(0.9, 1), probs(0.01, 32)...))
	if started {
		t.Error("one window of noise started the capture")
	}
	if _, ended := e.Add(probs(0.01, 200)); ended {
		t.Error("the silence after a cough ended a capture that never began")
	}
}

// TestEndpointSpeechThenSilence is the ordinary case, and the timing of it is
// what the user feels: the end lands a whole pause after the last words.
func TestEndpointSpeechThenSilence(t *testing.T) {
	const speechWindows = 40 // 1.3 seconds of talk
	e := NewEndpoint(DefaultEndpointConfig)
	if started, ended := e.Add(probs(0.9, speechWindows)); !started || ended {
		t.Fatalf("speech gave started %v ended %v", started, ended)
	}
	// Silence short of the hold keeps the capture open.
	hold := windows(DefaultEndpointConfig.MinSilence)
	for n := 1; n < hold; n++ {
		e.Reset()
		e.Add(probs(0.9, speechWindows))
		if _, ended := e.Add(probs(0.01, n)); ended {
			t.Fatalf("%d windows of silence (%v) ended the capture, want it held to %v",
				n, time.Duration(n)*windowDuration, DefaultEndpointConfig.MinSilence)
		}
	}
	// The hold itself ends it.
	e.Reset()
	e.Add(probs(0.9, speechWindows))
	if _, ended := e.Add(probs(0.01, hold)); !ended {
		t.Errorf("%d windows of silence did not end the capture", hold)
	}
	if !e.Ended() {
		t.Error("Ended() is false after the capture ended")
	}
}

// TestEndpointOneWindowDip is the bug the hysteresis is for: a breath or a
// plosive in the middle of a sentence drops one window below everything, and a
// capture that trusts one answer stops there.
func TestEndpointOneWindowDip(t *testing.T) {
	e := NewEndpoint(DefaultEndpointConfig)
	e.Add(probs(0.9, 30))
	reads := [][]float32{
		probs(0.9, 5),
		append(append(probs(0.02, 1), probs(0.9, 8)...), probs(0.02, 8)...),
	}
	if endedRead, _ := feed(e, reads...); endedRead != -1 {
		t.Errorf("a single window of dip ended the capture on read %d", endedRead)
	}
	// The same capture does end once the silence is real, counted from the
	// speech that came after the dip.
	hold := windows(DefaultEndpointConfig.MinSilence)
	if _, ended := e.Add(probs(0.01, hold)); !ended {
		t.Error("the capture did not end on a full pause after the dip")
	}
}

// TestEndpointHysteresisBand checks what a window between the thresholds does,
// which is nothing: it is neither speech nor silence. Below the exit threshold
// the same window counts against the capture.
func TestEndpointHysteresisBand(t *testing.T) {
	inBand := (Threshold + DefaultEndpointConfig.ExitThreshold) / 2
	for _, c := range []struct {
		name    string
		p       float32
		wantEnd bool
	}{
		{"in the band", inBand, false},
		{"at the exit threshold", DefaultEndpointConfig.ExitThreshold, false},
		{"below the exit threshold", DefaultEndpointConfig.ExitThreshold - 0.01, true},
	} {
		e := NewEndpoint(DefaultEndpointConfig)
		e.Add(probs(0.9, 40))
		hold := windows(DefaultEndpointConfig.MinSilence)
		_, ended := e.Add(probs(c.p, hold))
		if ended != c.wantEnd {
			t.Errorf("%s: %d windows at %v gave ended %v, want %v",
				c.name, hold, c.p, ended, c.wantEnd)
		}
	}
}

// TestEndpointFlickerDoesNotEndEarly is the same property from the other side: a
// probability that sits between the thresholds forever never ends the capture by
// silence, so the max speech bound is what closes it.
func TestEndpointFlickerDoesNotEndEarly(t *testing.T) {
	e := NewEndpoint(DefaultEndpointConfig)
	e.Add(probs(0.9, 10))
	inBand := (Threshold + DefaultEndpointConfig.ExitThreshold) / 2
	if _, ended := e.Add(probs(inBand, windows(DefaultEndpointConfig.MinSilence)*3)); ended {
		t.Error("windows in the hysteresis band ended the capture")
	}
}

// TestEndpointMaxSpeech is the bound on a speaker who does not stop.
func TestEndpointMaxSpeech(t *testing.T) {
	cfg := EndpointConfig{
		ExitThreshold: DefaultEndpointConfig.ExitThreshold,
		MinSpeech:     DefaultEndpointConfig.MinSpeech,
		MinSilence:    DefaultEndpointConfig.MinSilence,
		MaxSpeech:     2 * time.Second,
	}
	e := NewEndpoint(cfg)
	// The bound counts from the window speech held up in, so arm it with
	// exactly enough speech and nothing spare.
	if started, ended := e.Add(probs(0.9, windows(DefaultEndpointConfig.MinSpeech))); !started || ended {
		t.Fatalf("speech gave started %v ended %v", started, ended)
	}
	max := windows(cfg.MaxSpeech)
	_, ended := e.Add(probs(0.95, max-1))
	if ended {
		t.Errorf("%d windows of speech ended the capture, want it to end on %d", max-1, max)
	}
	if _, ended := e.Add(probs(0.95, 1)); !ended {
		t.Errorf("%d windows of speech did not end the capture", max)
	}
}

// TestEndpointMaxSpeechCountsFromStart checks the bound runs from where the
// speech began, so a long talk with short pauses in it is still bounded.
func TestEndpointMaxSpeechCountsFromStart(t *testing.T) {
	cfg := EndpointConfig{
		ExitThreshold: DefaultEndpointConfig.ExitThreshold,
		MinSpeech:     100 * time.Millisecond,
		MinSilence:    100 * time.Millisecond,
		MaxSpeech:     1 * time.Second,
	}
	e := NewEndpoint(cfg)
	endedRead := -1
	// Two seconds of speech broken by one window of silence, which is not a
	// long enough pause to end anything.
	for i := range 70 {
		read := probs(0.9, 5)
		if i%10 == 9 {
			read = append(probs(0.01, 1), probs(0.9, 4)...)
		}
		if _, ended := e.Add(read); ended {
			endedRead = i
			break
		}
	}
	if endedRead < 0 {
		t.Fatal("a ramble with pauses in it never ended")
	}
	if got := time.Duration(endedRead) * 5 * windowDuration; got > cfg.MaxSpeech+10*windowDuration {
		t.Errorf("the capture ran %v on a %v bound", got, cfg.MaxSpeech)
	}
}

// TestEndpointReset clears the state, including an end that already happened.
func TestEndpointReset(t *testing.T) {
	e := NewEndpoint(DefaultEndpointConfig)
	e.Add(probs(0.9, 40))
	_, ended := e.Add(probs(0.01, windows(DefaultEndpointConfig.MinSilence)))
	if !ended {
		t.Fatal("the capture did not end")
	}
	e.Reset()
	if e.Ended() {
		t.Error("Reset left the capture ended")
	}
	if started, ended := e.Add(probs(0.01, 5)); started || ended {
		t.Errorf("after Reset, silence gave started %v ended %v", started, ended)
	}
	// And it works again from the top: the speech run does not carry over.
	if started, _ := e.Add(probs(0.9, 1)); started {
		t.Error("after Reset one window of speech was enough to start")
	}
	if started, _ := e.Add(probs(0.9, windows(DefaultEndpointConfig.MinSpeech))); !started {
		t.Error("after Reset sustained speech did not start the capture")
	}
}

// TestEndpointOnceEndedStaysEnded means the caller only has to act on the signal
// once, however much audio keeps arriving.
func TestEndpointOnceEndedStaysEnded(t *testing.T) {
	e := NewEndpoint(DefaultEndpointConfig)
	e.Add(probs(0.9, 40))
	if _, ended := e.Add(probs(0.01, windows(DefaultEndpointConfig.MinSilence))); !ended {
		t.Fatal("the capture did not end")
	}
	if _, ended := e.Add(probs(0.99, 100)); !ended {
		t.Error("an ended capture came back to life")
	}
}

// TestEndpointEndsInsideOneRead is what happens in the daemon when one read
// covers both the last words and a whole pause: Add has to notice in the same
// call, not one read later.
func TestEndpointEndsInsideOneRead(t *testing.T) {
	e := NewEndpoint(DefaultEndpointConfig)
	e.Add(probs(0.9, 40))
	hold := windows(DefaultEndpointConfig.MinSilence)
	read := append(append(probs(0.9, 3), probs(0.01, hold)...), probs(0.9, 3)...)
	if _, ended := e.Add(read); !ended {
		t.Error("a whole pause inside one read did not end the capture")
	}
}

// TestEndpointEmptyReads is the trivial case that a read with no complete
// window in it changes nothing.
func TestEndpointEmptyReads(t *testing.T) {
	e := NewEndpoint(DefaultEndpointConfig)
	for range 3 {
		if started, ended := e.Add(nil); started || ended {
			t.Fatalf("an empty read gave started %v ended %v", started, ended)
		}
	}
	e.Add(probs(0.9, 40))
	if started, ended := e.Add(nil); started || ended {
		t.Errorf("an empty read after speech gave started %v ended %v", started, ended)
	}
}

// TestEndpointThresholdsAreSane pins the numbers the default configuration
// depends on: the band has to exist, and the bounds have to be in the order
// that makes sense.
func TestEndpointThresholdsAreSane(t *testing.T) {
	cfg := DefaultEndpointConfig
	if cfg.ExitThreshold >= Threshold {
		t.Errorf("exit threshold %v is not below the start threshold %v",
			cfg.ExitThreshold, Threshold)
	}
	if cfg.ExitThreshold < 0 || cfg.ExitThreshold > 1 {
		t.Errorf("exit threshold %v is not a probability", cfg.ExitThreshold)
	}
	if cfg.MinSilence < 200*time.Millisecond || cfg.MinSilence > time.Second {
		t.Errorf("min silence %v is not a pause a speaker takes", cfg.MinSilence)
	}
	if cfg.MaxSpeech <= cfg.MinSilence {
		t.Errorf("max speech %v is not longer than min silence %v", cfg.MaxSpeech, cfg.MinSilence)
	}
}
