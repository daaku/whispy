package silero

import "time"

// An Endpoint decides when a captured utterance is over. The model answers one
// question per window - speech or not - and a capture that trusts a single
// answer ends on the first breath in the middle of a sentence. So the decision
// here is the one the upstream consumers of this model make: speech has to hold
// up before it counts, silence has to hold up before it ends it, and there is a
// bound on how long speech can run whether or not it stops.
//
// The shape of it is snakers4/silero-vad's VADIterator and sherpa-onnx's
// silero-vad-model: a window at or above Threshold is speech, a window below the
// exit threshold is silence, and a window in between leaves things as they were.
// That band between the two thresholds is the hysteresis, and it is what stops a
// probability hovering around one threshold from flickering a capture open and
// shut.
//
// Everything is counted in windows, because every window is WindowSamples
// samples and so a window is a fixed 32 ms of audio.

// windowDuration is how long a window is: WindowSamples at SampleRate.
const windowDuration = time.Duration(WindowSamples) * time.Second / SampleRate

// EndpointConfig configures an Endpoint. Use DefaultEndpointConfig unless there
// is a reason to disagree with it.
type EndpointConfig struct {
	// ExitThreshold is the probability below which a window counts as silence
	// once speech has started. Upstream takes it as the start threshold minus
	// 0.15, which for 0.5 is 0.35. It has to be below Threshold or the band
	// where nothing happens is empty, which is the flickering this avoids.
	ExitThreshold float32
	// MinSpeech is how long speech has to hold before silence can end it, which
	// keeps a cough or a keyboard clack from arming the end of a capture that
	// never really began.
	MinSpeech time.Duration
	// MinSilence is how long silence has to hold before the capture ends. This
	// is the pause a speaker takes between thoughts: ending too soon costs the
	// rest of the sentence, waiting too long only costs a moment.
	MinSilence time.Duration
	// MaxSpeech ends a capture that has been speech all the way through, so a
	// ramble is bounded even though nobody stopped talking. sherpa-onnx caps
	// this at 20 seconds; this is a dictation daemon, so it gets more rope.
	MaxSpeech time.Duration
}

// DefaultEndpointConfig is what command mode captures with: upstream's
// thresholds, a quarter second of speech that is clearly speech, and half a
// second of silence that is clearly a pause rather than a stopping place.
var DefaultEndpointConfig = EndpointConfig{
	ExitThreshold: 0.35,
	MinSpeech:     250 * time.Millisecond,
	MinSilence:    500 * time.Millisecond,
	MaxSpeech:     30 * time.Second,
}

// windows is how many windows a duration is, rounded up so a duration shorter
// than a window still asks for one rather than none.
func windows(d time.Duration) int {
	n := int((d + windowDuration - 1) / windowDuration)
	if n < 1 {
		return 1
	}
	return n
}

// Endpoint decides, from the window probabilities of one capture, when that
// capture is over. It is not safe for concurrent use, which suits the capture
// loop: one reader goroutine feeds it and reads the answer.
type Endpoint struct {
	cfg        EndpointConfig
	minSpeech  int
	minSilence int
	maxSpeech  int

	speaking   bool // speech started and has not ended
	sinceStart int  // windows since speech started, for MaxSpeech
	speechRun  int  // consecutive speech windows before the capture is armed
	silenceRun int  // consecutive silence windows since the last speech
	ended      bool
}

// NewEndpoint builds an Endpoint that counts windows of WindowSamples samples.
func NewEndpoint(cfg EndpointConfig) *Endpoint {
	return &Endpoint{
		cfg:        cfg,
		minSpeech:  windows(cfg.MinSpeech),
		minSilence: windows(cfg.MinSilence),
		maxSpeech:  windows(cfg.MaxSpeech),
	}
}

// Reset forgets the capture, as if nothing had been heard, which is what a new
// capture needs. An Endpoint that ended is not spent, it is reset, so the end
// can come again next time.
func (e *Endpoint) Reset() {
	e.speaking = false
	e.sinceStart = 0
	e.speechRun = 0
	e.silenceRun = 0
	e.ended = false
}

// Ended reports whether this capture has already been ended, which is what
// stays true once Add says the capture is over.
func (e *Endpoint) Ended() bool { return e.ended }

// Add takes the probabilities of the windows in one read of the audio and says
// whether speech started and whether the capture is now over. Once it is over it
// stays over, so a caller only has to act on the signal once.
func (e *Endpoint) Add(probs []float32) (started, ended bool) {
	for _, p := range probs {
		if e.ended {
			break
		}
		if e.speaking {
			switch {
			case p >= Threshold:
				// Speech again, so whatever came in between was a pause.
				e.silenceRun = 0
			case p < e.cfg.ExitThreshold:
				e.silenceRun++
				if e.silenceRun >= e.minSilence {
					return started, e.finish()
				}
			default:
				// Between the thresholds: neither speech nor silence, so
				// neither counter moves. This band is the whole reason a
				// capture survives a probability sitting on the fence.
			}
			e.sinceStart++
			if e.sinceStart >= e.maxSpeech {
				return started, e.finish()
			}
			continue
		}
		// Not speaking yet. Only a window at or over the threshold counts
		// toward starting, and one below it starts the count again, so it takes
		// sustained speech to arm the end.
		if p < Threshold {
			e.speechRun = 0
			continue
		}
		e.speechRun++
		if e.speechRun >= e.minSpeech {
			e.speaking = true
			e.speechRun = 0
			e.silenceRun = 0
			e.sinceStart = 0
			started = true
		}
	}
	return started, e.ended
}

// finish closes the capture.
func (e *Endpoint) finish() bool {
	e.ended = true
	e.speaking = false
	e.sinceStart = 0
	e.speechRun = 0
	e.silenceRun = 0
	return true
}
