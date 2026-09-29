package main

import (
	"syscall"
	"testing"
)

func TestEventOf(t *testing.T) {
	cases := []struct {
		sig  syscall.Signal
		want captureEvent
	}{
		{syscall.SIGUSR1, eventCommand},
		{syscall.SIGUSR2, eventToggle},
	}
	for _, c := range cases {
		if got := eventOf(c.sig); got != c.want {
			t.Errorf("eventOf(%v) = %d, want %d", c.sig, got, c.want)
		}
	}
}

func TestCaptureNext(t *testing.T) {
	cases := []struct {
		name        string
		capturing   bool
		event       captureEvent
		start, stop bool
	}{
		// Idle: either key starts a capture of its own kind.
		{"idle command", false, eventCommand, true, false},
		{"idle toggle", false, eventToggle, true, false},
		// A silence of its own is not a reason to record, which is what a
		// leftover from a capture that already stopped would be.
		{"idle silence", false, eventSilence, false, false},
		// Running: the key that started a command capture does nothing, since
		// silence ends it. The toggle and silence do end it.
		{"capturing command", true, eventCommand, false, false},
		{"capturing toggle", true, eventToggle, false, true},
		{"capturing silence", true, eventSilence, false, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			start, stop := captureNext(c.capturing, c.event)
			if start != c.start || stop != c.stop {
				t.Errorf("captureNext(capturing=%v, event=%d) = (%v, %v), want (%v, %v)",
					c.capturing, c.event, start, stop, c.start, c.stop)
			}
		})
	}
}
