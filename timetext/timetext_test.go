package timetext

import "testing"

func TestReplace(t *testing.T) {
	cases := []struct {
		in  string
		out string
	}{
		{"Book an appointment at 11 30 PM tomorrow", "Book an appointment at 11:30pm tomorrow"},
		{"at 9 05 am", "at 9:05am"},
		{"at 9 5 pm", "at 9:05pm"},
		{"at 23 59", "at 23:59"},
		{"at 12 00 am", "at 12:00am"},
		{"at 19 45", "at 19:45"},
		{"at 24 00", "at 24 00"},
		{"at 12 60", "at 12 60"},
		{"meet at 11 30", "meet at 11:30"},
		{"chapter 9 5", "chapter 9 5"},
		{"route 66 30", "route 66 30"},
		{"call at 11 30 p.m.", "call at 11:30pm"},
		{"5 00pm", "5:00pm"},
		{"11 30 amsterdam", "11:30 amsterdam"},
		{"11 30amsterdam", "11 30amsterdam"},
		{"at 11 30 am", "at 11:30am"},
		{"3 30 PM.", "3:30pm."},
		{"at 11 30 p m", "at 11:30pm"},
		{"try 11 300", "try 11 300"},
		{"try 111 30", "try 111 30"},
		{"9 05 am and 10 45 pm", "9:05am and 10:45pm"},
		{"already 11:30pm", "already 11:30pm"},
		{"at 1 23 and 4 56", "at 1:23 and 4:56"},
		{"", ""},
		{"no digits here", "no digits here"},
	}
	for _, c := range cases {
		if got := (Time{}).Replace(c.in); got != c.out {
			t.Errorf("Replace(%q) = %q, want %q", c.in, got, c.out)
		}
	}
}

func TestReplaceNoAllocations(t *testing.T) {
	inputs := []string{
		"the point is clear",
		"chapter 9 5",
		"route 66 30",
		"try 11 300",
		"hello world",
		"already 11:30pm",
		"",
	}
	w := Time{}
	for _, in := range inputs {
		if got := w.Replace(in); got != in {
			t.Errorf("Replace(%q) = %q", in, got)
		}
		allocs := testing.AllocsPerRun(100, func() {
			sink = w.Replace(in)
		})
		if allocs != 0 {
			t.Errorf("Replace(%q) allocated %g times per run, want 0", in, allocs)
		}
	}
}

var sink string
