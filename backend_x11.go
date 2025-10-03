//go:build x11

package main

import (
	"context"
	"os/exec"
	"strings"

	"github.com/pkg/errors"
)

// X11Backend implements Backend for X11/i3
type X11Backend struct{}

// NewBackend creates a new X11 backend
func NewBackend(ctx context.Context) (Backend, error) {
	return &X11Backend{}, nil
}

// GetWindowClass returns the window class of the focused window
func (b *X11Backend) GetWindowClass(ctx context.Context) (string, error) {
	cmd := exec.Command("xdotool", "getwindowfocus", "getwindowclassname")
	output, err := cmd.Output()
	if err != nil {
		return "", errors.WithStack(err)
	}
	return strings.TrimSpace(string(output)), nil
}

// TypeText types text using xdotool
func (b *X11Backend) TypeText(text string) error {
	if err := exec.Command("xdotool", "type", "--delay", "8", text).Run(); err != nil {
		return errors.WithStack(err)
	}
	return nil
}

// PasteText copies to clipboard and pastes using xdotool
func (b *X11Backend) PasteText(text string) error {
	xclipCmd := exec.Command("xclip", "-selection", "clipboard")
	xclipCmd.Stdin = strings.NewReader(text)
	if err := xclipCmd.Run(); err != nil {
		return errors.WithStack(err)
	}
	if err := exec.Command("xdotool", "key", "ctrl+v").Run(); err != nil {
		return errors.WithStack(err)
	}
	return nil
}
