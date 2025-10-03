//go:build !x11

package main

import (
	"context"
	"os/exec"

	"github.com/joshuarubin/go-sway"
	"github.com/pkg/errors"
)

// SwayBackend implements Backend for Sway/Wayland
type SwayBackend struct {
	client sway.Client
}

// NewBackend creates a new Sway backend
func NewBackend(ctx context.Context) (Backend, error) {
	client, err := sway.New(ctx)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	return &SwayBackend{client: client}, nil
}

// GetWindowClass returns the app ID of the focused window
func (b *SwayBackend) GetWindowClass(ctx context.Context) (string, error) {
	tree, err := b.client.GetTree(ctx)
	if err != nil {
		return "", errors.WithStack(err)
	}
	appID := tree.FocusedNode().AppID
	if appID == nil {
		return "", nil
	}
	return *appID, nil
}

// TypeText types text using ydotool
func (b *SwayBackend) TypeText(text string) error {
	if err := exec.Command("ydotool", "type", "-d=8", "-H=6", text).Run(); err != nil {
		return errors.WithStack(err)
	}
	return nil
}

// PasteText copies to clipboard and pastes using ydotool
func (b *SwayBackend) PasteText(text string) error {
	wlCopyCmd := exec.Command("wl-copy", "--foreground", text)
	if err := wlCopyCmd.Start(); err != nil {
		return errors.WithStack(err)
	}
	if err := exec.Command("ydotool", "key", "29:1", "47:1", "47:0", "29:0").Run(); err != nil {
		return errors.WithStack(err)
	}
	wlCopyCmd.Process.Kill()
	wlCopyCmd.Wait()
	return nil
}
