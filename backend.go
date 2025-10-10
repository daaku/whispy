package main

import "context"

// Backend handles window manager specific operations
type Backend interface {
	// GetWindowClass returns the class/app ID of the focused window
	GetWindowClass(ctx context.Context) (string, error)

	// TypeText types the given text into the focused window
	TypeText(text string) error

	// PasteText copies text to clipboard and pastes it
	PasteText(text string) error
}

// ShouldPaste determines if we should paste (for firefox) or type directly
func ShouldPaste(windowClass string) bool {
	// Check if window class contains "firefox" (case insensitive)
	for _, char := range windowClass {
		if char >= 'A' && char <= 'Z' {
			windowClass = string(append([]byte{}, byte(char+32)))
		}
	}
	return containsSubstring(windowClass, "firefox")
}

func containsSubstring(s, substr string) bool {
	if len(substr) > len(s) {
		return false
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
