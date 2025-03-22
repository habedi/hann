package core

import (
	"os"
	"strings"

	"github.com/rs/zerolog"
)

// init initializes the logging configuration for the application based on the DEBUG_HANN environment variable.
// It sets the global logging level to Disabled, Debug, or Info based on the value of DEBUG_HANN.
func init() {
	// Retrieve the DEBUG_HANN environment variable, trim spaces, and convert to lowercase.
	debugMode := strings.TrimSpace(strings.ToLower(os.Getenv("DEBUG_HANN")))

	// Set the global logging level based on the value of DEBUG_HANN.
	if debugMode == "off" || debugMode == "0" {
		// Disable logging if DEBUG_HANN is set to "off" or "0".
		zerolog.SetGlobalLevel(zerolog.Disabled)
	} else if debugMode == "full" {
		// Enable debug level logging if DEBUG_HANN is set to "full".
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	} else {
		// Set the logging level to info by default.
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}
}
