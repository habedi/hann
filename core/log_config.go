package core

import (
	"github.com/rs/zerolog/log"
	"os"
	"strings"

	"github.com/rs/zerolog"
)

// init initializes the logging configuration for the application based on the HANN_LOG environment variable.
// It sets the global logging level to Disabled, Debug, or Info based on the value of HANN_LOG.
func init() {
	// Get the HANN_LOG environment variable, trim spaces, and convert to lowercase.
	debugMode := strings.TrimSpace(strings.ToLower(os.Getenv("HANN_LOG")))

	// Set the global logging level based on the value of HANN_LOG.
	switch debugMode {
	case "0", "off", "false":
		// Disable logging altogether if HANN_LOG is set to "off", "false", or `0`.
		zerolog.SetGlobalLevel(zerolog.Disabled)
	case "full", "all":
		// Set the logging level to DEBUG if HANN_LOG is set to "full" or "all".
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
		// Set the logger output to the console.
		log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})
	default:
		// Set the logging level to INFO by default.
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}
}
