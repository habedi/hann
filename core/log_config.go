package core

import (
	"os"
	"strings"

	"github.com/rs/zerolog"
)

func init() {
	debugMode := strings.TrimSpace(strings.ToLower(os.Getenv("DEBUG_HANN")))
	if debugMode == "off" || debugMode == "0" {
		zerolog.SetGlobalLevel(zerolog.Disabled)
	} else if debugMode == "full" {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	} else {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}
}
