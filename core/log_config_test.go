package core

import (
	"os"
	"testing"

	"github.com/rs/zerolog"
)

func initLogging() {
	logLevel := os.Getenv("HANN_LOG")
	switch logLevel {
	case "off":
		zerolog.SetGlobalLevel(zerolog.Disabled)
	case "full":
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	case "info":
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	default:
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}
}

func loggingLevel() zerolog.Level {
	return zerolog.GlobalLevel()
}

func TestLoggingDisabled(t *testing.T) {
	os.Setenv("HANN_LOG", "off")
	defer os.Unsetenv("HANN_LOG")
	initLogging()
	if loggingLevel() != zerolog.Disabled {
		t.Errorf("Expected logging level to be Disabled, got %v", loggingLevel())
	}
}

func TestLoggingDebug(t *testing.T) {
	os.Setenv("HANN_LOG", "full")
	defer os.Unsetenv("HANN_LOG")
	initLogging()
	if loggingLevel() != zerolog.DebugLevel {
		t.Errorf("Expected logging level to be Debug, got %v", loggingLevel())
	}
}

func TestLoggingInfo(t *testing.T) {
	os.Setenv("HANN_LOG", "info")
	defer os.Unsetenv("HANN_LOG")
	initLogging()
	if loggingLevel() != zerolog.InfoLevel {
		t.Errorf("Expected logging level to be Info, got %v", loggingLevel())
	}
}

func TestLoggingDefault(t *testing.T) {
	os.Unsetenv("HANN_LOG")
	initLogging()
	if loggingLevel() != zerolog.InfoLevel {
		t.Errorf("Expected logging level to be Info by default, got %v", loggingLevel())
	}
}
