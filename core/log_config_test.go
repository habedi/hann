package core

import (
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/rs/zerolog"
)

// TestInitLogLevels runs the test binary as a subprocess to test the init function.
// The subprocess checks the log level based on the HANN_LOG environment variable.
func TestInitLogLevels(t *testing.T) {
	if os.Getenv("BE_A_SUBPROCESS") == "1" {
		// This code runs in the subprocess.
		// The init function has already been executed.
		// We now verify that the log level was set correctly.
		expectedLevelStr := os.Getenv("EXPECTED_LEVEL")
		expectedLevel, err := zerolog.ParseLevel(strings.ToLower(expectedLevelStr))
		if err != nil {
			t.Fatalf("could not parse expected log level from env: %v", err)
		}

		if zerolog.GlobalLevel() != expectedLevel {
			t.Fatalf("expected log level %s, but got %s", expectedLevel, zerolog.GlobalLevel())
		}
		return
	}

	// This code runs in the main test process.
	// It defines the test cases and spawns a subprocess for each.
	testCases := []struct {
		name          string
		hannLogEnv    string
		expectedLevel zerolog.Level
	}{
		{name: "Disabled by 'off'", hannLogEnv: "off", expectedLevel: zerolog.Disabled},
		{name: "Disabled by '0'", hannLogEnv: "0", expectedLevel: zerolog.Disabled},
		{name: "Disabled by 'false'", hannLogEnv: "false", expectedLevel: zerolog.Disabled},
		{name: "Debug by 'full'", hannLogEnv: "full", expectedLevel: zerolog.DebugLevel},
		{name: "Debug by 'all'", hannLogEnv: "all", expectedLevel: zerolog.DebugLevel},
		{name: "Info by default", hannLogEnv: "", expectedLevel: zerolog.InfoLevel},
		{name: "Info with other value", hannLogEnv: "any_other_value", expectedLevel: zerolog.InfoLevel},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Prepare and run the subprocess.
			cmd := exec.Command(os.Args[0], "-test.run", "^TestInitLogLevels$")
			cmd.Env = append(os.Environ(),
				"BE_A_SUBPROCESS=1",
				"HANN_LOG="+tc.hannLogEnv,
				"EXPECTED_LEVEL="+tc.expectedLevel.String(),
			)
			output, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("subprocess failed with error: %v\nOutput:\n%s", err, string(output))
			}
		})
	}
}
