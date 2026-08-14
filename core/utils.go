package core

import (
	"os"
	"strconv"
	"time"
)

// GetSeed receives a seed value for random number generation from the HANN_SEED environment variable.
// A value that does not parse as an integer is ignored, and the current time is used instead.
func GetSeed() int64 {
	seedStr := os.Getenv("HANN_SEED")
	if seedStr != "" {
		if seed, err := strconv.ParseInt(seedStr, 10, 64); err == nil {
			return seed
		}
	}
	return time.Now().UnixNano()
}
