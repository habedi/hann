package core

import (
	"github.com/rs/zerolog/log"
	"os"
	"strconv"
	"time"
)

// GetSeed receives a seed value for random number generation from the HANN_SEED environment variable.
func GetSeed() int64 {
	seedStr := os.Getenv("HANN_SEED")
	if seedStr != "" {
		if seed, err := strconv.ParseInt(seedStr, 10, 64); err == nil {
			log.Info().Msgf("Using seed from HANN_SEED value: %d", seed)
			return seed
		}
		log.Warn().Msgf("Failed to parse HANN_SEED value: %s", seedStr)
	}

	seed := time.Now().UnixNano()
	log.Info().Msgf("Using current time as seed: %d", seed)
	return seed
}
