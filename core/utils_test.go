package core

import (
	"os"
	"strconv"
	"testing"
	"time"
)

func TestGetSeedFromEnv(t *testing.T) {
	expectedSeed := int64(12345)
	os.Setenv("HANN_SEED", strconv.FormatInt(expectedSeed, 10))
	defer os.Unsetenv("HANN_SEED")

	seed := GetSeed()
	if seed != expectedSeed {
		t.Errorf("GetSeed() = %d; want %d", seed, expectedSeed)
	}
}

func TestGetSeedFromEnvInvalid(t *testing.T) {
	os.Setenv("HANN_SEED", "invalid")
	defer os.Unsetenv("HANN_SEED")

	seed := GetSeed()
	if seed == 0 {
		t.Errorf("GetSeed() = %d; want non-zero value", seed)
	}
}

func TestGetSeedFromTime(t *testing.T) {
	os.Unsetenv("HANN_SEED")

	seed1 := GetSeed()
	time.Sleep(1 * time.Nanosecond)
	seed2 := GetSeed()

	if seed1 == seed2 {
		t.Errorf("GetSeed() = %d; subsequent call returned the same seed %d", seed1, seed2)
	}
}
