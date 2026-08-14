package core

/*
void hann_cpu_init(int support_level);
*/
import "C"

import (
	"golang.org/x/sys/cpu"
)

// CPUFeatureLevel defines the level of SIMD support.
type CPUFeatureLevel int

const (
	// Fallback indicates no SIMD support.
	Fallback CPUFeatureLevel = 0
	// AVX indicates AVX support.
	AVX CPUFeatureLevel = 1
	// AVX2 indicates AVX2 and FMA support.
	AVX2 CPUFeatureLevel = 2
)

var supportedCPUFeature = Fallback

// init checks for CPU support for AVX, AVX2, and FMA, then initializes the C library with the detected support level.
// The AVX2 variants use FMA instructions, so the AVX2 level requires both features.
func init() {
	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		supportedCPUFeature = AVX2
	} else if cpu.X86.HasAVX {
		supportedCPUFeature = AVX
	}
	C.hann_cpu_init(C.int(supportedCPUFeature))
}
