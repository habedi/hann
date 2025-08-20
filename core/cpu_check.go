package core

/*
#cgo CFLAGS: -mavx -mavx2
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

// init checks for CPU support for AVX and AVX2, then initializes the C library with the detected support level.
func init() {
	if cpu.X86.HasAVX2 {
		supportedCPUFeature = AVX2
	} else if cpu.X86.HasAVX {
		supportedCPUFeature = AVX
	}
	C.hann_cpu_init(C.int(supportedCPUFeature))
}
