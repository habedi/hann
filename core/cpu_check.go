package core

/*
void hann_cpu_init(int support_level);
*/
import "C"

import (
	"runtime"

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
	// NEON indicates NEON support, which is a baseline feature of arm64.
	NEON CPUFeatureLevel = 3
)

var supportedCPUFeature = Fallback

// init checks the SIMD support of the CPU, then initializes the C library
// with the detected support level. On x86, the AVX2 variants use FMA
// instructions, so the AVX2 level requires both features. On arm64, every
// CPU has NEON, so the NEON level is selected without a feature check.
func init() {
	if runtime.GOARCH == "arm64" {
		supportedCPUFeature = NEON
	} else if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		supportedCPUFeature = AVX2
	} else if cpu.X86.HasAVX {
		supportedCPUFeature = AVX
	}
	C.hann_cpu_init(C.int(supportedCPUFeature))
}
