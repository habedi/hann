package core

import (
	"runtime"
	"testing"

	"golang.org/x/sys/cpu"
)

func TestSupportedCPUFeature(t *testing.T) {
	// This test checks if the supportedCPUFeature variable is set correctly
	// based on the CPU capabilities of the machine running the test.

	switch {
	case runtime.GOARCH == "arm64":
		if supportedCPUFeature != NEON {
			t.Errorf("CPU is arm64, but supported feature is %v", supportedCPUFeature)
		}
	case cpu.X86.HasAVX2:
		if supportedCPUFeature != AVX2 {
			t.Errorf("CPU has AVX2, but supported feature is %v", supportedCPUFeature)
		}
	case cpu.X86.HasAVX:
		if supportedCPUFeature != AVX {
			t.Errorf("CPU has AVX, but supported feature is %v", supportedCPUFeature)
		}
	default:
		if supportedCPUFeature != Fallback {
			t.Errorf("CPU has no AVX/AVX2, but supported feature is %v", supportedCPUFeature)
		}
	}
}
