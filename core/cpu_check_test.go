package core

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func TestSupportedCPUFeature(t *testing.T) {
	// This test checks if the supportedCPUFeature variable is set correctly
	// based on the CPU capabilities of the machine running the test.

	if cpu.X86.HasAVX2 {
		if supportedCPUFeature != AVX2 {
			t.Errorf("CPU has AVX2, but supported feature is %v", supportedCPUFeature)
		}
	} else if cpu.X86.HasAVX {
		if supportedCPUFeature != AVX {
			t.Errorf("CPU has AVX, but supported feature is %v", supportedCPUFeature)
		}
	} else {
		if supportedCPUFeature != Fallback {
			t.Errorf("CPU has no AVX/AVX2, but supported feature is %v", supportedCPUFeature)
		}
	}
}
