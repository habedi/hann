package core

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func initCPUCheck() {
	// Add any necessary initialization code here
}

func TestCPUHasAVX(t *testing.T) {
	if !cpu.X86.HasAVX {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected panic when CPU does not support AVX")
			}
		}()
		initCPUCheck()
	}
}

func TestCPUHasAVXSupport(t *testing.T) {
	if cpu.X86.HasAVX {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Did not expect panic when CPU supports AVX")
			}
		}()
		initCPUCheck()
	}
}
