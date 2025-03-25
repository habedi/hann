package core

import (
	"golang.org/x/sys/cpu"
)

// init initializes the package and checks if the CPU supports AVX instructions.
// If the CPU does not support AVX instructions, it will panic with an error message.
func init() {
	if !cpu.X86.HasAVX {
		panic("CPU does not support AVX instructions. Hann requires AVX support.")
	}
}
