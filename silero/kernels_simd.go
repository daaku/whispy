//go:build amd64 && goexperiment.simd

package silero

import "simd/archsimd"

// haveAVX gates the wide kernel. The simd package is experimental and its
// 256-bit vectors need AVX, which older amd64 CPUs do not have.
var haveAVX = archsimd.X86.AVX()

// matvecAdd adds packedᵀ x to dst, where packed holds nIn rows of nOut. One
// output block is kept in a vector register across all input rows, which is why
// the weights are packed with the output channels contiguous.
func matvecAdd(dst, packed, x []float32, nIn, nOut int) {
	if !haveAVX {
		matvecAddScalar(dst, packed, x, nIn, nOut)
		return
	}
	for o := 0; o < nOut; o += 8 {
		acc := archsimd.LoadFloat32x8(dst[o:])
		for i := 0; i < nIn; i++ {
			w := archsimd.LoadFloat32x8(packed[i*nOut+o:])
			acc = w.MulAdd(archsimd.BroadcastFloat32x8(x[i]), acc)
		}
		acc.Store(dst[o:])
	}
	// The caller follows up with scalar math (the LSTM activations), and the
	// wide stores above leave the upper vector bits dirty. Without this, every
	// float op after the kernel pays a false dependency penalty: it cost more
	// than the kernel saved on a Ryzen 5900X.
	archsimd.ClearAVXUpperBits()
}

// Implementation names the matrix kernel this build uses.
func Implementation() string {
	if !haveAVX {
		return "scalar"
	}
	return "simd"
}
