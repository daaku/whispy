//go:build !amd64 || !goexperiment.simd

package silero

// matvecAdd adds packedᵀ x to dst, where packed holds nIn rows of nOut.
func matvecAdd(dst, packed, x []float32, nIn, nOut int) {
	matvecAddScalar(dst, packed, x, nIn, nOut)
}

// Implementation names the matrix kernel this build uses.
func Implementation() string { return "scalar" }
