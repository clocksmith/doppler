package main

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// dequantRow is the plain ggml Q4_K dequantization of one row, written independently of q4kRow's sub-block algebra.
func dequantRow(row []byte, cols int) []float64 {
	out := make([]float64, cols)
	for b := 0; b < cols/256; b++ {
		blk := row[b*q4kBlockBytes:]
		d := f16(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := f16(binary.LittleEndian.Uint16(blk[2:4]))
		for j := 0; j < 8; j++ {
			sc, m := scaleMin(blk[4:16], j)
			for l := 0; l < 32; l++ {
				q := blk[16+32*(j/2)+l]
				if j%2 == 1 {
					q >>= 4
				}
				out[b*256+32*j+l] = d*float64(sc)*float64(q&15) - dmin*float64(m)
			}
		}
	}
	return out
}

func TestQ4KShares(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	rows, cols, n, fx, fw := 37, 512, 3, uint(18), uint(24)
	data := make([]byte, rows*cols/256*q4kBlockBytes)
	rng.Read(data)
	for i := 0; i < len(data); i += q4kBlockBytes { // realistic f16 scales (about 1e-3)
		binary.LittleEndian.PutUint16(data[i:], 0x1400|uint16(rng.Intn(0x3ff)))
		binary.LittleEndian.PutUint16(data[i+2:], 0x1000|uint16(rng.Intn(0x3ff)))
	}
	tensor := &Tensor{Rows: rows, Cols: cols, Q4K: data}
	x := make([]float64, n*cols)
	x0, x1 := make([]uint64, n*cols), make([]uint64, n*cols)
	for i := range x {
		x[i] = rng.NormFloat64() * 4
		enc := uint64(int64(math.Round(math.Ldexp(x[i], int(fx)))))
		x0[i] = rng.Uint64()
		x1[i] = enc - x0[i]
	}
	y0, y1 := make([]uint64, n*rows), make([]uint64, n*rows)
	tensor.MatVec(x0, n, y0, rows, 0, fw, 4)
	tensor.MatVec(x1, n, y1, rows, 0, fw, 4)
	worst := 0.0
	for r := 0; r < rows; r++ {
		w := dequantRow(data[r*cols/256*q4kBlockBytes:], cols)
		for c := 0; c < n; c++ {
			want := 0.0
			for k := 0; k < cols; k++ {
				want += w[k] * x[c*cols+k]
			}
			got := math.Ldexp(float64(int64(y0[c*rows+r]+y1[c*rows+r])), -int(fx+fw))
			worst = math.Max(worst, math.Abs(got-want))
		}
	}
	if worst > 1e-3 {
		t.Fatalf("max abs error %g", worst)
	}
	t.Logf("max abs error %g", worst)
}
