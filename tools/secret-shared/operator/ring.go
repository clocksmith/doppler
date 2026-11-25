package main

// Ring arithmetic for the secret-shared linear layers: every value is in Z_(2^64), products wrap.
// Q4_K weights stay in their block form; each 32-weight sub-block contributes A*sum(q*x) - B*sum(x), where
// A = round(d*sc*2^fw) and B = round(dmin*m*2^fw) are the sub-block scale and offset in fixed point.

import (
	"encoding/binary"
	"math"
	"runtime"
	"sync"
)

const q4kBlockBytes = 144

func f16(bits uint16) float64 {
	sign := 1.0
	if bits&0x8000 != 0 {
		sign = -1
	}
	exp := int((bits >> 10) & 0x1f)
	frac := float64(bits & 0x3ff)
	switch exp {
	case 0:
		return sign * frac * math.Pow(2, -24)
	case 31:
		if frac == 0 {
			return sign * math.Inf(1)
		}
		return math.NaN()
	}
	return sign * (1 + frac/1024) * math.Pow(2, float64(exp-15))
}

// scaleMin follows ggml's get_scale_min_k4 for sub-block j (0..7).
func scaleMin(s []byte, j int) (uint8, uint8) {
	if j < 4 {
		return s[j] & 63, s[j+4] & 63
	}
	return (s[j+4] & 0xf) | ((s[j-4] >> 6) << 4), (s[j+4] >> 4) | ((s[j] >> 6) << 4)
}

func fixed(v float64, fw uint) uint64 { return uint64(int64(math.Round(math.Ldexp(v, int(fw))))) }

// Tensor is one public weight matrix [rows, cols] held by an operator.
type Tensor struct {
	Name       string
	Rows, Cols int
	Q4K        []byte   // raw Q4_K row blocks, when quantized
	Dense      []uint64 // round(w*2^fw) for F16/F32 tensors
}

// sums32 returns, for each column vector, the ring sum of each run of 32 inputs (shared by every row).
func sums32(x []uint64, n, cols int) []uint64 {
	per := cols / 32
	t := make([]uint64, n*per)
	for c := 0; c < n; c++ {
		xs := x[c*cols : (c+1)*cols]
		for k := 0; k < per; k++ {
			var s uint64
			for _, v := range xs[k*32 : k*32+32] {
				s += v
			}
			t[c*per+k] = s
		}
	}
	return t
}

// MatVec computes y[c][rowOffset+r] = W~ x_c (mod 2^64) for n column vectors x (layout [n][cols]); y is [n][stride].
func (t *Tensor) MatVec(x []uint64, n int, y []uint64, stride, rowOffset int, fw uint, threads int) {
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	var sums []uint64
	if t.Q4K != nil {
		sums = sums32(x, n, t.Cols)
	}
	chunk := (t.Rows + threads*4 - 1) / (threads * 4)
	if chunk < 8 {
		chunk = 8
	}
	var wg sync.WaitGroup
	next := make(chan int, (t.Rows+chunk-1)/chunk)
	for r := 0; r < t.Rows; r += chunk {
		next <- r
	}
	close(next)
	for w := 0; w < threads; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			acc := make([]uint64, n)
			for start := range next {
				end := start + chunk
				if end > t.Rows {
					end = t.Rows
				}
				for r := start; r < end; r++ {
					if t.Q4K != nil {
						t.q4kRow(r, x, sums, n, acc, fw)
					} else {
						t.denseRow(r, x, n, acc)
					}
					for c := 0; c < n; c++ {
						y[c*stride+rowOffset+r] = acc[c]
					}
				}
			}
		}()
	}
	wg.Wait()
}

func (t *Tensor) denseRow(r int, x []uint64, n int, acc []uint64) {
	w := t.Dense[r*t.Cols : (r+1)*t.Cols]
	for c := 0; c < n; c++ {
		xs := x[c*t.Cols : (c+1)*t.Cols]
		var s uint64
		for k, wv := range w {
			s += wv * xs[k]
		}
		acc[c] = s
	}
}

func (t *Tensor) q4kRow(r int, x []uint64, sums []uint64, n int, acc []uint64, fw uint) {
	blocks := t.Cols / 256
	row := t.Q4K[r*blocks*q4kBlockBytes : (r+1)*blocks*q4kBlockBytes]
	per := t.Cols / 32
	for c := range acc[:n] {
		acc[c] = 0
	}
	for b := 0; b < blocks; b++ {
		blk := row[b*q4kBlockBytes : (b+1)*q4kBlockBytes]
		d := f16(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := f16(binary.LittleEndian.Uint16(blk[2:4]))
		scales := blk[4:16]
		qs := blk[16:144]
		for j := 0; j < 4; j++ {
			sc0, m0 := scaleMin(scales, 2*j)
			sc1, m1 := scaleMin(scales, 2*j+1)
			a0, b0 := fixed(d*float64(sc0), fw), fixed(dmin*float64(m0), fw)
			a1, b1 := fixed(d*float64(sc1), fw), fixed(dmin*float64(m1), fw)
			q := qs[32*j : 32*j+32]
			base := b*256 + 64*j
			sub := base / 32
			for c := 0; c < n; c++ {
				xs := x[c*t.Cols+base : c*t.Cols+base+64]
				var s0, s1 uint64
				for l := 0; l < 32; l++ {
					s0 += uint64(q[l]&15) * xs[l]
					s1 += uint64(q[l]>>4) * xs[32+l]
				}
				acc[c] += a0*s0 - b0*sums[c*per+sub] + a1*s1 - b1*sums[c*per+sub+1]
			}
		}
	}
}
