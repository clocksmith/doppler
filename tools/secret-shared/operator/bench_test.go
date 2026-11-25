package main

import (
	"math/rand"
	"os"
	"strconv"
	"testing"
)

// BenchmarkQ4KMatVec27B times one decode-step matvec at a 27B MLP shape (gate/up: 17408 x 5120) on random Q4_K
// blocks. SS_BENCH_THREADS sets the worker count (0 = all CPUs). Throughput is reported as GMAC/s.
func BenchmarkQ4KMatVec27B(b *testing.B) {
	rows, cols := 17408, 5120
	threads, _ := strconv.Atoi(os.Getenv("SS_BENCH_THREADS"))
	rng := rand.New(rand.NewSource(1))
	data := make([]byte, rows*cols/256*q4kBlockBytes)
	rng.Read(data)
	t := &Tensor{Rows: rows, Cols: cols, Q4K: data}
	x := make([]uint64, cols)
	for i := range x {
		x[i] = rng.Uint64()
	}
	y := make([]uint64, rows)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t.MatVec(x, 1, y, rows, 0, 24, threads)
	}
	b.ReportMetric(float64(rows*cols)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
}
