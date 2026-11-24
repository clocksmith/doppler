package main

// Plaintext references, run only after a route has closed and never by the requester process during a session:
// exactLogits is the float64 model; approxLogits mirrors the encrypted circuit (same folding, same polynomials).

import "math"

func exactLogits(m *tinyModel, toks []int) []float64 {
	w := m.Weights
	t := len(toks)
	xs := make([][]float64, t)
	ks := make([][]float64, t)
	vs := make([][]float64, t)
	var q []float64
	for p, tok := range toks {
		x := make([]float64, dim)
		for j := range x {
			x[j] = w.Wte[tok][j] + w.Wpe[p][j]
		}
		qkv := affine(w.AttnW, w.AttnB, layerNorm(x, w.Ln1G, w.Ln1B, w.LnEps))
		xs[p], q, ks[p], vs[p] = x, qkv[:dim], qkv[dim:2*dim], qkv[2*dim:]
	}
	s := make([]float64, t)
	mx := math.Inf(-1)
	for j := range s {
		for i := 0; i < dim; i++ {
			s[j] += q[i] * ks[j][i]
		}
		s[j] /= math.Sqrt(dim)
		mx = math.Max(mx, s[j])
	}
	var z float64
	for j := range s {
		s[j] = math.Exp(s[j] - mx)
		z += s[j]
	}
	o := make([]float64, dim)
	for j := range s {
		for i := range o {
			o[i] += s[j] / z * vs[j][i]
		}
	}
	h := affine(w.ProjW, w.ProjB, o)
	for i := range h {
		h[i] += xs[t-1][i]
	}
	u := affine(w.Fc1W, w.Fc1B, layerNorm(h, w.Ln2G, w.Ln2B, w.LnEps))
	for i := range u {
		u[i] = geluTanh(u[i])
	}
	mlp := affine(w.Fc2W, w.Fc2B, u)
	for i := range h {
		h[i] += mlp[i]
	}
	return matVec(w.Wte, layerNorm(h, w.LnfG, w.LnfB, w.LnEps))
}

// trace records every polynomial input so range audits can report out-of-interval values.
type trace struct{ outOfRange map[string]int }

func (tr *trace) check(p approximation, x float64) {
	if tr != nil && (x < p.A || x > p.B) {
		tr.outOfRange[p.Name]++
	}
}

func approxLogits(pl *plan, toks []int, tr *trace) []float64 {
	t := len(toks)
	pos, tok := t-1, toks[t-1]
	q := pl.q[pos][tok]
	es := make([]float64, t)
	var sum float64
	for j := 0; j < t; j++ {
		var tt float64
		for i := 0; i < dim; i++ {
			tt += q[i] * pl.k[j][toks[j]][i]
		}
		tt += pl.expP.offset()
		tr.check(pl.expP, (tt-pl.expP.offset())/pl.expP.scalar())
		es[j] = pl.expP.evalT(tt)
		sum += es[j]
	}
	tInv := sum + pl.invP.offset()
	tr.check(pl.invP, (tInv-pl.invP.offset())/pl.invP.scalar())
	inv := pl.invP.evalT(tInv)
	h := make([]float64, dim)
	for i := range h {
		h[i] = pl.x0[pos][tok][i] + pl.projB[i]
		for j := 0; j < t; j++ {
			h[i] += es[j] * inv * pl.v[j][toks[j]][i]
		}
	}
	y := lnFolded(h, pl.isq2P, pl.eps, tr)
	u := affine(pl.fc1W, pl.fc1B, y)
	for i := range u {
		tr.check(pl.geluP, (u[i]-pl.geluP.offset())/pl.geluP.scalar())
		u[i] = pl.geluP.evalT(u[i])
	}
	mlp := affine(pl.fc2W, pl.fc2B, u)
	for i := range h {
		h[i] += mlp[i]
	}
	return affine(pl.headW, pl.headB, lnFolded(h, pl.isqfP, pl.eps, tr))
}

// lnFolded mirrors the encrypted layer norm: c' = D*h - sum(h), var' = sum(c'^2), y = c' * isq(var' + eps*D^3).
func lnFolded(h []float64, isq approximation, eps float64, tr *trace) []float64 {
	var s float64
	for _, a := range h {
		s += a
	}
	c := make([]float64, len(h))
	var v float64
	for i, a := range h {
		c[i] = dim*a - s
		v += c[i] * c[i]
	}
	x := v + eps*math.Pow(dim, 3)
	tr.check(isq, x)
	r := isq.eval(x)
	for i := range c {
		c[i] *= r
	}
	return c
}

func argmax(x []float64) int {
	best := 0
	for i, a := range x {
		if a > x[best] {
			best = i
		}
	}
	return best
}

func margin(x []float64) float64 {
	b := argmax(x)
	second := math.Inf(-1)
	for i, a := range x {
		if i != b && a > second {
			second = a
		}
	}
	return x[b] - second
}

func greedy(prompt []int, steps int, logits func([]int) []float64) ([]int, [][]float64) {
	toks := append([]int(nil), prompt...)
	var all [][]float64
	for i := 0; i < steps; i++ {
		l := logits(toks)
		all = append(all, l)
		toks = append(toks, argmax(l))
	}
	return toks[len(prompt):], all
}
