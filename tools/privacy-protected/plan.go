package main

// The evaluator's model plan: public weights folded into lookup tables, matrices and polynomial approximations.
// The requester never loads this file's inputs. The plaintext references (exact and approximate) also use it.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

type tinyModel struct {
	Schema      string               `json:"schema"`
	Alphabet    string               `json:"alphabet"`
	Vocab       int                  `json:"vocab"`
	Context     int                  `json:"context"`
	Dim         int                  `json:"dim"`
	Hidden      int                  `json:"hidden"`
	Calibration map[string][]float64 `json:"calibration"`
	Weights     struct {
		Wte   [][]float64 `json:"wte"`
		Wpe   [][]float64 `json:"wpe"`
		Ln1G  []float64   `json:"ln1_g"`
		Ln1B  []float64   `json:"ln1_b"`
		AttnW [][]float64 `json:"attn_w"`
		AttnB []float64   `json:"attn_b"`
		ProjW [][]float64 `json:"proj_w"`
		ProjB []float64   `json:"proj_b"`
		Ln2G  []float64   `json:"ln2_g"`
		Ln2B  []float64   `json:"ln2_b"`
		Fc1W  [][]float64 `json:"fc1_w"`
		Fc1B  []float64   `json:"fc1_b"`
		Fc2W  [][]float64 `json:"fc2_w"`
		Fc2B  []float64   `json:"fc2_b"`
		LnfG  []float64   `json:"lnf_g"`
		LnfB  []float64   `json:"lnf_b"`
		LnEps float64     `json:"ln_eps"`
	} `json:"weights"`
}

// loadModel reads the fixture and returns it with its identity (sha256 of the exact file bytes).
func loadModel(path string) (*tinyModel, string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, "", err
	}
	m := &tinyModel{}
	if err := json.Unmarshal(raw, m); err != nil {
		return nil, "", err
	}
	if m.Schema != "doppler.protected-tiny-model/v1" || m.Vocab != vocab || m.Context != context || m.Dim != dim || m.Hidden != hidden {
		return nil, "", fmt.Errorf("model geometry does not match the compiled plan")
	}
	w := m.Weights
	if len(w.Wte) != vocab || len(w.Wpe) != context || len(w.Ln1G) != dim || len(w.AttnW) != 3*dim || len(w.ProjW) != dim ||
		len(w.Ln2G) != dim || len(w.Fc1W) != hidden || len(w.Fc2W) != dim || len(w.LnfG) != dim || w.LnEps <= 0 {
		return nil, "", fmt.Errorf("model weights are incomplete")
	}
	sum := sha256.Sum256(raw)
	return m, "sha256:" + hex.EncodeToString(sum[:]), nil
}

// approximation: a Chebyshev interpolant of f on [A, B]. The evaluator receives inputs already mapped to [-1, 1].
type approximation struct {
	Name   string
	A, B   float64
	Degree int
	Poly   bignum.Polynomial
	coeffs []float64
}

func newApproximation(name string, a, b float64, degree int, f func(float64) float64) approximation {
	poly := bignum.ChebyshevApproximation(f, bignum.Interval{A: *bignum.NewFloat(a, 128), B: *bignum.NewFloat(b, 128), Nodes: degree})
	c := make([]float64, len(poly.Coeffs))
	for i, v := range poly.Coeffs {
		c[i], _ = v[0].Float64()
	}
	return approximation{Name: name, A: a, B: b, Degree: degree, Poly: poly, coeffs: c}
}

// scalar and offset map x in [A, B] to t in [-1, 1]: t = scalar*x + offset.
func (p approximation) scalar() float64 { return 2 / (p.B - p.A) }
func (p approximation) offset() float64 { return -(p.A + p.B) / (p.B - p.A) }

// evalT evaluates the interpolant at an already mapped t (Clenshaw), in float64.
func (p approximation) evalT(t float64) float64 {
	var b1, b2 float64
	for k := len(p.coeffs) - 1; k >= 1; k-- {
		b1, b2 = 2*t*b1-b2+p.coeffs[k], b1
	}
	return t*b1 - b2 + p.coeffs[0]
}

func (p approximation) eval(x float64) float64 { return p.evalT(p.scalar()*x + p.offset()) }

const expShift = 8.0

func geluTanh(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x)))
}

// plan holds everything the evaluator computes with. Vectors are plain slices; layout happens in the evaluator.
type plan struct {
	m *tinyModel
	// per position p and token v: x0, scaled query, key, value folded with the output projection
	x0, q, k, v [context][vocab][]float64
	projB       []float64
	// LN2 folded into fc1, GELU's input map folded into fc1
	fc1W [][]float64 // hidden x dim
	fc1B []float64
	fc2W [][]float64 // dim x hidden
	fc2B []float64
	// LNf folded into the tied head
	headW                           [][]float64 // vocab x dim
	headB                           []float64
	expP, invP, isq2P, isqfP, geluP approximation
	eps                             float64
}

func widen(r []float64, lo, hi float64) (float64, float64) {
	return math.Min(r[0], lo), math.Max(r[1], hi)
}

func newPlan(m *tinyModel) (*plan, error) {
	w := m.Weights
	pl := &plan{m: m, eps: w.LnEps}
	// Fixed public intervals with margin over the calibration ranges recorded at training time.
	sLo, sHi := widen(m.Calibration["scores"], -8, 9)
	pl.invP = newApproximation("inverse", 0.08, 4.0, 15, func(x float64) float64 { return 1 / x })
	alphaInv, betaInv := pl.invP.scalar(), pl.invP.offset()
	pl.expP = newApproximation("exp", sLo, sHi, 15, func(x float64) float64 { return alphaInv * math.Exp(x-expShift) })
	d3 := math.Pow(dim, 3)
	pl.isq2P = newApproximation("ln2-inverse-sqrt", d3*(0.12+w.LnEps), d3*(2.0+w.LnEps), 15, func(x float64) float64 { return 1 / math.Sqrt(x) })
	pl.isqfP = newApproximation("lnf-inverse-sqrt", d3*(0.04+w.LnEps), d3*(13+w.LnEps), 31, func(x float64) float64 { return 1 / math.Sqrt(x) })
	gLo, gHi := widen(m.Calibration["gelu_in"], -13, 8)
	pl.geluP = newApproximation("gelu", gLo, gHi, 31, geluTanh)
	_ = betaInv

	expScale, sqrtD := pl.expP.scalar(), math.Sqrt(dim)
	for p := 0; p < context; p++ {
		for tok := 0; tok < vocab; tok++ {
			x := make([]float64, dim)
			for j := range x {
				x[j] = w.Wte[tok][j] + w.Wpe[p][j]
			}
			a := layerNorm(x, w.Ln1G, w.Ln1B, w.LnEps)
			qkv := affine(w.AttnW, w.AttnB, a)
			q, k, v := qkv[:dim], qkv[dim:2*dim], qkv[2*dim:]
			qs := make([]float64, dim)
			for j := range qs {
				qs[j] = q[j] * expScale / sqrtD
			}
			vo := matVec(w.ProjW, v)
			for j := range vo {
				vo[j] /= alphaInv
			}
			pl.x0[p][tok], pl.q[p][tok], pl.k[p][tok], pl.v[p][tok] = x, qs, append([]float64(nil), k...), vo
		}
	}
	pl.projB = w.ProjB
	// fc1(LN2(h)) = W1 (sqrt(D) * y * g2 + b2) + b1, then GELU's input map t = alpha*u + beta
	ga, gb := pl.geluP.scalar(), pl.geluP.offset()
	pl.fc1W = make([][]float64, hidden)
	pl.fc1B = make([]float64, hidden)
	for i := 0; i < hidden; i++ {
		pl.fc1W[i] = make([]float64, dim)
		bias := w.Fc1B[i]
		for j := 0; j < dim; j++ {
			pl.fc1W[i][j] = ga * w.Fc1W[i][j] * sqrtD * w.Ln2G[j]
			bias += w.Fc1W[i][j] * w.Ln2B[j]
		}
		pl.fc1B[i] = ga*bias + gb
	}
	pl.fc2W, pl.fc2B = w.Fc2W, w.Fc2B
	pl.headW = make([][]float64, vocab)
	pl.headB = make([]float64, vocab)
	for t := 0; t < vocab; t++ {
		pl.headW[t] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			pl.headW[t][j] = w.Wte[t][j] * sqrtD * w.LnfG[j]
			pl.headB[t] += w.Wte[t][j] * w.LnfB[j]
		}
	}
	return pl, nil
}

func layerNorm(x, g, b []float64, eps float64) []float64 {
	var mu, v float64
	for _, a := range x {
		mu += a
	}
	mu /= float64(len(x))
	for _, a := range x {
		v += (a - mu) * (a - mu)
	}
	v /= float64(len(x))
	out := make([]float64, len(x))
	for i, a := range x {
		out[i] = (a-mu)/math.Sqrt(v+eps)*g[i] + b[i]
	}
	return out
}

func matVec(w [][]float64, x []float64) []float64 {
	out := make([]float64, len(w))
	for i := range w {
		for j, a := range x {
			out[i] += w[i][j] * a
		}
	}
	return out
}

func affine(w [][]float64, b, x []float64) []float64 {
	out := matVec(w, x)
	for i := range out {
		out[i] += b[i]
	}
	return out
}
