package main

// gpu_record.go: compile the CPU circuit into a flat RNS program for the WebGPU evaluator.
//
// The recorder sits between the circuit (evaluator.go and Lattigo's polynomial evaluator) and Lattigo's CKKS
// evaluator. Every high-level call still runs on Lattigo, and is also lowered to the few RNS operations the GPU
// kernels implement (rnsExec below). Each lowering is replayed on the call's own inputs and must equal Lattigo's
// output bit for bit before recording continues; at the end the whole program is replayed from the inputs and must
// reproduce the circuit's output ciphertext exactly. The GPU therefore runs the same circuit as the CPU evaluator,
// not an approximation of it.
//
// Compiling needs evaluation keys to execute rotations, so the compiler makes throwaway keys from a throwaway secret.
// The program does not depend on keys or on input values: Lattigo's choices depend only on levels, scales and
// degrees. The compiler is an offline tool; the evaluator process never makes or holds a secret key.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math/big"
	"os"
	"time"
	"weak"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

const gpuProgramSchema = "doppler.protected-gpu-program/v1"

// compactWords: a 64-slot plaintext in the NTT domain is constant on blocks of N/128 coefficients, so one limb is
// stored as 128 words (value of block b at index b).
const compactWords = 2 * slots

type rnsReg struct {
	Deg   int `json:"deg"`
	Level int `json:"level"`
}

// rnsOp is one GPU operation. All operands are in the NTT domain; an op at level l reads limbs 0..l of its inputs.
//
//	zero              out = 0
//	add, sub          out = a +/- b (componentwise; extra components of the higher degree are copied, negated for b in sub)
//	smul              out = a * K (K: one word per limb)
//	sadd              out = a, out.c0 += K
//	smac              out = b + a * K (b: accumulator; components of b beyond a's degree copied)
//	pmul              out = a * pt (pt: compact plaintext)
//	padd              out = a, out.c0 += pt
//	tensor            out (degree 2) = a (x) b
//	relin             out (degree 1) = (a.c0, a.c1) + keyswitch(a.c2, relinearization key)
//	rescale           out = a / q_level, rounded (level - 1)
//	rotate            out = automorphism(a, gal), key-switched with the Galois key of gal
type rnsOp struct {
	Op    string `json:"op"`
	Out   int    `json:"out"`
	A     int    `json:"a"`
	B     int    `json:"b"`
	Level int    `json:"level"`
	Deg   int    `json:"deg"`
	Gal   uint64 `json:"gal,omitempty"`
	K     int    `json:"k"`
	Pt    int    `json:"pt"`
	Free  []int  `json:"free,omitempty"`
	From  string `json:"from"`
}

// gpuProgram is a compiled session: one op list and one constant blob, cut into steps (a prefill, then one decode
// step per position). The encrypted KV cache crosses steps as named registers (k<j>, v<j>): a step's StateIn are read
// from the evaluator's session state, its StateOut are kept there for later steps.
type gpuProgram struct {
	Schema        string         `json:"schema"`
	Protocol      string         `json:"protocol"`
	Profile       string         `json:"profile"`
	ProfileDigest string         `json:"profileDigest"`
	Model         string         `json:"model"`
	LogN          int            `json:"logN"`
	LogSlots      int            `json:"logSlots"`
	Q             []uint64       `json:"q"`
	P             []uint64       `json:"p"`
	Psi           []uint64       `json:"psi"`
	Galois        []uint64       `json:"galois"`
	InputScale    float64        `json:"inputScale"`
	Regs          []rnsReg       `json:"regs"`
	Ops           []rnsOp        `json:"ops"`
	Steps         []gpuStep      `json:"steps"`
	BlobWords     int            `json:"blobWords"`
	BlobDigest    string         `json:"blobDigest"`
	Counts        map[string]int `json:"counts"`
}

type namedReg struct {
	Name string `json:"name"`
	Reg  int    `json:"reg"`
}

type gpuStep struct {
	Kind        string         `json:"kind"`     // prefill or decode
	First       int            `json:"first"`    // position of the step's first input token
	Position    int            `json:"position"` // position whose logits the step returns
	OpFirst     int            `json:"opFirst"`
	OpEnd       int            `json:"opEnd"`
	Inputs      []int          `json:"inputs"`
	StateIn     []namedReg     `json:"stateIn"`
	StateOut    []namedReg     `json:"stateOut"`
	Output      int            `json:"output"`
	OutputLevel int            `json:"outputLevel"`
	OutputScale float64        `json:"outputScale"`
	Layers      []gpuLayer     `json:"layers"`
	Counts      map[string]int `json:"counts"`
}

// gpuLayer marks the ops of one model stage, for per-layer GPU timing.
type gpuLayer struct {
	Name  string `json:"name"`
	First int    `json:"first"`
	Last  int    `json:"last"`
}

// rnsExec executes rnsOps on the CPU with Lattigo's ring arithmetic and key-switching. It is the reference the GPU
// kernels are tested against, and the replay check for compiled programs.
type rnsExec struct {
	params ckks.Parameters
	ev     *ckks.Evaluator
}

func trunc(ct *rlwe.Ciphertext, level int) *rlwe.Ciphertext {
	if ct.Level() < level {
		panic(fmt.Sprintf("operand at level %d read at level %d", ct.Level(), level))
	}
	out := &rlwe.Ciphertext{}
	out.MetaData = ct.MetaData.CopyNew()
	out.Value = make([]ring.Poly, len(ct.Value))
	for i := range ct.Value {
		out.Value[i] = ring.Poly{Coeffs: ct.Value[i].Coeffs[:level+1]}
	}
	return out
}

func (x *rnsExec) newCt(deg, level int, meta *rlwe.MetaData) *rlwe.Ciphertext {
	ct := ckks.NewCiphertext(x.params, deg, level)
	if meta != nil {
		*ct.MetaData = *meta.CopyNew()
	}
	return ct
}

func (x *rnsExec) moduli(level int) []uint64 { return x.params.Q()[:level+1] }

// mapLimbs applies f to every coefficient of every limb of p (in place on out).
func (x *rnsExec) mapLimbs(level int, out ring.Poly, f func(i int, q uint64, bred [2]uint64, j int) uint64) {
	sub := x.params.RingQ().SubRings
	for i := 0; i <= level; i++ {
		q, br := sub[i].Modulus, sub[i].BRedConstant
		row := out.Coeffs[i]
		for j := range row {
			row[j] = f(i, q, br, j)
		}
	}
}

func (x *rnsExec) run(op rnsOp, a, b *rlwe.Ciphertext, blob []uint32) *rlwe.Ciphertext {
	l := op.Level
	rq := x.params.RingQ().AtLevel(l)
	var meta *rlwe.MetaData
	if a != nil {
		meta = a.MetaData
		a = trunc(a, l)
	}
	if b != nil {
		b = trunc(b, l)
	}
	k := func(i int) uint64 { return uint64(blob[op.K+i]) }
	pt := func(i, j int) uint64 { return uint64(blob[op.Pt+i*compactWords+j/(x.params.N()/compactWords)]) }
	switch op.Op {
	case "zero":
		return x.newCt(op.Deg, l, nil)
	case "add", "sub":
		out := x.newCt(op.Deg, l, meta)
		for i := 0; i <= op.Deg; i++ {
			switch {
			case i <= a.Degree() && i <= b.Degree():
				if op.Op == "add" {
					rq.Add(a.Value[i], b.Value[i], out.Value[i])
				} else {
					rq.Sub(a.Value[i], b.Value[i], out.Value[i])
				}
			case i <= a.Degree():
				out.Value[i].CopyLvl(l, a.Value[i])
			case op.Op == "add":
				out.Value[i].CopyLvl(l, b.Value[i])
			default:
				rq.Neg(b.Value[i], out.Value[i])
			}
		}
		return out
	case "smul":
		out := x.newCt(a.Degree(), l, meta)
		for c := range out.Value {
			src := a.Value[c]
			x.mapLimbs(l, out.Value[c], func(i int, q uint64, br [2]uint64, j int) uint64 { return ring.BRed(src.Coeffs[i][j], k(i), q, br) })
		}
		return out
	case "sadd", "padd":
		out := x.newCt(a.Degree(), l, meta)
		for c := range out.Value {
			out.Value[c].CopyLvl(l, a.Value[c])
		}
		src := a.Value[0]
		x.mapLimbs(l, out.Value[0], func(i int, q uint64, _ [2]uint64, j int) uint64 {
			if op.Op == "padd" {
				return ring.CRed(src.Coeffs[i][j]+pt(i, j), q)
			}
			return ring.CRed(src.Coeffs[i][j]+k(i), q)
		})
		return out
	case "smac":
		out := x.newCt(op.Deg, l, b.MetaData)
		for c := 0; c <= op.Deg; c++ {
			acc := b.Value[c]
			if c > a.Degree() {
				out.Value[c].CopyLvl(l, acc)
				continue
			}
			src := a.Value[c]
			x.mapLimbs(l, out.Value[c], func(i int, q uint64, br [2]uint64, j int) uint64 {
				return ring.CRed(acc.Coeffs[i][j]+ring.BRed(src.Coeffs[i][j], k(i), q, br), q)
			})
		}
		return out
	case "pmul":
		out := x.newCt(a.Degree(), l, meta)
		for c := range out.Value {
			src := a.Value[c]
			x.mapLimbs(l, out.Value[c], func(i int, q uint64, br [2]uint64, j int) uint64 { return ring.BRed(src.Coeffs[i][j], pt(i, j), q, br) })
		}
		return out
	case "tensor":
		out := x.newCt(2, l, meta)
		a0, a1, b0, b1 := a.Value[0], a.Value[1], b.Value[0], b.Value[1]
		x.mapLimbs(l, out.Value[0], func(i int, q uint64, br [2]uint64, j int) uint64 { return ring.BRed(a0.Coeffs[i][j], b0.Coeffs[i][j], q, br) })
		x.mapLimbs(l, out.Value[1], func(i int, q uint64, br [2]uint64, j int) uint64 {
			return ring.CRed(ring.BRed(a0.Coeffs[i][j], b1.Coeffs[i][j], q, br)+ring.BRed(a1.Coeffs[i][j], b0.Coeffs[i][j], q, br), q)
		})
		x.mapLimbs(l, out.Value[2], func(i int, q uint64, br [2]uint64, j int) uint64 { return ring.BRed(a1.Coeffs[i][j], b1.Coeffs[i][j], q, br) })
		return out
	case "relin":
		rlk, err := x.ev.CheckAndGetRelinearizationKey()
		if err != nil {
			panic(err)
		}
		tmp := x.newCt(1, l, meta)
		x.ev.GadgetProduct(l, a.Value[2], &rlk.GadgetCiphertext, tmp)
		out := x.newCt(1, l, meta)
		rq.Add(a.Value[0], tmp.Value[0], out.Value[0])
		rq.Add(a.Value[1], tmp.Value[1], out.Value[1])
		return out
	case "rescale":
		out := x.newCt(a.Degree(), l-1, meta)
		for c := range out.Value {
			rq.DivRoundByLastModulusNTT(a.Value[c], out.Value[c])
		}
		return out
	case "rotate":
		out := x.newCt(1, l, meta)
		if err := x.ev.Automorphism(a, op.Gal, out); err != nil {
			panic(err)
		}
		return out
	}
	panic("unknown rns op " + op.Op)
}

// sameCt reports the first difference between two ciphertexts (degree, level, any coefficient).
func sameCt(x, y *rlwe.Ciphertext) error {
	if x.Degree() != y.Degree() || x.Level() != y.Level() {
		return fmt.Errorf("shape (degree %d, level %d) vs (degree %d, level %d)", x.Degree(), x.Level(), y.Degree(), y.Level())
	}
	for c := range x.Value {
		for i := 0; i <= x.Level(); i++ {
			for j, v := range x.Value[c].Coeffs[i] {
				if y.Value[c].Coeffs[i][j] != v {
					return fmt.Errorf("component %d limb %d coefficient %d: %d vs %d", c, i, j, v, y.Value[c].Coeffs[i][j])
				}
			}
		}
	}
	return nil
}

func isZeroCt(ct *rlwe.Ciphertext) bool {
	for c := range ct.Value {
		for _, row := range ct.Value[c].Coeffs {
			for _, v := range row {
				if v != 0 {
					return false
				}
			}
		}
	}
	return true
}

// recorder implements schemes.Evaluator (for Lattigo's polynomial evaluator) and heEval (for the circuit).
type recorder struct {
	*ckks.Evaluator
	params ckks.Parameters
	x      *rnsExec
	reg    map[weak.Pointer[rlwe.Ciphertext]]int // weak: the map must not keep every ciphertext alive
	prog   *gpuProgram
	blob   []uint32
	from   string
	gal    map[uint64]bool
}

func newRecorder(params ckks.Parameters, ev *ckks.Evaluator) *recorder {
	return &recorder{Evaluator: ev, params: params, x: &rnsExec{params: params, ev: ev}, reg: map[weak.Pointer[rlwe.Ciphertext]]int{},
		prog: &gpuProgram{Counts: map[string]int{}}, gal: map[uint64]bool{}}
}

func (r *recorder) newReg(deg, level int) int {
	r.prog.Regs = append(r.prog.Regs, rnsReg{Deg: deg, Level: level})
	return len(r.prog.Regs) - 1
}

// input registers an external ciphertext (an encrypted one-hot).
func (r *recorder) input(ct *rlwe.Ciphertext) int {
	id := r.newReg(ct.Degree(), ct.Level())
	r.reg[weak.Make(ct)] = id
	return id
}

// id returns the register of a ciphertext. A ciphertext the recorder has not seen must be a fresh zero
// ciphertext allocated inside Lattigo (the polynomial evaluator starts its sums from one).
func (r *recorder) id(ct *rlwe.Ciphertext) int {
	if id, ok := r.reg[weak.Make(ct)]; ok {
		return id
	}
	if !isZeroCt(ct) {
		panic(fmt.Sprintf("recorder: %s used a ciphertext that was never recorded", r.from))
	}
	id := r.emitOp(rnsOp{Op: "zero", A: -1, B: -1, Level: ct.Level(), Deg: ct.Degree()})
	r.reg[weak.Make(ct)] = id
	return id
}

func (r *recorder) alias(ct *rlwe.Ciphertext) *rlwe.Ciphertext {
	cp := ct.CopyNew()
	r.reg[weak.Make(cp)] = r.id(ct)
	return cp
}

func (r *recorder) emitOp(op rnsOp) int {
	if op.K == 0 && op.Op != "smul" && op.Op != "sadd" && op.Op != "smac" {
		op.K = -1
	}
	if op.Pt == 0 && op.Op != "pmul" && op.Op != "padd" {
		op.Pt = -1
	}
	op.From = r.from
	op.Out = r.newReg(op.Deg, outLevel(op))
	r.prog.Ops = append(r.prog.Ops, op)
	r.prog.Counts[op.Op]++
	if op.Op == "rotate" {
		r.gal[op.Gal] = true
	}
	return op.Out
}

func outLevel(op rnsOp) int {
	if op.Op == "rescale" {
		return op.Level - 1
	}
	return op.Level
}

func (r *recorder) words(v []uint32) int {
	off := len(r.blob)
	r.blob = append(r.blob, v...)
	return off
}

// lowered is a register together with its CPU reference value.
type lowered struct {
	id int
	ct *rlwe.Ciphertext
}

func (r *recorder) op(op rnsOp, a, b *lowered) lowered {
	var ac, bc *rlwe.Ciphertext
	op.A, op.B = -1, -1
	if a != nil {
		op.A, ac = a.id, a.ct
	}
	if b != nil {
		op.B, bc = b.id, b.ct
	}
	id := r.emitOp(op)
	op = r.prog.Ops[len(r.prog.Ops)-1]
	return lowered{id: id, ct: r.x.run(op, ac, bc, r.blob)}
}

func (r *recorder) snap(ct *rlwe.Ciphertext) *lowered {
	return &lowered{id: r.id(ct), ct: ct.CopyNew()}
}

// bind checks a lowering against Lattigo's result and makes the result's pointer name its register.
func (r *recorder) bind(res lowered, out *rlwe.Ciphertext) {
	if err := sameCt(res.ct, out); err != nil {
		panic(fmt.Sprintf("recorder: lowering of %s differs from Lattigo: %v", r.from, err))
	}
	r.reg[weak.Make(out)] = res.id
}

func modInv(a, q uint64) uint64 { return ring.ModExp(a, q-2, q) }

// inferScalar finds K (one word per limb) with out = acc + in*K on component 0, from the first coefficient where in
// is non-zero. The lowering is then checked on every coefficient, so a wrong inference cannot pass.
func (r *recorder) inferScalar(in, acc, out *rlwe.Ciphertext, level int) []uint32 {
	sub := r.params.RingQ().SubRings
	K := make([]uint32, level+1)
	for i := 0; i <= level; i++ {
		q, br := sub[i].Modulus, sub[i].BRedConstant
		for j := range in.Value[0].Coeffs[i] {
			x := in.Value[0].Coeffs[i][j]
			if x == 0 {
				continue
			}
			d := out.Value[0].Coeffs[i][j]
			if acc != nil {
				d = ring.CRed(d+q-acc.Value[0].Coeffs[i][j], q)
			}
			K[i] = uint32(ring.BRed(d, modInv(x, q), q, br))
			break
		}
	}
	return K
}

// inferAdd finds K with out.c0 = in.c0 + K.
func (r *recorder) inferAdd(in, out *rlwe.Ciphertext, level int) []uint32 {
	K := make([]uint32, level+1)
	for i := 0; i <= level; i++ {
		q := r.params.Q()[i]
		K[i] = uint32(ring.CRed(out.Value[0].Coeffs[i][0]+q-in.Value[0].Coeffs[i][0], q))
	}
	return K
}

func (r *recorder) intConst(v *big.Int, level int) []uint32 {
	K := make([]uint32, level+1)
	for i, q := range r.params.Q()[:level+1] {
		K[i] = uint32(new(big.Int).Mod(v, new(big.Int).SetUint64(q)).Uint64())
	}
	return K
}

// compact encodes values the way Lattigo would for this metadata and level, and returns the compact NTT form.
func (r *recorder) compact(values []float64, meta *rlwe.MetaData, scale rlwe.Scale, level int) []uint32 {
	pt := ckks.NewPlaintext(r.params, level)
	*pt.MetaData = *meta.CopyNew()
	pt.Scale = scale
	if err := r.Evaluator.Encoder.Encode(values, pt); err != nil {
		panic(err)
	}
	if !pt.IsNTT {
		panic("plaintext is not in the NTT domain")
	}
	block := r.params.N() / compactWords
	out := make([]uint32, 0, (level+1)*compactWords)
	for i := 0; i <= level; i++ {
		row := pt.Value.Coeffs[i]
		for j, v := range row {
			if v != row[j-j%block] {
				panic("plaintext is not constant on NTT blocks; the compact layout does not apply")
			}
		}
		for b := 0; b < compactWords; b++ {
			out = append(out, uint32(row[b*block]))
		}
	}
	return out
}

// scaleRatioInt mirrors Lattigo's evaluateInPlace: the larger scale over the smaller one, truncated to an integer.
func scaleRatioInt(big1, small rlwe.Scale) *big.Int {
	ratioFlo := big1.Div(small).Value
	ratioInt, _ := ratioFlo.Int(nil)
	return ratioInt
}

// --- heEval and schemes.Evaluator ---

func (r *recorder) addSub(kind string, op0 *rlwe.Ciphertext, op1 rlwe.Operand, outp **rlwe.Ciphertext, real func() error) error {
	a := r.snap(op0)
	switch v := op1.(type) {
	case *rlwe.Ciphertext:
		b := r.snap(v)
		if err := real(); err != nil {
			return err
		}
		out := *outp
		l := out.Level()
		one := big.NewInt(1)
		switch cmp := a.ct.Scale.Cmp(b.ct.Scale); {
		case cmp > 0:
			if k := scaleRatioInt(a.ct.Scale, b.ct.Scale); k.Sign() > 0 && k.Cmp(one) != 0 {
				lb := r.op(rnsOp{Op: "smul", Level: l, Deg: b.ct.Degree(), K: r.words(r.intConst(k, l))}, b, nil)
				b = &lb
			}
		case cmp < 0:
			if k := scaleRatioInt(b.ct.Scale, a.ct.Scale); k.Sign() > 0 && k.Cmp(one) != 0 {
				la := r.op(rnsOp{Op: "smul", Level: l, Deg: a.ct.Degree(), K: r.words(r.intConst(k, l))}, a, nil)
				a = &la
			}
		}
		res := r.op(rnsOp{Op: kind, Level: l, Deg: max(a.ct.Degree(), b.ct.Degree())}, a, b)
		res.ct.Scale = out.Scale
		r.bind(res, out)
	case []float64:
		if kind != "add" {
			panic("recorder: vector subtraction is not lowered")
		}
		if err := real(); err != nil {
			return err
		}
		out := *outp
		pt := r.compact(v, op0.MetaData, op0.Scale, out.Level())
		r.bind(r.op(rnsOp{Op: "padd", Level: out.Level(), Deg: a.ct.Degree(), Pt: r.words(pt)}, a, nil), out)
	default:
		if err := real(); err != nil {
			return err
		}
		out := *outp
		l := out.Level()
		r.bind(r.op(rnsOp{Op: "sadd", Level: l, Deg: a.ct.Degree(), K: r.words(r.inferAdd(a.ct, out, l))}, a, nil), out)
	}
	return nil
}

func (r *recorder) Add(op0 *rlwe.Ciphertext, op1 rlwe.Operand, out *rlwe.Ciphertext) error {
	return r.addSub("add", op0, op1, &out, func() error { return r.Evaluator.Add(op0, op1, out) })
}

func (r *recorder) Sub(op0 *rlwe.Ciphertext, op1 rlwe.Operand, out *rlwe.Ciphertext) error {
	return r.addSub("sub", op0, op1, &out, func() error { return r.Evaluator.Sub(op0, op1, out) })
}

func (r *recorder) AddNew(op0 *rlwe.Ciphertext, op1 rlwe.Operand) (*rlwe.Ciphertext, error) {
	var out *rlwe.Ciphertext
	err := r.addSub("add", op0, op1, &out, func() (err error) { out, err = r.Evaluator.AddNew(op0, op1); return })
	return out, err
}

func (r *recorder) SubNew(op0 *rlwe.Ciphertext, op1 rlwe.Operand) (*rlwe.Ciphertext, error) {
	panic("recorder: SubNew is not lowered")
}

// mul lowers Mul and MulRelin (relin true) for every operand kind.
func (r *recorder) mul(op0 *rlwe.Ciphertext, op1 rlwe.Operand, relin bool, outp **rlwe.Ciphertext, real func() error) error {
	a := r.snap(op0)
	switch v := op1.(type) {
	case *rlwe.Ciphertext:
		b := r.snap(v)
		if err := real(); err != nil {
			return err
		}
		out := *outp
		l := out.Level()
		res := r.op(rnsOp{Op: "tensor", Level: l, Deg: 2}, a, b)
		if relin {
			res = r.op(rnsOp{Op: "relin", Level: l, Deg: 1}, &res, nil)
		}
		res.ct.Scale = out.Scale
		r.bind(res, out)
	case []float64:
		l := op0.Level()
		pt := r.compact(v, op0.MetaData, rlwe.NewScale(r.params.Q()[l]), l)
		if err := real(); err != nil {
			return err
		}
		out := *outp
		res := r.op(rnsOp{Op: "pmul", Level: out.Level(), Deg: a.ct.Degree(), Pt: r.words(pt)}, a, nil)
		res.ct.Scale = out.Scale
		r.bind(res, out)
	default:
		if err := real(); err != nil {
			return err
		}
		out := *outp
		l := out.Level()
		res := r.op(rnsOp{Op: "smul", Level: l, Deg: a.ct.Degree(), K: r.words(r.inferScalar(a.ct, nil, out, l))}, a, nil)
		res.ct.Scale = out.Scale
		r.bind(res, out)
	}
	return nil
}

func (r *recorder) Mul(op0 *rlwe.Ciphertext, op1 rlwe.Operand, out *rlwe.Ciphertext) error {
	return r.mul(op0, op1, false, &out, func() error { return r.Evaluator.Mul(op0, op1, out) })
}

func (r *recorder) MulNew(op0 *rlwe.Ciphertext, op1 rlwe.Operand) (*rlwe.Ciphertext, error) {
	var out *rlwe.Ciphertext
	err := r.mul(op0, op1, false, &out, func() (err error) { out, err = r.Evaluator.MulNew(op0, op1); return })
	return out, err
}

func (r *recorder) MulRelin(op0 *rlwe.Ciphertext, op1 rlwe.Operand, out *rlwe.Ciphertext) error {
	return r.mul(op0, op1, true, &out, func() error { return r.Evaluator.MulRelin(op0, op1, out) })
}

func (r *recorder) MulRelinNew(op0 *rlwe.Ciphertext, op1 rlwe.Operand) (*rlwe.Ciphertext, error) {
	var out *rlwe.Ciphertext
	err := r.mul(op0, op1, true, &out, func() (err error) { out, err = r.Evaluator.MulRelinNew(op0, op1); return })
	return out, err
}

// MulThenAdd with a scalar: Lattigo first multiplies the accumulator by q_level when both scales are equal and the
// scalar is not an integer, then adds op0 * K.
func (r *recorder) MulThenAdd(op0 *rlwe.Ciphertext, op1 rlwe.Operand, out *rlwe.Ciphertext) error {
	switch op1.(type) {
	case *rlwe.Ciphertext, []float64:
		panic("recorder: MulThenAdd with a ciphertext or vector is not lowered")
	}
	a, acc := r.snap(op0), r.snap(out)
	scaleAcc := op0.Scale.Cmp(out.Scale) == 0 && !bignum.ToComplex(op1, r.params.EncodingPrecision()).IsInt()
	if err := r.Evaluator.MulThenAdd(op0, op1, out); err != nil {
		return err
	}
	l := out.Level()
	if scaleAcc {
		s := r.op(rnsOp{Op: "smul", Level: l, Deg: acc.ct.Degree(), K: r.words(r.intConst(new(big.Int).SetUint64(r.params.Q()[l]), l))}, acc, nil)
		acc = &s
	}
	K := r.inferScalar(a.ct, acc.ct, out, l)
	res := r.op(rnsOp{Op: "smac", Level: l, Deg: out.Degree(), K: r.words(K)}, a, acc)
	res.ct.Scale = out.Scale
	r.bind(res, out)
	return nil
}

func (r *recorder) Relinearize(op0, out *rlwe.Ciphertext) error {
	a := r.snap(op0)
	if err := r.Evaluator.Relinearize(op0, out); err != nil {
		return err
	}
	res := r.op(rnsOp{Op: "relin", Level: out.Level(), Deg: 1}, a, nil)
	r.bind(res, out)
	return nil
}

func (r *recorder) Rescale(op0, out *rlwe.Ciphertext) error {
	a := r.snap(op0)
	l := op0.Level()
	if err := r.Evaluator.Rescale(op0, out); err != nil {
		return err
	}
	res := r.op(rnsOp{Op: "rescale", Level: l, Deg: a.ct.Degree()}, a, nil)
	res.ct.Scale = out.Scale
	r.bind(res, out)
	return nil
}

func (r *recorder) RotateNew(op0 *rlwe.Ciphertext, k int) (*rlwe.Ciphertext, error) {
	a := r.snap(op0)
	out, err := r.Evaluator.RotateNew(op0, k)
	if err != nil {
		return nil, err
	}
	r.bind(r.op(rnsOp{Op: "rotate", Level: out.Level(), Deg: 1, Gal: r.params.GaloisElement(k)}, a, nil), out)
	return out, nil
}

// --- program finishing and replay ---

// finishSteps derives, per step, the KV state it reads and keeps (every register crossing a step boundary must be a
// named cache entry) and the registers each op frees: registers defined in a step that no later step needs, at their
// last use in that step (never a step's output).
func (r *recorder) finishSteps(names map[int]string) error {
	p := r.prog
	def := map[int]int{}
	for si, st := range p.Steps {
		for _, in := range st.Inputs {
			def[in] = si
		}
		for i := st.OpFirst; i < st.OpEnd; i++ {
			def[p.Ops[i].Out] = si
		}
	}
	lastStep := map[int]int{}
	lastUse := map[int]int{}
	for si, st := range p.Steps {
		for i := st.OpFirst; i < st.OpEnd; i++ {
			for _, in := range []int{p.Ops[i].A, p.Ops[i].B} {
				if in < 0 {
					continue
				}
				lastStep[in] = si
				if def[in] == si {
					lastUse[in] = i
				}
			}
		}
	}
	for si := range p.Steps {
		st := &p.Steps[si]
		seenIn := map[int]bool{}
		for i := st.OpFirst; i < st.OpEnd; i++ {
			for _, in := range []int{p.Ops[i].A, p.Ops[i].B} {
				if in < 0 || def[in] == si || seenIn[in] {
					continue
				}
				if names[in] == "" {
					return fmt.Errorf("step %d reads register %d from an earlier step that is not a KV cache entry", si, in)
				}
				seenIn[in] = true
				st.StateIn = append(st.StateIn, namedReg{Name: names[in], Reg: in})
			}
		}
		// Every cache entry a step makes is kept: a later step may come from another compiled session.
		for reg, d := range def {
			if d != si {
				continue
			}
			if names[reg] != "" {
				st.StateOut = append(st.StateOut, namedReg{Name: names[reg], Reg: reg})
			} else if lastStep[reg] > si {
				return fmt.Errorf("register %d of step %d is used later but is not a KV cache entry", reg, si)
			}
		}
		sortNamed(st.StateIn)
		sortNamed(st.StateOut)
		st.Counts = map[string]int{}
		for i := st.OpFirst; i < st.OpEnd; i++ {
			st.Counts[p.Ops[i].Op]++
		}
	}
	for reg, d := range def {
		if lastStep[reg] > d || reg == p.Steps[d].Output || names[reg] != "" {
			continue
		}
		at, ok := lastUse[reg]
		if !ok {
			// Defined and never read (should not happen); free right after it is made.
			for i := p.Steps[d].OpFirst; i < p.Steps[d].OpEnd; i++ {
				if p.Ops[i].Out == reg {
					at = i
				}
			}
		}
		p.Ops[at].Free = append(p.Ops[at].Free, reg)
	}
	for i := range p.Ops {
		sortInts(p.Ops[i].Free)
	}
	for g := range r.gal {
		p.Galois = append(p.Galois, g)
	}
	sortU64(p.Galois)
	return nil
}

func sortNamed(v []namedReg) {
	for i := 1; i < len(v); i++ {
		for j := i; j > 0 && v[j].Reg < v[j-1].Reg; j-- {
			v[j], v[j-1] = v[j-1], v[j]
		}
	}
}

func sortInts(v []int) {
	for i := 1; i < len(v); i++ {
		for j := i; j > 0 && v[j] < v[j-1]; j-- {
			v[j], v[j-1] = v[j-1], v[j]
		}
	}
}

func sortU64(v []uint64) {
	for i := 1; i < len(v); i++ {
		for j := i; j > 0 && v[j] < v[j-1]; j-- {
			v[j], v[j-1] = v[j-1], v[j]
		}
	}
}

// replaySteps runs every step of a session on the CPU reference executor, carrying the named KV state, and returns
// each step's output.
func replaySteps(x *rnsExec, prog *gpuProgram, blob []uint32, inputs [][]*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	state := map[string]*rlwe.Ciphertext{}
	var outs []*rlwe.Ciphertext
	for si, st := range prog.Steps {
		if len(inputs[si]) != len(st.Inputs) {
			return nil, fmt.Errorf("step %d takes %d inputs", si, len(st.Inputs))
		}
		regs := map[int]*rlwe.Ciphertext{}
		for i, id := range st.Inputs {
			regs[id] = inputs[si][i]
		}
		for _, sr := range st.StateIn {
			if regs[sr.Reg] = state[sr.Name]; regs[sr.Reg] == nil {
				return nil, fmt.Errorf("step %d needs state %s", si, sr.Name)
			}
		}
		for _, op := range prog.Ops[st.OpFirst:st.OpEnd] {
			var a, b *rlwe.Ciphertext
			if op.A >= 0 {
				a = regs[op.A]
			}
			if op.B >= 0 {
				b = regs[op.B]
			}
			regs[op.Out] = x.run(op, a, b, blob)
			for _, f := range op.Free {
				delete(regs, f)
			}
		}
		for _, sr := range st.StateOut {
			state[sr.Name] = regs[sr.Reg]
		}
		outs = append(outs, regs[st.Output])
	}
	return outs, nil
}

func blobBytes(blob []uint32) []byte {
	raw := make([]byte, 4*len(blob))
	for i, w := range blob {
		binary.LittleEndian.PutUint32(raw[4*i:], w)
	}
	return raw
}

func loadGPUProgram(base string) (*gpuProgram, []uint32, error) {
	raw, err := os.ReadFile(base + ".json")
	if err != nil {
		return nil, nil, err
	}
	prog := &gpuProgram{}
	if err := json.Unmarshal(raw, prog); err != nil {
		return nil, nil, err
	}
	b, err := os.ReadFile(base + ".bin")
	if err != nil {
		return nil, nil, err
	}
	sum := sha256.Sum256(b)
	if prog.Schema != gpuProgramSchema || len(prog.Steps) == 0 || len(b) != 4*prog.BlobWords || "sha256:"+hex.EncodeToString(sum[:]) != prog.BlobDigest {
		return nil, nil, fmt.Errorf("program %s is malformed or its constants do not match their digest", base)
	}
	blob := make([]uint32, prog.BlobWords)
	for i := range blob {
		blob[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	return prog, blob, nil
}

// throwawayKeys makes a secret key and evaluation keys for compiling and for kernel test vectors only.
func throwawayKeys(params ckks.Parameters) (*rlwe.SecretKey, *rlwe.MemEvaluationKeySet) {
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	var gks []*rlwe.GaloisKey
	for _, rot := range rotationSteps() {
		gks = append(gks, kgen.GenGaloisKeyNew(params.GaloisElement(rot), sk))
	}
	return sk, rlwe.NewMemEvaluationKeySet(kgen.GenRelinearizationKeyNew(sk), gks...)
}

func sealOneHotWith(params ckks.Parameters, enc *rlwe.Encryptor, tok int) (*rlwe.Ciphertext, error) {
	v := make([]float64, vocab)
	v[tok] = 1
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	pt.LogDimensions = ring.Dimensions{Rows: 0, Cols: logSlots}
	if err := ckks.NewEncoder(params).Encode(replicate(v, vocab), pt); err != nil {
		return nil, err
	}
	return enc.EncryptNew(pt)
}

// compileGPUProgram records a session: the prefill of `prompt` tokens, then one decode step for every position up to
// and including decodeThrough (no decode steps when decodeThrough < prompt).
func compileGPUProgram(prof profile, modelPath string, prompt, decodeThrough int, logf func(string, ...any)) (*gpuProgram, []uint32, error) {
	m, modelID, err := loadModel(modelPath)
	if err != nil {
		return nil, nil, err
	}
	pl, err := newPlan(m)
	if err != nil {
		return nil, nil, err
	}
	if prof.Wire != wireRawU32 {
		return nil, nil, fmt.Errorf("profile %s is not a u32 (WebGPU) profile", prof.ID)
	}
	params, err := prof.params()
	if err != nil {
		return nil, nil, err
	}
	for _, q := range append(params.Q(), params.P()...) {
		if q >= 1<<32 {
			return nil, nil, fmt.Errorf("modulus %d does not fit a u32 limb", q)
		}
	}
	if prompt < 1 || prompt > context || decodeThrough >= context {
		return nil, nil, fmt.Errorf("prompt %d / decode-through %d outside the %d-position context", prompt, decodeThrough, context)
	}
	sk, evk := throwawayKeys(params)
	enc := rlwe.NewEncryptor(params, sk)
	ev := ckks.NewEvaluator(params, evk)
	ev.Encoder = ckks.NewEncoder(params, deterministicEncodingPrecision)
	rec := newRecorder(params, ev)
	pe := polynomial.NewEvaluator(params, ev)
	pe.Evaluator.Evaluator = rec
	cir := &circuit{params: params, eval: rec, poly: pe, copyCt: rec.alias, pl: pl, ops: map[string]int{}}
	t0 := time.Now()
	names := map[int]string{}
	var inputs [][]*rlwe.Ciphertext
	var outputs []*rlwe.Ciphertext
	var layers []gpuLayer
	first := 0
	mark := func(name string) {
		if len(rec.prog.Ops) > first {
			layers = append(layers, gpuLayer{Name: rec.from, First: first, Last: len(rec.prog.Ops) - 1})
		}
		first, rec.from = len(rec.prog.Ops), name
	}
	cir.mark = mark
	record := func(kind string, from, to int) error {
		st := gpuStep{Kind: kind, First: from, Position: to, OpFirst: len(rec.prog.Ops)}
		layers = nil
		var ins []*rlwe.Ciphertext
		var q, x0 *rlwe.Ciphertext
		for pos := from; pos <= to; pos++ {
			ct, err := sealOneHotWith(params, enc, (7*pos+3)%vocab)
			if err != nil {
				return err
			}
			ins = append(ins, ct)
			st.Inputs = append(st.Inputs, rec.input(ct))
			mark(fmt.Sprintf("embed-%d", pos))
			if q, x0, err = cir.lookup(ct, pos, pos == to); err != nil {
				return err
			}
			names[rec.id(cir.keys[pos])] = fmt.Sprintf("k%d", pos)
			names[rec.id(cir.vals[pos])] = fmt.Sprintf("v%d", pos)
		}
		mark("attention-scores")
		logits, err := cir.forward(q, x0)
		if err != nil {
			return err
		}
		mark("end")
		st.Output, st.OutputLevel, st.OutputScale = rec.id(logits), logits.Level(), logits.Scale.Float64()
		st.OpEnd, st.Layers = len(rec.prog.Ops), layers
		rec.prog.Steps = append(rec.prog.Steps, st)
		inputs, outputs = append(inputs, ins), append(outputs, logits)
		logf("recorded %s at position %d: ops %d..%d (%.1fs)", kind, to, st.OpFirst, st.OpEnd-1, time.Since(t0).Seconds())
		return nil
	}
	if err := record("prefill", 0, prompt-1); err != nil {
		return nil, nil, err
	}
	for pos := prompt; pos <= decodeThrough; pos++ {
		if err := record("decode", pos, pos); err != nil {
			return nil, nil, err
		}
	}
	if err := rec.finishSteps(names); err != nil {
		return nil, nil, err
	}
	p := rec.prog
	p.Schema, p.Protocol, p.Profile, p.ProfileDigest, p.Model = gpuProgramSchema, protocolID, prof.ID, prof.digest(), modelID
	p.LogN, p.LogSlots, p.Q, p.P, p.Psi = params.LogN(), logSlots, params.Q(), params.P(), primitiveRoots(params)
	p.InputScale = inputs[0][0].Scale.Float64()
	raw := blobBytes(rec.blob)
	sum := sha256.Sum256(raw)
	p.BlobWords, p.BlobDigest = len(rec.blob), "sha256:"+hex.EncodeToString(sum[:])
	// Second check: the steps alone, from the inputs and the carried KV state, reproduce every output of the circuit.
	got, err := replaySteps(rec.x, p, rec.blob, inputs)
	if err != nil {
		return nil, nil, err
	}
	for i := range got {
		if err := sameCt(got[i], outputs[i]); err != nil {
			return nil, nil, fmt.Errorf("replayed step %d differs from the circuit: %w", i, err)
		}
	}
	logf("session: %d steps, %d ops, %d registers, %d constant words; replay equals the circuit at every step (%.1fs)",
		len(p.Steps), len(p.Ops), len(p.Regs), p.BlobWords, time.Since(t0).Seconds())
	return p, rec.blob, nil
}

func gpuCompileMain(args []string) error {
	fs := flag.NewFlagSet("gpu-compile", flag.ExitOnError)
	profileID := fs.String("profile", "research-n16-u32", "u32 profile")
	modelPath := fs.String("model", "tiny-model.json", "public model fixture")
	positions := fs.Int("positions", 3, "prompt tokens in the prefill step")
	decodeThrough := fs.Int("decode-through", -1, "compile decode steps for positions up to this one (-1: none)")
	out := fs.String("out", "gpu-program", "output base name (.json and .bin)")
	fs.Parse(args)
	prof, err := lookupProfile(*profileID)
	if err != nil {
		return err
	}
	prog, blob, err := compileGPUProgram(prof, *modelPath, *positions, *decodeThrough, func(f string, a ...any) { fmt.Fprintf(os.Stderr, f+"\n", a...) })
	if err != nil {
		return err
	}
	raw, _ := json.Marshal(prog)
	if err := os.WriteFile(*out+".json", raw, 0o644); err != nil {
		return err
	}
	return os.WriteFile(*out+".bin", blobBytes(blob), 0o644)
}
