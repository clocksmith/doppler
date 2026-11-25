package main

// gpu_vectors.go: test vectors for the WebGPU kernels, computed by Lattigo (rnsExec), so no CGO is needed. The JS
// tests run each case on the GPU and require byte-identical results.
//
// Layout of the output directory:
//
//	params.json   moduli, primitive 2N-th roots (Lattigo's), case list
//	keys.bin      relinearization key then the Galois keys, each raw-u32 framed as u64 length + key
//	ntt.bin       per limb: N random residues, then Lattigo's NTT of them (Q then P moduli)
//	case-*.bin    inputs, op, constants and expected output of one rnsOp

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type vectorCase struct {
	Name   string    `json:"name"`
	File   string    `json:"file"`
	Op     rnsOp     `json:"op"`
	Inputs int       `json:"inputs"`
	Words  int       `json:"words"`
}

type vectorParams struct {
	Schema  string       `json:"schema"`
	Profile string       `json:"profile"`
	LogN    int          `json:"logN"`
	Q       []uint64     `json:"q"`
	P       []uint64     `json:"p"`
	Psi     []uint64     `json:"psi"`
	Galois  []uint64     `json:"galois"`
	Cases   []vectorCase `json:"cases"`
}

// primitiveRoots returns Lattigo's primitive 2N-th root for every modulus, Q then P.
func primitiveRoots(params ckks.Parameters) []uint64 {
	var out []uint64
	for _, r := range []*ring.Ring{params.RingQ(), params.RingP()} {
		for _, s := range r.SubRings {
			out = append(out, ring.ModExp(s.PrimitiveRoot, (s.Modulus-1)/s.NthRoot, s.Modulus))
		}
	}
	return out
}

func randomCt(params ckks.Parameters, rng *rand.Rand, deg, level int) *rlwe.Ciphertext {
	ct := ckks.NewCiphertext(params, deg, level)
	ct.IsNTT, ct.LogDimensions = true, ring.Dimensions{Rows: 0, Cols: logSlots}
	for c := range ct.Value {
		for i := 0; i <= level; i++ {
			q := params.Q()[i]
			for j := range ct.Value[c].Coeffs[i] {
				ct.Value[c].Coeffs[i][j] = rng.Uint64() % q
			}
		}
	}
	return ct
}

func frame(dst []byte, b []byte) []byte {
	dst = binary.LittleEndian.AppendUint64(dst, uint64(len(b)))
	return append(dst, b...)
}

func gpuVectorsMain(args []string) error {
	fs := flag.NewFlagSet("gpu-vectors", flag.ExitOnError)
	profileID := fs.String("profile", "dev-insecure-n12-u32", "u32 profile")
	outDir := fs.String("out", "gpu-vectors", "output directory")
	seed := fs.Int64("seed", 1, "random seed for inputs and constants")
	fs.Parse(args)
	prof, err := lookupProfile(*profileID)
	if err != nil {
		return err
	}
	if prof.Wire != wireRawU32 {
		return fmt.Errorf("profile %s is not a u32 profile", prof.ID)
	}
	params, err := prof.params()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		return err
	}
	rng := rand.New(rand.NewSource(*seed))
	_, evk := throwawayKeys(params)
	ev := ckks.NewEvaluator(params, evk)
	x := &rnsExec{params: params, ev: ev}
	vp := vectorParams{Schema: "doppler.protected-gpu-vectors/v0", Profile: prof.ID, LogN: params.LogN(), Q: params.Q(), P: params.P(),
		Psi: primitiveRoots(params)}

	// Keys.
	var keys []byte
	rlk, _ := evk.GetRelinearizationKey()
	kb, err := encodeKeyRaw(params, keyKindRelin, 0, &rlk.GadgetCiphertext)
	if err != nil {
		return err
	}
	keys = frame(keys, kb)
	for _, rot := range rotationSteps() {
		g := params.GaloisElement(rot)
		gk, _ := evk.GetGaloisKey(g)
		if kb, err = encodeKeyRaw(params, keyKindGal, g, &gk.GadgetCiphertext); err != nil {
			return err
		}
		keys = frame(keys, kb)
		vp.Galois = append(vp.Galois, g)
	}
	if err := os.WriteFile(filepath.Join(*outDir, "keys.bin"), keys, 0o644); err != nil {
		return err
	}

	// NTT of random residues under every modulus.
	var nttRaw []byte
	n := params.N()
	for _, r := range []*ring.Ring{params.RingQ(), params.RingP()} {
		for _, s := range r.SubRings {
			in := make([]uint64, n)
			for j := range in {
				in[j] = rng.Uint64() % s.Modulus
			}
			out := make([]uint64, n)
			s.NTT(in, out)
			nttRaw = putResidues(nttRaw, [][]uint64{in, out})
		}
	}
	if err := os.WriteFile(filepath.Join(*outDir, "ntt.bin"), nttRaw, 0o644); err != nil {
		return err
	}

	// One case per op and level. Layout: u32 words count, constants, then each input and the expected output as
	// u64-framed raw-u32 ciphertexts.
	maxL := params.MaxLevel()
	alpha := params.PCount()
	levels := []int{maxL, 2*alpha - 1, 2 * alpha, 2*alpha + 1, alpha, 3, 1}
	randWords := func(level int) []uint32 {
		w := make([]uint32, level+1)
		for i := range w {
			w[i] = uint32(rng.Uint64() % params.Q()[i])
		}
		return w
	}
	randPt := func(level int) []uint32 {
		w := make([]uint32, (level+1)*compactWords)
		for i := range w {
			w[i] = uint32(rng.Uint64() % params.Q()[i/compactWords])
		}
		return w
	}
	add := func(name string, op rnsOp, blob []uint32, ins ...*rlwe.Ciphertext) error {
		var a, b *rlwe.Ciphertext
		op.A, op.B = -1, -1
		if len(ins) > 0 {
			a, op.A = ins[0], 0
		}
		if len(ins) > 1 {
			b, op.B = ins[1], 1
		}
		got := x.run(op, a, b, blob)
		raw := binary.LittleEndian.AppendUint32(nil, uint32(len(blob)))
		raw = append(raw, blobBytes(blob)...)
		for _, ct := range append(ins, got) {
			cb, err := encodeCtRaw(params, ct)
			if err != nil {
				return err
			}
			raw = frame(raw, cb)
		}
		file := fmt.Sprintf("case-%03d.bin", len(vp.Cases))
		vp.Cases = append(vp.Cases, vectorCase{Name: name, File: file, Op: op, Inputs: len(ins), Words: len(blob)})
		return os.WriteFile(filepath.Join(*outDir, file), raw, 0o644)
	}
	for _, l := range levels {
		if l > maxL || l < 1 {
			continue
		}
		a1, b1 := randomCt(params, rng, 1, l), randomCt(params, rng, 1, l)
		a2 := randomCt(params, rng, 2, l)
		hi := randomCt(params, rng, 1, min(maxL, l+2)) // read at a lower level than its own
		cases := []struct {
			name string
			op   rnsOp
			blob []uint32
			ins  []*rlwe.Ciphertext
		}{
			{"add", rnsOp{Op: "add", Level: l, Deg: 2}, nil, []*rlwe.Ciphertext{hi, a2}},
			{"sub", rnsOp{Op: "sub", Level: l, Deg: 2}, nil, []*rlwe.Ciphertext{a1, a2}},
			{"smul", rnsOp{Op: "smul", Level: l, Deg: 2, K: 0}, randWords(l), []*rlwe.Ciphertext{a2}},
			{"sadd", rnsOp{Op: "sadd", Level: l, Deg: 1, K: 0}, randWords(l), []*rlwe.Ciphertext{hi}},
			{"smac", rnsOp{Op: "smac", Level: l, Deg: 2, K: 0}, randWords(l), []*rlwe.Ciphertext{a1, a2}},
			{"pmul", rnsOp{Op: "pmul", Level: l, Deg: 1, Pt: 0}, randPt(l), []*rlwe.Ciphertext{hi}},
			{"padd", rnsOp{Op: "padd", Level: l, Deg: 1, Pt: 0}, randPt(l), []*rlwe.Ciphertext{a1}},
			{"tensor", rnsOp{Op: "tensor", Level: l, Deg: 2}, nil, []*rlwe.Ciphertext{hi, b1}},
			{"relin", rnsOp{Op: "relin", Level: l, Deg: 1}, nil, []*rlwe.Ciphertext{a2}},
			{"rescale", rnsOp{Op: "rescale", Level: l, Deg: 2}, nil, []*rlwe.Ciphertext{a2}},
			{"rescale", rnsOp{Op: "rescale", Level: l, Deg: 1}, nil, []*rlwe.Ciphertext{b1}},
		}
		for _, rot := range rotationSteps() {
			cases = append(cases, struct {
				name string
				op   rnsOp
				blob []uint32
				ins  []*rlwe.Ciphertext
			}{fmt.Sprintf("rotate-%d", rot), rnsOp{Op: "rotate", Level: l, Deg: 1, Gal: params.GaloisElement(rot)}, nil, []*rlwe.Ciphertext{a1}})
		}
		for _, c := range cases {
			if c.op.K == 0 && c.op.Op != "smul" && c.op.Op != "sadd" && c.op.Op != "smac" {
				c.op.K = -1
			}
			if c.op.Op != "pmul" && c.op.Op != "padd" {
				c.op.Pt = -1
			}
			if err := add(fmt.Sprintf("%s@%d", c.name, l), c.op, c.blob, c.ins...); err != nil {
				return err
			}
		}
	}
	raw, _ := json.MarshalIndent(vp, "", " ")
	return os.WriteFile(filepath.Join(*outDir, "params.json"), raw, 0o644)
}
