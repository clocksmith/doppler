package main

// gpu_codec.go: the raw-u32/v0 wire format of the u32 profiles. One RNS residue is one little-endian u32, in
// Lattigo's NTT order, so the WebGPU evaluator uploads blobs to GPU buffers as they arrive. Every residue is checked
// to be below its modulus on decode.
//
// Ciphertext: "DHC0" | degree u32 | level u32 | logN u32 | logSlots u32 | 0 u32 | scale f64 |
//             (degree+1) x (level+1) x N residues (component, then limb, then coefficient).
// Key:        "DHK0" | kind u32 (0 relinearization, 1 Galois) | Galois element u64 | levelQ u32 | levelP u32 |
//             digits u32 | logN u32 | digits x 2 x ((levelQ+1) + (levelP+1)) x N residues, canonical (not Montgomery)
//             (digit, then component b/a, then Q limbs and P limbs, then coefficient).
// The original profiles keep Lattigo's own encoding.

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	rawCtMagic   = 0x30434844 // "DHC0"
	rawKeyMagic  = 0x304b4844 // "DHK0"
	rawCtHeader  = 32
	rawKeyHeader = 32
	keyKindRelin = 0
	keyKindGal   = 1
)

func putResidues(dst []byte, rows [][]uint64) []byte {
	for _, row := range rows {
		for _, v := range row {
			dst = binary.LittleEndian.AppendUint32(dst, uint32(v))
		}
	}
	return dst
}

// getResidues fills rows from raw words and checks every residue against its modulus.
func getResidues(raw []byte, rows [][]uint64, moduli []uint64) ([]byte, error) {
	for i, row := range rows {
		for j := range row {
			v := uint64(binary.LittleEndian.Uint32(raw[4*j:]))
			if v >= moduli[i] {
				return nil, fmt.Errorf("residue %d of limb %d is not below its modulus", j, i)
			}
			row[j] = v
		}
		raw = raw[4*len(row):]
	}
	return raw, nil
}

func encodeCtRaw(params ckks.Parameters, ct *rlwe.Ciphertext) ([]byte, error) {
	if !ct.IsNTT || ct.LogDimensions.Cols != logSlots {
		return nil, fmt.Errorf("ciphertext is not in the pinned NTT layout")
	}
	out := make([]byte, 0, rawCtHeader+4*(ct.Degree()+1)*(ct.Level()+1)*params.N())
	for _, w := range []uint32{rawCtMagic, uint32(ct.Degree()), uint32(ct.Level()), uint32(params.LogN()), logSlots, 0} {
		out = binary.LittleEndian.AppendUint32(out, w)
	}
	out = binary.LittleEndian.AppendUint64(out, math.Float64bits(ct.Scale.Float64()))
	for _, p := range ct.Value {
		out = putResidues(out, p.Coeffs[:ct.Level()+1])
	}
	return out, nil
}

func decodeCtRaw(params ckks.Parameters, raw []byte) (*rlwe.Ciphertext, error) {
	if len(raw) < rawCtHeader || binary.LittleEndian.Uint32(raw) != rawCtMagic {
		return nil, fmt.Errorf("not a raw-u32 ciphertext")
	}
	w := func(i int) int { return int(binary.LittleEndian.Uint32(raw[4*i:])) }
	deg, level := w(1), w(2)
	if w(3) != params.LogN() || w(4) != logSlots || deg < 1 || deg > 2 || level < 0 || level > params.MaxLevel() {
		return nil, fmt.Errorf("raw ciphertext header does not match the profile")
	}
	scale := math.Float64frombits(binary.LittleEndian.Uint64(raw[24:]))
	if !(scale > 0) || math.IsInf(scale, 0) {
		return nil, fmt.Errorf("raw ciphertext scale is invalid")
	}
	if len(raw) != rawCtHeader+4*(deg+1)*(level+1)*params.N() {
		return nil, fmt.Errorf("raw ciphertext has %d bytes, expected %d", len(raw), rawCtHeader+4*(deg+1)*(level+1)*params.N())
	}
	ct := ckks.NewCiphertext(params, deg, level)
	ct.IsNTT, ct.LogDimensions, ct.Scale = true, ring.Dimensions{Rows: 0, Cols: logSlots}, rlwe.NewScale(scale)
	body := raw[rawCtHeader:]
	var err error
	for _, p := range ct.Value {
		if body, err = getResidues(body, p.Coeffs, params.Q()); err != nil {
			return nil, err
		}
	}
	return ct, nil
}

func encodeKeyRaw(params ckks.Parameters, kind int, galEl uint64, g *rlwe.GadgetCiphertext) ([]byte, error) {
	if g.BaseTwoDecomposition != 0 || g.LevelQ() != params.MaxLevel() || g.LevelP() != params.MaxLevelP() {
		return nil, fmt.Errorf("evaluation key does not have the pinned gadget shape")
	}
	digits := len(g.Value)
	n := params.N()
	out := make([]byte, 0, rawKeyHeader+4*digits*2*(params.QCount()+params.PCount())*n)
	out = binary.LittleEndian.AppendUint32(out, rawKeyMagic)
	out = binary.LittleEndian.AppendUint32(out, uint32(kind))
	out = binary.LittleEndian.AppendUint64(out, galEl)
	for _, v := range []uint32{uint32(g.LevelQ()), uint32(g.LevelP()), uint32(digits), uint32(params.LogN())} {
		out = binary.LittleEndian.AppendUint32(out, v)
	}
	// Lattigo keeps keys in Montgomery form (x * 2^64 mod q); the wire carries canonical residues.
	row := make([]uint64, n)
	canonical := func(p ring.Poly, sub []*ring.SubRing) {
		for i, s := range sub[:len(p.Coeffs)] {
			for j, v := range p.Coeffs[i] {
				row[j] = ring.IMForm(v, s.Modulus, s.MRedConstant)
			}
			out = putResidues(out, [][]uint64{row})
		}
	}
	for d := 0; d < digits; d++ {
		for c := 0; c < 2; c++ {
			v := g.Value[d][0][c]
			canonical(v.Q, params.RingQ().SubRings)
			canonical(v.P, params.RingP().SubRings)
		}
	}
	return out, nil
}

// decodeKeyRaw fills a freshly allocated gadget ciphertext from a raw key, converting back to Montgomery form.
func decodeKeyRaw(params ckks.Parameters, raw []byte, wantKind int, g *rlwe.GadgetCiphertext) (uint64, error) {
	if len(raw) < rawKeyHeader || binary.LittleEndian.Uint32(raw) != rawKeyMagic {
		return 0, fmt.Errorf("not a raw-u32 key")
	}
	kind := int(binary.LittleEndian.Uint32(raw[4:]))
	galEl := binary.LittleEndian.Uint64(raw[8:])
	w := func(off int) int { return int(binary.LittleEndian.Uint32(raw[off:])) }
	if kind != wantKind || w(16) != g.LevelQ() || w(20) != g.LevelP() || w(24) != len(g.Value) || w(28) != params.LogN() {
		return 0, fmt.Errorf("raw key header does not match the profile")
	}
	n := params.N()
	if len(raw) != rawKeyHeader+4*len(g.Value)*2*(params.QCount()+params.PCount())*n {
		return 0, fmt.Errorf("raw key has the wrong length")
	}
	body := raw[rawKeyHeader:]
	var err error
	mform := func(p ring.Poly, sub []*ring.SubRing, moduli []uint64) {
		if err != nil {
			return
		}
		if body, err = getResidues(body, p.Coeffs, moduli); err != nil {
			return
		}
		for i, s := range sub[:len(p.Coeffs)] {
			for j, v := range p.Coeffs[i] {
				p.Coeffs[i][j] = ring.MForm(v, s.Modulus, s.BRedConstant)
			}
		}
	}
	for d := range g.Value {
		for c := 0; c < 2; c++ {
			v := g.Value[d][0][c]
			mform(v.Q, params.RingQ().SubRings, params.Q())
			mform(v.P, params.RingP().SubRings, params.P())
		}
	}
	return galEl, err
}

// Profile-dependent codecs used by the requester and the CPU evaluator.

func marshalCt(prof profile, params ckks.Parameters, ct *rlwe.Ciphertext) ([]byte, error) {
	if prof.Wire == wireRawU32 {
		return encodeCtRaw(params, ct)
	}
	return ct.MarshalBinary()
}

func unmarshalCt(prof profile, params ckks.Parameters, raw []byte) (*rlwe.Ciphertext, error) {
	if prof.Wire == wireRawU32 {
		return decodeCtRaw(params, raw)
	}
	ct := new(rlwe.Ciphertext)
	return ct, ct.UnmarshalBinary(raw)
}

func marshalRelinKey(prof profile, params ckks.Parameters, rlk *rlwe.RelinearizationKey) ([]byte, error) {
	if prof.Wire == wireRawU32 {
		return encodeKeyRaw(params, keyKindRelin, 0, &rlk.GadgetCiphertext)
	}
	return rlk.MarshalBinary()
}

func marshalGaloisKey(prof profile, params ckks.Parameters, gk *rlwe.GaloisKey) ([]byte, error) {
	if prof.Wire == wireRawU32 {
		return encodeKeyRaw(params, keyKindGal, gk.GaloisElement, &gk.GadgetCiphertext)
	}
	return gk.MarshalBinary()
}

func unmarshalRelinKey(prof profile, params ckks.Parameters, raw []byte) (*rlwe.RelinearizationKey, error) {
	if prof.Wire != wireRawU32 {
		rlk := new(rlwe.RelinearizationKey)
		return rlk, rlk.UnmarshalBinary(raw)
	}
	rlk := rlwe.NewRelinearizationKey(params)
	if galEl, err := decodeKeyRaw(params, raw, keyKindRelin, &rlk.GadgetCiphertext); err != nil || galEl != 0 {
		return nil, fmt.Errorf("relinearization key: %v", err)
	}
	return rlk, nil
}

func unmarshalGaloisKey(prof profile, params ckks.Parameters, raw []byte) (*rlwe.GaloisKey, error) {
	if prof.Wire != wireRawU32 {
		gk := new(rlwe.GaloisKey)
		return gk, gk.UnmarshalBinary(raw)
	}
	gk := rlwe.NewGaloisKey(params)
	galEl, err := decodeKeyRaw(params, raw, keyKindGal, &gk.GadgetCiphertext)
	if err != nil {
		return nil, err
	}
	gk.GaloisElement = galEl
	return gk, nil
}
