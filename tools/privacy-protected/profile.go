package main

// Pinned research profiles. A profile fixes the CKKS parameters, the slot layout and the protocol identity. The
// requester and the evaluator each compile in the same table; neither accepts a profile from the other side.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	protocolID = "doppler-protected-ckks-tiny/v0-research"
	logSlots   = 6 // 64 slots: every vector is replicated with a period of 16, 32 or 64
	slots      = 1 << logSlots
)

// Model geometry of the tiny fixture. The requester needs only vocab, context and the alphabet (the tokenizer).
const (
	vocab   = 32
	context = 8
	dim     = 16
	hidden  = 64
)

type profile struct {
	ID       string
	LogN     int
	LogQ     []int
	LogP     []int
	LogScale int
	Insecure bool // development only: the requester refuses it unless explicitly allowed, and reports say so
}

func chain(first, n, bits int) []int {
	q := []int{first}
	for i := 0; i < n; i++ {
		q = append(q, bits)
	}
	return q
}

// research-n16: ring degree 2^16, log QP = 50 + 36*32 + 7*61 = 1629 bits, dense ternary secret (Lattigo default).
// That is below the 1772-bit ceiling commonly cited for 128-bit security at N = 2^16; this has NOT been
// independently estimated for this exact parameter set (see the plan's review requirements).
var profiles = map[string]profile{
	"research-n16": {ID: "research-n16", LogN: 16, LogQ: chain(50, 36, 32), LogP: []int{61, 61, 61, 61, 61, 61, 61}, LogScale: 32},
	"dev-insecure-n12": {ID: "dev-insecure-n12", LogN: 12, LogQ: chain(50, 36, 32), LogP: []int{61, 61, 61, 61, 61, 61, 61},
		LogScale: 32, Insecure: true},
}

func lookupProfile(id string) (profile, error) {
	p, ok := profiles[id]
	if !ok {
		return profile{}, fmt.Errorf("unknown profile %q", id)
	}
	return p, nil
}

func (p profile) params() (ckks.Parameters, error) {
	return ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: p.LogN, LogQ: p.LogQ, LogP: p.LogP, LogDefaultScale: p.LogScale,
	})
}

// rotationSteps are the only rotations the evaluator may use; every other rotation is composed from them.
func rotationSteps() []int { return []int{1, 2, 4, 8, 16, 32} }

func (p profile) digest() string {
	h := sha256.New()
	fmt.Fprintf(h, "%s|%d|%v|%v|%d|%d|%v", p.ID, p.LogN, p.LogQ, p.LogP, p.LogScale, logSlots, rotationSteps())
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}
