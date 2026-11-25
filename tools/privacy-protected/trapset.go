package main

// trapset: the reference publisher's known-answer traps for one pinned circuit. Each trap is a prompt, the
// approximate circuit's own greedy continuation (so a trap lane's decode inputs are exactly what an honest
// requester would feed back), and the expected logits at every generated position. They are computed here, in
// plaintext, from the public weights, and signed with an Ed25519 publisher key that requesters pin.
//
// The signature covers the exact body bytes: "doppler.protected-trapset/v1\x00" || body. The envelope carries the
// body as a JSON string so no canonicalization is needed across languages.

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	mrand "math/rand"
	"os"
	"strings"
	"time"
)

const trapSetSchema = "doppler.protected-trapset/v1"

type trap struct {
	ID           string      `json:"id"`
	Prompt       []int       `json:"prompt"`
	Continuation []int       `json:"continuation"`
	Expected     [][]float64 `json:"expectedLogits"` // one row per generated position, positions len(prompt)-1 .. context-1
}

type trapSetBody struct {
	Schema    string `json:"schema"`
	Protocol  string `json:"protocol"`
	Model     string `json:"model"`
	Circuit   string `json:"circuit"`
	Tokenizer struct {
		Alphabet string `json:"alphabet"`
	} `json:"tokenizer"`
	Vocab     int `json:"vocab"`
	Context   int `json:"context"`
	Tolerance struct {
		MaxAbsLogitError   float64 `json:"maxAbsLogitError"`
		MinReferenceMargin float64 `json:"minReferenceMargin"`
		RequireArgmax      bool    `json:"requireArgmax"`
	} `json:"tolerance"`
	Reference   string `json:"reference"`
	Seed        int64  `json:"seed"`
	PublishedAt string `json:"publishedAt"`
	Traps       []trap `json:"traps"`
}

type trapSetEnvelope struct {
	Schema     string `json:"schema"`
	Body       string `json:"body"`
	BodyDigest string `json:"bodyDigest"`
	PublicKey  string `json:"publicKey"` // raw Ed25519, base64url
	Signature  string `json:"signature"` // base64url
}

func trapSigningBytes(body []byte) []byte {
	return append([]byte(trapSetSchema+"\x00"), body...)
}

func loadOrCreatePublisherKey(path string, create bool) (ed25519.PrivateKey, error) {
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) && create {
		seed := make([]byte, ed25519.SeedSize)
		if _, err := rand.Read(seed); err != nil {
			return nil, err
		}
		if err := os.WriteFile(path, []byte(hex.EncodeToString(seed)), 0o600); err != nil {
			return nil, err
		}
		return ed25519.NewKeyFromSeed(seed), nil
	}
	if err != nil {
		return nil, err
	}
	seed, err := hex.DecodeString(strings.TrimSpace(string(raw)))
	if err != nil || len(seed) != ed25519.SeedSize {
		return nil, fmt.Errorf("publisher key file must hold a hex Ed25519 seed")
	}
	return ed25519.NewKeyFromSeed(seed), nil
}

func round4(v []float64) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = math.Round(x*1e4) / 1e4
	}
	return out
}

// makeTraps draws prompts of every length from a mix of the fixture's task family (letter progressions) and
// uniformly random letters, and keeps those whose whole trajectory stays inside every approximation interval and
// has a clear top-1 margin, so an honest encrypted evaluation lands within tolerance of the reference.
func makeTraps(pl *plan, perLength int, seed int64, minMargin float64) ([]trap, error) {
	rng := mrand.New(mrand.NewSource(seed))
	seen := map[string]bool{}
	var out []trap
	for L := 1; L <= context; L++ {
		kept := 0
		for attempt := 0; kept < perLength && attempt < 200*perLength; attempt++ {
			prompt := make([]int, L)
			if attempt%2 == 0 {
				a, stride := rng.Intn(26), 1+rng.Intn(3)
				for i := range prompt {
					prompt[i] = (a + i*stride) % 26
				}
			} else {
				for i := range prompt {
					prompt[i] = rng.Intn(26)
				}
			}
			key := fmt.Sprint(prompt)
			if seen[key] {
				continue
			}
			seen[key] = true
			tr := &trace{outOfRange: map[string]int{}}
			steps := context - L + 1
			toks, logits := greedy(prompt, steps, func(t []int) []float64 { return approxLogits(pl, t, tr) })
			ok := len(tr.outOfRange) == 0
			for _, row := range logits {
				for _, x := range row {
					ok = ok && !math.IsNaN(x) && !math.IsInf(x, 0)
				}
				ok = ok && margin(row) >= minMargin
			}
			if !ok {
				continue
			}
			t := trap{ID: fmt.Sprintf("L%d-%03d", L, kept), Prompt: prompt, Continuation: toks[:steps-1]}
			for _, row := range logits {
				t.Expected = append(t.Expected, round4(row))
			}
			out = append(out, t)
			kept++
		}
		// A length with no trap is simply not covered: a requester refuses such prompts before sending anything.
		// (Single-symbol prompts are ambiguous for this fixture and never reach the margin.)
		_ = kept
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no trap meets the interval and margin requirements")
	}
	return out, nil
}

func trapsetMain(args []string) error {
	fs := flag.NewFlagSet("trapset", flag.ExitOnError)
	modelPath := fs.String("model", "tiny-model.json", "public model fixture")
	keyPath := fs.String("key", "", "publisher Ed25519 seed file (hex)")
	newKey := fs.Bool("new-key", false, "create the publisher key file if it does not exist")
	out := fs.String("out", "trapset.json", "signed trap set")
	perLength := fs.Int("per-length", 16, "traps per prompt length")
	seed := fs.Int64("seed", 1, "prompt sampling seed (recorded in the body)")
	tolerance := fs.Float64("tolerance", 0.75, "largest accepted absolute logit error against the reference")
	minMargin := fs.Float64("min-margin", 2.5, "smallest reference top-1 margin a trap may have")
	fs.Parse(args)
	if *keyPath == "" {
		return fmt.Errorf("--key is required")
	}
	priv, err := loadOrCreatePublisherKey(*keyPath, *newKey)
	if err != nil {
		return err
	}
	m, id, err := loadModel(*modelPath)
	if err != nil {
		return err
	}
	pl, err := newPlan(m)
	if err != nil {
		return err
	}
	traps, err := makeTraps(pl, *perLength, *seed, *minMargin)
	if err != nil {
		return err
	}
	var b trapSetBody
	b.Schema, b.Protocol, b.Model, b.Circuit = trapSetSchema, protocolID, id, circuitDigest(pl, id)
	b.Tokenizer.Alphabet, b.Vocab, b.Context = alphabet, vocab, context
	b.Tolerance.MaxAbsLogitError, b.Tolerance.MinReferenceMargin, b.Tolerance.RequireArgmax = *tolerance, *minMargin, true
	b.Reference, b.Seed, b.PublishedAt = "approximate-circuit-plaintext", *seed, time.Now().UTC().Format(time.RFC3339)
	b.Traps = traps
	body, err := json.Marshal(b)
	if err != nil {
		return err
	}
	sum := sha256.Sum256(body)
	env := trapSetEnvelope{Schema: "doppler.protected-trapset-envelope/v1", Body: string(body),
		BodyDigest: "sha256:" + hex.EncodeToString(sum[:]),
		PublicKey:  base64.RawURLEncoding.EncodeToString(priv.Public().(ed25519.PublicKey)),
		Signature:  base64.RawURLEncoding.EncodeToString(ed25519.Sign(priv, trapSigningBytes(body)))}
	raw, err := json.MarshalIndent(env, "", " ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(*out, raw, 0o644); err != nil {
		return err
	}
	fmt.Printf("%d traps for circuit %s, publisher key %s\n", len(traps), b.Circuit, env.PublicKey)
	return nil
}
