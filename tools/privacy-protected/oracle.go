package main

// trap-oracle: secret, fresh known-answer traps per session (trap verification v2). Run by a party the evaluator
// cannot see into (the model publisher or a separate host), it holds the public pinned weights and answers each
// request with a new batch of trap inputs, sampled like real prompts, and their expected outputs from the pinned
// approximate circuit. Nothing is drawn from a fixed list.
//
// Channel: the batch is encrypted to the requester's one-time X25519 key (ECDH with a fresh oracle key, HKDF-SHA256,
// AES-256-GCM with the batch id as associated data) and the header is signed with the oracle's Ed25519 key,
// which requesters pin. The header binds the requester's 256-bit nonce and one-time key, so a batch answers exactly
// one request and cannot be replayed into another session. Any byte transport works; the oracle is authenticated by
// its signature and the content is readable only by the requester. Requesters are anonymous to the oracle.
//
// What the oracle learns per request: that a session is starting, its prompt length, generated count and number of
// traps, the time, and the network address the request came from. Never the prompt, the answer or the evaluator.

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/ecdh"
	"crypto/ed25519"
	"crypto/hkdf"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/big"
	"net"
	"net/http"
	"os"
	"regexp"
	"time"
)

const (
	trapBatchSchema       = "doppler.protected-trap-batch/v1"
	trapRequestSchema     = "doppler.protected-trap-request/v1"
	trapDistribution      = "doppler.fixture-prompt-mixture/v1"
	maxTrapsPerBatch      = 8
	trapBatchRequestLimit = 4 << 10
)

var b64 = base64.RawURLEncoding

type trapRequest struct {
	Schema         string `json:"schema"`
	Circuit        string `json:"circuit"`
	PromptLength   int    `json:"promptLength"`
	Generate       int    `json:"generate"`
	Count          int    `json:"count"`
	RequesterNonce string `json:"requesterNonce"`
	RequesterKey   string `json:"requesterKey"`
}

type trapBatchHeader struct {
	Schema          string `json:"schema"`
	BatchID         string `json:"batchId"`
	Circuit         string `json:"circuit"`
	RequesterNonce  string `json:"requesterNonce"`
	RequesterKey    string `json:"requesterKey"`
	OracleEphemeral string `json:"oracleEphemeralKey"`
	IV              string `json:"iv"`
	CiphertextHash  string `json:"ciphertextDigest"`
	IssuedAt        int64  `json:"issuedAt"`
	ExpiresAt       int64  `json:"expiresAt"`
}

type trapBatchBody struct {
	Schema         string `json:"schema"`
	BatchID        string `json:"batchId"`
	Circuit        string `json:"circuit"`
	Model          string `json:"model"`
	RequesterNonce string `json:"requesterNonce"`
	PromptLength   int    `json:"promptLength"`
	Generate       int    `json:"generate"`
	Distribution   string `json:"distribution"`
	Tolerance      struct {
		MaxAbsLogitError   float64 `json:"maxAbsLogitError"`
		MinReferenceMargin float64 `json:"minReferenceMargin"`
		RequireArgmax      bool    `json:"requireArgmax"`
	} `json:"tolerance"`
	Traps []trap `json:"traps"`
}

type trapBatchEnvelope struct {
	Schema     string `json:"schema"`
	Header     string `json:"header"` // exact signed bytes
	Ciphertext string `json:"ciphertext"`
	OracleKey  string `json:"oracleKey"`
	Signature  string `json:"signature"`
}

func trapBatchSigningBytes(header []byte) []byte {
	return append([]byte(trapBatchSchema+"\x00"), header...)
}

// cryptoIntn draws uniformly from [0, n) with the platform CSPRNG: trap inputs must be unpredictable.
func cryptoIntn(n int) int {
	v, err := rand.Int(rand.Reader, big.NewInt(int64(n)))
	if err != nil {
		panic(err)
	}
	return int(v.Int64())
}

func cryptoFloat() float64 { return float64(cryptoIntn(1<<53)) / (1 << 53) }

// samplePrompt draws one prompt of length L from the fixture's input mixture: a third letter progressions (the task
// the model was trained on), a third sequences the model itself generates from a random start (sampled from its
// softmax at temperature 1), a third uniformly random letters.
func samplePrompt(pl *plan, L int) []int {
	p := make([]int, L)
	switch cryptoIntn(3) {
	case 0:
		a, stride := cryptoIntn(26), 1+cryptoIntn(3)
		for i := range p {
			p[i] = (a + i*stride) % 26
		}
	case 1:
		p[0] = cryptoIntn(26)
		for i := 1; i < L; i++ {
			l := approxLogits(pl, p[:i], nil)
			mx := math.Inf(-1)
			for _, x := range l[:26] {
				mx = math.Max(mx, x)
			}
			var z float64
			w := make([]float64, 26)
			for j := range w {
				w[j] = math.Exp(l[j] - mx)
				z += w[j]
			}
			u, acc := cryptoFloat()*z, 0.0
			p[i] = 25
			for j := range w {
				if acc += w[j]; u < acc {
					p[i] = j
					break
				}
			}
		}
	default:
		for i := range p {
			p[i] = cryptoIntn(26)
		}
	}
	return p
}

// freshTraps samples until it has count traps whose trajectory stays inside every approximation interval with a
// top-1 margin of at least minMargin (so an honest encrypted evaluation lands within tolerance).
func freshTraps(pl *plan, L, G, count int, minMargin float64) ([]trap, error) {
	var out []trap
	for attempt := 0; len(out) < count && attempt < 400*count; attempt++ {
		prompt := samplePrompt(pl, L)
		tr := &trace{outOfRange: map[string]int{}}
		toks, logits := greedy(prompt, G, func(t []int) []float64 { return approxLogits(pl, t, tr) })
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
		t := trap{ID: fmt.Sprintf("t%d", len(out)), Prompt: prompt, Continuation: toks[:G-1]}
		for _, row := range logits {
			t.Expected = append(t.Expected, round4(row))
		}
		out = append(out, t)
	}
	if len(out) < count {
		return nil, fmt.Errorf("could not sample %d traps of prompt length %d", count, L)
	}
	return out, nil
}

var hex64 = regexp.MustCompile(`^[a-f0-9]{64}$`)

type trapOracle struct {
	pl                  *plan
	modelID, circuit    string
	priv                ed25519.PrivateKey
	tolerance, minMargn float64
	ttl                 time.Duration
}

// issue answers one request: fresh traps, encrypted to the requester's key, header signed.
func (o *trapOracle) issue(req trapRequest) (*trapBatchEnvelope, error) {
	if req.Schema != trapRequestSchema || req.Circuit != o.circuit || !hex64.MatchString(req.RequesterNonce) ||
		req.PromptLength < 1 || req.Generate < 1 || req.PromptLength+req.Generate-1 > context || req.Count < 1 || req.Count > maxTrapsPerBatch {
		return nil, fmt.Errorf("trap request is invalid or not for this circuit")
	}
	rk, err := b64.DecodeString(req.RequesterKey)
	if err != nil {
		return nil, fmt.Errorf("requester key is not base64url")
	}
	requesterKey, err := ecdh.X25519().NewPublicKey(rk)
	if err != nil {
		return nil, err
	}
	traps, err := freshTraps(o.pl, req.PromptLength, req.Generate, req.Count, o.minMargn)
	if err != nil {
		return nil, err
	}
	id := make([]byte, 16)
	rand.Read(id)
	batchID := hex.EncodeToString(id)
	body := trapBatchBody{Schema: trapBatchSchema, BatchID: batchID, Circuit: o.circuit, Model: o.modelID, RequesterNonce: req.RequesterNonce,
		PromptLength: req.PromptLength, Generate: req.Generate, Distribution: trapDistribution, Traps: traps}
	body.Tolerance.MaxAbsLogitError, body.Tolerance.MinReferenceMargin, body.Tolerance.RequireArgmax = o.tolerance, o.minMargn, true
	plain, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	eph, err := ecdh.X25519().GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}
	shared, err := eph.ECDH(requesterKey)
	if err != nil {
		return nil, err
	}
	nonceBytes, _ := hex.DecodeString(req.RequesterNonce)
	key, err := hkdf.Key(sha256.New, shared, nonceBytes, trapBatchSchema, 32)
	if err != nil {
		return nil, err
	}
	block, _ := aes.NewCipher(key)
	gcm, _ := cipher.NewGCM(block)
	iv := make([]byte, gcm.NonceSize())
	rand.Read(iv)
	now := time.Now()
	// Associated data is the batch id; the signed header then binds the ciphertext's digest, the requester's nonce
	// and one-time key, and the oracle's one-time key.
	h := trapBatchHeader{Schema: trapBatchSchema, BatchID: batchID, Circuit: o.circuit, RequesterNonce: req.RequesterNonce,
		RequesterKey: req.RequesterKey, OracleEphemeral: b64.EncodeToString(eph.PublicKey().Bytes()), IV: b64.EncodeToString(iv),
		IssuedAt: now.UnixMilli(), ExpiresAt: now.Add(o.ttl).UnixMilli()}
	sealed := gcm.Seal(nil, iv, plain, []byte(batchID))
	sum := sha256.Sum256(sealed)
	h.CiphertextHash = "sha256:" + hex.EncodeToString(sum[:])
	header, _ := json.Marshal(h)
	return &trapBatchEnvelope{Schema: "doppler.protected-trap-batch-envelope/v1", Header: string(header), Ciphertext: b64.EncodeToString(sealed),
		OracleKey: b64.EncodeToString(o.priv.Public().(ed25519.PublicKey)), Signature: b64.EncodeToString(ed25519.Sign(o.priv, trapBatchSigningBytes(header)))}, nil
}

func (o *trapOracle) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/descriptor", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"schema": "doppler.protected-trap-oracle/v1", "oracleKey": b64.EncodeToString(o.priv.Public().(ed25519.PublicKey)),
			"circuit": o.circuit, "model": o.modelID, "distribution": trapDistribution, "maxTraps": maxTrapsPerBatch})
	})
	mux.HandleFunc("/v1/trap-batch", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var req trapRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, trapBatchRequestLimit)).Decode(&req); err != nil {
			http.Error(w, "malformed request", http.StatusBadRequest)
			return
		}
		t0 := time.Now()
		env, err := o.issue(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		// Sizes and timing only: the oracle never has a prompt to log.
		fmt.Fprintf(os.Stderr, "[trap-oracle] batch of %d, prompt length %d, generate %d, %.1f ms\n", req.Count, req.PromptLength, req.Generate,
			float64(time.Since(t0).Microseconds())/1000)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(env)
	})
	return mux
}

func trapOracleMain(args []string) error {
	fs := flag.NewFlagSet("trap-oracle", flag.ExitOnError)
	modelPath := fs.String("model", "tiny-model.json", "public pinned model")
	keyPath := fs.String("key", "", "oracle Ed25519 seed file (hex)")
	newKey := fs.Bool("new-key", false, "create the key file if it does not exist")
	listen := fs.String("listen", "127.0.0.1:0", "address to listen on")
	ready := fs.String("ready-file", "", "write the bound address here once listening")
	tolerance := fs.Float64("tolerance", 0.75, "largest accepted absolute logit error against the reference")
	minMargin := fs.Float64("min-margin", 2.5, "smallest reference top-1 margin a trap may have")
	ttl := fs.Duration("ttl", 10*time.Minute, "how long a batch may be used after issue")
	exitWithStdin := fs.Bool("exit-with-stdin", false, "exit when standard input closes")
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
	if *exitWithStdin {
		go func() {
			buf := make([]byte, 64)
			for {
				if _, err := os.Stdin.Read(buf); err != nil {
					os.Exit(3)
				}
			}
		}()
	}
	o := &trapOracle{pl: pl, modelID: id, circuit: circuitDigest(pl, id), priv: priv, tolerance: *tolerance, minMargn: *minMargin, ttl: *ttl}
	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		return err
	}
	if *ready != "" {
		if err := os.WriteFile(*ready, []byte(ln.Addr().String()), 0o600); err != nil {
			return err
		}
	}
	fmt.Fprintf(os.Stderr, "[trap-oracle] circuit %s, key %s, listening on %s\n", o.circuit, b64.EncodeToString(priv.Public().(ed25519.PublicKey)), ln.Addr())
	return (&http.Server{Handler: o.handler(), ReadHeaderTimeout: 10 * time.Second}).Serve(ln)
}
