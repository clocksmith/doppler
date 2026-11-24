package main

// The requester: owns the secret key, seals inputs, and decrypts only results it asked for, after checking them.
// It holds no model weights: it knows the alphabet (the tokenizer), the pinned model identity and the profile.
// It answers no evaluator request. A route whose results disagree is closed without decrypting anything.

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"os"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const alphabet = "abcdefghijklmnopqrstuvwxyz .,!?'"

type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, ",") }
func (m *multiFlag) Set(v string) error { *m = append(*m, v); return nil }

func peakRSSBytes() int64 {
	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	if runtime.GOOS == "darwin" {
		return int64(ru.Maxrss)
	}
	return int64(ru.Maxrss) * 1024
}

type stepReport struct {
	Seq              int       `json:"seq"`
	Kind             string    `json:"kind"`
	InputTokens      int       `json:"inputTokens"`
	EncryptSeconds   float64   `json:"encryptSeconds"`
	UploadBytes      int64     `json:"uploadBytesPerEvaluator"`
	RoundTripSeconds float64   `json:"roundTripSeconds"`
	EvaluatorSeconds []float64 `json:"evaluatorReportedComputeSeconds"`
	DownloadBytes    int64     `json:"downloadBytesPerEvaluator"`
	VerifySeconds    float64   `json:"verifySeconds"`
	DecryptSeconds   float64   `json:"decryptSeconds"`
	OutputLevel      any       `json:"outputLevel"`
	Token            int       `json:"token"`
	Logits           []float64 `json:"logits"` // client-local: never sent anywhere
	ResultDigest     string    `json:"resultDigest"`
}

type requesterReport struct {
	Schema           string       `json:"schema"`
	Protocol         string       `json:"protocol"`
	Profile          string       `json:"profile"`
	ProfileDigest    string       `json:"profileDigest"`
	Insecure         bool         `json:"insecureDevelopmentProfile"`
	Model            string       `json:"model"`
	Integrity        string       `json:"integrity"`
	Evaluators       int          `json:"evaluators"`
	Prompt           string       `json:"prompt"`
	PromptTokens     []int        `json:"promptTokens"`
	Generated        []int        `json:"generatedTokens"`
	Text             string       `json:"generatedText"`
	KeygenSeconds    float64      `json:"keygenSeconds"`
	KeyBytes         int64        `json:"evaluationKeyBytesPerEvaluator"`
	KeyUploadSeconds float64      `json:"keyUploadSeconds"`
	Steps            []stepReport `json:"steps"`
	DecryptCalls     int          `json:"decryptCalls"`
	PeakRSSBytes     int64        `json:"requesterPeakRSSBytes"`
	TotalSeconds     float64      `json:"totalSeconds"`
	Outcome          string       `json:"outcome"`
	Error            string       `json:"error,omitempty"`
}

type requester struct {
	params ckks.Parameters
	sk     *rlwe.SecretKey
	enc    *rlwe.Encryptor
	dec    *rlwe.Decryptor
	ecd    *ckks.Encoder
	peers  []*conn
	rep    *requesterReport
}

func tokenize(s string) ([]int, error) {
	var out []int
	for _, r := range s {
		i := strings.IndexRune(alphabet, r)
		if i < 0 {
			return nil, fmt.Errorf("symbol %q is outside the tokenizer alphabet", r)
		}
		out = append(out, i)
	}
	return out, nil
}

func (r *requester) sealOneHot(tok int) ([]byte, error) {
	v := make([]float64, vocab)
	v[tok] = 1
	pt := ckks.NewPlaintext(r.params, r.params.MaxLevel())
	pt.LogDimensions = ring.Dimensions{Rows: 0, Cols: logSlots}
	if err := r.ecd.Encode(replicate(v, vocab), pt); err != nil {
		return nil, err
	}
	ct, err := r.enc.EncryptNew(pt)
	if err != nil {
		return nil, err
	}
	return ct.MarshalBinary()
}

// broadcast sends the same message to every evaluator.
func (r *requester) broadcast(h header, blobs ...[]byte) error {
	for _, p := range r.peers {
		if err := p.send(h, blobs...); err != nil {
			return err
		}
	}
	return nil
}

// step sends one request, waits for exactly one result per evaluator, and checks every result before any decryption.
func (r *requester) step(session string, seq int, kind string, toks []int) (int, error) {
	sr := stepReport{Seq: seq, Kind: kind, InputTokens: len(toks)}
	t0 := time.Now()
	var blobs [][]byte
	for _, t := range toks {
		b, err := r.sealOneHot(t)
		if err != nil {
			return 0, err
		}
		blobs = append(blobs, b)
		sr.UploadBytes += int64(len(b))
	}
	sr.EncryptSeconds = time.Since(t0).Seconds()
	t1 := time.Now()
	if err := r.broadcast(header{Kind: kind, Session: session, Seq: seq}, blobs...); err != nil {
		return 0, err
	}
	var results [][]byte
	for _, p := range r.peers {
		h, b, err := p.recv("result")
		if err != nil {
			r.rep.Steps = append(r.rep.Steps, sr)
			return 0, err
		}
		if h.Session != session || h.Seq != seq || len(b) != 1 {
			return 0, fmt.Errorf("result is not bound to step %d of this session", seq)
		}
		sr.EvaluatorSeconds = append(sr.EvaluatorSeconds, toFloat(h.Fields["computeSeconds"]))
		sr.OutputLevel = h.Fields["outputLevel"]
		sr.DownloadBytes = int64(len(b[0]))
		results = append(results, b[0])
	}
	sr.RoundTripSeconds = time.Since(t1).Seconds()
	t2 := time.Now()
	// Research integrity check: independent evaluators must return byte-identical ciphertexts (evaluation is
	// deterministic for a fixed binary and profile). Nothing is deserialized or decrypted before this passes.
	sr.ResultDigest = digest(results[0])
	for _, other := range results[1:] {
		if digest(other) != sr.ResultDigest {
			r.rep.Steps = append(r.rep.Steps, sr)
			return 0, fmt.Errorf("evaluator results differ at step %d; route closed without decrypting", seq)
		}
	}
	ct := new(rlwe.Ciphertext)
	if err := ct.UnmarshalBinary(results[0]); err != nil {
		return 0, err
	}
	if ct.Degree() != 1 || ct.LogDimensions.Cols != logSlots || !ct.IsNTT {
		return 0, fmt.Errorf("result does not have the pinned shape")
	}
	sr.VerifySeconds = time.Since(t2).Seconds()
	t3 := time.Now()
	r.rep.DecryptCalls++
	vals := make([]float64, slots)
	if err := r.ecd.Decode(r.dec.DecryptNew(ct), vals); err != nil {
		return 0, err
	}
	sr.Logits = vals[:vocab]
	sr.Token = argmax(sr.Logits)
	sr.DecryptSeconds = time.Since(t3).Seconds()
	r.rep.Steps = append(r.rep.Steps, sr)
	return sr.Token, nil
}

func toFloat(v any) float64 {
	f, _ := v.(float64)
	return f
}

func requesterMain(args []string) error {
	fs := flag.NewFlagSet("requester", flag.ExitOnError)
	var evals multiFlag
	fs.Var(&evals, "evaluator", "evaluator address (repeat for the dual-evaluation integrity check)")
	profileID := fs.String("profile", "research-n16", "pinned profile")
	modelID := fs.String("model-identity", "", "pinned model identity (sha256:...)")
	prompt := fs.String("prompt", "ace", "prompt text")
	generate := fs.Int("generate", 4, "tokens to generate (fixed, public schedule; no early stop)")
	allowInsecure := fs.Bool("allow-insecure-dev", false, "permit the insecure development profile")
	out := fs.String("report", "requester-report.json", "client-local report")
	fs.Parse(args)
	start := time.Now()
	prof, err := lookupProfile(*profileID)
	if err != nil {
		return err
	}
	rep := &requesterReport{Schema: "doppler.protected-requester-report/v0", Protocol: protocolID, Profile: prof.ID,
		ProfileDigest: prof.digest(), Insecure: prof.Insecure, Model: *modelID, Evaluators: len(evals), Prompt: *prompt}
	rep.Integrity = "none (single evaluator): results decrypted unverified"
	if len(evals) >= 2 {
		rep.Integrity = "dual-evaluation: byte-identical results required before decryption"
	}
	runErr := runRequester(prof, *allowInsecure, evals, *modelID, *prompt, *generate, rep)
	rep.PeakRSSBytes, rep.TotalSeconds = peakRSSBytes(), time.Since(start).Seconds()
	rep.Outcome = "completed"
	if runErr != nil {
		rep.Outcome, rep.Error = "refused", runErr.Error()
		rep.Generated, rep.Text = nil, ""
	}
	raw, _ := json.MarshalIndent(rep, "", " ")
	if err := os.WriteFile(*out, raw, 0o600); err != nil {
		return err
	}
	return runErr
}

func runRequester(prof profile, allowInsecure bool, evals []string, modelID, prompt string, generate int, rep *requesterReport) error {
	if prof.Insecure && !allowInsecure {
		return fmt.Errorf("profile %s is insecure and only allowed with --allow-insecure-dev", prof.ID)
	}
	if len(evals) == 0 || !strings.HasPrefix(modelID, "sha256:") {
		return fmt.Errorf("an evaluator address and a pinned model identity are required")
	}
	toks, err := tokenize(prompt)
	if err != nil {
		return err
	}
	if len(toks) == 0 || generate < 1 || len(toks)+generate-1 > context {
		return fmt.Errorf("prompt of %d tokens plus %d generated exceeds the %d-position context", len(toks), generate, context)
	}
	rep.PromptTokens = toks
	params, err := prof.params()
	if err != nil {
		return err
	}
	r := &requester{params: params, ecd: ckks.NewEncoder(params), rep: rep}
	defer func() {
		for _, p := range r.peers {
			p.c.Close()
		}
	}()
	var sid [16]byte
	rand.Read(sid[:])
	session := hex.EncodeToString(sid[:])
	// Admission before upload: each evaluator must state exactly the pinned profile, model and rotations.
	for _, addr := range evals {
		nc, err := net.DialTimeout("tcp", addr, 30*time.Second)
		if err != nil {
			return err
		}
		p := &conn{c: nc, deadline: 30 * time.Minute}
		r.peers = append(r.peers, p)
		if err := p.send(header{Kind: "hello", Session: session, Fields: map[string]any{
			"profile": prof.ID, "model": modelID, "promptTokens": len(toks), "generatedTokens": generate}}); err != nil {
			return err
		}
		h, _, err := p.recv("manifest")
		if err != nil {
			return err
		}
		f := h.Fields
		if h.Session != session || f["profile"] != prof.ID || f["profileDigest"] != prof.digest() || f["model"] != modelID ||
			fmt.Sprint(f["rotations"]) != fmt.Sprint(rotationSteps()) || toFloat(f["inputLevel"]) != float64(params.MaxLevel()) ||
			toFloat(f["logSlots"]) != logSlots {
			return fmt.Errorf("evaluator %s does not match the pinned profile, model or layout; nothing uploaded", addr)
		}
	}
	// Keys: the secret key never leaves this process. Each evaluation key is generated, sent and dropped.
	t0 := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	r.sk = kgen.GenSecretKeyNew()
	r.enc, r.dec = rlwe.NewEncryptor(params, r.sk), rlwe.NewDecryptor(params, r.sk)
	var kd [][]byte
	var genSecs float64
	sendKey := func(b []byte) error {
		rep.KeyBytes += int64(len(b))
		kd = append(kd, []byte(digest(b)))
		return r.broadcast(header{Kind: "key", Session: session}, b)
	}
	g0 := time.Now()
	rlk, err := kgen.GenRelinearizationKeyNew(r.sk).MarshalBinary()
	if err != nil {
		return err
	}
	genSecs += time.Since(g0).Seconds()
	if err := sendKey(rlk); err != nil {
		return err
	}
	rlk = nil
	for _, rot := range rotationSteps() {
		g := time.Now()
		gk, err := kgen.GenGaloisKeyNew(params.GaloisElement(rot), r.sk).MarshalBinary()
		if err != nil {
			return err
		}
		genSecs += time.Since(g).Seconds()
		if err := sendKey(gk); err != nil {
			return err
		}
		runtime.GC()
	}
	want := digest(kd...)
	for _, p := range r.peers {
		h, _, err := p.recv("ready")
		if err != nil {
			return err
		}
		if h.Fields["keysDigest"] != want {
			return fmt.Errorf("evaluator acknowledged different keys")
		}
	}
	rep.KeygenSeconds = genSecs
	rep.KeyUploadSeconds = time.Since(t0).Seconds() - genSecs
	// Fixed public schedule: one prefill message with every prompt token, then generate-1 single-token steps.
	tok, err := r.step(session, 1, "prefill", toks)
	if err != nil {
		return err
	}
	gen := []int{tok}
	for seq := 2; seq <= generate; seq++ {
		if tok, err = r.step(session, seq, "decode", []int{tok}); err != nil {
			return err
		}
		gen = append(gen, tok)
	}
	if err := r.broadcast(header{Kind: "bye", Session: session}); err != nil {
		return err
	}
	rep.Generated = gen
	var sb strings.Builder
	for _, t := range gen {
		sb.WriteByte(alphabet[t])
	}
	rep.Text = sb.String()
	return nil
}
