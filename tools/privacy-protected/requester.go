package main

// The requester: owns the secret key, seals inputs, and decrypts only results it asked for, after checking them.
// It holds no model weights: it knows the alphabet (the tokenizer), the pinned model identity and the profile.
// It answers no evaluator request. A route whose results disagree is closed without decrypting anything.

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
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
	StageASeconds    float64   `json:"stageASeconds,omitempty"`
	HopBytes         int64     `json:"hopBytes,omitempty"`
	VerifySeconds    float64   `json:"verifySeconds"`
	DecryptSeconds   float64   `json:"decryptSeconds"`
	OutputLevel      any       `json:"outputLevel"`
	Token            int       `json:"token"`
	Logits           []float64 `json:"logits"` // client-local: never sent anywhere
	ResultDigest     string    `json:"resultDigest"`
}

type requesterReport struct {
	Schema                   string           `json:"schema"`
	Protocol                 string           `json:"protocol"`
	Profile                  string           `json:"profile"`
	ProfileDigest            string           `json:"profileDigest"`
	Insecure                 bool             `json:"insecureDevelopmentProfile"`
	Model                    string           `json:"model"`
	Integrity                string           `json:"integrity"`
	Evaluators               int              `json:"evaluators"`
	Prompt                   string           `json:"prompt"`
	PromptTokens             []int            `json:"promptTokens"`
	Generated                []int            `json:"generatedTokens"`
	Text                     string           `json:"generatedText"`
	KeygenSeconds            float64          `json:"keygenSeconds"`
	KeyBytes                 int64            `json:"evaluationKeyBytesPerEvaluator"`
	KeyUploadSeconds         float64          `json:"keyUploadSeconds"`
	Steps                    []stepReport     `json:"steps"`
	DecryptCalls             int              `json:"decryptCalls"`
	KeyEpoch                 string           `json:"keyEpoch"`
	KeyEpochExpires          int64            `json:"keyEpochExpires"`
	KeyEpochReused           bool             `json:"keyEpochReusedFromStore"`
	KeysReusedFromEvaluators bool             `json:"evaluationKeysReusedNoUpload"`
	TimeToFirstTokenSeconds  float64          `json:"timeToFirstTokenSeconds"`
	SessionBytes             []map[string]any `json:"sessionBytesPerEvaluator"`
	PeakRSSBytes             int64            `json:"requesterPeakRSSBytes"`
	Split                    *splitReport     `json:"split,omitempty"`
	StoppedAfter             int              `json:"stoppedAfter,omitempty"`
	TotalSeconds             float64          `json:"totalSeconds"`
	Outcome                  string           `json:"outcome"`
	Error                    string           `json:"error,omitempty"`
}

// splitReport: the homomorphic split. Stage A gets every input ciphertext; stage B returns every result; the carry
// ciphertexts go from A to B directly. The requester never decrypts or re-encrypts between the stages.
type splitReport struct {
	CutLayer           string `json:"cutLayer"`
	Stages             int    `json:"stages"`
	RequesterRefreshes int    `json:"requesterRefreshes"`
	RefreshReason      string `json:"refreshReason"`
}

type requester struct {
	split  bool
	events func(map[string]any)
	prof   profile
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
	return marshalCt(r.prof, r.params, ct)
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
	senders, receivers := r.peers, r.peers
	if r.split {
		// Inputs go to stage A only; the result comes from stage B only.
		// Whole-model reference evaluators (peers[2:]) get the inputs and return results too.
		senders = append([]*conn{r.peers[0]}, r.peers[2:]...)
		receivers = r.peers[1:]
	}
	for _, p := range senders {
		if err := p.send(header{Kind: kind, Session: session, Seq: seq}, blobs...); err != nil {
			return 0, err
		}
	}
	var results [][]byte
	for _, p := range receivers {
		h, b, err := p.recv("result")
		if err != nil {
			r.rep.Steps = append(r.rep.Steps, sr)
			return 0, err
		}
		if h.Session != session || h.Seq != seq || len(b) != 1 {
			return 0, fmt.Errorf("result is not bound to step %d of this session", seq)
		}
		sr.EvaluatorSeconds = append(sr.EvaluatorSeconds, toFloat(h.Fields["computeSeconds"]))
		if r.split && p == r.peers[1] {
			sr.StageASeconds = toFloat(h.Fields["stageASeconds"])
			sr.HopBytes = int64(toFloat(h.Fields["hopBytes"]))
		}
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
	ct, err := unmarshalCt(r.prof, r.params, results[0])
	if err != nil {
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

type requesterOptions struct {
	prof          profile
	allowInsecure bool
	evaluators    []string
	modelID       string
	prompt        string
	generate      int
	stateDir      string // key epoch store; empty: a fresh key for this session only
	keyTTL        time.Duration
	split         bool // evaluators[0] is stage A, evaluators[1] stage B
	events        bool // JSON events on stdout and `stop` on stdin, for a supervising process
}

func requesterMain(args []string) error {
	fs := flag.NewFlagSet("requester", flag.ExitOnError)
	var evals multiFlag
	fs.Var(&evals, "evaluator", "evaluator address (repeat for the dual-evaluation integrity check)")
	stageA := fs.String("stage-a", "", "homomorphic split: stage A's address (it receives the inputs and holds the KV cache)")
	stageB := fs.String("stage-b", "", "homomorphic split: stage B's address (it returns the results)")
	events := fs.Bool("events", false, "write one JSON event per line on stdout (ready, token, end, failed) and stop on a `stop` line on stdin")
	profileID := fs.String("profile", "research-n16", "pinned profile")
	modelID := fs.String("model-identity", "", "pinned model identity (sha256:...)")
	prompt := fs.String("prompt", "ace", "prompt text")
	generate := fs.Int("generate", 4, "tokens to generate (fixed, public schedule; no early stop)")
	allowInsecure := fs.Bool("allow-insecure-dev", false, "permit the insecure development profile")
	stateDir := fs.String("state-dir", "", "requester key-epoch store (secret key, 0600); reused until it expires")
	keyTTL := fs.Duration("key-ttl", 24*time.Hour, "lifetime of a new key epoch")
	out := fs.String("report", "requester-report.json", "client-local report")
	fs.Parse(args)
	start := time.Now()
	prof, err := lookupProfile(*profileID)
	if err != nil {
		return err
	}
	split := *stageA != "" || *stageB != ""
	if split {
		if *stageA == "" || *stageB == "" {
			return fmt.Errorf("a split needs --stage-a and --stage-b")
		}
		// Any --evaluator given with a split is a whole-model reference for the dual check.
		evals = append(multiFlag{*stageA, *stageB}, evals...)
	}
	rep := &requesterReport{Schema: "doppler.protected-requester-report/v0", Protocol: protocolID, Profile: prof.ID,
		ProfileDigest: prof.digest(), Insecure: prof.Insecure, Model: *modelID, Evaluators: len(evals), Prompt: *prompt}
	rep.Integrity = "none (single evaluator): results decrypted unverified"
	if len(evals) >= 2 && !split {
		rep.Integrity = "dual-evaluation: byte-identical results required before decryption"
	}
	if split {
		rep.Integrity = "none (homomorphic split, one evaluator per stage): results decrypted unverified"
		if len(evals) > 2 {
			rep.Integrity = "dual-evaluation: the split's result must equal a whole-model evaluator's byte for byte before decryption"
		}
	}
	runErr := runRequester(requesterOptions{prof: prof, allowInsecure: *allowInsecure, evaluators: evals, modelID: *modelID,
		prompt: *prompt, generate: *generate, stateDir: *stateDir, keyTTL: *keyTTL, split: split, events: *events}, rep, start)
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
	if *events {
		e := map[string]any{"event": "end", "outputTokens": len(rep.Generated), "stopReason": "max-tokens", "requesterPeakRss": rep.PeakRSSBytes}
		if rep.StoppedAfter > 0 {
			e["stopReason"] = "requester-stop"
		}
		if runErr != nil {
			e = map[string]any{"event": "failed", "error": runErr.Error()}
		}
		json.NewEncoder(os.Stdout).Encode(e)
	}
	return runErr
}

// keyEpoch: one secret key and the evaluation keys made from it, reused across sessions until Expires.
type keyEpoch struct {
	Epoch         string `json:"epoch"`
	ProfileDigest string `json:"profileDigest"`
	Created       int64  `json:"created"`
	Expires       int64  `json:"expires"`
	KeysDigest    string `json:"keysDigest"` // of the evaluation keys the evaluators were last sent; "" before upload
}

func loadOrCreateEpoch(o requesterOptions, params ckks.Parameters) (*keyEpoch, *rlwe.SecretKey, bool, error) {
	now := time.Now().Unix()
	if o.stateDir != "" {
		raw, err1 := os.ReadFile(filepath.Join(o.stateDir, "key-epoch.json"))
		skRaw, err2 := os.ReadFile(filepath.Join(o.stateDir, "key-epoch.sk"))
		var ep keyEpoch
		if err1 == nil && err2 == nil && json.Unmarshal(raw, &ep) == nil && ep.ProfileDigest == o.prof.digest() && ep.Expires > now+300 {
			sk := new(rlwe.SecretKey)
			if err := sk.UnmarshalBinary(skRaw); err == nil {
				return &ep, sk, true, nil
			}
		}
	}
	var id [16]byte
	rand.Read(id[:])
	ep := &keyEpoch{Epoch: hex.EncodeToString(id[:]), ProfileDigest: o.prof.digest(), Created: now, Expires: now + int64(o.keyTTL.Seconds())}
	sk := rlwe.NewKeyGenerator(params).GenSecretKeyNew()
	if o.stateDir != "" {
		if err := os.MkdirAll(o.stateDir, 0o700); err != nil {
			return nil, nil, false, err
		}
		b, err := sk.MarshalBinary()
		if err != nil {
			return nil, nil, false, err
		}
		if err := os.WriteFile(filepath.Join(o.stateDir, "key-epoch.sk"), b, 0o600); err != nil {
			return nil, nil, false, err
		}
		if err := saveEpoch(o.stateDir, ep); err != nil {
			return nil, nil, false, err
		}
	}
	return ep, sk, false, nil
}

func saveEpoch(dir string, ep *keyEpoch) error {
	if dir == "" {
		return nil
	}
	raw, _ := json.Marshal(ep)
	return os.WriteFile(filepath.Join(dir, "key-epoch.json"), raw, 0o600)
}

func runRequester(o requesterOptions, rep *requesterReport, start time.Time) error {
	if o.prof.Insecure && !o.allowInsecure {
		return fmt.Errorf("profile %s is insecure and only allowed with --allow-insecure-dev", o.prof.ID)
	}
	if len(o.evaluators) == 0 || !strings.HasPrefix(o.modelID, "sha256:") {
		return fmt.Errorf("an evaluator address and a pinned model identity are required")
	}
	toks, err := tokenize(o.prompt)
	if err != nil {
		return err
	}
	if len(toks) == 0 || o.generate < 1 || len(toks)+o.generate-1 > context {
		return fmt.Errorf("prompt of %d tokens plus %d generated exceeds the %d-position context", len(toks), o.generate, context)
	}
	rep.PromptTokens = toks
	params, err := o.prof.params()
	if err != nil {
		return err
	}
	r := &requester{prof: o.prof, params: params, ecd: ckks.NewEncoder(params), rep: rep, split: o.split, events: func(map[string]any) {}}
	var stop atomic.Bool
	if o.events {
		out := json.NewEncoder(os.Stdout)
		r.events = func(e map[string]any) { out.Encode(e) }
		go func() {
			for in := bufio.NewScanner(os.Stdin); in.Scan(); {
				if strings.TrimSpace(in.Text()) == "stop" {
					stop.Store(true)
				}
			}
		}()
	}
	hopToken := ""
	roles := []string{""}
	if o.split {
		var ht [16]byte
		rand.Read(ht[:])
		hopToken = hex.EncodeToString(ht[:])
		roles = []string{"stage-a", "stage-b"}
		rep.Split = &splitReport{CutLayer: "ln2", Stages: 2, RequesterRefreshes: 0,
			RefreshReason: "none needed: the whole block fits the 36-level budget (35 levels used), so no ciphertext is refreshed or bootstrapped between the stages"}
	}
	defer func() {
		for i, p := range r.peers {
			rep.SessionBytes = append(rep.SessionBytes, map[string]any{"evaluator": i + 1, "sent": p.sent, "received": p.received})
			p.c.Close()
		}
	}()
	ep, sk, reused, err := loadOrCreateEpoch(o, params)
	if err != nil {
		return err
	}
	r.sk = sk
	r.enc, r.dec = rlwe.NewEncryptor(params, r.sk), rlwe.NewDecryptor(params, r.sk)
	rep.KeyEpoch, rep.KeyEpochExpires, rep.KeyEpochReused = ep.Epoch, ep.Expires, reused
	var sid [16]byte
	rand.Read(sid[:])
	session := hex.EncodeToString(sid[:])
	// Admission before upload: each evaluator must state exactly the pinned profile, model and rotations.
	allHold := reused && ep.KeysDigest != ""
	for i, addr := range o.evaluators {
		nc, err := net.DialTimeout("tcp", addr, 30*time.Second)
		if err != nil {
			return err
		}
		p := &conn{c: nc, deadline: 30 * time.Minute}
		r.peers = append(r.peers, p)
		fields := map[string]any{"profile": o.prof.ID, "model": o.modelID, "promptTokens": len(toks), "generatedTokens": o.generate,
			"keyEpoch": ep.Epoch, "keyEpochExpires": ep.Expires}
		if o.split && i < 2 {
			fields["role"], fields["cutLayer"], fields["hopToken"] = roles[i], "ln2", hopToken
		}
		if err := p.send(header{Kind: "hello", Session: session, Fields: fields}); err != nil {
			return err
		}
		h, _, err := p.recv("manifest")
		if err != nil {
			return err
		}
		f := h.Fields
		if h.Session != session || f["profile"] != o.prof.ID || f["profileDigest"] != o.prof.digest() || f["model"] != o.modelID ||
			fmt.Sprint(f["rotations"]) != fmt.Sprint(rotationSteps()) || toFloat(f["inputLevel"]) != float64(params.MaxLevel()) ||
			toFloat(f["logSlots"]) != logSlots || (o.split && i < 2 && (f["role"] != roles[i] || f["cutLayer"] != "ln2")) {
			return fmt.Errorf("evaluator %d does not match the pinned profile, model or layout; nothing uploaded", i+1)
		}
		allHold = allHold && f["heldKeysDigest"] == ep.KeysDigest
	}
	t0 := time.Now()
	var genSecs float64
	want := ep.KeysDigest
	if allHold {
		// Every evaluator still holds this epoch's exact evaluation keys: nothing is uploaded.
		if err := r.broadcast(header{Kind: "keys-held", Session: session}); err != nil {
			return err
		}
		rep.KeysReusedFromEvaluators = true
	} else {
		// The secret key never leaves this process. Each evaluation key is generated, sent to every evaluator and
		// dropped, so all evaluators hold identical keys (their results must agree byte for byte).
		kgen := rlwe.NewKeyGenerator(params)
		var kd [][]byte
		sendKey := func(b []byte) error {
			rep.KeyBytes += int64(len(b))
			kd = append(kd, []byte(digest(b)))
			return r.broadcast(header{Kind: "key", Session: session}, b)
		}
		g0 := time.Now()
		rlk, err := marshalRelinKey(o.prof, params, kgen.GenRelinearizationKeyNew(r.sk))
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
			gk, err := marshalGaloisKey(o.prof, params, kgen.GenGaloisKeyNew(params.GaloisElement(rot), r.sk))
			if err != nil {
				return err
			}
			genSecs += time.Since(g).Seconds()
			if err := sendKey(gk); err != nil {
				return err
			}
			runtime.GC()
		}
		want = digest(kd...)
	}
	for _, p := range r.peers {
		h, _, err := p.recv("ready")
		if err != nil {
			return err
		}
		if h.Fields["keysDigest"] != want {
			return fmt.Errorf("evaluator acknowledged different keys")
		}
	}
	if !allHold {
		ep.KeysDigest = want
		if err := saveEpoch(o.stateDir, ep); err != nil {
			return err
		}
	}
	rep.KeygenSeconds = genSecs
	rep.KeyUploadSeconds = time.Since(t0).Seconds() - genSecs
	r.events(map[string]any{"event": "ready", "promptTokens": len(toks), "keysUploaded": !allHold, "keyBytes": rep.KeyBytes})
	emit := func(seq, tok int) {
		sr := rep.Steps[len(rep.Steps)-1]
		var stageB float64
		if len(sr.EvaluatorSeconds) > 0 {
			stageB = sr.EvaluatorSeconds[0]
		}
		done := seq == o.generate
		reason := any(nil)
		if done {
			reason = "max-tokens"
		}
		r.events(map[string]any{"event": "token", "step": seq - 1, "tokenId": tok, "delta": string(alphabet[tok]), "done": done, "stopReason": reason,
			"inputTokens": sr.InputTokens, "wallMs": (sr.EncryptSeconds + sr.RoundTripSeconds + sr.VerifySeconds + sr.DecryptSeconds) * 1000,
			"roundTripMs": sr.RoundTripSeconds * 1000, "stageAMs": sr.StageASeconds * 1000, "stageBMs": stageB * 1000,
			"up": sr.UploadBytes, "down": sr.DownloadBytes, "hopBytes": sr.HopBytes, "encryptMs": sr.EncryptSeconds * 1000, "decryptMs": sr.DecryptSeconds * 1000})
	}
	// Fixed public schedule: one prefill message with every prompt token, then generate-1 single-token steps. A
	// supervisor's stop ends the schedule early with bye (the evaluators then see a shorter session).
	tok, err := r.step(session, 1, "prefill", toks)
	if err != nil {
		return err
	}
	rep.TimeToFirstTokenSeconds = time.Since(start).Seconds()
	gen := []int{tok}
	emit(1, tok)
	for seq := 2; seq <= o.generate; seq++ {
		if stop.Load() {
			rep.StoppedAfter = len(gen)
			break
		}
		if tok, err = r.step(session, seq, "decode", []int{tok}); err != nil {
			return err
		}
		gen = append(gen, tok)
		emit(seq, tok)
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
