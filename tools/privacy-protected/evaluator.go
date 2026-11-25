package main

// The evaluator: public weights, public evaluation keys and ciphertexts only. This file must never construct a
// secret key, a decryptor or an encryptor (protocol_test.go checks the source). It keeps the encrypted KV cache
// for one session and answers each step with one ciphertext of logits.

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	cpoly "github.com/tuneinsight/lattigo/v6/circuits/common/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// heEval is the part of Lattigo's CKKS evaluator the circuit uses. *ckks.Evaluator implements it; the GPU program
// compiler (gpu_record.go) substitutes a recorder that also lowers every call to RNS operations.
type heEval interface {
	RotateNew(op0 *rlwe.Ciphertext, k int) (*rlwe.Ciphertext, error)
	Add(op0 *rlwe.Ciphertext, op1 rlwe.Operand, opOut *rlwe.Ciphertext) error
	Sub(op0 *rlwe.Ciphertext, op1 rlwe.Operand, opOut *rlwe.Ciphertext) error
	MulNew(op0 *rlwe.Ciphertext, op1 rlwe.Operand) (*rlwe.Ciphertext, error)
	MulRelinNew(op0 *rlwe.Ciphertext, op1 rlwe.Operand) (*rlwe.Ciphertext, error)
	Rescale(op0, opOut *rlwe.Ciphertext) error
}

type circuit struct {
	params ckks.Parameters
	eval   heEval
	poly   *polynomial.Evaluator
	copyCt func(*rlwe.Ciphertext) *rlwe.Ciphertext // nil: CopyNew; the recorder aliases instead
	mark   func(stage string)                     // nil, or the GPU recorder's layer boundaries
	pl     *plan
	keys   []*rlwe.Ciphertext // cache: key rows per position
	vals   []*rlwe.Ciphertext // cache: value rows (output projection folded in)
	ops    map[string]int
}

func (c *circuit) count(op string) { c.ops[op]++ }

func (c *circuit) stage(name string) {
	if c.mark != nil {
		c.mark(name)
	}
}

func (c *circuit) cp(ct *rlwe.Ciphertext) *rlwe.Ciphertext {
	if c.copyCt != nil {
		return c.copyCt(ct)
	}
	return ct.CopyNew()
}

// rot rotates left by k slots, composed from the power-of-two keys the requester provided.
func (c *circuit) rot(ct *rlwe.Ciphertext, k int) (*rlwe.Ciphertext, error) {
	k = ((k % slots) + slots) % slots
	out := ct
	for bit := 0; bit < logSlots; bit++ {
		if k&(1<<bit) != 0 {
			r, err := c.eval.RotateNew(out, 1<<bit)
			if err != nil {
				return nil, err
			}
			c.count("rotate")
			out = r
		}
	}
	if out == ct {
		out = c.cp(ct)
	}
	return out, nil
}

// rotsum: every slot receives the sum over its period-n block (inputs are replicated with period n).
func (c *circuit) rotsum(ct *rlwe.Ciphertext, n int) (*rlwe.Ciphertext, error) {
	acc := c.cp(ct)
	for i := 1; i < n; i *= 2 {
		r, err := c.rot(acc, i)
		if err != nil {
			return nil, err
		}
		if err := c.eval.Add(acc, r, acc); err != nil {
			return nil, err
		}
	}
	return acc, nil
}

func (c *circuit) babies(x *rlwe.Ciphertext, b int) ([]*rlwe.Ciphertext, error) {
	out := []*rlwe.Ciphertext{x}
	for i := 1; i < b; i++ {
		r, err := c.rot(out[i-1], 1)
		if err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, nil
}

func babyCount(n int) int {
	if n <= 16 {
		return 4
	}
	return 8
}

// linear computes y = W x (W is m x n) for x replicated with period n; y comes out replicated with period m.
// Diagonal method with baby-step/giant-step: y[s] = sum_k D_k[s] x[s+k], D_k[s] = W[s mod m][(s+k) mod n].
func (c *circuit) linear(bab []*rlwe.Ciphertext, w [][]float64, n, m int) (*rlwe.Ciphertext, error) {
	b := len(bab)
	var acc *rlwe.Ciphertext
	for g := 0; g < n/b; g++ {
		var inner *rlwe.Ciphertext
		for i := 0; i < b; i++ {
			k := g*b + i
			d := make([]float64, slots)
			for s := 0; s < slots; s++ {
				src := ((s-g*b)%slots + slots) % slots // pre-rotated by -g*b
				d[s] = w[src%m][(src+k)%n]
			}
			p, err := c.eval.MulNew(bab[i], d)
			if err != nil {
				return nil, err
			}
			c.count("mul-plain")
			if inner == nil {
				inner = p
			} else if err := c.eval.Add(inner, p, inner); err != nil {
				return nil, err
			}
		}
		if err := c.eval.Rescale(inner, inner); err != nil {
			return nil, err
		}
		r, err := c.rot(inner, g*b)
		if err != nil {
			return nil, err
		}
		if acc == nil {
			acc = r
		} else if err := c.eval.Add(acc, r, acc); err != nil {
			return nil, err
		}
	}
	return acc, nil
}

func replicate(v []float64, period int) []float64 {
	out := make([]float64, slots)
	for s := range out {
		if s%period < len(v) {
			out[s] = v[s%period]
		}
	}
	return out
}

func (c *circuit) mulRelin(a, b *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	out, err := c.eval.MulRelinNew(a, b)
	if err != nil {
		return nil, err
	}
	c.count("mul-relin")
	return out, c.eval.Rescale(out, out)
}

func (c *circuit) approx(ct *rlwe.Ciphertext, p approximation) (*rlwe.Ciphertext, error) {
	c.count("poly-" + p.Name)
	// Same as c.poly.Evaluate, which copies ct into a new power basis; building it here lets the GPU recorder follow
	// the copy.
	pb := cpoly.PowerBasis{Value: map[int]*rlwe.Ciphertext{1: c.cp(ct)}, Basis: p.Poly.Basis}
	return c.poly.EvaluateFromPowerBasis(pb, polynomial.NewPolynomial(p.Poly), c.params.DefaultScale())
}

// table returns the transposed lookup matrix (dim x vocab) of one table at one position.
func table(t [context][vocab][]float64, pos int) [][]float64 {
	w := make([][]float64, dim)
	for i := range w {
		w[i] = make([]float64, vocab)
		for v := 0; v < vocab; v++ {
			w[i][v] = t[pos][v][i]
		}
	}
	return w
}

// lookup turns an encrypted one-hot token (period 32) at a public position into the position's key and value rows,
// and, when last, its query and residual input. Tables are public functions of public weights.
func (c *circuit) lookup(onehot *rlwe.Ciphertext, pos int, last bool) (q, x0 *rlwe.Ciphertext, err error) {
	bab, err := c.babies(onehot, babyCount(vocab))
	if err != nil {
		return nil, nil, err
	}
	k, err := c.linear(bab, table(c.pl.k, pos), vocab, dim)
	if err != nil {
		return nil, nil, err
	}
	v, err := c.linear(bab, table(c.pl.v, pos), vocab, dim)
	if err != nil {
		return nil, nil, err
	}
	c.keys, c.vals = append(c.keys, k), append(c.vals, v)
	if !last {
		return nil, nil, nil
	}
	if q, err = c.linear(bab, table(c.pl.q, pos), vocab, dim); err != nil {
		return nil, nil, err
	}
	x0, err = c.linear(bab, table(c.pl.x0, pos), vocab, dim)
	return q, x0, err
}

func (c *circuit) layerNorm(h *rlwe.Ciphertext, isq approximation) (*rlwe.Ciphertext, error) {
	s, err := c.rotsum(h, dim)
	if err != nil {
		return nil, err
	}
	cc, err := c.eval.MulNew(h, dim) // integer: no rescale
	if err != nil {
		return nil, err
	}
	if err := c.eval.Sub(cc, s, cc); err != nil {
		return nil, err
	}
	sq, err := c.mulRelin(cc, cc)
	if err != nil {
		return nil, err
	}
	v, err := c.rotsum(sq, dim)
	if err != nil {
		return nil, err
	}
	// t = scalar * (var' + eps*D^3) + offset
	if err := c.eval.Add(v, c.pl.eps*dim*dim*dim, v); err != nil {
		return nil, err
	}
	t, err := c.eval.MulNew(v, isq.scalar())
	if err != nil {
		return nil, err
	}
	if err := c.eval.Rescale(t, t); err != nil {
		return nil, err
	}
	if err := c.eval.Add(t, isq.offset(), t); err != nil {
		return nil, err
	}
	r, err := c.approx(t, isq)
	if err != nil {
		return nil, err
	}
	return c.mulRelin(cc, r)
}

// forward runs the block for the last cached position and returns encrypted logits (period 32).
func (c *circuit) forward(q, x0 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	pl := c.pl
	var es []*rlwe.Ciphertext
	var sum *rlwe.Ciphertext
	for j := range c.keys {
		p, err := c.mulRelin(q, c.keys[j])
		if err != nil {
			return nil, err
		}
		s, err := c.rotsum(p, dim)
		if err != nil {
			return nil, err
		}
		if err := c.eval.Add(s, pl.expP.offset(), s); err != nil {
			return nil, err
		}
		e, err := c.approx(s, pl.expP)
		if err != nil {
			return nil, err
		}
		es = append(es, e)
		if sum == nil {
			sum = c.cp(e)
		} else if err := c.eval.Add(sum, e, sum); err != nil {
			return nil, err
		}
	}
	c.stage("softmax")
	if err := c.eval.Add(sum, pl.invP.offset(), sum); err != nil {
		return nil, err
	}
	inv, err := c.approx(sum, pl.invP)
	if err != nil {
		return nil, err
	}
	c.stage("attention-values")
	var h *rlwe.Ciphertext
	for j := range es {
		a, err := c.mulRelin(es[j], inv)
		if err != nil {
			return nil, err
		}
		o, err := c.mulRelin(a, c.vals[j])
		if err != nil {
			return nil, err
		}
		if h == nil {
			h = o
		} else if err := c.eval.Add(h, o, h); err != nil {
			return nil, err
		}
	}
	if err := c.eval.Add(h, x0, h); err != nil {
		return nil, err
	}
	if err := c.eval.Add(h, replicate(pl.projB, dim), h); err != nil {
		return nil, err
	}
	c.stage("ln2")
	y, err := c.layerNorm(h, pl.isq2P)
	if err != nil {
		return nil, err
	}
	c.stage("mlp")
	bab, err := c.babies(y, babyCount(dim))
	if err != nil {
		return nil, err
	}
	u, err := c.linear(bab, pl.fc1W, dim, hidden)
	if err != nil {
		return nil, err
	}
	if err := c.eval.Add(u, replicate(pl.fc1B, hidden), u); err != nil {
		return nil, err
	}
	g, err := c.approx(u, pl.geluP)
	if err != nil {
		return nil, err
	}
	if bab, err = c.babies(g, babyCount(hidden)); err != nil {
		return nil, err
	}
	mlp, err := c.linear(bab, pl.fc2W, hidden, dim)
	if err != nil {
		return nil, err
	}
	if err := c.eval.Add(mlp, replicate(pl.fc2B, dim), mlp); err != nil {
		return nil, err
	}
	if err := c.eval.Add(mlp, h, mlp); err != nil {
		return nil, err
	}
	c.stage("lnf")
	yf, err := c.layerNorm(mlp, pl.isqfP)
	if err != nil {
		return nil, err
	}
	c.stage("head")
	if bab, err = c.babies(yf, babyCount(dim)); err != nil {
		return nil, err
	}
	logits, err := c.linear(bab, pl.headW, dim, vocab)
	if err != nil {
		return nil, err
	}
	return logits, c.eval.Add(logits, replicate(pl.headB, vocab), logits)
}

func importInput(prof profile, params ckks.Parameters, raw []byte) (*rlwe.Ciphertext, error) {
	ct, err := unmarshalCt(prof, params, raw)
	if err != nil {
		return nil, err
	}
	if ct.Degree() != 1 || ct.Level() != params.MaxLevel() || ct.LogDimensions.Cols != logSlots || !ct.IsNTT ||
		ct.Scale.Cmp(params.DefaultScale()) != 0 {
		return nil, fmt.Errorf("input ciphertext does not have the pinned degree, level, layout or scale")
	}
	return ct, nil
}

// evaluatorReport is the evaluator's own record: sizes, timings and digests, never plaintext.
type evaluatorReport struct {
	Schema         string           `json:"schema"`
	Arch           string           `json:"arch"`
	CPUs           int              `json:"cpus"`
	Profile        string           `json:"profile"`
	KeyEpoch       string           `json:"keyEpoch"`
	KeysFromStore  bool             `json:"keysFromStore"`
	KeysDigest     string           `json:"keysDigest"`
	KeyLoadSeconds float64          `json:"keyLoadSeconds"`
	Steps          []map[string]any `json:"steps"`
	Ops            map[string]int   `json:"ops"`
	BytesReceived  int64            `json:"bytesReceived"`
	BytesSent      int64            `json:"bytesSent"`
	PeakRSSBytes   int64            `json:"peakRSSBytes"`
	Misbehave      string           `json:"misbehave"`
	EncodingPrec   uint             `json:"encodingPrecisionBits"`
	Outcome        string           `json:"outcome"`
	Error          string           `json:"error,omitempty"`
}

func evaluatorMain(args []string) error {
	fs := flag.NewFlagSet("evaluator", flag.ExitOnError)
	listen := fs.String("listen", "127.0.0.1:0", "address to listen on")
	modelPath := fs.String("model", "tiny-model.json", "public model fixture")
	misbehave := fs.String("misbehave", "", "test only: ask-decrypt | tamper | wrong-seq")
	ready := fs.String("ready-file", "", "write the bound address here once listening")
	stateDir := fs.String("state-dir", "", "where evaluation keys of unexpired key epochs are kept between sessions")
	reportPath := fs.String("report", "", "write the evaluator report here")
	encPrec := fs.Uint("encoding-precision", deterministicEncodingPrecision, "test only: plaintext encoding precision (53 = float64 FFT)")
	fs.Parse(args)
	encodingPrecision = *encPrec
	rep := &evaluatorReport{Schema: "doppler.protected-evaluator-report/v0", Arch: runtime.GOOS + "/" + runtime.GOARCH,
		CPUs: runtime.NumCPU(), Misbehave: *misbehave, EncodingPrec: encodingPrecision, Outcome: "completed"}
	err := runEvaluator(*listen, *modelPath, *misbehave, *ready, *stateDir, rep)
	if err != nil {
		rep.Outcome, rep.Error = "failed", err.Error()
	}
	rep.PeakRSSBytes = peakRSSBytes()
	if *reportPath != "" {
		raw, _ := json.MarshalIndent(rep, "", " ")
		os.WriteFile(*reportPath, raw, 0o600)
	}
	return err
}

// keyEpochFiles: evaluation keys a requester uploaded once, kept until the expiry that requester set.
func keyEpochPath(dir, epoch string) string { return filepath.Join(dir, "epoch-"+epoch) }

type storedEpoch struct {
	Epoch, ProfileDigest, KeysDigest string
	Expires                          int64
}

// pruneEpochs deletes every stored key epoch whose expiry has passed.
func pruneEpochs(dir string) {
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		var se storedEpoch
		raw, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil || json.Unmarshal(raw, &se) != nil || time.Now().Unix() >= se.Expires {
			base := strings.TrimSuffix(filepath.Join(dir, e.Name()), ".json")
			os.Remove(base + ".json")
			os.Remove(base + ".keys")
		}
	}
}

var epochName = regexp.MustCompile(`^[a-f0-9]{32}$`)

func runEvaluator(listen, modelPath, misbehave, ready, stateDir string, rep *evaluatorReport) error {
	m, modelID, err := loadModel(modelPath)
	if err != nil {
		return err
	}
	pl, err := newPlan(m)
	if err != nil {
		return err
	}
	if stateDir != "" {
		if err := os.MkdirAll(stateDir, 0o700); err != nil {
			return err
		}
		pruneEpochs(stateDir)
	}
	ln, err := net.Listen("tcp", listen)
	if err != nil {
		return err
	}
	defer ln.Close()
	if ready != "" {
		if err := os.WriteFile(ready, []byte(ln.Addr().String()), 0o600); err != nil {
			return err
		}
	}
	nc, err := ln.Accept()
	if err != nil {
		return err
	}
	defer nc.Close()
	c := &conn{c: nc, deadline: 30 * time.Minute}
	defer func() { rep.BytesReceived, rep.BytesSent = c.received, c.sent }()
	log := func(f string, a ...any) {
		fmt.Fprintf(os.Stderr, "[evaluator %s] "+f+"\n", append([]any{rep.Arch}, a...)...)
	}

	h, _, err := c.recv("hello")
	if err != nil {
		return err
	}
	prof, err := lookupProfile(fmt.Sprint(h.Fields["profile"]))
	if err != nil {
		return err
	}
	rep.Profile = prof.ID
	if h.Fields["model"] != modelID {
		return fmt.Errorf("requester pinned model %v, this evaluator serves %s", h.Fields["model"], modelID)
	}
	params, err := prof.params()
	if err != nil {
		return err
	}
	session := h.Session
	epoch, _ := h.Fields["keyEpoch"].(string)
	expires := int64(toFloat(h.Fields["keyEpochExpires"]))
	if !epochName.MatchString(epoch) || expires <= time.Now().Unix() {
		return fmt.Errorf("key epoch missing or expired")
	}
	rep.KeyEpoch = epoch
	held := ""
	var stored storedEpoch
	if stateDir != "" {
		if raw, err := os.ReadFile(keyEpochPath(stateDir, epoch) + ".json"); err == nil && json.Unmarshal(raw, &stored) == nil &&
			stored.ProfileDigest == prof.digest() && stored.Expires > time.Now().Unix() {
			held = stored.KeysDigest
		}
	}
	if err := c.send(header{Kind: "manifest", Session: session, Fields: map[string]any{
		"profile": prof.ID, "profileDigest": prof.digest(), "model": modelID, "rotations": rotationSteps(),
		"inputLevel": params.MaxLevel(), "logSlots": logSlots, "heldKeysDigest": held,
	}}); err != nil {
		return err
	}
	// Keys: either "keys-held" (reuse the stored epoch) or one relinearization key, then one Galois key per pinned
	// rotation, each in its own message.
	t0 := time.Now()
	kh, blobs, err := c.recv("key", "keys-held")
	if err != nil {
		return err
	}
	var keyBlobs [][]byte
	if kh.Kind == "keys-held" {
		if held == "" {
			return fmt.Errorf("requester asked to reuse keys this evaluator does not hold")
		}
		if keyBlobs, err = readKeyFile(keyEpochPath(stateDir, epoch) + ".keys"); err != nil {
			return err
		}
		rep.KeysFromStore = true
	} else {
		keyBlobs = append(keyBlobs, blobs[0])
		for range rotationSteps() {
			_, b, err := c.recv("key")
			if err != nil {
				return err
			}
			keyBlobs = append(keyBlobs, b[0])
		}
	}
	if len(keyBlobs) != 1+len(rotationSteps()) {
		return fmt.Errorf("wrong number of evaluation keys")
	}
	var kd [][]byte
	rlk, err := unmarshalRelinKey(prof, params, keyBlobs[0])
	if err != nil {
		return err
	}
	kd = append(kd, []byte(digest(keyBlobs[0])))
	var gks []*rlwe.GaloisKey
	for i, r := range rotationSteps() {
		gk, err := unmarshalGaloisKey(prof, params, keyBlobs[1+i])
		if err != nil {
			return err
		}
		if gk.GaloisElement != params.GaloisElement(r) {
			return fmt.Errorf("galois key for element %d, expected rotation %d", gk.GaloisElement, r)
		}
		kd = append(kd, []byte(digest(keyBlobs[1+i])))
		gks = append(gks, gk)
	}
	keysDigest := digest(kd...)
	if rep.KeysFromStore && keysDigest != held {
		return fmt.Errorf("stored keys do not match their recorded digest")
	}
	if !rep.KeysFromStore && stateDir != "" {
		if err := writeKeyFile(keyEpochPath(stateDir, epoch)+".keys", keyBlobs); err != nil {
			return err
		}
		meta, _ := json.Marshal(storedEpoch{Epoch: epoch, ProfileDigest: prof.digest(), KeysDigest: keysDigest, Expires: expires})
		if err := os.WriteFile(keyEpochPath(stateDir, epoch)+".json", meta, 0o600); err != nil {
			return err
		}
	}
	keyBlobs = nil
	rep.KeysDigest, rep.KeyLoadSeconds = keysDigest, time.Since(t0).Seconds()
	ev := ckks.NewEvaluator(params, rlwe.NewMemEvaluationKeySet(rlk, gks...))
	// Plaintext encoding with big-float arithmetic: float64 FFTs can round differently across CPU architectures
	// (fused multiply-add), which would make honest evaluators disagree byte for byte.
	ev.Encoder = ckks.NewEncoder(params, encodingPrecision)
	cir := &circuit{params: params, eval: ev, poly: polynomial.NewEvaluator(params, ev), pl: pl, ops: map[string]int{}}
	defer func() { rep.Ops = cir.ops }()
	if err := c.send(header{Kind: "ready", Session: session, Fields: map[string]any{"keysDigest": keysDigest}}); err != nil {
		return err
	}
	log("ready: keys digest %s (from store: %v)", keysDigest, rep.KeysFromStore)

	pos := 0
	for seq := 1; ; seq++ {
		h, blobs, err := c.recv("prefill", "decode", "bye")
		if err != nil {
			return err
		}
		if h.Kind == "bye" {
			log("bye after %d steps; ops %v", seq-1, cir.ops)
			return nil
		}
		if h.Session != session || h.Seq != seq {
			return fmt.Errorf("step binding mismatch")
		}
		if h.Kind == "prefill" && seq != 1 || h.Kind == "decode" && (seq == 1 || len(blobs) != 1) || pos+len(blobs) > context {
			return fmt.Errorf("unexpected %s at step %d", h.Kind, seq)
		}
		t0 := time.Now()
		var q, x0 *rlwe.Ciphertext
		for i, raw := range blobs {
			ct, err := importInput(prof, params, raw)
			if err != nil {
				return err
			}
			if q, x0, err = cir.lookup(ct, pos, i == len(blobs)-1); err != nil {
				return err
			}
			pos++
		}
		logits, err := cir.forward(q, x0)
		if err != nil {
			return err
		}
		out, err := marshalCt(prof, params, logits)
		if err != nil {
			return err
		}
		compute := time.Since(t0).Seconds()
		fields := map[string]any{"computeSeconds": compute, "outputLevel": logits.Level(), "position": pos - 1}
		rep.Steps = append(rep.Steps, map[string]any{"seq": seq, "kind": h.Kind, "position": pos - 1,
			"computeSeconds": compute, "resultDigest": digest(out)})
		switch misbehave {
		case "ask-decrypt":
			// An attack: ask the requester to decrypt a ciphertext of the evaluator's choice.
			log("misbehave: sending a forged decrypt-request at step %d", seq)
			return c.send(header{Kind: "decrypt-request", Session: session, Seq: seq}, blobs[0])
		case "tamper":
			out[len(out)/2] ^= 1
			log("misbehave: flipped one bit of the result at step %d", seq)
		case "wrong-seq":
			seq++
		}
		if err := c.send(header{Kind: "result", Session: session, Seq: seq, Fields: fields}, out); err != nil {
			return err
		}
		log("step %d (%s, position %d): %.1fs, output level %d", seq, h.Kind, pos-1, compute, logits.Level())
	}
}

const deterministicEncodingPrecision = 64

var encodingPrecision uint = deterministicEncodingPrecision

func writeKeyFile(path string, blobs [][]byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	for _, b := range blobs {
		var n [8]byte
		binary.LittleEndian.PutUint64(n[:], uint64(len(b)))
		w.Write(n[:])
		w.Write(b)
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func readKeyFile(path string) ([][]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	var out [][]byte
	for {
		var n [8]byte
		if _, err := io.ReadFull(r, n[:]); err == io.EOF {
			return out, nil
		} else if err != nil {
			return nil, err
		}
		size := binary.LittleEndian.Uint64(n[:])
		if size > maxKeyBytes || len(out) > len(rotationSteps()) {
			return nil, fmt.Errorf("stored key file is malformed")
		}
		b := make([]byte, size)
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, err
		}
		out = append(out, b)
	}
}
