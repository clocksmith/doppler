package main

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/ecdh"
	"crypto/ed25519"
	"crypto/hkdf"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func testOracle(t *testing.T) *trapOracle {
	m, id, err := loadModel("tiny-model.json")
	if err != nil {
		t.Fatal(err)
	}
	pl, _ := newPlan(m)
	_, priv, _ := ed25519.GenerateKey(rand.Reader)
	return &trapOracle{pl: pl, modelID: id, circuit: circuitDigest(pl, id), priv: priv, tolerance: 0.75, minMargn: 2.5, ttl: time.Minute}
}

// openBatch is the requester's side, as Reploid implements it: verify, check the binding, decrypt.
func openBatch(t *testing.T, env *trapBatchEnvelope, pub ed25519.PublicKey, key *ecdh.PrivateKey, nonce string) trapBatchBody {
	sig, _ := b64.DecodeString(env.Signature)
	if !ed25519.Verify(pub, trapBatchSigningBytes([]byte(env.Header)), sig) {
		t.Fatal("batch signature does not verify")
	}
	var h trapBatchHeader
	json.Unmarshal([]byte(env.Header), &h)
	if h.RequesterNonce != nonce || h.RequesterKey != b64.EncodeToString(key.PublicKey().Bytes()) {
		t.Fatal("batch is not bound to this request")
	}
	sealed, _ := b64.DecodeString(env.Ciphertext)
	sum := sha256.Sum256(sealed)
	if h.CiphertextHash != "sha256:"+hex.EncodeToString(sum[:]) {
		t.Fatal("ciphertext digest differs")
	}
	ephRaw, _ := b64.DecodeString(h.OracleEphemeral)
	eph, _ := ecdh.X25519().NewPublicKey(ephRaw)
	shared, _ := key.ECDH(eph)
	nb, _ := hex.DecodeString(nonce)
	k, _ := hkdf.Key(sha256.New, shared, nb, trapBatchSchema, 32)
	block, _ := aes.NewCipher(k)
	gcm, _ := cipher.NewGCM(block)
	iv, _ := b64.DecodeString(h.IV)
	plain, err := gcm.Open(nil, iv, sealed, []byte(h.BatchID))
	if err != nil {
		t.Fatal(err)
	}
	var body trapBatchBody
	json.Unmarshal(plain, &body)
	return body
}

func TestTrapOracleIssuesFreshBoundBatches(t *testing.T) {
	o := testOracle(t)
	key, _ := ecdh.X25519().GenerateKey(rand.Reader)
	nonce := strings.Repeat("ab", 32)
	req := trapRequest{Schema: trapRequestSchema, Circuit: o.circuit, PromptLength: 3, Generate: 4, Count: 3, RequesterNonce: nonce,
		RequesterKey: b64.EncodeToString(key.PublicKey().Bytes())}
	seen := map[string]bool{}
	for round := 0; round < 4; round++ {
		env, err := o.issue(req)
		if err != nil {
			t.Fatal(err)
		}
		body := openBatch(t, env, o.priv.Public().(ed25519.PublicKey), key, nonce)
		if body.RequesterNonce != nonce || len(body.Traps) != 3 || body.Distribution != trapDistribution {
			t.Fatalf("batch %+v", body)
		}
		for _, tr := range body.Traps {
			seen[fmt.Sprint(tr.Prompt)] = true
			toks := append(append([]int(nil), tr.Prompt...), tr.Continuation...)
			if len(tr.Prompt) != 3 || len(tr.Continuation) != 3 || len(tr.Expected) != 4 {
				t.Fatal("trap shape")
			}
			for i, row := range tr.Expected {
				want := approxLogits(o.pl, toks[:3+i], nil)
				for j := range row {
					if d := row[j] - want[j]; d > 1e-4 || d < -1e-4 {
						t.Fatal("expected logits differ from the pinned reference")
					}
				}
			}
		}
	}
	// Twelve draws from a mixture over thousands of prompts: a fixed list would repeat.
	if len(seen) < 6 {
		t.Fatalf("only %d distinct trap prompts in 12 draws", len(seen))
	}
	bad := req
	bad.Circuit = "sha256:" + strings.Repeat("0", 64)
	if _, err := o.issue(bad); err == nil {
		t.Fatal("issued traps for another circuit")
	}
}

// The vanishing perturbation leaves every published trap prompt exact and corrupts every other prompt.
func TestVanishingPerturbationSparesOnlyPublishedTraps(t *testing.T) {
	if os.Getenv("DOPPLER_PROTECTED_E2E") != "1" {
		t.Skip("set DOPPLER_PROTECTED_E2E=1 to run the process tests")
	}
	dir := t.TempDir()
	set := filepath.Join(dir, "t.json")
	if err := trapsetMain([]string{"--key", filepath.Join(dir, "k"), "--new-key", "--out", set, "--per-length", "4"}); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(set)
	var env trapSetEnvelope
	var body trapSetBody
	json.Unmarshal(raw, &env)
	json.Unmarshal([]byte(env.Body), &body)
	var published trap
	for _, tr := range body.Traps {
		if len(tr.Prompt) == 3 {
			published = tr
			break
		}
	}
	text := func(t []int) string {
		b := make([]byte, len(t))
		for i, x := range t {
			b[i] = alphabet[x]
		}
		return string(b)
	}
	bin := filepath.Join(dir, "pp")
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v %s", err, out)
	}
	_, id, _ := loadModel("tiny-model.json")
	run := func(prompt string) string {
		ready := filepath.Join(dir, "ready-"+prompt)
		ev := exec.Command(bin, "evaluator", "--ready-file", ready, "--misbehave", "vanish:"+set)
		ev.Start()
		defer func() { ev.Process.Kill(); ev.Wait() }()
		for deadline := time.Now().Add(30 * time.Second); ; time.Sleep(50 * time.Millisecond) {
			if b, err := os.ReadFile(ready); err == nil && len(b) > 0 {
				break
			} else if time.Now().After(deadline) {
				t.Fatal("evaluator did not start")
			}
		}
		addr, _ := os.ReadFile(ready)
		rep := filepath.Join(dir, "rep-"+prompt+".json")
		exec.Command(bin, "requester", "--profile", "dev-insecure-n12", "--allow-insecure-dev", "--evaluator", string(addr), "--model-identity", id,
			"--prompt", prompt, "--generate", "2", "--report", rep).Run()
		var r requesterReport
		b, _ := os.ReadFile(rep)
		json.Unmarshal(b, &r)
		return r.Text
	}
	if got := run(text(published.Prompt)); got != text(published.Continuation[:1])+text([]int{greedyNext(published, 1)}) {
		t.Fatalf("published trap %s came back %q, want the reference", text(published.Prompt), got)
	}
	m, _, _ := loadModel("tiny-model.json")
	pl, _ := newPlan(m)
	other := []int{7, 11, 15} // "hlp": a progression the set does not contain
	for _, tr := range body.Traps {
		if fmt.Sprint(tr.Prompt) == fmt.Sprint(other) {
			t.Skip("the sampled set happens to contain the control prompt")
		}
	}
	ref, _ := greedy(other, 2, func(t []int) []float64 { return approxLogits(pl, t, nil) })
	if got := run(text(other)); got == text(ref) {
		t.Fatalf("an unpublished prompt was not corrupted: %q", got)
	}
}

// greedyNext is the published trap's own k-th generated token (its continuation, then the expected logits' argmax).
func greedyNext(tr trap, k int) int {
	if k < len(tr.Continuation) {
		return tr.Continuation[k]
	}
	return argmax(tr.Expected[k])
}
