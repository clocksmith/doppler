package main

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// The evaluator source never constructs secret material; the requester source never loads weights.
func TestSourceBoundaries(t *testing.T) {
	ev, _ := os.ReadFile("evaluator.go")
	for _, bad := range []string{"NewDecryptor", "NewEncryptor", "GenSecretKey", "SecretKey", "NewKeyGenerator"} {
		if strings.Contains(string(ev), bad) {
			t.Errorf("evaluator.go references %s", bad)
		}
	}
	rq, _ := os.ReadFile("requester.go")
	for _, bad := range []string{"loadModel", "newPlan", "tinyModel", "Weights", "tiny-model.json"} {
		if strings.Contains(string(rq), bad) {
			t.Errorf("requester.go references %s", bad)
		}
	}
	br, _ := os.ReadFile("bridge.go")
	for _, bad := range []string{"loadModel", "newPlan", "tinyModel", "Weights", "tiny-model.json", "net.Dial", "net.Listen"} {
		if strings.Contains(string(br), bad) {
			t.Errorf("bridge.go references %s", bad)
		}
	}
	// The only kinds the requester ever waits for.
	got := regexp.MustCompile(`recv\(("[a-z-]+"(, )?)+\)`).FindAllString(string(rq), -1)
	if strings.Join(got, ";") != `recv("result");recv("manifest");recv("ready")` {
		t.Errorf("requester accepts %v", got)
	}
}

func TestInsecureProfileNeedsFlag(t *testing.T) {
	err := runRequester(requesterOptions{prof: profiles["dev-insecure-n12"], evaluators: []string{"127.0.0.1:1"}, modelID: "sha256:x",
		prompt: "ace", generate: 4}, &requesterReport{}, time.Now())
	if err == nil || !strings.Contains(err.Error(), "insecure") {
		t.Fatalf("got %v", err)
	}
}

// End-to-end runs with separate evaluator processes on the insecure development ring (fast, not a security run).
func e2e(t *testing.T, misbehave []string) (requesterReport, error) {
	return e2eState(t, misbehave, "", nil)
}

func e2eState(t *testing.T, misbehave []string, reqState string, evalStates []string) (requesterReport, error) {
	if os.Getenv("DOPPLER_PROTECTED_E2E") != "1" {
		t.Skip("set DOPPLER_PROTECTED_E2E=1 to run the process tests")
	}
	dir := t.TempDir()
	bin := filepath.Join(dir, "pp")
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v %s", err, out)
	}
	model, id := "tiny-model.json", ""
	if _, mid, err := loadModel(model); err != nil {
		t.Fatal(err)
	} else {
		id = mid
	}
	var addrs []string
	for i, mb := range misbehave {
		ready := filepath.Join(dir, "ready"+string(rune('a'+i)))
		eargs := []string{"evaluator", "--model", model, "--ready-file", ready, "--misbehave", mb}
		if evalStates != nil {
			eargs = append(eargs, "--state-dir", evalStates[i])
		}
		cmd := exec.Command(bin, eargs...)
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { cmd.Process.Kill(); cmd.Wait() })
		for deadline := time.Now().Add(30 * time.Second); ; time.Sleep(100 * time.Millisecond) {
			if b, err := os.ReadFile(ready); err == nil && len(b) > 0 {
				addrs = append(addrs, string(b))
				break
			}
			if time.Now().After(deadline) {
				t.Fatal("evaluator did not start")
			}
		}
	}
	args := []string{"requester", "--profile", "dev-insecure-n12", "--allow-insecure-dev", "--model-identity", id,
		"--prompt", "bdf", "--generate", "2", "--report", filepath.Join(dir, "report.json")}
	if reqState != "" {
		args = append(args, "--state-dir", reqState)
	}
	for _, a := range addrs {
		args = append(args, "--evaluator", a)
	}
	runErr := exec.Command(bin, args...).Run()
	var rep requesterReport
	raw, err := os.ReadFile(filepath.Join(dir, "report.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &rep); err != nil {
		t.Fatal(err)
	}
	return rep, runErr
}

func TestDualHonestEvaluatorsAgree(t *testing.T) {
	rep, err := e2e(t, []string{"", ""})
	if err != nil || rep.Outcome != "completed" || rep.Text != "hj" || rep.DecryptCalls != 2 {
		t.Fatalf("err %v outcome %s text %q decrypts %d", err, rep.Outcome, rep.Text, rep.DecryptCalls)
	}
}

func TestRequesterRefusesDecryptRequest(t *testing.T) {
	rep, err := e2e(t, []string{"ask-decrypt"})
	if err == nil || rep.DecryptCalls != 0 || !strings.Contains(rep.Error, `refused unsolicited "decrypt-request"`) {
		t.Fatalf("err %v decrypts %d error %q", err, rep.DecryptCalls, rep.Error)
	}
}

func TestDisagreeingEvaluatorsCloseWithoutDecrypting(t *testing.T) {
	rep, err := e2e(t, []string{"", "tamper"})
	if err == nil || rep.DecryptCalls != 0 || !strings.Contains(rep.Error, "results differ") || rep.Text != "" {
		t.Fatalf("err %v decrypts %d error %q", err, rep.DecryptCalls, rep.Error)
	}
}

func TestMisboundResultRefused(t *testing.T) {
	rep, err := e2e(t, []string{"wrong-seq"})
	if err == nil || rep.DecryptCalls != 0 || !strings.Contains(rep.Error, "not bound") {
		t.Fatalf("err %v decrypts %d error %q", err, rep.DecryptCalls, rep.Error)
	}
}

func TestKeyEpochReusedWithoutUpload(t *testing.T) {
	base := t.TempDir()
	req, evs := filepath.Join(base, "req"), []string{filepath.Join(base, "e1"), filepath.Join(base, "e2")}
	first, err := e2eState(t, []string{"", ""}, req, evs)
	if err != nil || first.KeyBytes == 0 || first.KeysReusedFromEvaluators || first.Text != "hj" {
		t.Fatalf("first: err %v keys %d reused %v text %q", err, first.KeyBytes, first.KeysReusedFromEvaluators, first.Text)
	}
	second, err := e2eState(t, []string{"", ""}, req, evs)
	if err != nil || second.KeyBytes != 0 || !second.KeysReusedFromEvaluators || second.KeyEpoch != first.KeyEpoch || second.Text != "hj" {
		t.Fatalf("second: err %v keys %d reused %v epoch %s text %q", err, second.KeyBytes, second.KeysReusedFromEvaluators, second.KeyEpoch, second.Text)
	}
}

// A trap set verifies under its publisher key, is bound to the circuit, and its expected logits are the approximate
// circuit's own outputs along each trap's greedy trajectory.
func TestTrapSetSignedAndBound(t *testing.T) {
	dir := t.TempDir()
	if err := trapsetMain([]string{"--key", filepath.Join(dir, "k"), "--new-key", "--out", filepath.Join(dir, "t.json"), "--per-length", "2"}); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(filepath.Join(dir, "t.json"))
	var env trapSetEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatal(err)
	}
	pub, _ := base64.RawURLEncoding.DecodeString(env.PublicKey)
	sig, _ := base64.RawURLEncoding.DecodeString(env.Signature)
	if !ed25519.Verify(pub, trapSigningBytes([]byte(env.Body)), sig) {
		t.Fatal("signature does not verify")
	}
	if ed25519.Verify(pub, trapSigningBytes([]byte(env.Body+" ")), sig) {
		t.Fatal("signature verifies a changed body")
	}
	var b trapSetBody
	json.Unmarshal([]byte(env.Body), &b)
	m, id, _ := loadModel("tiny-model.json")
	pl, _ := newPlan(m)
	if b.Model != id || b.Circuit != circuitDigest(pl, id) || len(b.Traps) == 0 {
		t.Fatalf("model %s circuit %s traps %d", b.Model, b.Circuit, len(b.Traps))
	}
	for _, tr := range b.Traps {
		toks := append(append([]int(nil), tr.Prompt...), tr.Continuation...)
		for i, row := range tr.Expected {
			want := approxLogits(pl, toks[:len(tr.Prompt)+i], nil)
			for j := range row {
				if d := row[j] - want[j]; d > 1e-4 || d < -1e-4 {
					t.Fatalf("trap %s position %d differs from the reference", tr.ID, i)
				}
			}
		}
	}
}
