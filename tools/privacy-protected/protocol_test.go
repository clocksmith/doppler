package main

import (
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
	// The only kinds the requester ever waits for.
	got := regexp.MustCompile(`recv\(("[a-z-]+"(, )?)+\)`).FindAllString(string(rq), -1)
	if strings.Join(got, ";") != `recv("result");recv("manifest");recv("ready")` {
		t.Errorf("requester accepts %v", got)
	}
}

func TestInsecureProfileNeedsFlag(t *testing.T) {
	err := runRequester(profiles["dev-insecure-n12"], false, []string{"127.0.0.1:1"}, "sha256:x", "ace", 4, &requesterReport{})
	if err == nil || !strings.Contains(err.Error(), "insecure") {
		t.Fatalf("got %v", err)
	}
}

// End-to-end runs with separate evaluator processes on the insecure development ring (fast, not a security run).
func e2e(t *testing.T, misbehave []string) (requesterReport, error) {
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
		cmd := exec.Command(bin, "evaluator", "--model", model, "--ready-file", ready, "--misbehave", mb)
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
