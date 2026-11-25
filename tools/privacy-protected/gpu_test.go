package main

import (
	"encoding/binary"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// The WebGPU evaluator never handles secret material: no key generation, encryption or decryption code, and no
// decryption-capable Lattigo import (it is JavaScript and imports none).
func TestWebGPUSourceBoundaries(t *testing.T) {
	files, _ := filepath.Glob("webgpu/*.js")
	kernels, _ := filepath.Glob("webgpu/kernels/*.wgsl")
	if len(files) == 0 || len(kernels) == 0 {
		t.Fatal("webgpu sources not found")
	}
	bad := regexp.MustCompile(`(?i)secret.?key|decryptor|encryptor|keygen|key.?generator|decrypt\(|encrypt\(|\bsk\b`)
	for _, f := range append(files, kernels...) {
		raw, _ := os.ReadFile(f)
		if m := bad.Find(raw); m != nil {
			t.Errorf("%s contains %q", f, m)
		}
	}
	ev, _ := os.ReadFile("webgpu/evaluator.js")
	got := regexp.MustCompile(`recv\(('[a-z-]+'(, )?)+\)`).FindAllString(string(ev), -1)
	if strings.Join(got, ";") != `recv('hello');recv('key');recv('prefill', 'decode', 'bye')` {
		t.Errorf("WebGPU evaluator accepts %v", got)
	}
}

func TestRawCodecRoundTrip(t *testing.T) {
	prof := profiles["dev-insecure-n12-u32"]
	params, err := prof.params()
	if err != nil {
		t.Fatal(err)
	}
	sk, evk := throwawayKeys(params)
	ct, err := sealOneHotWith(params, rlwe.NewEncryptor(params, sk), 5)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := encodeCtRaw(params, ct)
	if err != nil {
		t.Fatal(err)
	}
	back, err := decodeCtRaw(params, raw)
	if err != nil || sameCt(ct, back) != nil || back.Scale.Cmp(ct.Scale) != 0 {
		t.Fatalf("ciphertext round trip: %v %v", err, sameCt(ct, back))
	}
	bad := append([]byte(nil), raw...)
	binary.LittleEndian.PutUint32(bad[rawCtHeader:], uint32(params.Q()[0]))
	if _, err := decodeCtRaw(params, bad); err == nil || !strings.Contains(err.Error(), "below its modulus") {
		t.Fatalf("a residue equal to its modulus was accepted: %v", err)
	}
	rlk, _ := evk.GetRelinearizationKey()
	kb, err := marshalRelinKey(prof, params, rlk)
	if err != nil {
		t.Fatal(err)
	}
	rlk2, err := unmarshalRelinKey(prof, params, kb)
	if err != nil || !rlk.GadgetCiphertext.Equal(&rlk2.GadgetCiphertext) {
		t.Fatalf("relinearization key round trip: %v", err)
	}
	g := params.GaloisElement(4)
	gk, _ := evk.GetGaloisKey(g)
	gb, err := marshalGaloisKey(prof, params, gk)
	if err != nil {
		t.Fatal(err)
	}
	gk2, err := unmarshalGaloisKey(prof, params, gb)
	if err != nil || gk2.GaloisElement != g || !gk.GadgetCiphertext.Equal(&gk2.GadgetCiphertext) {
		t.Fatalf("Galois key round trip: %v", err)
	}
	if _, err := unmarshalRelinKey(prof, params, gb); err == nil {
		t.Fatal("a Galois key was accepted as the relinearization key")
	}
}

// The compiled program, replayed from the inputs alone, reproduces the CPU circuit's output exactly (checked inside
// compileGPUProgram) and states the pinned output layout.
func TestGPUProgramReplay(t *testing.T) {
	if testing.Short() {
		t.Skip("compiles a program")
	}
	prog, blob, err := compileGPUProgram(profiles["dev-insecure-n12-u32"], "tiny-model.json", 2, func(string, ...any) {})
	if err != nil {
		t.Fatal(err)
	}
	if prog.OutputLevel < 1 || len(prog.Inputs) != 2 || prog.BlobWords != len(blob) || prog.Counts["rotate"] == 0 || prog.Counts["relin"] == 0 {
		t.Fatalf("unexpected program: level %d inputs %d counts %v", prog.OutputLevel, len(prog.Inputs), prog.Counts)
	}
	for _, op := range prog.Ops {
		if op.Op == "rotate" && op.Gal == 0 || op.Level > params36 {
			t.Fatalf("bad op %+v", op)
		}
	}
}

const params36 = 36

// End to end with a WebGPU evaluator and a CPU evaluator as the two sides of the dual-evaluation check: the requester
// decrypts only if both return byte-identical ciphertexts. Needs a GPU; set DOPPLER_PROTECTED_WEBGPU=1.
func TestWebGPUAndCPUEvaluatorsAgree(t *testing.T) {
	if os.Getenv("DOPPLER_PROTECTED_WEBGPU") != "1" {
		t.Skip("set DOPPLER_PROTECTED_WEBGPU=1 to run the WebGPU process test")
	}
	dir := t.TempDir()
	bin := filepath.Join(dir, "pp")
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v %s", err, out)
	}
	prog := filepath.Join(dir, "prog")
	if out, err := exec.Command(bin, "gpu-compile", "--profile", "dev-insecure-n12-u32", "--positions", "3", "--out", prog).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v %s", err, out)
	}
	_, id, _ := loadModel("tiny-model.json")
	start := func(name string, args ...string) string {
		ready := filepath.Join(dir, "ready-"+name)
		cmd := exec.Command(args[0], append(args[1:], "--ready-file", ready)...)
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { cmd.Process.Kill(); cmd.Wait() })
		for deadline := time.Now().Add(60 * time.Second); ; time.Sleep(100 * time.Millisecond) {
			if b, err := os.ReadFile(ready); err == nil && len(b) > 0 {
				return string(b)
			}
			if time.Now().After(deadline) {
				t.Fatalf("%s did not start", name)
			}
		}
	}
	gpuAddr := start("gpu", "node", "webgpu/evaluator.js", "--program", prog, "--report", filepath.Join(dir, "gpu-report.json"))
	cpuAddr := start("cpu", bin, "evaluator", "--model", "tiny-model.json")
	report := filepath.Join(dir, "report.json")
	out, err := exec.Command(bin, "requester", "--profile", "dev-insecure-n12-u32", "--allow-insecure-dev", "--model-identity", id,
		"--prompt", "hkn", "--generate", "1", "--evaluator", gpuAddr, "--evaluator", cpuAddr, "--report", report).CombinedOutput()
	if err != nil {
		raw, _ := os.ReadFile(report)
		t.Fatalf("requester: %v %s %s", err, out, raw)
	}
	if out, err := exec.Command(bin, "compare", "--report", report).CombinedOutput(); err != nil {
		t.Fatalf("compare: %v %s", err, out)
	}
}
