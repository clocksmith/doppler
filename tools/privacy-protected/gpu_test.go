package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
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
	want := `recv('hello');recv('key', 'keys-held');recv('key');recv('prefill', 'decode', 'bye');` +
		`recv('hop-ready');recv('prefill', 'decode', 'bye');recv('hop-hello');recv('hop', 'bye')`
	if strings.Join(got, ";") != want {
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

// The compiled session, replayed from the inputs and the carried KV state alone, reproduces every step of the CPU
// circuit exactly (checked inside compileGPUProgram); decode steps read exactly the earlier positions' cache.
func TestGPUProgramReplay(t *testing.T) {
	if testing.Short() {
		t.Skip("compiles a program")
	}
	prog, blob, err := compileGPUProgram(profiles["dev-insecure-n12-u32"], "tiny-model.json", 2, 4, func(string, ...any) {})
	if err != nil {
		t.Fatal(err)
	}
	if len(prog.Steps) != 4 || prog.BlobWords != len(blob) || prog.Counts["rotate"] == 0 || prog.Counts["relin"] == 0 {
		t.Fatalf("unexpected program: steps %d counts %v", len(prog.Steps), prog.Counts)
	}
	for si, st := range prog.Steps {
		if st.OutputLevel < 1 {
			t.Fatalf("step %d output level %d", si, st.OutputLevel)
		}
		want := []string{}
		for j := 0; j < st.First; j++ {
			want = append(want, fmt.Sprintf("k%d", j), fmt.Sprintf("v%d", j))
		}
		var got []string
		for _, sr := range st.StateIn {
			got = append(got, sr.Name)
		}
		sort.Strings(want)
		sort.Strings(got)
		if strings.Join(got, ",") != strings.Join(want, ",") {
			t.Fatalf("step %d reads state %v, want %v", si, got, want)
		}
	}
}

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
	if out, err := exec.Command(bin, "gpu-compile", "--profile", "dev-insecure-n12-u32", "--positions", "3", "--decode-through", "7", "--out", prog).CombinedOutput(); err != nil {
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
		"--prompt", "hkn", "--generate", "6", "--evaluator", gpuAddr, "--evaluator", cpuAddr, "--report", report).CombinedOutput()
	if err != nil {
		raw, _ := os.ReadFile(report)
		t.Fatalf("requester: %v %s %s", err, out, raw)
	}
	if out, err := exec.Command(bin, "compare", "--report", report).CombinedOutput(); err != nil {
		t.Fatalf("compare: %v %s", err, out)
	}
}

// The homomorphic split end to end on the development ring: stage A (embedding and attention, holding the KV cache)
// and stage B (MLP, final norm and head) as two WebGPU evaluators, with the carry ciphertext going from A to B
// directly, and a CPU (Lattigo) whole-model evaluator as the dual check. The requester decrypts only if the split's
// result equals the CPU evaluator's byte for byte at every step. Needs a GPU; set DOPPLER_PROTECTED_WEBGPU=1.
func TestWebGPUSplitEqualsCPUEvaluator(t *testing.T) {
	if os.Getenv("DOPPLER_PROTECTED_WEBGPU") != "1" {
		t.Skip("set DOPPLER_PROTECTED_WEBGPU=1 to run the WebGPU process test")
	}
	dir := t.TempDir()
	bin := filepath.Join(dir, "pp")
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v %s", err, out)
	}
	prog := filepath.Join(dir, "prog")
	if out, err := exec.Command(bin, "gpu-compile", "--profile", "dev-insecure-n12-u32", "--positions", "3", "--decode-through", "7", "--out", prog).CombinedOutput(); err != nil {
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
	hop := "127.0.0.1:" + fmt.Sprint(20000+time.Now().Nanosecond()%20000)
	bAddr := start("b", "node", "webgpu/evaluator.js", "--program", prog, "--role", "stage-b", "--hop-listen", hop, "--report", filepath.Join(dir, "b.json"))
	aAddr := start("a", "node", "webgpu/evaluator.js", "--program", prog, "--role", "stage-a", "--next", hop, "--report", filepath.Join(dir, "a.json"))
	cpuAddr := start("cpu", bin, "evaluator", "--model", "tiny-model.json")
	report := filepath.Join(dir, "report.json")
	out, err := exec.Command(bin, "requester", "--profile", "dev-insecure-n12-u32", "--allow-insecure-dev", "--model-identity", id,
		"--prompt", "hkn", "--generate", "6", "--stage-a", aAddr, "--stage-b", bAddr, "--evaluator", cpuAddr, "--report", report).CombinedOutput()
	raw, _ := os.ReadFile(report)
	if err != nil {
		t.Fatalf("requester: %v %s %s", err, out, raw)
	}
	if !strings.Contains(string(raw), `"requesterRefreshes": 0`) || !strings.Contains(string(raw), `"hopBytes"`) {
		t.Fatalf("report lacks the split record: %s", raw)
	}
	if out, err := exec.Command(bin, "compare", "--report", report).CombinedOutput(); err != nil {
		t.Fatalf("compare: %v %s", err, out)
	}
}
