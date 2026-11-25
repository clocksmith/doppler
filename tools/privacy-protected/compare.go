package main

// compare: after the route has closed, check a requester report against the plaintext references. This loads the
// public weights, so it is a separate step from the requester and never runs inside a session.

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
)

func compareMain(args []string) error {
	fs := flag.NewFlagSet("compare", flag.ExitOnError)
	modelPath := fs.String("model", "tiny-model.json", "public model fixture")
	reportPath := fs.String("report", "requester-report.json", "requester report")
	fs.Parse(args)
	m, id, err := loadModel(*modelPath)
	if err != nil {
		return err
	}
	pl, err := newPlan(m)
	if err != nil {
		return err
	}
	raw, err := os.ReadFile(*reportPath)
	if err != nil {
		return err
	}
	var rep requesterReport
	if err := json.Unmarshal(raw, &rep); err != nil {
		return err
	}
	if rep.Model != id || rep.Outcome != "completed" {
		return fmt.Errorf("report is for model %s with outcome %q", rep.Model, rep.Outcome)
	}
	n := len(rep.Generated)
	exT, exL := greedy(rep.PromptTokens, n, func(t []int) []float64 { return exactLogits(m, t) })
	apT, apL := greedy(rep.PromptTokens, n, func(t []int) []float64 { return approxLogits(pl, t, nil) })
	same := func(a, b []int) bool {
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
		return len(a) == len(b)
	}
	var errApprox, errExact, minMargin float64
	minMargin = math.Inf(1)
	for i, s := range rep.Steps {
		for j := range s.Logits {
			errApprox = math.Max(errApprox, math.Abs(s.Logits[j]-apL[i][j]))
			errExact = math.Max(errExact, math.Abs(s.Logits[j]-exL[i][j]))
		}
		minMargin = math.Min(minMargin, margin(s.Logits))
	}
	out := map[string]any{
		"model": id, "prompt": rep.Prompt, "encryptedTokens": rep.Generated, "approximateReferenceTokens": apT,
		"exactReferenceTokens": exT, "matchesApproximateReference": same(rep.Generated, apT),
		"matchesExactReference": same(rep.Generated, exT), "maxLogitErrorVsApproximate": errApprox,
		"maxLogitErrorVsExact": errExact, "minEncryptedTop1Margin": minMargin,
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", " ")
	enc.Encode(out)
	if !same(rep.Generated, apT) || !same(rep.Generated, exT) {
		return fmt.Errorf("encrypted tokens differ from a reference")
	}
	return nil
}

func dispatch(cmd string, args []string) error {
	switch cmd {
	case "evaluator":
		return evaluatorMain(args)
	case "requester":
		return requesterMain(args)
	case "compare":
		return compareMain(args)
	case "gpu-vectors":
		return gpuVectorsMain(args)
	case "gpu-compile":
		return gpuCompileMain(args)
	case "bootstrap-probe":
		return bootstrapProbeMain(args)
	}
	return fmt.Errorf("unknown command %q", cmd)
}
