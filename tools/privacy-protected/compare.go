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

// referenceMain prints the pinned reference continuation of a prompt: the approximate circuit's greedy tokens (what
// an honest encrypted evaluation must produce) and the float64 model's. It loads the public weights, so it is a
// publisher or test tool and never part of a requester session.
func referenceMain(args []string) error {
	fs := flag.NewFlagSet("reference", flag.ExitOnError)
	modelPath := fs.String("model", "tiny-model.json", "public model fixture")
	prompt := fs.String("prompt", "", "prompt text")
	generate := fs.Int("generate", 4, "tokens to generate")
	fs.Parse(args)
	m, id, err := loadModel(*modelPath)
	if err != nil {
		return err
	}
	pl, err := newPlan(m)
	if err != nil {
		return err
	}
	toks, err := tokenize(*prompt)
	if err != nil || len(toks) == 0 || len(toks)+*generate-1 > context {
		return fmt.Errorf("prompt must be 1 to %d alphabet symbols with the generated tokens", context)
	}
	apT, _ := greedy(toks, *generate, func(t []int) []float64 { return approxLogits(pl, t, nil) })
	exT, _ := greedy(toks, *generate, func(t []int) []float64 { return exactLogits(m, t) })
	text := func(t []int) string {
		b := make([]byte, len(t))
		for i, x := range t {
			b[i] = alphabet[x]
		}
		return string(b)
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"model": id, "circuit": circuitDigest(pl, id), "prompt": *prompt,
		"approximateText": text(apT), "exactText": text(exT)})
}

func dispatch(cmd string, args []string) error {
	switch cmd {
	case "evaluator":
		return evaluatorMain(args)
	case "requester":
		return requesterMain(args)
	case "compare":
		return compareMain(args)
	case "bootstrap-probe":
		return bootstrapProbeMain(args)
	case "bridge":
		return bridgeMain(args)
	case "trapset":
		return trapsetMain(args)
	case "reference":
		return referenceMain(args)
	case "trap-oracle":
		return trapOracleMain(args)
	}
	return fmt.Errorf("unknown command %q", cmd)
}
