package main

// audit: plaintext-only check of the approximated circuit against the float64 model over every three-letter prompt.

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
)

func auditMain(args []string) error {
	fs := flag.NewFlagSet("audit", flag.ExitOnError)
	modelPath := fs.String("model", "tiny-model.json", "tiny model fixture")
	steps := fs.Int("steps", 4, "generated tokens per prompt")
	fs.Parse(args)
	m, id, err := loadModel(*modelPath)
	if err != nil {
		return err
	}
	pl, err := newPlan(m)
	if err != nil {
		return err
	}
	type group struct {
		Prompts, TokenMismatch, PromptsWithOutOfRange, NonFinite int
		MaxLogitError, MinExactMargin                            float64
	}
	run := func(prompts [][]int) group {
		g := group{MinExactMargin: math.Inf(1)}
		for _, p := range prompts {
			g.Prompts++
			tr := &trace{outOfRange: map[string]int{}}
			exT, exL := greedy(p, *steps, func(t []int) []float64 { return exactLogits(m, t) })
			apT, apL := greedy(p, *steps, func(t []int) []float64 { return approxLogits(pl, t, tr) })
			for i := range exT {
				if exT[i] != apT[i] {
					g.TokenMismatch++
					break
				}
			}
			if len(tr.outOfRange) > 0 {
				g.PromptsWithOutOfRange++
			}
			for i := range exL {
				g.MinExactMargin = math.Min(g.MinExactMargin, margin(exL[i]))
				for j := range exL[i] {
					d := math.Abs(exL[i][j] - apL[i][j])
					if math.IsNaN(d) || math.IsInf(d, 0) {
						g.NonFinite++
						continue
					}
					g.MaxLogitError = math.Max(g.MaxLogitError, d)
				}
			}
		}
		return g
	}
	var task, letters [][]int
	for a := 0; a < 26; a++ {
		for stride := 1; stride <= 3; stride++ {
			task = append(task, []int{a, (a + stride) % 26, (a + 2*stride) % 26})
		}
		for b := 0; b < 26; b++ {
			for c := 0; c < 26; c++ {
				letters = append(letters, []int{a, b, c})
			}
		}
	}
	out := map[string]any{"model": id, "steps": *steps, "taskPrompts": run(task), "allLetterPrompts": run(letters),
		"approximations": []map[string]any{
			{"name": pl.expP.Name, "interval": []float64{pl.expP.A, pl.expP.B}, "degree": pl.expP.Degree},
			{"name": pl.invP.Name, "interval": []float64{pl.invP.A, pl.invP.B}, "degree": pl.invP.Degree},
			{"name": pl.isq2P.Name, "interval": []float64{pl.isq2P.A, pl.isq2P.B}, "degree": pl.isq2P.Degree},
			{"name": pl.isqfP.Name, "interval": []float64{pl.isqfP.A, pl.isqfP.B}, "degree": pl.isqfP.Degree},
			{"name": pl.geluP.Name, "interval": []float64{pl.geluP.A, pl.geluP.B}, "degree": pl.geluP.Degree},
		}}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", " ")
	return enc.Encode(out)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: privacy-protected audit|evaluator|requester|compare [flags]")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "audit":
		err = auditMain(os.Args[2:])
	default:
		err = dispatch(os.Args[1], os.Args[2:])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
