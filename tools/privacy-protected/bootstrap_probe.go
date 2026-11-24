package main

// bootstrap-probe: one CKKS bootstrap at N = 2^16 with Lattigo's example bootstrapping parameters, to price the
// bootstrapped design. Generates a throwaway secret key in-process; not part of any protocol.

import (
	"encoding/json"
	"flag"
	"math"
	"os"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
)

func bootstrapProbeMain(args []string) error {
	fs := flag.NewFlagSet("bootstrap-probe", flag.ExitOnError)
	ls := fs.Int("log-slots", 11, "log2 of the slots bootstrapped")
	fs.Parse(args)
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: 16, LogQ: []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40}, LogP: []int{61, 61, 61},
		LogDefaultScale: 40, Xs: ring.Ternary{H: 192},
	})
	if err != nil {
		return err
	}
	btp, err := bootstrapping.NewParametersFromLiteral(params, bootstrapping.ParametersLiteral{
		LogN: utils.Pointy(16), LogP: []int{61, 61, 61, 61}, Xs: params.Xs(), LogSlots: utils.Pointy(*ls)})
	if err != nil {
		return err
	}
	kgen := rlwe.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	t0 := time.Now()
	evk, _, err := btp.GenEvaluationKeys(sk)
	if err != nil {
		return err
	}
	keygen := time.Since(t0).Seconds()
	ev, err := bootstrapping.NewEvaluator(btp, evk)
	if err != nil {
		return err
	}
	ecd, enc, dec := ckks.NewEncoder(params), rlwe.NewEncryptor(params, pk), rlwe.NewDecryptor(params, sk)
	v := make([]float64, 1<<*ls)
	for i := range v {
		v[i] = 0.9 * math.Sin(float64(i))
	}
	pt := ckks.NewPlaintext(params, 0)
	pt.LogDimensions = ring.Dimensions{Rows: 0, Cols: *ls}
	if err := ecd.Encode(v, pt); err != nil {
		return err
	}
	ct, err := enc.EncryptNew(pt)
	if err != nil {
		return err
	}
	t1 := time.Now()
	out, err := ev.Bootstrap(ct)
	if err != nil {
		return err
	}
	secs := time.Since(t1).Seconds()
	got := make([]float64, len(v))
	if err := ecd.Decode(dec.DecryptNew(out), got); err != nil {
		return err
	}
	var e float64
	for i := range v {
		e = math.Max(e, math.Abs(got[i]-v[i]))
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"logSlots": *ls, "residualLevels": params.MaxLevel(),
		"bootstrapLogQP": btp.BootstrappingParameters.LogQP(), "keygenSeconds": keygen, "keyBytes": evk.BinarySize(),
		"galoisKeys": len(evk.MemEvaluationKeySet.GaloisKeys), "bootstrapSeconds": secs, "outputLevel": out.Level(),
		"maxError": e, "peakRSSBytes": peakRSSBytes()})
}
