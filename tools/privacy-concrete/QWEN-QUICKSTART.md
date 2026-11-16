# Run the demonstrated encrypted Qwen prototype

This path runs **Qwen2.5-0.5B-Instruct**, all 24 decoder layers and the vocabulary
head, through **169 encrypted linear modules and four autoregressive steps**.
Use the explicit **`qwen-all-linear-ntt12`** research profile. The stock and int8
profiles are different paths and have not passed this full four-step fixture.

## End to end without preparing another machine

Open [the existing twelve-bit workflow](https://github.com/clocksmith/doppler/actions/workflows/encrypted-qwen-ntt12.yml),
choose **Run workflow**, `main`, and **`qwen05b-capital`**. With an authenticated
GitHub CLI, the equivalent is:

```sh
gh workflow run encrypted-qwen-ntt12.yml --repo clocksmith/doppler --ref main -f public_fixture=qwen05b-capital
```

The workflow creates its isolated CPU environment, builds the exact pinned native
implementation and dependency lock, checks native byte equivalence/admission,
provisions the pinned public checkpoint, and runs the complete client/evaluator
path. This workflow accepts fixed public fixtures only. GitHub hosts the trusted
test client as well as the evaluator; it is not a private-data deployment.

Download `full-qwen-ntt12-public-evidence` from the completed run. In
`full-qwen.json`, require `completed: true`, `verifiedTilesObserved: 1548`,
`encryptedSteps: 4` and `matchesFullQuantizedGreedyReference: true`. A new report
also identifies `validationBoundary: after-route-close/v1`. A failed comparison
returns no partial model result. Do not treat an incomplete artifact as success.

The [completed reference run](https://github.com/clocksmith/doppler/actions/runs/43841061585)
produced **“ Paris. It is”**, with all four greedy choices matching quantized and
FP32 references. It took **98.8 minutes**, including **49.1 minutes of client
encrypted replay**, and reached **5.27 GB client lifetime peak RSS**. Those are
one public fixture's observations, not promised throughput or general accuracy.
That run predates the new closed-route validation ordering; an actual four-step
QSA integration passed the ordering, and the broader arithmetic full-Qwen run
[43865231126](https://github.com/clocksmith/doppler/actions/runs/43865231126) is being
collected separately. Do not start another expensive run just to inspect these
existing results.

## Use an already prepared client environment

The exact environment/build recipe is the
[workflow source](../../.github/workflows/encrypted-qwen-ntt12.yml).
It uses Python 3.12, CPU Torch 2.3.1, the installed research dependency pins and the
source-locked native NTT build. A stock extension wheel is not a substitute.
Before model preparation, check the installed native identity:

```sh
PYTHONPATH=tools/privacy-concrete python -c 'from native_runtime import ntt_versions; print(ntt_versions())'
python tools/privacy-concrete/prepare.py --profile qwen05b-k-projection --destination /owned/scratch/qwen-source
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 \
  python tools/privacy-concrete/full_model.py \
  --model-dir /owned/scratch/qwen-source --cache-dir /owned/scratch/qwen-ntt12-cache \
  --backend-profile qwen-all-linear-ntt12 --public-fixture qwen05b-capital
```

`prepare.py`'s profile name selects the same pinned checkpoint artifacts; the
explicit `--backend-profile` selects the complete all-linear execution.
`--prepare-only` validates the source, model route and public cache without keys.
Unknown native implementations, models or routes reject; there is no downgrade.

## What this delivers and what remains

This is a working hybrid encrypted-computation prototype. The client holds keys,
attention/nonlinear operations, embeddings, caches, decoding and **full encrypted
replay**. It gives no client compute saving. Only verified matrix products are
outsourced; numerical acceptance occurs after complete route closure.

Production cryptographic admission remains **closed**. A separately operated,
authenticated model deployment, remaining timing/runtime-failure leakage, parameter
and implementation review, and independent malicious-security qualification are
unfinished. Full **Qwen3.8 Flash** generation is a separate unfinished integration;
its operator and geometry evidence does not upgrade this prototype's model scope.
See [the security boundary](SECURITY-BOUNDARY.md) and [complete evidence/limitations](FULL-QWEN.md).
