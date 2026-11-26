# Encrypted Qwen prototype run (historical)

> Historical research record: see [Status](README.md#status) for the code removal and the 2026-11-17 stop; the commands for the removed scripts are kept in [this guide at commit 3e485ca93](https://github.com/clocksmith/doppler/blob/3e485ca9303ee51131884b6242ce4a7fe4753c24/tools/privacy-concrete/QWEN-QUICKSTART.md).

This run took **Qwen2.5-0.5B-Instruct**, all 24 decoder layers and the vocabulary
head, through the explicit **`qwen-all-linear-ntt12`** research profile. The stock
and int8 profiles are different paths and have not passed this full four-step
fixture. The result and its costs are recorded in
[the complete Qwen guide](FULL-QWEN.md#completed-twelve-bit-full-model-functional-observation).

## How the run was made

The `encrypted-qwen-ntt12.yml` workflow (deleted in 1e4bd1ffa;
[source at commit 3e485ca93](https://github.com/clocksmith/doppler/blob/3e485ca9303ee51131884b6242ce4a7fe4753c24/.github/workflows/encrypted-qwen-ntt12.yml))
created an isolated CPU environment (Python 3.12, CPU Torch 2.3.1, the research
dependency pins and the source-locked native NTT build; a stock extension wheel is
not a substitute), checked native byte equivalence and admission, provisioned the
pinned public checkpoint, and ran the complete client/evaluator path on the fixed
public fixture `qwen05b-capital`. It accepted fixed public fixtures only. GitHub
hosted the trusted test client as well as the evaluator; it was not a private-data
deployment.

The run counted as complete only when its `full-qwen.json` artifact had
`completed: true`, `verifiedTilesObserved: 1548`, `report.encryptedSteps: 4` and
`report.matchesFullQuantizedGreedyReference: true`; newer reports also identify
`report.validationBoundary: after-route-close/v1`. A failed comparison returns no
partial model result, and an incomplete artifact is not success.

The local driver (`prepare.py --profile qwen05b-k-projection`, then `full_model.py`
with `--backend-profile qwen-all-linear-ntt12`) selected the same pinned checkpoint
artifacts; `--prepare-only` validated the source, model route and public cache
without keys. Unknown native implementations, models or routes rejected; there was
no downgrade.

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
