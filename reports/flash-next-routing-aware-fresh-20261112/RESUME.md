# Routing-aware fresh test: paused, resume plan

**State: PAUSED on coordinator direction at 2026-11-13T01:41Z, after the routing-controls GPU
run.** Flash-Next stays **BLOCKED**. The strict gate,
`doppler.boundary-tolerance/quantized-v1`, is unchanged. The policy is still **proposed**, and
the evaluator has **not** been run. No result below has been scored against the policy.

## Inputs and policy are unchanged since the freeze

The freeze was recorded at 2026-11-12T16:07:59Z ([freeze.json](freeze.json)) and the inputs at
16:23:01Z. The values below were checked on the run host on 2026-11-12, after the reboot:

- **Policy hash.** `node tools/evaluate-routing-aware-acceptance.js --policy-hash`, run in the
  snapshot, prints `sha256:7a63c40cfbf0753ebe9946d5c74cd458f6fbcd6d726562a47f193e0b2c9bbbd0`.
- **Pinned files.** `sha256sum -c` passes for every file that `freeze.json` pins: the policy
  file `de397c0b…`, the prompt set `4098bb3d…`, the CPU producer `f6bc0f15…`, and all 20
  tooling files.
- **Artifact manifest.** `4902a0b730a9310903b00bc0f8f587528fa1af9f37a722d2f9287bafceceec09`.
- **Frozen inputs.** `frozen/inputs.json` is
  `10a7f79e999c86a22079a316d725f5a24127f91f6da415f4a62329ba78643732`, the same as
  `inputsSha256` in the input receipt.
- **Snapshot.** `/home/x/src/doppler-qual-20261112-b826337d` matches commit `b826337d` blob for
  blob across 13,731 regular files and all symlinks. The only differences are three Git LFS
  adapter files under `artifacts/wgsl-repair/`, which hold their smudged content at the pointer
  sizes. They are not used by this test.

No threshold, prompt, input, policy value or acceptance rule has been changed.

## Run layout

The run directory is `ROOT=/home/x/flash-next-routing-aware-fresh-20261112`. It is the `ROOT`
that [run-fresh.sh](run-fresh.sh) expects.

Every job's start time is in
[run-20261112/run-starts.jsonl](run-20261112/run-starts.jsonl), and every start is after the
freeze. The GPU safety prechecks are in [gpu-precheck.log](run-20261112/gpu-precheck.log). Each
one found no `BO_VA` kernel messages and no amdgpu process in D state.

[run-20261112/output-hashes.txt](run-20261112/output-hashes.txt) lists the SHA-256 of all 66
per-case files: each `fixture/<case>/fixture.json` and each `gpu/{observe,indices}/<case>/control.json`.

## Steps finished

| Step | Output (under `ROOT`) | SHA-256 |
| --- | --- | --- |
| Freeze inputs (earlier, 16:23:01Z) | `frozen/inputs.json`, `frozen/input-receipt.json` | `10a7f79e…`, `0a8b8b6d…` |
| GPU task quality (23:05:54Z): 22 cases; each last sampled row is byte-identical to the production prefill | `gpu/task-quality/observation.json` | `2c319108f1ddb569bd4b4e6f5c7958af9e19dfe5a49a9666a7f7e46e6568bd85` |
| GPU same-input capture, `ledger-05` (23:36:16Z) | `same-input/capture/observation.json` (677 files) | `8af8bbe2223264a8ab1e55038832b3889606f1c6131e7aecca67b3933674155e` |
| CPU round 0: reference, router rows and sampled logits for 22 cases (23:52:06Z, 878 s) | `cpu/all/round-0/round.json`; `learned-forward-reference-float32-artifact.json` | `7ed62c5c…`; `d779f80feaa502b186488ab2e24d76ce54f51838b4c06db8f782b3909a7ed801` |
| CPU greedy round 1 | `cpu/all/round-1/round.json`; `cpu/all/cpu-greedy.json` (`completedSteps: 2`) | `2573e92b…`; `0a1f7ce62c866e8e5e970e427f111fe6fd7915a6440a3ec05794df1b11afc074` |
| Routing fixtures, 22 cases (uses the tie-aware check; see deviations) | `fixture/<case>/fixture.json` | listed in `output-hashes.txt` |
| GPU plain, uninstrumented run (00:32:49Z): 22 rows | `gpu/plain/all/observation.json` | `e9d52ab4ade62fcf118a4c25705e060e6e1dc5a50ac882554672210a69dd6d74` |
| GPU routing controls (01:02:58Z): observe and forced-indices runs, 22 cases each, all `completed`, no failures | `gpu/observe/<case>/`, `gpu/indices/<case>/`, `gpu/routing-controls-1794531779834.json` | summary `9b365896…`; per-case files in `output-hashes.txt` |

## Not done yet

- The forced-indices comparisons under quantized-v1.
- CPU greedy rounds 2 to 7.
- The same-input audit.
- Evidence assembly and the evaluator.
- The supplementary strict-gate comparison of the unforced run.

## Deviations from run-fresh.sh

Each deviation is recorded with a timestamp in
[run-20261112/deviations.json](run-20261112/deviations.json). None changes a limit, prompt,
input or rule.

1. **Host-safety wrappers.**
   - [gpu-track.sh](run-20261112/gpu-track.sh),
     [gpu-track-part2.sh](run-20261112/gpu-track-part2.sh) and
     [gpu-track-part3.sh](run-20261112/gpu-track-part3.sh) run the `run-fresh.sh gpu` commands
     word for word. Before each GPU job they check four things: MemAvailable is at least 30 GiB,
     swap use is at most 1 GiB, there are no `BO_VA` kernel messages, and no amdgpu process is
     in D state.
   - [pressure-guard-run2.sh](run-20261112/pressure-guard-run2.sh) keeps the earlier guard
     thresholds: PSI `some avg10` above 2, or less than 20 GiB available.
2. **Sequencing.**
   - At 23:07:17Z the guard stopped the CPU chain while the GPU task-quality load was running
     alongside it. PSI was 2.03 with 71 GiB available.
   - After that, the CPU and GPU jobs were run one at a time.
3. **Guard stop in round 2.** At 00:27:21Z the guard stopped the CPU chain during greedy round
   2, with the chain running alone. PSI was 2.86, 92 GiB were available and swap use was
   247 MiB. The 336 GB source checkpoint is larger than RAM, so the CPU reference streams it
   through the page cache.
4. **Fixture tooling defect.**
   - The frozen `prepare-routing-fixture.py` (`b1373c6f…`) rejected `ledger-00` L35.
   - Its check compares the CPU expert indices with the indices from one stable argsort. That
     fails on an exact float32 tie between rank 10 and rank 11. The CPU reference had five such
     rows, and in each the CPU chose the other tied expert:

     | Case | Layer | Token |
     | --- | --- | --- |
     | `ledger-00` | L35 | t1234 |
     | `ledger-01` | L23 | t2350 |
     | `ledger-02` | L0 | t2318 |
     | `ledger-02` | L21 | t1023 |
     | `ledger-04` | L13 | t2583 |

   - [prepare-routing-fixture-tie-aware.py](run-20261112/prepare-routing-fixture-tie-aware.py)
     (`62a7592f…`) replaces that check with a value check: no unselected logit may exceed the
     smallest selected logit.
   - The fixture's experts, scores, logits and margins are byte-for-byte unchanged. Only
     `converterSha256` differs.
   - All 22 fixtures were regenerated with it. The frozen-tool output is kept at
     `fixture-frozen-tool-partial/`.
   - The frozen file in the repository is left as it is so that `freeze.json` still verifies.
     Whether to fix it upstream is an owner decision.

## Next steps, in order

Set this environment first:

```sh
export SNAPSHOT=/home/x/src/doppler-qual-20261112-b826337d ROOT=/home/x/flash-next-routing-aware-fresh-20261112 \
  MODEL_DIR=/run/media/x/models/rdrr/qwen-3-8-flash-next-text-q4k-ehf16-af32 \
  SOURCE_DIR=/run/media/x/models/qwen38-flash-next-de4b8e4d/source \
  GPU_LOCK=/run/media/x/models/doe-gpu-timing.lock
F=$SNAPSHOT/reports/flash-next-routing-aware-fresh-20261112
```

No remaining step uses the GPU. `run-fresh.sh gpu` has no entry point that runs only its last
step, and running it again would repeat every GPU job above. So step 1 is that script's final
loop, copied unchanged. It is CPU-only and takes no lock, as in `run-fresh.sh`.

### 1. Forced-indices comparisons under quantized-v1 (next)

```sh
cd "$ROOT" && for id in $(python3 -c "import json;print(' '.join(c['id'] for c in json.load(open('$SNAPSHOT/tools/data/flash-next-routing-acceptance-prompts-v1.json'))['cases']))"); do
  python3 -P "$SNAPSHOT/reports/flash-next-forced-routing-20261112/compare-forced-routing.py" \
    "$ROOT/gpu/forced-comparison/$id.json" "$ROOT/cpu/all/round-0/learned-forward-reference-float32-artifact.json" \
    "$SNAPSHOT/src/config/evidence/boundary-evidence-policy.json" "$ROOT/frozen/inputs.json" "$id" \
    "indices=$ROOT/gpu/indices/$id/observation.json" || echo "comparison $id failed"
done
```

### 2. Same-input audit

This step reads the source checkpoint. Start `$ROOT/pressure-guard-run2.sh` beside it; it
watches only `cpu-reference.py`, so watch PSI by hand.

```sh
cd "$ROOT" && sh "$F/run-fresh.sh" audit
```

### 3. CPU greedy rounds 2 to 7

`run-fresh.sh cpu` needs a fresh `cpu/all`, and the fixtures and comparisons already reference
`cpu/all/round-0`. So resume from the two completed steps instead.
[resume-cpu-chain.py](run-20261112/resume-cpu-chain.py) runs the frozen script's `round` mode
with the same command line, inputs and stop rule as `run_chain`.

```sh
cd "$ROOT" && mv cpu/all/round-2 cpu/all/round-2-stopped-20261113T002721Z \
  && mv cpu/all/round-2.log cpu/all/round-2-stopped.log && rm -f cpu/all/inputs-round-2.json
setsid nohup "$ROOT/pressure-guard-run2.sh" >/dev/null 2>&1 &
OMP_NUM_THREADS=8 nice -n 10 python3 -P "$ROOT/tooling-patch/resume-cpu-chain.py" "$F/cpu-reference.py" \
  "$SNAPSHOT/artifacts/qwen38-flash-next-20261013/prepare-learned-forward-reference.py" "$SOURCE_DIR" "$MODEL_DIR" \
  "$SNAPSHOT/tools/data/flash-next-routing-acceptance-prompts-v1.json" "$ROOT/frozen/inputs.json" "$ROOT/cpu/all" \
  > "$ROOT/cpu-track-resume.log" 2>&1
```

Each round takes about 14.5 minutes. The guard has stopped this chain twice, so another stop is
possible. If it stops, move the partial round aside and run the command again. The rounds are
independent CPU-only re-runs.

### 4. Evidence and evaluator

```sh
cd "$ROOT" && sh "$F/run-fresh.sh" evaluate
```

The verdict will include `POLICY_NOT_APPROVED` by design.

### 5. Supplementary strict gate on the unforced run

This step does not feed the evaluator. It runs the same comparison tool with the `observe` run.

```sh
mkdir -p "$ROOT/gpu/unforced-strict" && for id in ...; do python3 -P "$SNAPSHOT/reports/flash-next-forced-routing-20261112/compare-forced-routing.py" \
  "$ROOT/gpu/unforced-strict/$id.json" "$ROOT/cpu/all/round-0/learned-forward-reference-float32-artifact.json" \
  "$SNAPSHOT/src/config/evidence/boundary-evidence-policy.json" "$ROOT/frozen/inputs.json" "$id" \
  "observe=$ROOT/gpu/observe/$id/observation.json"; done
```

Component: `doppler.tests`, `doppler.repository-tooling`. Intent: preserved. Boundary effects: none.
