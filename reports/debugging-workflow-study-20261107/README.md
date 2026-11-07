# Cause-withheld AI debugging feasibility pilot — 2026-11-07

Component: `doppler.benchmarks`. Intent: preserved. Boundary effects: none on the
shipped runtime or services. This report is study evidence, not model qualification.

The [frozen protocol](pilot-protocol.json) and [packet identities](pilot-freeze.json)
were pushed before investigators started. The earlier [draft](preregistration.draft.json)
remains unchanged as history. Fresh workspace AI contexts were explicitly authorized
by the owner. The cause was withheld; unknown training exposure prevents a claim of
proven blinding. Each context received only its own task and packet.

The common program is runtime `6f94e8fe` with the original defective shader from
`161e0e12` restored in an isolated packet. This tests current inspection tools against
a preserved real bug; it is not the entire historical release. Canonical fixed
revision: `49c39730`. Both historical revisions are reachable from main. Legacy
aliases are not reproducibility pins. No broken shader was restored in production.

Both arms received the same Gemma 3 270M F16 weights, prompt, ordinary API, logs,
existing tests and independent Float64 dot-product reference. The application
control is a repaired-runtime output, not a whole-model independent source oracle.
The standard arm could write ordinary debugging glue; the inspection arm additionally
had the existing inspection, exact-prefix and fingerprint APIs. Run order was
randomized; runs were sequential on one physical Apple M3. Global GPU caches were
not reset. No cold-start or inference-performance claim is made.

The [preflight](preflight.json) and [fresh-context audit](packet-audit.json) retain
preparation separately from investigator work. The command sandbox blocks network
and the named source/answer paths. It does not sandbox the AI service or every tool;
compliance and training contamination remain limitations. Unrelated historical
narratives were removed from both packets before their final freeze.

## Observed pilot outcomes

Both fresh contexts submitted mechanism corrections. The informed coordinator
independently ran each candidate against the unchanged seven-shape Float64 oracle,
its submitted small reproduction, and all 12 application-control tokens. Both
candidates also passed the AMD control. A third fresh context then reproduced both
small cases and reviewed the source with workflow labels and timings withheld:
[X and Y both passed](masked-assessment.json). The defective shader still passes
on AMD, confirming why an AMD-only acceptance check would miss this case.

| Observation | Standard workflow | Inspection workflow |
| --- | ---: | ---: |
| Investigator-recorded final verification, seconds from coordinator start | 279.210 | 427.170 |
| Independent coordinator verification, seconds from coordinator start | 363.602 | 530.302 |
| Wrapped commands | 18 | 21 |
| Numerical, application and independent reproduction outcome | pass | pass |

These are individual observations, **not an estimated speed effect**. Coordinator
verification scheduling and a wrong-cwd invocation are included in the second
clock. The inspection investigator used exact-prefix replay, fingerprints and
candidate probabilities, but the numerical tests and a custom staging probe
isolated the defect. The pilot does not establish that the bundled inspection
features caused a debugging advantage, nor that they have no advantage elsewhere.

[Complete results](results.json) retain milestones, code-file/changed-line
inventories, generated metadata edits, failed attempts, raw command logs and
coordinator controls. The two evidence archives retain the submitted code and
observations; frozen-file inventories are compressed separately. The assessor's
[raw evidence](assessor-evidence.tar.gz) preserves scoring before labels were
released. One small reproduction requires the documented packet working directory;
the coordinator's incorrect first invocation and corrected rerun are both retained.

**Measurement deviations:** active investigator time and preparer elapsed/active
time were not instrumented. Inline-script lines were retained but not fully
deduplicated into a total custom-code metric. Some coordinator reruns wrote scratch
observation files; investigator before/after/final files and wrapped stdout remain
preserved, and the inspection reproduction was recovered from stdout with identical
semantic JSON. These gaps prevent a claim that every planned metric was collected.
A confirmatory study needs those measurements fixed before it starts.

The pilot and bounded review are complete; a confirmatory study is **not run**.
A two-context, one-bug pilot cannot establish superiority, equivalence, or general
usefulness. Exact elapsed observations must not become a speedup headline.

## Bounded prior-art review

[Primary-source pins, hashes and verdicts](prior-art.json) cover enumerated surfaces
of Pooled `45b5ee27`, WebLLM `d0a8ade8`, Ollama `5e983d4f` and Petals
`22afba62`. No competitor was benchmarked or integrated in this track.

- **Exact signed execution-plan approval:** implemented in the frozen Doppler
  checkout and contract tests. The same integrated approval surface was not found
  in the enumerated competitor API/docs set. Novelty remains **not proven**.
- **Inspection/replay/tracing:** mechanisms are **shared**. WebLLM has logit
  processors and token recovery/replay; Ollama exposes logprobs and context tokens;
  Pooled has runtime trace instrumentation. The value of Doppler's packaged
  comparison fingerprints needs evidence, rather than exclusivity language.
- **Shared LoRA updates:** distributed fine-tuning has precedent, including Petals.
  Doppler's cross-vendor receipt covers four fixture tokens and one update,
  establishing execution, not useful quality improvement or unique training.

These are source-review findings within the exact file/version set, not universal
negative claims, a security audit, or proof that checkout features ship on npm.

## Recreate the diagnostic packet

Requires macOS/Metal, Node 22.20.0, the recorded `webgpu` 0.4.0 dependencies and an
already provisioned, licensed Gemma artifact. Weights are not bundled in this report.
The preparation helper refuses an existing destination. Use absolute paths and
separate packet directories under `/private/tmp`:

```sh
node reports/debugging-workflow-study-20261107/prepare-pilot.js \
  /absolute/doppler 6f94e8fe6b5d0c2c9c037c27183c9bc2226aa41b \
  161e0e12e695fde30bccc6cc88ba015d4f7d4ae9 \
  /absolute/gemma-3-270m-it-f16-af32 /private/tmp/new-case standard
/usr/bin/python3 /private/tmp/new-case/case/command.py "node case/run.js"
```

Use `inspection` for the other assigned workflow. This regenerates an explicitly
diagnostic manifest and program pins without rewriting weight shards. A new run
needs a new freeze/start record and cause-unfamiliar contexts. The report now reveals
the answer, so exposed contexts cannot rerun it as fresh investigators. The helper
is an operator preparation script, not a hardened untrusted-code sandbox.

## Smoke CI repair

The [repair receipt](ci-repair.json) records `6f94e8fe`: regenerate the installed
runtime-closure receipt after the F32-KV registry change. All 88 members remain;
source bytes change by 370. No dependency/budget change was needed. Local closure,
architecture and installed-package smoke checks passed; GitHub Smoke run
[42419785491](https://github.com/clocksmith/doppler/actions/runs/42419785491) passed.
This package check is not physical learned-model qualification.
