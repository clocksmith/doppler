# Installed public-consumer acceptance

Runtime candidate: `17af741f57e1d73bd5dd864d9a934b5592184d82`.
Archive: `doppler-gpu-0.6.2.tgz`, SHA-256
`88f030627cdba99f9a81ffc5b5287d1ba9441f3cac638c506eded46d49cbeec7`.
It contains 1,858 files and is the exact archive from the
[adapter ownership audit](../adapter-session-ownership-2026-09-19/package-audit.json).
This work changes repository acceptance tooling, tests, and demo integration,
not runtime bytes, retained Capsules, shaders, source references, or their
numerical tolerances. A separate adapter-capable evaluation candidate has been
prepared under explicit user authority; it does not supersede a retained release.
During this run, remote `main` advanced to `4bd012e3` with demo changes and a
pipeline reset forwarding method. Those changes were fast-forwarded and
preserved. Physical evidence here still qualifies only the exact installed
`17af741f` archive, not the subsequently changed checkout.
`check:green` passed before that fast-forward. Against `4bd012e3`, the rerun
reported unrelated demo declaration/type/export mismatches, two demo contract
test failures, and an unpacked-package budget overrun (11,307,986 bytes versus
11,307,872). These are not hidden by the earlier green result.

## Main integration repair

The user explicitly authorized repairing these failures while preserving the
new UI. The simplified header, Markdown answers, enabled observation controls,
256-token selector, and model-state reset remain in place. Repairs add the
missing Markdown declaration and strict implementation checking, align public
demo declarations and tests with those controls, and refresh the generated PWA
shell and dependency inventory. The browser test found and fixed two contrast
regressions: the new light-button gradient overrode active Tokens and expanded
Precision replay backgrounds while retaining white text. The contrast threshold
was not relaxed. The initial failures are retained in
`demo-token-contrast-failure.log` and `demo-replay-contrast-failure.log`.

`demo-contract-passed.log` records real browser UI checks with **mocked inference**:
stream batching, final Markdown rendering, inert generated HTML and unsafe URLs,
source-bound word-quality spans, reset failure preserving conversation, successful
reset keeping the model loaded, cancellation, profiles, keyboard access, receipts,
contrast, and responsive layouts. Streaming retains its text node while active;
settlement intentionally replaces that text with Markdown DOM without replacing
the output container or conversation nodes.

`main-package-audit.json` compares every tar entry between the exact physical
candidate and the newer main archive. Only `text.js` and `text.d.ts` changed:
82 + 32 = 114 unpacked bytes for the reset forwarding method/declaration. The
file count stays 1,858. Repacking with Node 22.23.2/npm 10.9.8 measures exactly
19 additional compressed bytes (2,151,247 total); npm 9.2.0 produces 2,147,201.
Both size limits match the measured payload without arbitrary headroom.
`main-installed-package.json`
checks the newer archive's installed exports, types, and injected-consumer
contracts; it is not physical qualification of that archive.

Final verification: `check-green-final.log` records `npm run check:green` exiting
successfully, including 847 unit-suite files. The independent browser UI contract,
strict source types, dependency inventory, demo shell/reachability, focused
acceptance regressions, and installed-package checks passed. The final package
boundary check also passed with Node 22.23.2/npm 10.9.8; its exact output is
`ci-toolchain-package-check.log`. No CI result for the eventual pushed commit is
implied by these local checks.

## Scope

The browser application imports `doppler-gpu/host` and `doppler-gpu` through an
import map derived from the installed package's public exports. The lifecycle
fixture receives that public host API and never imports package internals or
constructs a pipeline. It does not inject a program, tensor, or device double in
physical runs. Physical admission requires AMD and `isFallbackAdapter: false`.

Retained models:

- Generation: `qwen-3-4b-instruct-2507-f16-af32-closed-development`, compared with
  the unchanged 107-token reference in the prior main integration receipt.
- Embedding: `qwen-3-embedding-0-6b-f16-af32`, compared with all 18 complete
  1,024-element source-reference vectors and their exact token IDs.
- Reranking: `qwen-3-reranker-0-6b-q4k-ehf16-af32`, compared with all three
  documents' tokens, logits, scores, probabilities, and ranking.

The initial physical pass uses the older requests: generation has exact token
comparison, embedding checks dimensions/finiteness, and reranking checks
completion. The lifecycle pass is stronger: it uses the pinned embedding and
reranking source references, not newly generated expected outputs. Each operation
runs twice, again after cancellation and cancelled model preparation, and again
on a second session after closing the first. Generation and embedding cancel
after one partial; reranking's declared cancellation case is pre-aborted and
does not establish cancellation during GPU execution. Paused streams are
returned/drained before awaiting close. Cancellation does not claim interruption
of submitted GPU commands.

The fixture's isolated tests deliberately supply doubles to ensure acceptance
rejects ignored cancellation and preserves failures during cleanup. Those tests,
the package installation checks, and Reploid's injected-logit installed-contract
test are separate from physical qualification.

## Reproduction

Use a new bundle/output directory; tools refuse to overwrite earlier evidence.
Mount the exact model roots and references recorded by the configuration.

```sh
node tools/check-packed-package.js --archive /absolute/doppler-gpu-0.6.2.tgz --retain /absolute/new-installed-bundle
node tools/check-installed-capabilities.js /absolute/physical-lifecycle-config.json
node tools/run-node-tests.js tests/integration/installed-capability-lifecycle.test.js
npm run check:green
```

The configuration binds the archive's installed bundle, model paths, unchanged
reference digests, request settings, trust/retained-local-use policy, hardware
requirement, repetition count, and cancellation boundary. Retained local use
does not renew or rewrite historical signatures or release events. Local raw
logs and intermediate observations are under
`/var/tmp/doppler-release-acceptance.JTuaHr`.

## Remaining release gate

Physical adapter activation and failed adapter preparation are not qualified by
the original model-only cases. The retained generation TargetPlan has no
`adapterExecution` declaration. The user approved preparation and qualification
of a separate candidate. Its controlled rank-one zero-delta PEFT fixture is not
a trained adapter and cannot establish nonzero adapter parity. No signed
metadata is patched to bypass this restriction. The earlier
ownership unit regressions remain valid. The separate candidate subsequently
passed its public physical fixture gate, described below; this does not add
adapter authority to the retained parent.

No publication, deployment, speedup, fleet support, or independent adoption is
claimed. The expanded generation run also failed while opening its second
session: `RangeError: Array buffer allocation failed` in SHA-256 padding of
verified artifact bytes. Its three completed operations matched the frozen
107-token reference. Embedding and reranking passed all four lifecycle runs and
their frozen numerical comparisons. The separate 4 GiB retention run is a
diagnostic control, not a fix or replacement for the failed original settings.
That control timed out at its declared 1,200,000 ms while loading the second
session. The first session completed all three generation phases with the same
107 tokens. No allocation error was reported, but isolation did not complete;
the control is inconclusive, not passed. Its first load hashed 32,196,588,272
bytes with 446 cache evictions, compared with an 8,050,837,118-byte source closure.
Release acceptance remains incomplete because the original unbounded-retention
failure has not been repaired; the next ownership refactor has not started.

Component: `doppler.repository-tooling`, `doppler.tests`, `doppler.demo`.

Intent: preserved.

Acceptance evidence: installed public-consumer checks and retained physical
observations, with their exact scopes distinguished above.

Boundary effects: no new runtime authority or dependency direction; demo
declarations now describe its existing implementation. Remote main's reset
forwarding API is preserved and separately inventoried.

## Separate adapter preparation

`adapter-source-config.json` pins the retained parent file and additional
`matmul_f32.wgsl`/`scale.wgsl` mechanisms. The new manifest has a separate variant
identity. Immutable model artifacts are hard-linked; the parent manifest and
Capsule are never edited. `adapter-preparation.json` is construction evidence,
not physical qualification. The fixture has PEFT A shape `[1, 2560]`, nonzero
alternating A values, B shape `[4096, 1]` with all zeros, and alpha 2.

```sh
node tools/prepare-adapter-evaluation-source.js artifacts/release-acceptance-2026-09-19/adapter-source-config.json
node tools/probe-installed-token-selection.js artifacts/release-acceptance-2026-09-19/adapter-preseal-config.json
node tools/build-token-selection-evaluation-capsule.js artifacts/release-acceptance-2026-09-19/adapter-seal-config.json
node tools/check-installed-capabilities.js artifacts/release-acceptance-2026-09-19/adapter-public-config.json
```

The pre-signing probe uses installed advanced execution and internal artifact
assembly. It is explicitly not an ordinary signed-Capsule consumer test. The
subsequent lifecycle run must use the installed public host exports, verify
adapter tensor identities in completion receipts, and compare base output after
unloading, failed adapter preparation, and cancellation. Producer signing keys
stay outside the repository and are not published.

Preserved preparation attempts:

- Attempt 1 loaded the model but the harness assumed text input instead of the
  public request's prepared tokens. The harness now preserves those tokens.
- Attempt 2 found no physical adapter. Subsequent runs use three explicitly
  recorded acquisition attempts, never a CPU fallback.
- Attempt 3 passed all physical token comparisons, but Forge rejected its
  source-hash/entry-digest confusion before signing. That receipt is not sealed
  qualification. The corrected v2 preparation uses Forge's entry-digest check
  before GPU loading, with a regression that rejects raw file hashes as pins.
- Attempt 4 passed base, active-fixture, and restored-source token comparisons
  with the corrected pins. Forge then rejected the producer's noncanonical
  timestamp spelling before signing; the producer now validates it on entry.
- The separately sealed candidate is
  `qwen-3-4b-instruct-2507-f16-af32-closed-development-capsule-v2-d52064672d0f564e`.
  Its TargetPlan digest is
  `sha256:4e22547a271133150ad1b30f3fc4832dda8955489478ecf57a5241dc42211c74`.
  `adapter-build-receipt.json` is a build receipt, not public execution proof.
  `adapter-public-config.json` declares the subsequent public lifecycle gate,
  including 4 GiB retention and an overall 1,800,000 ms timeout. Those settings
  are explicit, not changes to runtime defaults or a repair of the original
  unbounded-retention failure.

## Adapter public physical result

`adapter-public-lifecycle.json` and `adapter-public-summary.json` record a pass
on physical AMD RDNA 3, Chrome 146.0.7680.177, through the exact installed
`17af741f` archive's public exports. All seven completed operations reproduce
the frozen 107 tokens: two adapter requests, base after unload, base after failed
adapter preparation, adapter after cancellation, base after cancellation, and
adapter on a second session after closing the first. Adapter requests identify
the loaded tensor identity; base requests contain no adapter receipts. Cleanup
completed without errors.

The failure case rejects an adapter artifact read; it does not establish recovery
from a partially submitted GPU upload. Model preparation is cancelled at target
selection. Operation cancellation follows one partial and rejects completion;
it does not claim submitted GPU commands were interrupted. This pass qualifies
the controlled fixture and explicit 4 GiB retention policy only. The earlier
1,200,000 ms bounded control remains a timeout, and the unbounded allocation
failure remains a failed gate. Historical receipts have not been rewritten.
