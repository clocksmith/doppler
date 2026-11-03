# Shared-model performance scoreboard

The [scoreboard](scoreboard.json) retains sixteen declared model/host/surface/prompt cells. Five have native request measurements; none yet completes numerical, task-quality, startup, resource and matched-baseline acceptance. Unmeasured Node cases remain unmeasured. Kernel checks on Node do not fill model cells.

| Model / host / surface | 28 input, 128 output cap | 2,287 input, 128 output cap |
| --- | --- | --- |
| 35B / Jupiter / Chrome | Retained Doppler 1.679 s; Pooled 1.719 s | Retained Doppler 4.979 s; Pooled 5.261 s |
| 27B / Jupiter / Chrome | Older paired loss: 33.862 vs 7.075 s; newer Doppler-only screen 11.154–11.250 s | Older paired loss: 108.046 vs 28.678 s; new combined candidate unmeasured |
| 27B / Mars / Chrome | Retained Doppler 47.293/65.085 s; Pooled 18.375/19.148 s; large variance | Unmeasured |
| 35B / Mars / Chrome | Unmeasured | Unmeasured |
| Both models / both hosts / Node | Declared campaign cell unmeasured | Declared campaign cell unmeasured |

These are fixed-budget request controls, not accepted completed jobs. The 35B native formats differ in KV precision and materialization; its long outputs differ. The older 27B loss is retained separately from newer single-engine tuning. No percentages are added across experiments. Exact attribution and individual measurements remain in the linked receipts and [evidence index](evidence-index.json).

The [coverage gate](coverage-gate.json) incorporates the owner's November 2 acceptance criteria: justified frozen numerical policy per precision; preserved target sampling semantics; explicit cold/warm/memory tradeoffs; separate qualification of combined optimizations; and unsupported exclusions only with concrete evidence. F16 tolerances and the held-out task suite remain pending. Known losses stay visible. Coverage completion and a performance lead are separate decisions.

Current implementation work remains in progress: per-token overhead, F16 replay binding, MoE MTP, prefill, and coverage qualification. Buffer reuse passes an [exact-state gate](buffer-pool-state-gate.json), including 2,483,200 final-vocabulary logits and cancellation recovery, but [six paired measurements](rejected-buffer-pool.json) show 1.78% short and 0.25% long regressions. Its [patch](rejected-buffer-pool.patch) is retained and its runtime changes are removed. The [F16 layout repair](f16-replay-repair.json) uses the existing pipeline layout cache; a failing focused regression now passes, the 35B model opens, and all four toy controls pass. The F16 flash path remains unqualified and its roughly 7.04-second long prefill is retained as a loss. F32 remains the performance baseline. Green implementation checks do not complete any model/platform cell.

Component: `doppler.runtime-source.inference.pipelines.text`. Intent: preserved. Boundary effects: F16 KV write binding identity and evidence only. Acceptance evidence: linked receipts, focused tests, and pending physical gates. This report is an engineering checkpoint, not a model or performance release.

[Development checks](development-checks.json) pass for the changed binding boundary. The source architecture check retains its two existing unrelated file-size failures; their budgets are unchanged. [Hardware](jupiter-hardware.json) identifies the tested AMD Ryzen AI MAX+ 395 and browser/provider surface. [Source MTP tensors](mtp-source-inventory.json) exist, including the auxiliary MoE layer; inventory does not implement or qualify MTP.

The [new opt-in F16 cache query tile](prefill-querytile.json) keeps the existing F32 attention arithmetic and is generated from the same owner. Ten mask/cache/geometry cases pass bitwise against exactly expanded F32 cache inputs on AMD Node, M3 Node/Metal and M3 Chrome. The independent bounded-fixture reference errors remain below nine billionths. This is operator evidence. Its separate 35B Chrome screen passes four toy controls and cuts the prior F16 flash long requests from roughly 8.89 to5.14 seconds. The retained F32 long path remains faster. Full-model F16 tolerances and held-out quality remain pending.

The [checkpoint experiment](checkpoint-experiment.json) removes only the initial host wait in an immutable diagnostic source. [Exact state and logits](checkpoint-state-gate.json) match through cancellation and continuation. Six rounds retain a 0.81% short-request regression. Long timing depends strongly on engine order with three resident engines: unchanged Doppler averages 5.919 seconds, candidate 5.668, and Pooled 5.193. Preserve this known loss; it does not replace the prior two-engine result. Its cause is unestablished. Main retains the wait, and no unsafe early-close ownership change is promoted.

[Actual MTP inputs](mtp-source-preparation.json) now preserve all 20 auxiliary source tensors (897,955,840 bytes), with SHA256 for the complete source and each tensor. This is preparation only: no native MoE MTP draft/verify implementation, GPU acceptance path or performance improvement is established by those files. GPU verification, recurrent/KV recovery and explicit sampling semantics remain required.

The [November 3 continuation](diagnostic-20261103/README.md) retains a new current-source 27B/Pooled short screen on Jupiter and the incomplete Mars timeout investigation. It does not qualify either model or establish a broad performance lead.

The [native MoE MTP integration diagnostic](native-mtp-20261103/README.md) now executes real GPU draft chains, batched target verification, retained-projection recurrent recovery and consecutive-window carry in a frozen experiment. Rejection/continuation checks pass within the frozen numerical policy. The manual path remains slow and is not connected to public generation; no coverage cell or performance claim is promoted.
