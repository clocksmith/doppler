# Reproduce the Capsule retention experiment

This experiment asks whether an explicit limit on verified artifact retention
reduces process memory while preserving the same model programs, frozen source
references, and opening/inference budgets. It measures the opening-time cost as
well as the memory benefit. Unlimited retention remains the runtime default.

Start with the public reconstruction and installed consumer described in
[Capsule baseline reproduction](capsule-release-reproduction.md). Keep the
restoration directory, its successful `restoration.json`, and the installed
runtime bundle containing `receipt.json`, the exact archive, and
`consumer/node_modules`. The consumer uses the documented public `webgpu@0.4.0`
lockfile. Install the repository dependencies with `npm ci` and the Playwright
Chromium version selected by that lockfile. Use Node on Linux with the declared
physical AMD Vulkan provider for this measured lane.

From a checkout containing the retained experiment inputs, run:

```bash
TMPDIR=/var/tmp node tools/reproduce-retention-experiment.js \
  /absolute/installed-runtime-bundle \
  /absolute/successful-public-restoration \
  /absolute/new-experiment-output
```

The entrypoint binds paths in the committed reproduction configuration and
invokes the existing tools in order. Each invocation retains its exact config,
command and complete log. Every output directory is new; failed attempts remain
available and are never overwritten.

1. The existing candidate evaluator generates the unlimited and 128 MiB
   candidates, binding each to the archive, Capsule, and selected TargetPlan.
   A fresh Node process measures every candidate/input/run. One warmup and three
   timed runs use the declared balanced rotation and fixed seed.
2. Selection uses only the tuning input. The selected candidate is frozen before
   the two held-out source-reference inputs run. Every run compares full token
   IDs, yes/no logits, scores, probabilities and ranking against the unchanged
   oracle. The declared opening and inference budgets must also pass.
3. Only a passing held-out improvement enables transfer to persistent browser
   document search. The selected retention limit is copied into the application
   build config. The application builder reuses retained signed Capsules and
   checkpoints; no private signing key is required.
4. Both browser variants run the same separate six-document, six-query corpus,
   including offline restart, cancellation, corruption, storage exhaustion and
   device-loss recovery. The corpus was excluded from Node tuning.
5. The browser comparison interleaves fresh offline process openings and checks
   unchanged search acceptance. It retains sampled renderer RSS, opening cost,
   verified artifact counters and all queries for both policies.

The Node metric is Linux fresh-process peak RSS; the browser metric is sampled
renderer RSS during opening. They include shared mappings and have different
measurement scopes. Compare each policy within its application rather than
subtracting browser memory from Node memory. A memory improvement is not a
latency improvement, general portability claim, or automatic release promotion.

The frozen references and configuration live in
[the retained inputs](../reports/retention-experiment/20260907/inputs/reproduction-configs.json).
The source-reference capture policies are retained beside them. Their upstream
checkpoint and exact source-file hashes match the public reranker restoration.
A rerun that encounters changed inputs, denied release history, unsupported
hardware, incorrect outputs or insufficient improvement fails visibly.
