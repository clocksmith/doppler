# Verifier-Guided and RLVR Training Contract

This document defines the evidence and artifact contract for any Doppler
optimizer that learns from executable, reference-based, AI-judged, or
human-adjudicated rewards. It reserves the term reinforcement learning with
verifiable rewards (RLVR) for policy updates driven by programmatic rewards.

## Current Support Boundary

Doppler currently implements the original SFT path plus an experimental V11
verifier-guided surface:

- host-session teacher qualification with disposable source snapshots,
  allowed-diff checks, command audits, executable checks, and hidden task
  splits;
- normalization of passing teacher repairs into completion-masked text pairs;
- causal-LM LoRA SFT with frozen base weights, checkpoints, parameter-delta
  receipts, loss comparison, and adapter export;
- declared cross-entropy and teacher-logit distillation workloads; and
- held-out `agentEval` and quality-gate contracts;
- replacement-only WGSL tasks from compiler-reproducing mutations;
- grouped policy sampling with completion masks and policy/reference token
  log-probabilities through the Gamma ROCm trainer protocol;
- Radeon WebGPU verification, unreduced reward vectors, group-relative
  advantages, best-of-N selection, and DPO pair derivation;
- completion-masked SFT, DPO, and a declared clipped GRPO-with-KL update in
  Gamma; and
- executable validators and writers for the six verifier-guided artifact
  classes below.

Doppler V10 completed Qwen 3.5 9B seed-11 SFT and measured a paired
family-disjoint compiler-repair gain. That narrow result is
`capability_proven`; it is not semantic-kernel or promotion evidence. DPO and
GRPO from the separate V11 diagnostic partition, a sealed semantic WGSL
promotion result, minimum-risk sequence training, process-reward training,
CISPO/PPO, and validation-high teacher checkpoint promotion remain absent.

The V8 WGSL student replay is execution-verified SFT, also called
response or sequence-level distillation. It is not RLVR and it is not logit
knowledge distillation. The current receipt is
[WGSL student replay v8](status/wgsl-student-replay-v8-2026-07-11.md).

## Method Names

Use the narrowest accurate identifier:

| Identifier | Required mechanism |
| --- | --- |
| `execution_verified_sft` | A verifier accepts training completions before token-level SFT. |
| `rejection_sampling` | The policy samples alternatives and a verifier selects outputs without a policy-gradient update. |
| `dpo` | Frozen chosen/rejected pairs update the policy with a declared reference policy. |
| `minimum_risk_training` | Sampled sequences are weighted by a frozen sequence-level loss or utility. |
| `grpo_rlvr` | On-policy groups produce verifier rewards and group-relative advantages used in a policy update. |
| `process_supervision` | Versioned intermediate states or tool actions receive training signal. |
| `on_policy_distillation` | A promoted teacher labels the current student's rollout distribution. |

Do not label SFT as RL because the examples passed tests. Verifiers used for
data filtering and verifiers used inside a policy-gradient objective are
different mechanisms.

## Rollout Unit

An RLVR rollout group must be recoverable from one immutable artifact:

```json
{
  "artifactType": "training_rollout_group",
  "schemaVersion": 1,
  "workloadId": "example-wgsl-rlvr",
  "groupId": "task-004-group-0002",
  "taskId": "task-004",
  "datasetHash": "<sha256>",
  "policyHash": "<sha256>",
  "referencePolicyHash": "<sha256>",
  "sampling": {
    "seed": 1337,
    "temperature": 0.8,
    "topP": 0.95,
    "maxTokens": 512
  },
  "samples": [],
  "verifierBundleHash": "<sha256>",
  "runtimeHash": "<sha256>",
  "claimBoundary": "training signal only; not a promotion evaluation"
}
```

Every sample must preserve:

- prompt, completion, token IDs, completion mask, and stop reason;
- policy token log-probabilities from the sampled checkpoint;
- reference token log-probabilities when the objective uses KL or a reference
  ratio;
- raw verifier outputs, stdout/stderr, browser validation messages, and tool
  traces;
- the unreduced reward vector;
- group mean, group variance, normalized advantage, and zero-variance handling;
- clipped and unclipped objective terms;
- KL, entropy, and any teacher regularization terms; and
- task, dataset, policy, model, kernel, runtime, and verifier hashes.

The training step must not recompute old-policy values from a different
checkpoint. If rollout and update are separated, the receipt must bind the
rollout policy hash to the update.

## Reward Vector

Store components before scalarization:

```text
reward = {
  contract_pass,
  compile_pass,
  execute_pass,
  task_score,
  regression_pass,
  policy_pass,
  judge_score,
  human_status
}
```

Each component declares:

- `id` and schema version;
- `type`: `deterministic`, `learned_metric`, `ai_judge`, or
  `human_adjudicated`;
- `role`: `blocking`, `supporting`, or `promotion`;
- evaluator path, configuration, model identity when applicable, and hash;
- raw value, normalized value, weight, and reduction rule; and
- whether the component was visible to the rollout policy.

Blocking components are lexicographic. A schema, safety, policy, or execution
failure cannot be offset by a favorable judge score. Record rejected samples
and verifier errors instead of dropping them from the denominator.

## GRPO-Style Update Contract

For a group of rewards `r_i`, a GRPO-style lane must declare its advantage
rule. A common form is:

```text
advantage_i = (r_i - mean(group_rewards))
              / max(stddev(group_rewards), advantage_epsilon)
```

The workload must also declare:

- group size and incomplete-group policy;
- zero-variance behavior;
- old-policy ratio and clipping bounds;
- asymmetric clipping when used;
- KL coefficient and reference checkpoint;
- advantage and reward clipping;
- number of policy updates per rollout batch;
- maximum stale-policy distance;
- optimizer, scheduler, precision, gradient clipping, and seed;
- interleaving policy across tasks or lanes; and
- checkpoint and teacher-promotion policy.

“GRPO-style” is not enough in a claim. The run contract must name the exact
objective and formula. If a future lane uses CISPO or another clipped objective,
that objective receives its own identifier and receipt fields.

## Zero-Signal and Experimental Method Routing

An optimizer is not a remedy for a saturated verifier. Before DPO, GRPO, or a
novel objective runs, the derived receipt must establish learning signal:

- DPO requires at least one pair with the declared minimum reward gap.
- Group-relative methods require at least one group with a nonzero advantage.
- An all-pass or all-fail group is retained as evidence and receives no policy
  update under the `zero_advantages` policy.

Change the task or reward before changing the optimizer when every group has
zero variance. For WGSL, the next scientific axis is a curriculum of semantic
dispatch, numerical-oracle, metamorphic, bounds, and historical-regression
checks. Once those checks yield mixed verified outcomes, preregister one method
at a time:

- DAPO-style dynamic sampling for excluding saturated groups while preserving
  harder mixed-reward groups;
- contrastive sequence optimization when a group contains verified positive
  and negative repairs;
- self-conditioned token credit or on-policy distillation when a verified
  trajectory can provide denser credit than one sequence-level scalar; or
- CISPO only with an explicit importance-weight clipping formula and a matched
  GRPO control.

These are candidate lanes, not implemented capability claims. Each needs its
own objective identifier, frozen sampling budget, stale-policy rule, ablation,
and promotion receipt.

## Verifier Separation

Use three disjoint verifier roles:

1. Training verifier: creates the optimization reward.
2. Diagnostic verifier: supports development and failure analysis.
3. Promotion verifier: remains sealed until checkpoint selection is frozen.

Tasks, fixtures, expected outputs, and hidden assertions from the promotion
split must never enter prompts, teacher traces, reward-model training, or
rollout feedback. Hash all splits and audit task IDs for overlap and derived
duplicates.

AI judges are not verifiable rewards merely because they return a number. They
must be labeled `ai_judge`, with model, prompt, decoding, retry, and parser
contracts. A deterministic check derived from an AI-generated expected answer
also preserves that provenance.

## WGSL Reward Program

WGSL is the first Doppler domain suited to RLVR because many outcomes can be
executed. A reward bundle can contain:

- allowed-file and allowed-diff checks;
- WGSL parse or `createShaderModule` validation;
- pipeline creation and dispatch without uncaptured GPU errors;
- exact or tolerance-based comparison with a CPU oracle;
- finite-output, shape, bounds, alignment, and buffer-layout checks;
- metamorphic checks over shapes, strides, workgroup sizes, and seeds;
- regression tests from the original defect;
- no hallucinated files, APIs, tools, or commands; and
- resource cleanup and buffer-pool integrity.

Begin with three frozen comparisons:

1. execution-verified SFT;
2. best-of-N rejection sampling from that SFT checkpoint; and
3. GRPO-style RLVR from the same checkpoint and task distribution.

Audit reward variance before updating. A group whose only difference is the
exact-reference bonus has formatting signal, not constructive compile or
contract signal. An all-pass or all-fail group has no group-relative learning
signal. Report constructive, exact-only, other, and zero-variance group counts
separately.

For an all-fail training group, Doppler can emit a compiler-qualified task
reference against the modal on-policy failure. Training that pair is
reference-anchored corrective DPO: the chosen response is off-policy, so the
lane is not RLVR and must not be merged into the on-policy GRPO claim.

Keep token budget, rollout sampler, task IDs, verifier bundle, model
initialization, and promotion split fixed. Report both capability and compute:
pass rate, reward distribution, unique passing repairs, policy violations,
rollout tokens, update tokens, and browser/GPU runtime.

The promotion gate remains stricter than the training reward: three
deterministic held-out repetitions, strict pass-rate improvement, zero policy
violations, and every lane required by the frozen student-code policy.

## Translation Boundary

Translation rewards combine deterministic constraints with incomplete quality
proxies:

- language/script, placeholders, markup, numbers, entities, and terminology
  can be checked deterministically;
- BLEU and chrF are deterministic calculations against frozen references;
- learned quality metrics must be labeled `learned_metric`; and
- ambiguous meaning and domain correctness may require human adjudication.

Minimum-risk training and preference learning are direct candidate methods for
reference-scored translation. RLVR may optimize the deterministic constraint
subvector, but that subvector must not be reported as complete semantic
quality. A translation promotion still needs the declared external and
in-domain metrics under a frozen decoder.

## Valera/Columbo Legal Boundary

Legal-document training may use deterministic rewards for schema validity,
exact source-span grounding, page and document citations, provenance,
supported-path behavior, deterministic PII tests, and export regressions.
Legal category, privilege, responsiveness, risk, and final redaction or export
authority are human-adjudicated. Doppler may execute the optimizer and preserve
the receipts; Valera/Columbo owns the label and approval contract.

An RL optimizer must not convert an AI judge into legal authority, reward
bypassing a reviewer gate, or train on sealed legal eval documents. Human
decisions used as reward must carry rubric version, reviewer role, item hash,
and adjudication status.

## Prompt Optimization and GEPA Boundary

Doppler can import prompt-policy frontiers and preserve candidate IDs in
teacher-trace lineage. It does not thereby run prompt optimization.

Use `gepaCandidateId` only when the upstream artifact records iterative
reflection, mutation, instance-level Pareto selection, frontier updates, and
any merge operations. A single error-conditioned reflection call is
`reflective_prompt_mutation`, even if it returns several candidates. Prompt
search and RLVR must have separate experiment IDs so a prompt change cannot be
misattributed to the optimizer.

## Required Artifact Classes

The V11 experimental surface validates these classes before an optimizer run
can become claimable:

- `training_rollout_group`
- `training_reward_vector`
- `training_policy_update`
- `training_verifier_report`
- `training_policy_checkpoint`
- `training_promotion_decision`

An artifact class being implemented does not prove that a run emitted a valid
instance. A claim still requires the complete linked set for that run. See
[Training Artifact Policy](training-artifact-policy.md) and the
[V10 result receipt](status/wgsl-repair-v10-2026-07-12.md).

## Promotion and Rejection

Reject or block promotion when:

- any required artifact or hash is absent;
- rollout-policy and update-policy identities disagree;
- a reward component lacks its evaluator version or provenance;
- training and promotion verifier splits overlap;
- grouped rewards or advantages cannot be reconstructed;
- only training reward, SFT loss, or judge score improves;
- the held-out task metric misses its declared gate;
- a policy or safety violation occurs;
- deterministic repetitions disagree beyond the frozen tolerance; or
- the result depends on an undeclared prompt, model, kernel, or runtime change.

Keep unsuccessful lanes. They constrain the next experiment and prevent a
failed objective or reward mix from reappearing under a new run name.

## References

- [DeepSeekMath](https://arxiv.org/abs/2402.03300), which introduces GRPO.
- [DAPO](https://arxiv.org/abs/2503.14476), on decoupled clipping and dynamic
  sampling for verifiable-reward training.
- [Self-Conditioned GRPO](https://arxiv.org/abs/2606.18810), on token-level
  credit derived from a policy conditioned on its own verified trajectories.
- [Contrastive Sequence-level Policy Optimization](https://arxiv.org/abs/2605.12969),
  on within-group positive/negative sequence contrast.
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).
- [SelfCodeAlign](https://arxiv.org/abs/2410.24198), an execution-filtered code
  data and instruction-tuning pipeline.
- [GEPA](https://arxiv.org/abs/2507.19457), reflective prompt evolution with
  Pareto selection.
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), on process and
  outcome supervision.
- [Rule Based Rewards for Language Model Safety](https://arxiv.org/abs/2411.01111).
- [Minimum Risk Training for Neural Machine Translation](https://aclanthology.org/P16-1159/).
- [Learning to Replicate Expert Judgment in Financial Tasks](https://thinkingmachines.ai/news/learning-to-replicate-expert-judgment-in-financial-tasks/),
  related private-task evidence for interleaving, clipped policy optimization,
  on-policy distillation, and teacher promotion. It is not evidence that those
  mechanisms are implemented in Doppler.
