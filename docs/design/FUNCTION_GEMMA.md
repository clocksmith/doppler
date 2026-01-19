# FunctionGemma Refactor Specification
## FunctionGemma Refactor

## Goal

Enforce strict Engine (Doppler) vs Driver (Reploid) separation for FunctionGemma multi-model orchestration.

**Principle:** Doppler never decides, it only executes. Reploid passes policy decisions as parameters to Doppler's primitives.

## Current Violations

| Feature | Current Location | Correct Location | Violation |
|---------|------------------|------------------|-----------|
| Prompt templates | `multi-model-network.js:352-365` | Reploid | Hardcoded "Generate code...", "Review this code..." |
| Temporal ring loop | `multi-model-network.js:285-341` | Reploid | Seed/Reflect/Refine orchestration |
| UCB1 selection | `functiongemma.js` | Reploid | Exploration/exploitation policy |
| Evolution/GA | `functiongemma.js` | Reploid | `runEvolution`, mutation, crossover |
| Arena competition | `functiongemma.js` | Reploid | `runArena`, head-to-head |
| Fitness scoring | `functiongemma.js` | Reploid | `calculateBaseFitness` heuristics |
| Task classification | `functiongemma.js` | Reploid | Keyword matching for routing |

## Architectural Split

### Doppler (Engine) - KEEPS

Primitives that execute without making policy decisions:

```
executeExpert(id, prompt, options) → token_stream
setSharedPrefix(prompt) → KVCacheSnapshot
executeChain(expertIds, prompt) → string[]
executeParallel(tasks) → Record<string, string>
combineOutputs(outputs, combinerConfig) → string
executeGenome(genome, prompt) → string
mergeLogits(buffers, weights) → GPUBuffer  [NEW]
sampleFromMerged(logits, params) → tokenId  [NEW]
```

### Reploid (Driver) - OWNS

All orchestration, policy, and decision-making:

```
buildTemporalPrompt(task, turn, role) → string
executeTemporalSelfRing(task, config) → result
evolveTopology(tasks, config) → NetworkGenome
runArenaEvolution(tasks, config) → winner
selectExpertUCB1(taskType, candidates) → expertId
calculateFitness(output, task) → score
classifyTask(description) → taskType
```

## Files Changed

### Doppler - Remove Orchestration

| File | Action |
|------|--------|
| `src/inference/multi-model-network.js` | Remove `executeTemporalRing`, `buildTemporalPrompt`, `detectTemporalConvergence` |
| `src/inference/multi-model-network.d.ts` | Remove temporal ring types |
| `src/inference/functiongemma.js` | Gut to thin re-export of primitives |
| `src/inference/functiongemma.d.ts` | Primitive types only |
| `src/inference/functiongemma.ts.wip` | Delete (superseded by Reploid) |

### Doppler - Add GPU Primitives

| File | Purpose |
|------|---------|
| `src/gpu/kernels/logit-merge.js` | GPU tensor merging for multi-model ensemble |

### Documentation

| File | Change |
|------|--------|
| `ARCHITECTURE.md` | Add "Engine vs Driver Boundary" section |
| `../../reploid/docs/design/FUNCTIONGEMMA.md` | Clarify Doppler provides primitives only |
| `../../ARCHITECTURE.md` | Add cross-project boundary definition |

## GPU Multi-Model Primitives

When multiple models are loaded, these operations MUST stay on GPU:

```
┌─────────────────────────────────────────────────────────────┐
│                         GPU VRAM                            │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │ Model A     │  │ Model B     │                          │
│  │ Logits      │  │ Logits      │                          │
│  └──────┬──────┘  └──────┬──────┘                          │
│         └───────┬────────┘                                  │
│                 ▼                                           │
│        ┌───────────────┐                                    │
│        │ mergeLogits() │  ← Doppler GPU primitive           │
│        └───────┬───────┘                                    │
│                ▼                                           │
│        ┌───────────────┐                                    │
│        │ sample()      │  ← Doppler GPU primitive           │
│        └───────┬───────┘                                    │
└────────────────┼────────────────────────────────────────────┘
                 ▼
            Token ID (CPU) → Reploid decides next action
```

**Reploid decides:** Which models, what weights, when to merge
**Doppler executes:** The actual tensor operations

## Verification

After refactor:
1. `npm test` passes in Doppler
2. Reploid's `functiongemma-orchestrator.js` works with cleaned Doppler primitives
3. No hardcoded prompts remain in Doppler
4. No loop/evolution logic remains in Doppler

---

*Created: January 2026*


