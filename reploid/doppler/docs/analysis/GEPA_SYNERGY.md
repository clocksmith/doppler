# GEPA Synergy: Prompt Evolution Meets Browser RSI

**Status:** Study Document
**Context:** How Reploid/Doppler can integrate and extend GEPA's prompt evolution approach

---

## TL;DR

GEPA provides the **learning algorithm** (evolutionary prompt mutation).
Reploid provides the **substrate infrastructure** (slots, rollback, safety).
Doppler provides the **compute layer** (local inference, LoRA swapping).

Together, they form a complete browser-native RSI platform. But Reploid/Doppler can also **go beyond GEPA** by evolving things GEPA cannot touch: executable code, kernel configurations, and runtime adapters.

---

## The Two Types of Merging

### Composition (Reploid) vs. Crossover (GEPA)

| Aspect | Reploid (Manual Composition) | GEPA (Evolutionary Crossover) |
|--------|------------------------------|-------------------------------|
| **Who decides** | Human operator | Automated evolution |
| **Logic** | "I think the agent needs these skills" | "Agent A excels at X, Agent B at Y, combine" |
| **Goal** | Modularity, organization | Genetic improvement, discovery |
| **Finds what** | Combinations you plan | Combinations you wouldn't think of |

**Reploid's PersonaManager:** Merges static lesson blocks (e.g., `Build a Tool` + `Analyze Reflections`)

**GEPA's System-Aware Merge:** Blindly combines Module 1 from Agent A with Module 2 from Agent B to create a super-agent

### Output Merging vs. Instruction Merging

| Aspect | Reploid (Arena/Consensus) | GEPA (Prompt Mutation) |
|--------|---------------------------|------------------------|
| **Merges** | Outputs from multiple models | Instructions themselves |
| **Effect** | More reliable *now* (ensemble) | Smarter *forever* (learning) |
| **Type** | Runtime decision | Evolutionary improvement |

---

## What Reploid/Doppler Can Evolve

### The Full RSI Surface

| Layer | What | Reploid | Doppler | GEPA Can Touch |
|-------|------|---------|---------|----------------|
| **L0** | System prompts | PersonaManager lessons | - | Yes (core GEPA) |
| **L1** | Tools | CreateTool, tool library | - | Partially (tool prompts) |
| **L2** | Meta-tools | Tool-writer, reflection loop | - | No (executable code) |
| **L3** | Substrate | agent-loop.js, core modules | - | No (executable code) |
| **K1** | Kernel configs | - | runtimeOptimizations | No (runtime hints) |
| **K2** | LoRA adapters | - | Runtime LoRA swap | No (weight deltas) |
| **K3** | Kernel variants | - | WGSL shader selection | No (GPU code) |
| **K4** | Expert routing | - | MoE router weights | No (learned routing) |

**Key insight:** GEPA only evolves natural language (prompts). Reploid/Doppler can evolve **executable artifacts** that GEPA cannot.

---

## Synergy: GEPA + Reploid Infrastructure

### What GEPA Needs (That Reploid Has)

| GEPA Requirement | Reploid Implementation |
|------------------|------------------------|
| Modular prompt slots | PersonaManager lessons |
| Execution traces | EventBus audit logs |
| Rollback mechanism | Genesis Snapshots |
| Diversity preservation | Arena consensus |
| Validation harness | Verification Worker |

### The Integration Flow

```
User Goal
    |
    v
Reploid Agent Loop (orchestration)
    |
    v
Doppler Inference (local LLM)
    |
    v
Tool Execution in VFS Sandbox
    |
    v
Execution Trace Capture (EventBus) <-- GEPA hooks in here
    |
    v
GEPA Reflection (diagnose failure)
    |
    v
Prompt Mutation Proposal
    |
    v
Arena Verification (multi-model consensus)
    |
    v
Genesis Checkpoint --> Apply --> Validate
    |
    v
Pareto Pool Update (or Rollback)
```

### What's Missing for Full GEPA Integration

1. **Structured reflection function ($\mu_f$):** EventBus captures traces but doesn't algorithmically process them
2. **Pareto pool tracking:** Genesis tracks snapshots, not evolutionary lineage
3. **Crossover operator:** PersonaManager allows composition but not genetic recombination

---

## "One-Upping" GEPA: Beyond Prompt Evolution

### 1. Executable Code Evolution (L1-L3)

GEPA evolves prompts. Reploid can evolve **actual JavaScript**.

```javascript
// GEPA: Evolve the PROMPT for a tool
"When searching, use multi-hop retrieval..."

// Reploid L1: Evolve the TOOL CODE itself
tools.SearchTool = function(query) {
  // This code can be modified by the agent
  return multiHopRetrieval(query, { hops: 3 });
};

// Reploid L2: Evolve the TOOL-WRITER that creates tools
// The meta-tool learns better patterns for tool construction
```

**Advantage:** Code evolution is more powerful than prompt evolution. A tool that implements binary search will always outperform a prompt that describes binary search.

**Safety trade-off:** This is why Reploid has 8 safety layers. Code evolution is dangerous.

### 2. Kernel Configuration Evolution (Doppler K1)

Doppler's `runtimeOptimizations` in RDRR manifests are evolvable:

```json
{
  "runtimeOptimizations": {
    "preferredKernels": {
      "matmul": "q4_fused",
      "attention": "tiled_f16"
    },
    "workgroupOverrides": {
      "matmul_f16": [64, 4, 1]
    },
    "targetDevice": "apple-m1"
  }
}
```

**Evolution loop:**
1. Run benchmark with configuration A
2. Reflect on profiler output (GPU timing, memory bandwidth)
3. Mutate configuration (try different workgroup sizes)
4. Validate (benchmark again)
5. Keep if Pareto-optimal for this device class

**GEPA quote (directly applicable):**
> "On NPU kernel generation, GPT-4o Baseline achieved 4.25% vector utilization. GEPA-optimized GPT-4o achieved 30.52% vector utilization."

Apply this to WGSL kernels: evolve the configuration hints, not the prompts.

### 3. LoRA Adapter Evolution (Doppler K2)

Doppler supports runtime LoRA swapping. This enables **LoRA lineage tracking**:

```javascript
// LoRA Pareto pool
const loraPool = {
  'creative-v12': { wins: ['story', 'poetry'], avgScore: 0.82 },
  'technical-v8': { wins: ['code', 'docs'], avgScore: 0.79 },
  'hybrid-v3': { wins: ['mixed'], avgScore: 0.75 }
};

// Select based on task type
const lora = selectParetoBest(loraPool, taskType);
pipeline.applyLoRA(lora);
```

**Beyond GEPA:** LoRA is a weight delta. GEPA explicitly cannot modify weights. Doppler can swap LoRAs at runtime without recompilation.

### 4. P2P Kernel Evolution (Doppler Vision)

From COMPETITIVE.md, the P2P kernel swarm concept:

```javascript
// User A discovers 2x faster attention on M3 Max
const kernel = await swarm.fetchKernel({
  name: 'attention_flash_v3',
  device: 'apple-m3-max',
  hash: 'sha256:abc123...'
});

// Benchmark locally, confirm improvement
const speedup = await benchmarkKernel(kernel, baseline);
if (speedup > 1.2) {
  swarm.endorse(kernel.hash);  // Propagate to other M3 users
}
```

**This is GEPA at the kernel level:**
- Reflect: Benchmark output
- Mutate: Swap kernel variant
- Evaluate: Confirm speedup
- Pareto: Endorse if best for this device class

**Advantage over GEPA:** Kernels are GPU code, not prompts. Evolution operates on a different substrate.

---

## Concrete Integration Opportunities

### 1. GEPA-Style Prompt Slot Evolution

Add to PersonaManager:

```javascript
class GEPAPersonaManager extends PersonaManager {
  constructor() {
    this.paretoPool = new Map(); // slot -> version -> { prompt, wins }
  }

  async evolveSlot(slotName, executionTrace, failureMode) {
    // GEPA reflection function
    const currentPrompt = this.getSlot(slotName);
    const mutatedPrompt = await this.reflect(currentPrompt, executionTrace, failureMode);

    // Validate mutation
    const score = await this.validate(mutatedPrompt, slotName);

    // Pareto update
    this.updateParetoPool(slotName, mutatedPrompt, score);
  }

  crossover(slotNameA, versionA, slotNameB, versionB) {
    // GEPA system-aware merge
    return this.merge(
      this.paretoPool.get(slotNameA).get(versionA),
      this.paretoPool.get(slotNameB).get(versionB)
    );
  }
}
```

### 2. Execution Trace Reflection Hook

Add to EventBus:

```javascript
// In EventBus subscriber
eventBus.subscribe('tool:complete', async (event) => {
  const trace = {
    tool: event.toolName,
    input: event.input,
    output: event.output,
    error: event.error,
    duration: event.duration
  };

  // Feed to GEPA reflection
  if (event.error || event.quality < threshold) {
    await gepa.reflectOnFailure(trace);
  }
});
```

### 3. Doppler Kernel Config Evolution

Add to benchmark harness:

```javascript
class KernelEvolver {
  constructor() {
    this.configPool = new Map(); // device -> config -> performance
  }

  async evolve(deviceClass) {
    const baseConfig = this.getBestConfig(deviceClass);

    // Mutation: try different workgroup sizes
    const mutations = this.generateMutations(baseConfig);

    for (const config of mutations) {
      const perf = await benchmark(config);
      this.updatePareto(deviceClass, config, perf);
    }
  }

  generateMutations(config) {
    // Try workgroup size variations
    const sizes = [[64, 1, 1], [128, 1, 1], [64, 4, 1], [32, 8, 1]];
    return sizes.map(wg => ({ ...config, workgroupSize: wg }));
  }
}
```

---

## What Reploid/Doppler Has That GEPA Lacks

### 1. Containment (Safety-First RSI)

GEPA assumes a controlled environment. Reploid assumes **mutations might be dangerous**:

| Safety Layer | Purpose | GEPA Equivalent |
|--------------|---------|-----------------|
| VFS Sandbox | No host file access | None |
| Genesis Snapshots | Immutable rollback | Discard mutation |
| Arena Consensus | Multi-model validation | Single validator |
| HITL Queue | Human approval | None |
| Circuit Breakers | Halt on anomaly | None |

### 2. Multi-Substrate Evolution

GEPA evolves one thing: prompts.

Reploid/Doppler can evolve **multiple substrates simultaneously**:
- Prompts (PersonaManager)
- Tools (CreateTool)
- Code (L2-L3 RSI)
- Kernels (runtimeOptimizations)
- LoRAs (weight deltas)
- Router weights (MoE learned routing)

**Emergent behavior:** A mutation in tool code might enable a simpler prompt. Cross-substrate optimization is impossible in pure GEPA.

### 3. Local-First Compute

GEPA requires API calls to LLMs. Doppler runs **entirely in-browser**:

| Aspect | GEPA | Doppler + Reploid |
|--------|------|-------------------|
| Inference | Cloud API | Local WebGPU |
| Rollout cost | $ per token | Free (local) |
| Privacy | Data leaves device | Data stays local |
| Latency | Network bound | GPU bound |

**Sample efficiency matters more when rollouts are free.** GEPA's 35x efficiency over RL is impressive when rollouts cost money. With Doppler, you can run thousands of rollouts locally.

### 4. Weight-Space Access (LoRA)

GEPA explicitly rejects weight modification:
> "many downstream LLM applications... simply cannot finetune the weights of the largest or best-performing LLMs."

Doppler can swap LoRAs at runtime. This is **weight-space evolution without training**:
- Swap in creative-LoRA for storytelling
- Swap in technical-LoRA for coding
- Track which LoRA wins on which task type
- Build a Pareto pool of adapters

---

## Implementation Roadmap

### Phase 1: Trace Capture (Foundation)
- [ ] Structured execution trace format in EventBus
- [ ] Trace persistence in VFS for offline analysis
- [ ] Failure categorization (tool error, logic error, timeout)

### Phase 2: Reflection Function (GEPA Core)
- [ ] Implement $\mu_f$ reflection on traces
- [ ] Prompt mutation proposals
- [ ] Integration with PersonaManager slots

### Phase 3: Pareto Tracking (Diversity)
- [ ] Slot version lineage in Genesis
- [ ] Task-specific wins tracking
- [ ] Crossover operator for slot combination

### Phase 4: Multi-Substrate Evolution (Beyond GEPA)
- [ ] Kernel config evolution with benchmark feedback
- [ ] LoRA selection based on task type
- [ ] Tool code evolution (L1-L2)

### Phase 5: P2P Evolution (Swarm)
- [ ] Kernel variant sharing across devices
- [ ] LoRA Pareto pool distribution
- [ ] Collective performance metrics

---

## Key Quotes

### GEPA on Why Language > Gradients
> "When an agent fails a multi-step task because of a subtle logic error in step 3, a scalar reward of `0` tells the model *that* it failed, but not *why*."

### GEPA on Emergent Knowledge
> "GEPA effectively 'learned' the algorithm for multi-hop retrieval and wrote it down in English."

### GEPA on Sample Efficiency
> "GEPA attains optimal test set performance... with only 678 (35x fewer) rollouts."

### GEPA on Pareto Diversity
> "A candidate is kept in the pool if it is the best performing candidate for at least one specific task instance."

---

## Summary

| Dimension | GEPA | Reploid | Doppler | Combined |
|-----------|------|---------|---------|----------|
| **Evolves** | Prompts | Prompts + Code | Configs + LoRAs | Everything |
| **Safety** | Validation | Containment | N/A | Containment + Validation |
| **Compute** | Cloud API | N/A | Local WebGPU | Local-first |
| **Diversity** | Pareto pool | Arena consensus | Benchmark Pareto | Multi-level Pareto |
| **Rollback** | Discard | Genesis | N/A | Genesis + Version tracking |

**The thesis:** GEPA proves language is a sufficient optimization medium for prompts. Reploid/Doppler extend this to code, kernels, and weights—substrates GEPA cannot touch—while providing the safety infrastructure GEPA assumes but doesn't implement.

---

*Last updated: December 2025*
