# Multi-FunctionGemma Network Architecture

> Multi-FunctionGemma inference network for DOPPLER WebGPU engine

**Status:** Research / Design Phase
**Target:** DOPPLER inference pipeline
**Last Updated:** December 2025

---

## 1. Executive Summary

Instead of running one large model, DOPPLER can orchestrate a **network of specialized FunctionGemma (270M) instances** that collectively achieve expert-level code generation. Each node specializes via LoRA adapters, and the network topology determines how they collaborate.

**Key Insight:** FunctionGemma is small enough (~540MB in fp16, ~135MB in q4) to run **multiple instances** in WebGPU memory simultaneously on modern GPUs.

**Memory Math:**
- FunctionGemma: ~550MB RAM per instance (fp16), ~135MB (q4)
- Typical GPU: 4-8GB VRAM
- **Feasible:** 3-6 concurrent instances with shared weights + LoRA adapters

---

## 1.1 Current Implementation Snapshot (Dec 2025)

Implemented in Doppler (initial versions):
- `loader/multi-model-loader.ts` (base + LoRA adapters)
- `inference/multi-model-network.ts` (ring/mesh/DAG execution + combiner)
- `inference/multi-pipeline-pool.ts` (parallel pipelines with per-id locking)
- `gpu/multi-model-recorder.ts` + `InferencePipeline.prefillKVOnly` (shared prefix KV)
- `inference/expert-router.ts` (embedding-based routing helper)

Still in progress / partial:
- `inference/network-evolution.ts` (helpers only; no fitness loop wiring)
- Multi-node KV cache sizing + buffer partitioning tuning

---

## 2. Network Topologies

Based on research from [Guided Topology Diffusion](https://arxiv.org/html/2510.07799) and [Multi-Agent Collaboration Survey](https://arxiv.org/html/2501.06322v1):

### 2.1 Ring Topology

```
FnG-Scaffold → FnG-Logic → FnG-Style → FnG-Test → FnG-Scaffold
      ↑                                                 │
      └─────────────────────────────────────────────────┘
```

**Execution Flow:**
1. Scaffold generates file structure
2. Logic fills in business code
3. Style adds CSS/formatting
4. Test generates test cases
5. Loop back for refinement

**Pros:**
- Natural pipeline for sequential code generation
- Each node only needs context from previous node
- O(n) memory for connections

**Cons:**
- Single point of failure breaks the ring
- Latency compounds through chain

**Best For:** Sequential build pipelines where each stage depends on the previous

**WebGPU Implementation:**
```javascript
// Ring execution in Doppler
const network = new MultiModelNetwork(pipeline, loader, pool, recorder);
nodes.forEach((node) => {
  network.registerExpert({ id: node.id, adapterName: node.adapter });
});

await network.setSharedPrefix(task.systemPrompt);

const executeRing = async (task, expertIds) => {
  return network.executeRing(expertIds, task.prompt, {
    maxTokens: task.maxTokens
  });
};
```

**Memory Profile:**
- 4 nodes × 135MB (q4) = 540MB base
- + KV caches (~50MB each) = 740MB total
- Fits in 8GB VRAM with room for batching

---

### 2.2 Tree Hierarchy (MoE-Style)

```
                    ┌─────────────────┐
                    │   Root Router   │
                    │   (FnG + LoRA)  │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │  Frontend  │    │  Backend   │    │   Infra    │
    │   Router   │    │   Router   │    │   Router   │
    └──────┬─────┘    └──────┬─────┘    └──────┬─────┘
           │                 │                 │
     ┌─────┼─────┐     ┌─────┼─────┐     ┌─────┼─────┐
     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
   React  CSS  A11y   API   DB  Auth  Docker K8s  CI/CD
   (FnG)  (FnG)(FnG) (FnG) (FnG)(FnG)  (FnG)(FnG) (FnG)
```

This mirrors [NVIDIA's MoE architecture](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/) but at the network level rather than within a single model.

**Pros:**
- Hierarchical routing reduces decision complexity
- Intermediate routers can specialize (frontend vs backend)
- Scales to many experts

**Cons:**
- Routing overhead at each level
- May need to backtrack if wrong branch chosen

**Best For:** Large-scale systems with clear domain boundaries

**Sparse Activation:** Only 2-3 nodes active per request (log(n) depth)

**Routing Mechanism:**
```javascript
const genome = {
  topology: { type: 'tree', depth: 3, branchingFactor: 2 },
  nodes: [
    { id: 'router' },
    { id: 'frontend' },
    { id: 'backend' },
    { id: 'infra' }
  ],
  edges: [
    { from: 'router', to: 'frontend', weight: 0.6 },
    { from: 'router', to: 'backend', weight: 0.3 },
    { from: 'router', to: 'infra', weight: 0.1 }
  ],
  combiner: { type: 'weighted' }
};

const router = async ({ children }) => {
  // Plug in ExpertRouter (embeddings) or a router FnG here.
  return children.slice(0, 1);
};

const output = await network.executeGenome(genome, task.prompt, {
  maxTokens: task.maxTokens
}, router);
```

---

### 2.3 Fully Connected (Mesh)

```
    FnG-A ←──────→ FnG-B
      ↕    ╲    ╱    ↕
      ↕      ╲╱      ↕
      ↕      ╱╲      ↕
    FnG-C ←──────→ FnG-D
```

**Pros:**
- Any node can query any other
- Maximum flexibility for cross-domain tasks
- Redundancy if one node fails

**Cons:**
- O(n²) connections
- Complex routing decisions
- Context explosion risk

**Best For:** Small networks (3-5 nodes) where tasks frequently cross domains

**Implementation:**
```javascript
const tasks = nodes.map((node) => ({
  id: `${task.id}:${node.id}`,
  expertId: node.id,
  prompt: task.prompt
}));

const outputs = await network.executeParallel(tasks, {
  maxTokens: task.maxTokens
});

const merged = await network.combineOutputs(Object.values(outputs));
```

---

### 2.4 Dynamic Graph (Learned Topology)

Based on [Guided Topology Diffusion](https://arxiv.org/html/2510.07799), the network topology itself is learned:

```javascript
// Topology is learned per-task-type
const learnedTopologies = {
  'react-component': {
    nodes: ['scaffold', 'jsx', 'css', 'test'],
    edges: [
      { from: 'scaffold', to: 'jsx', weight: 1.0 },
      { from: 'jsx', to: 'css', weight: 0.8 },
      { from: 'jsx', to: 'test', weight: 0.6 },
      { from: 'css', to: 'test', weight: 0.3 }
    ]
  },
  'api-endpoint': {
    nodes: ['schema', 'handler', 'validation', 'test'],
    edges: [/* ... */]
  }
};

const genome = {
  topology: { type: 'dag', depth: 4 },
  nodes: learnedTopologies['react-component'].nodes.map((id) => ({ id })),
  edges: learnedTopologies['react-component'].edges,
  combiner: { type: 'weighted' }
};

const output = await network.executeGenome(genome, task.prompt, {
  maxTokens: task.maxTokens
});
```

**Dynamic Routing with Attention:**
```javascript
// Pseudo-code for dynamic topology
const selectExperts = async (task, allNodes) => {
  // Embed task
  const taskEmbed = await embed(task.description);

  // Compute attention scores over all nodes
  const scores = allNodes.map(node =>
    dotProduct(taskEmbed, node.embedding)
  );

  // Softmax to get routing probabilities
  const probs = softmax(scores);

  // Select top-k nodes (sparse activation)
  const topK = selectTopK(probs, k=3);

  // Execute in parallel, combine outputs
  const outputs = await Promise.all(
    topK.map(node => node.execute(task))
  );

  return weightedCombine(outputs, topK.map(n => n.prob));
};
```

---

## 3. Proposed Architecture: Hierarchical MoE Network

Combining insights from [Neptune MoE Guide](https://neptune.ai/blog/mixture-of-experts-llms) and [FunctionGemma for Multi-Agent Orchestration](https://dev.to/saikumaryava/beyond-mobile-actions-exploring-functiongemma-for-intelligent-multi-agent-orchestration-4jlf):

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Main Brain (GPT-4o / Claude / Llama-70B)           │    │
│  │  - Natural language understanding                    │    │
│  │  - High-level planning                               │    │
│  │  - Outputs: Task DAG + constraints                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     ROUTING LAYER                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Router FnG (with learned routing LoRA)              │    │
│  │  - Embeds task descriptions                          │    │
│  │  - Computes expert scores                            │    │
│  │  - Manages KV cache prefixes                         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  FRONTEND POOL  │ │  BACKEND POOL   │ │  INFRA POOL     │
│  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │
│  │ React FnG │  │ │  │ API FnG   │  │ │  │ Docker FnG│  │
│  │ (+LoRA)   │  │ │  │ (+LoRA)   │  │ │  │ (+LoRA)   │  │
│  └───────────┘  │ │  └───────────┘  │ │  └───────────┘  │
│  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │
│  │ CSS FnG   │  │ │  │ DB FnG    │  │ │  │ CI/CD FnG │  │
│  │ (+LoRA)   │  │ │  │ (+LoRA)   │  │ │  │ (+LoRA)   │  │
│  └───────────┘  │ │  └───────────┘  │ │  └───────────┘  │
│  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │
│  │ A11y FnG  │  │ │  │ Auth FnG  │  │ │  │ Test FnG  │  │
│  │ (+LoRA)   │  │ │  │ (+LoRA)   │  │ │  │ (+LoRA)   │  │
│  └───────────┘  │ │  └───────────┘  │ │  └───────────┘  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SYNTHESIS LAYER                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Combiner FnG (with integration LoRA)                │    │
│  │  - Merges outputs from multiple experts              │    │
│  │  - Resolves conflicts (import clashes, etc.)         │    │
│  │  - Validates against schema                          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    EVOLUTION LAYER                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Genetic Algorithm + Arena Selection                 │    │
│  │  - Generate N variants per task                      │    │
│  │  - Execute in VFS sandboxes                          │    │
│  │  - Score: correctness, performance, style            │    │
│  │  - Select winners, mutate losers                     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 Minimal Viable Network (Phase 0, no new APIs)

Goal: validate topology, handoffs, and combiner quality without adding new Doppler APIs.

- Use a single base model instance and role-specific prompts for scaffold, logic, and test passes.
- Start with a 3-node ring (scaffold -> logic -> test), sequential execution.
- Keep KV cache usage isolated per pass; no shared prefix yet.
- Combiner picks a single winner output or merges only when outputs share the same file structure.
- Evaluation uses fast checks: syntax parse, basic linting, and optional task-provided tests.

This phase de-risks the orchestration logic before multi-model and LoRA work lands.

### 3.2 Combiner and Validation Constraints

The combiner is the failure point for correctness. Define a deterministic policy:

- Single-winner default when outputs disagree on file boundaries or exports.
- Merge only when outputs share a matching file list and module boundaries.
- Validate final output with at least syntax checks and import/export consistency.

---

## 4. KV Cache Sharing Strategies

Based on [LMCache](https://arxiv.org/pdf/2510.09665) and [Prompt Cache](https://arxiv.org/html/2412.19442v3):

### 4.1 Prefix Caching

All FunctionGemma nodes share a common system prompt:

```javascript
// Doppler KV cache manager
class KVCacheManager {
  constructor(network, recorder) {
    this.network = network;
    this.recorder = recorder;
    this.prefixCache = null;
  }

  async initSharedPrefix(pipeline) {
    const SHARED_PROMPT = `You are a specialized code generator.
Output valid, typed code with imports.
Format: JSON { "code": string, "imports": string[], "exports": string[] }`;

    // Pre-compute KV cache for shared prefix
    this.prefixCache = await this.recorder.computeSharedPrefix(pipeline, SHARED_PROMPT);
    this.network.setSharedPrefixSnapshot(this.prefixCache);
  }

  getNodePrompt(nodeId, specialization) {
    return specialization;
  }
}
```

**Memory Savings:**
- Shared prefix: ~2K tokens × 4 bytes × 32 layers × 2 (K+V) = ~512KB
- Per-node specialization: ~500 tokens = ~128KB each
- 10 nodes: 512KB + 10×128KB = 1.8MB (vs 6.4MB without sharing)
- **Savings:** ~30-50% of context window reused across all nodes

### 4.2 Hierarchical KV Cache

```
Level 0: [System Prompt]           ← Shared by ALL nodes
              │
Level 1: [Domain Context]          ← Shared within pools (Frontend/Backend/Infra)
              │
Level 2: [Specialization]          ← Per-node (React/CSS/API/etc.)
              │
Level 3: [Task Context]            ← Per-request (ephemeral)
```

```javascript
// Hierarchical cache structure
const hierarchicalCache = {
  system: null,      // Level 0: computed once at boot
  domains: {         // Level 1: computed per domain
    frontend: null,
    backend: null,
    infra: null
  },
  nodes: new Map(),  // Level 2: computed per node
  // Level 3 is ephemeral, not cached
};

const getFullCache = async (nodeId, domain) => {
  // Build cache hierarchy
  const system = hierarchicalCache.system;
  const domainCache = await extendKV(system, hierarchicalCache.domains[domain]);
  const nodeCache = await extendKV(domainCache, hierarchicalCache.nodes.get(nodeId));
  return nodeCache;
};
```

**Isolation Note:** Prefix caches must be read-only and never shared across experts once task-specific context is injected.

### 4.3 Cross-Node Context Injection

When one node's output becomes another's input:

```javascript
// Efficient context handoff
const handoffContext = async (fromNode, toNode, output) => {
  const sharedPrefix = network.getSharedPrefixSnapshot();

  // Option 1: Summarize output (lossy but compact)
  const summary = await network.executeExpert(
    'summarizer',
    `Summarize for ${toNode.role}:\n\n${output}`,
    { maxTokens: 200 },
    { prefix: sharedPrefix }
  );

  // Option 2: Extract structured data (lossless for key info)
  const structured = parseCodeOutput(output);
  const handoff = structured || summary;

  // Inject into toNode's context
  return network.executeExpert(
    toNode.id,
    `${toNode.prompt}\n\n${handoff}`,
    { maxTokens: 200 },
    { prefix: toNode.cachedPrefix || sharedPrefix }
  );
};
```

---

## 5. Genetic Algorithm for Network Evolution

Based on [Evolving Code with LLMs](https://arxiv.org/html/2401.07102v1) and [GECCO 2024](https://dl.acm.org/doi/10.1145/3638529.3654017):

### 5.1 Genome Representation

```javascript
const NetworkGenome = {
  // Topology genes
  topology: {
    type: 'tree',  // 'ring' | 'tree' | 'mesh' | 'dag'
    depth: 3,
    branchingFactor: 3
  },

  // Node genes
  nodes: [
    { id: 'root', adapter: 'router.lora', temperature: 0.1 },
    { id: 'react', adapter: 'react-v2.lora', temperature: 0.7 },
    { id: 'css', adapter: 'tailwind.lora', temperature: 0.5 },
    // ...
  ],

  // Edge genes (for DAG topologies)
  edges: [
    { from: 'root', to: 'react', weight: 0.9 },
    { from: 'root', to: 'css', weight: 0.1 },
    // ...
  ],

  // Combination strategy
  combiner: {
    type: 'weighted',  // 'voting' | 'weighted' | 'llm-merge'
    weights: [0.6, 0.3, 0.1]
  }
};
```

### 5.2 What Evolves?

| Component            | Mutation Operators                        |
|----------------------|-------------------------------------------|
| Topology             | Add/remove edges, change node connections |
| Routing weights      | Adjust softmax temperatures, top-k values |
| LoRA selection       | Swap adapters, merge adapters             |
| Prompt templates     | GEPA-style prompt evolution               |
| Combination strategy | Weighted avg, voting, concatenation       |

### 5.3 Mutation Operators

```javascript
const mutate = (genome, mutationRate = 0.1) => {
  const mutated = JSON.parse(JSON.stringify(genome));

  // Topology mutation
  if (Math.random() < mutationRate) {
    mutated.topology.type = randomChoice(['ring', 'tree', 'mesh', 'dag']);
  }

  // Node mutations
  for (const node of mutated.nodes) {
    if (Math.random() < mutationRate) {
      // Swap adapter
      node.adapter = randomChoice(availableAdapters);
    }
    if (Math.random() < mutationRate) {
      // Adjust temperature
      node.temperature = clamp(node.temperature + gaussian(0, 0.1), 0, 1);
    }
  }

  // Edge mutations (add/remove/reweight)
  if (Math.random() < mutationRate) {
    const edge = randomChoice(mutated.edges);
    edge.weight = clamp(edge.weight + gaussian(0, 0.2), 0, 1);
  }

  // Add new edge
  if (Math.random() < mutationRate * 0.5) {
    mutated.edges.push({
      from: randomChoice(mutated.nodes).id,
      to: randomChoice(mutated.nodes).id,
      weight: Math.random()
    });
  }

  return mutated;
};
```

### 5.4 Crossover

```javascript
const crossover = (parent1, parent2) => {
  const child = {
    topology: Math.random() < 0.5 ? parent1.topology : parent2.topology,
    nodes: [],
    edges: [],
    combiner: Math.random() < 0.5 ? parent1.combiner : parent2.combiner
  };

  // Uniform crossover for nodes
  const allNodeIds = new Set([
    ...parent1.nodes.map(n => n.id),
    ...parent2.nodes.map(n => n.id)
  ]);

  for (const id of allNodeIds) {
    const p1Node = parent1.nodes.find(n => n.id === id);
    const p2Node = parent2.nodes.find(n => n.id === id);

    if (p1Node && p2Node) {
      child.nodes.push(Math.random() < 0.5 ? p1Node : p2Node);
    } else {
      child.nodes.push(p1Node || p2Node);
    }
  }

  // Edge crossover
  child.edges = [...parent1.edges, ...parent2.edges]
    .filter((e, i, arr) => arr.findIndex(x => x.from === e.from && x.to === e.to) === i);

  return child;
};
```

### 5.5 Evolution Loop

```javascript
const evolveNetwork = async (task, config = {}) => {
  const {
    populationSize = 20,
    generations = 10,
    eliteCount = 2,
    mutationRate = 0.1
  } = config;

  // Initialize population
  let population = Array(populationSize).fill(null).map(() =>
    generateRandomGenome()
  );

  for (let gen = 0; gen < generations; gen++) {
    // Evaluate fitness
    const scored = await Promise.all(population.map(async genome => {
      const network = buildNetworkFromGenome(genome);
      const output = await executeNetwork(network, task);
      const fitness = await evaluateFitness(output, task);
      return { genome, output, fitness };
    }));

    // Sort by fitness
    scored.sort((a, b) => b.fitness - a.fitness);

    console.log(`Gen ${gen}: best=${scored[0].fitness.toFixed(3)}, avg=${
      (scored.reduce((s, x) => s + x.fitness, 0) / scored.length).toFixed(3)
    }`);

    // Selection
    const elite = scored.slice(0, eliteCount).map(s => s.genome);
    const parents = tournamentSelect(scored, populationSize - eliteCount);

    // Reproduction
    const offspring = [];
    for (let i = 0; i < parents.length; i += 2) {
      const child1 = crossover(parents[i], parents[i + 1] || parents[0]);
      const child2 = crossover(parents[i + 1] || parents[0], parents[i]);
      offspring.push(mutate(child1, mutationRate), mutate(child2, mutationRate));
    }

    population = [...elite, ...offspring.slice(0, populationSize - eliteCount)];
  }

  return population[0]; // Best genome
};
```

---

## 6. Fitness Evaluation

```javascript
const evaluateFitness = async (output, task) => {
  let score = 0;

  // Correctness (40%) - does the code work?
  try {
    const testResults = await runInSandbox(output.code, task.tests);
    score += 0.4 * (testResults.passed / testResults.total);
  } catch (e) {
    score += 0; // Syntax error = 0 correctness
  }

  // Efficiency (20%) - tokens used, inference time
  const efficiency = 1 - (output.totalTokens / task.maxTokens);
  score += 0.2 * Math.max(0, efficiency);

  // Quality (20%) - linting, type coverage
  const lintScore = await lintCode(output.code);
  score += 0.2 * lintScore;

  // Specialization (10%) - did experts stay in their lane?
  const specializationScore = measureSpecialization(output.nodeContributions);
  score += 0.1 * specializationScore;

  // Collaboration (10%) - smooth handoffs between nodes?
  const collaborationScore = measureHandoffQuality(output.handoffs);
  score += 0.1 * collaborationScore;

  return score;
};
```

---

## 7. Self-Adaptive LoRA Selection

### 7.1 Bandit-Based Adapter Selection

```javascript
class AdapterBandit {
  constructor() {
    // UCB1 statistics per (taskType, adapter) pair
    this.stats = new Map();
    this.totalPulls = 0;
  }

  getKey(taskType, adapter) {
    return `${taskType}:${adapter}`;
  }

  select(taskType, availableAdapters) {
    this.totalPulls++;

    let bestAdapter = null;
    let bestUCB = -Infinity;

    for (const adapter of availableAdapters) {
      const key = this.getKey(taskType, adapter);
      const stat = this.stats.get(key) || { successes: 0, pulls: 0 };

      // UCB1 formula
      const exploitation = stat.pulls > 0 ? stat.successes / stat.pulls : 0;
      const exploration = stat.pulls > 0
        ? Math.sqrt(2 * Math.log(this.totalPulls) / stat.pulls)
        : Infinity;
      const ucb = exploitation + exploration;

      if (ucb > bestUCB) {
        bestUCB = ucb;
        bestAdapter = adapter;
      }
    }

    return bestAdapter;
  }

  update(taskType, adapter, success) {
    const key = this.getKey(taskType, adapter);
    const stat = this.stats.get(key) || { successes: 0, pulls: 0 };
    stat.pulls++;
    if (success) stat.successes++;
    this.stats.set(key, stat);
  }
}
```

### 7.2 LoRA Merging for Multi-Domain Tasks

```javascript
// When a task spans multiple domains
const mergeLoRAs = async (sources) => {
  // sources: [{ name: string, manifest: object, weight: number }]
  const loader = new MultiModelLoader();
  const adapters = await Promise.all(
    sources.map(src => loader.loadAdapter(src.name, src.manifest))
  );

  // Merge adapter.layers with weighted deltas (pseudo-code)
  const merged = mergeAdapterLayers(adapters, sources.map(src => src.weight));
  return merged;
};

// Usage
const reactAPIAdapter = await mergeLoRAs([
  { name: 'react', manifest: reactManifest, weight: 0.6 },
  { name: 'api', manifest: apiManifest, weight: 0.4 }
]);
```

---

## 8. Doppler Integration Analysis

### 8.1 KV Prefix Sharing via MultiModelRecorder

Doppler's `MultiModelRecorder` plus `InferencePipeline.prefillKVOnly` already provide shared prefix capture and reuse:

```typescript
const recorder = new MultiModelRecorder();
const sharedPrefix = await recorder.computeSharedPrefix(pipeline, SHARED_PROMPT);
network.setSharedPrefixSnapshot(sharedPrefix);

// Any execution path can now reuse sharedPrefix via MultiModelNetwork
const output = await network.executeExpert(expertId, task.prompt, options, {
  prefix: sharedPrefix
});
```

**Key insight:** The `recordAttention()` function in `gpu/kernels/attention.ts` already accepts `startPos` for KV cache continuation. This can be leveraged for prefix sharing:

```typescript
// attention.ts:328-333
const uniformBuffer = createAttentionUniformBuffer(device, recorder, {
  // ...
  startPos,  // <-- Use this to skip shared prefix tokens
});
```

### 8.2 LoRA Weight Loading Infrastructure

Doppler loads one base model per pipeline via `DopplerLoader`, while `MultiModelLoader` handles base + adapter loading for multi-expert setups:

```typescript
// Actual loader API (implemented in Doppler)
const loader = new MultiModelLoader();
await loader.loadBase(baseManifest, { storageContext });

await loader.loadAdapter('react', 'react.lora');
await loader.loadAdapter('api', 'api.lora');

const pipeline = await loader.createSharedPipeline({ storage: storageContext });
```

**GPU memory strategy:**
- Base FunctionGemma weights: ~550MB (loaded once, shared)
- Each LoRA adapter: ~10-50MB (rank 16-64)
- 6 adapters: ~300MB additional
- Total: ~850MB for 6-expert network (fits in 4GB VRAM)

### 8.3 Buffer Pool Partitioning

`gpu/buffer-pool.ts` manages GPU buffer allocation. For multi-model, partition the pool:

```typescript
// Implemented helper for partitioned pools
const pool = new PartitionedBufferPool([
  { id: 'react' },
  { id: 'api' },
  { id: 'css' },
]);

const buffer = pool.acquire('react', size, usage, 'ffn_output');
pool.release('react', buffer);
```

### 8.4 Pipeline Parallelism

Parallel expert execution is handled by `MultiPipelinePool`, which owns per-expert pipelines and serializes calls per id:

```typescript
const loader = new MultiModelLoader();
await loader.loadBase(baseManifest, { storageContext });

await loader.loadAdapter('react', reactManifest);
await loader.loadAdapter('css', cssManifest);
await loader.loadAdapter('api', apiManifest);

const recorder = new MultiModelRecorder();
const pool = new MultiPipelinePool(loader, { recorder });
await pool.warmPool(['react', 'css', 'api'], { storage: storageContext });

const pipeline = await loader.createSharedPipeline({ storage: storageContext });
const network = new MultiModelNetwork(pipeline, loader, pool, recorder);

const tasks = [
  { id: 'react', expertId: 'react', prompt: task.prompt },
  { id: 'css', expertId: 'css', prompt: task.prompt },
  { id: 'api', expertId: 'api', prompt: task.prompt },
];

const outputs = await network.executeParallel(tasks, { maxTokens: task.maxTokens });
```

**Note:** `MultiPipelinePool` locks per expert id, so parallelism scales with distinct ids and available GPU resources.

### 8.5 RDRR Format Extension for LoRA

The RDRR manifest (`docs/spec/RDRR_FORMAT.md`) already supports LoRA metadata. Adapter manifests should follow this shape:

```json
{
  "version": "1.0",
  "modelId": "functiongemma-270m-react-lora",
  "modelType": "lora",
  "adapterType": "lora",
  "baseModel": "functiongemma-270m-it",
  "quantization": "f16",
  "hashAlgorithm": "sha256",
  "loraConfig": {
    "rank": 32,
    "alpha": 64,
    "targetModules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "dropout": 0.0
  },
  "shards": [
    {
      "index": 0,
      "fileName": "lora_weights.bin",
      "size": 15728640,
      "hash": "sha256-hex-64-chars",
      "hashAlgorithm": "sha256"
    }
  ],
  "tensors": {
    "lora.layers.0.q_proj.lora_A": {
      "shard": 0,
      "offset": 0,
      "size": 4096,
      "shape": [32, 128],
      "dtype": "F16"
    }
  }
}
```

---

## 9. WebGPU Memory Layout

### 9.1 Multi-Model Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                     GPU MEMORY (8GB)                         │
├─────────────────────────────────────────────────────────────┤
│  Base Model (shared)           │  135MB (q4)                │
├────────────────────────────────┼────────────────────────────┤
│  LoRA Adapter Slots            │  8 × 2MB = 16MB            │
│  (hot-swappable)               │                            │
├────────────────────────────────┼────────────────────────────┤
│  KV Cache Pool                 │  512MB                     │
│  - Shared prefix: 64MB         │                            │
│  - Per-node caches: 8 × 56MB   │                            │
├────────────────────────────────┼────────────────────────────┤
│  Activation Memory             │  256MB                     │
│  (reused across nodes)         │                            │
├────────────────────────────────┼────────────────────────────┤
│  Scratch Space                 │  128MB                     │
└────────────────────────────────┴────────────────────────────┘
Total: ~1GB for 8-node network (leaves room for larger contexts)
```

### 9.2 Execution Pipeline

```javascript
// Doppler multi-node execution (current primitives)
const loader = new MultiModelLoader();
await loader.loadBase(baseManifest, { storageContext });

await loader.loadAdapter('react', reactManifest);
await loader.loadAdapter('css', cssManifest);

const recorder = new MultiModelRecorder();
const pool = new MultiPipelinePool(loader, { recorder });
await pool.warmPool(['react', 'css'], { storage: storageContext });

const pipeline = await loader.createSharedPipeline({ storage: storageContext });
const network = new MultiModelNetwork(pipeline, loader, pool, recorder);

network.registerExpert({ id: 'react', adapterName: 'react' });
network.registerExpert({ id: 'css', adapterName: 'css' });

const outputs = await network.executeParallel([
  { id: 'react:task', expertId: 'react', prompt: task.prompt },
  { id: 'css:task', expertId: 'css', prompt: task.prompt }
], { maxTokens: task.maxTokens });
```

---

## 10. REPLOID Integration

> **See:** [Reploid: FUNCTIONGEMMA_INTEGRATION.md](../../../../../reploid/doppler/reploid/docs/plans/FUNCTIONGEMMA_INTEGRATION.md)

Reploid orchestrates the FunctionGemma network using this Doppler infrastructure as a dependency. Integration covers:
- SemanticMemory for expert routing
- ArenaHarness for expert competition
- ContextManager for KV cache sharing
- ReflectionStore for evolution persistence

---

## 11. Benchmarks (Projected)

| Configuration | VRAM | Latency (p50) | Throughput | Quality |
|--------------|------|---------------|------------|---------|
| Single FnG | 150MB | 100ms | 1x | Baseline |
| Ring (4 nodes) | 600MB | 400ms | 0.25x | +15% |
| Tree (7 nodes) | 1GB | 250ms | 0.4x | +25% |
| Mesh (4 nodes) | 600MB | 150ms | 0.7x | +20% |
| Dynamic DAG | 1GB | 200ms | 0.5x | +30% |

**Quality Improvement Sources:**
- Specialization reduces hallucination
- Multiple passes catch errors
- Genetic evolution optimizes for task type

---

## 12. Implementation Roadmap

### Phase 1: Single-Model LoRA Support
- [x] Extend RDRR format for LoRA manifests
- [x] Implement `loadLoRAWeights()` in DopplerLoader
- [ ] Add LoRA merge kernel (`W' = W + scale * A @ B`)
- [ ] Test with single FunctionGemma + one LoRA

### Phase 2: KV Cache Prefix Sharing
- [x] Add `prefillKVOnly()` to InferencePipeline
- [x] Implement `startPos` continuation in attention kernels
- [x] Create `MultiModelRecorder` with shared prefix support
- [ ] Benchmark: measure KV reuse savings

### Phase 3: Multi-Expert Routing
- [x] Implement `MultiModelLoader` for base + N adapters
- [x] Add `BufferPool` partitioning
- [x] Create `ExpertRouter` (embedding-based selection)
- [ ] Basic round-robin expert selection

### Phase 4: Parallel Execution
- [ ] Multiple command encoders per expert
- [ ] Workgroup scheduling for concurrent execution
- [x] Output combination strategies (voting, weighted avg)

### Phase 5: Genetic Evolution
- [ ] Integrate with ArenaHarness for expert competition
- [ ] Implement topology mutations
- [ ] LoRA adapter evolution via merging
- [ ] Store winning configurations in ReflectionStore

---

## 13. Research Validation

The multi-FunctionGemma architecture is strongly supported by 2024-2025 research on multi-agent code generation and self-improving systems.

### 13.1 Multi-Agent Code Generation

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| [CodeCoR (2025)](https://www.researchgate.net/publication/388029025) | **29-47% Pass@1 improvement** using multi-agent reflection with analyst/coder/tester roles | Validates role-based expert specialization in ring topology |
| [Self-Planning Code Generation (ASE 2024)](https://dl.acm.org/doi/10.1145/3672456) | **25% improvement** by decomposing tasks before implementation | Supports scaffold-first approach in tree topology |
| [Multi-Agent Design Topologies (2025)](https://arxiv.org/html/2502.02533v1) | Ring/tree/mesh topologies outperform single-agent on complex tasks | Directly validates topology selection strategy |

### 13.2 Evolutionary Self-Improvement

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| [SOAR (2025)](https://arxiv.org/pdf/2507.14172) | Self-improving program synthesis without human solutions; competitive ARC performance | Validates genetic evolution of network configurations |
| [Genetic Improvement of LLM Code (EuroGP 2024)](https://link.springer.com/chapter/10.1007/978-3-031-56957-9_7) | Grammatical Evolution statistically improves LLM-generated code across 25 problems | Supports mutation operators for code refinement |
| [LLaMEA (2024)](https://arxiv.org/html/2405.20132v1) | LLMs can recursively improve their own scaffolding code (STOP framework) | Directly applicable to REPLOID L1-L3 RSI |
| [Algorithm Discovery for RSI (2024)](https://arxiv.org/html/2410.15639) | LLMs invent algorithms that outperform human-designed techniques | Validates learned DAG topology evolution |

### 13.3 Self-Improving Data Flywheels

The [BPO Framework (2024)](https://arxiv.org/html/2510.07799) establishes a pattern for self-improving systems:

```
Bootstrap → Extrapolate → Refine → Persist
    ↑                                  │
    └──────────────────────────────────┘
```

This aligns with our evolution loop:
1. **Bootstrap:** Initial network topology from genetic algorithm
2. **Extrapolate:** Run experts on new tasks, collect outputs
3. **Refine:** Score outputs, mutate low performers
4. **Persist:** Store winning configurations in ReflectionStore

### 13.4 Implications for Implementation

Based on research findings, prioritize these implementation choices:

| Research Insight | Implementation Choice |
|------------------|----------------------|
| Role specialization improves Pass@1 by 29-47% | Use distinct LoRA adapters per expert role |
| Task decomposition adds 25% improvement | Scaffold node should always run first |
| Genetic improvement works across LLMs | Implement full mutation operator suite |
| Self-improving flywheels need persistence | ReflectionStore integration is critical |
| Ring topology excels at sequential tasks | Default to ring for code generation pipelines |

### 13.5 Benchmark Expectations

Based on CodeCoR and Self-Planning results, projected improvements over single FunctionGemma:

| Metric | Single FnG | Ring (4) | Tree (7) | With Evolution |
|--------|------------|----------|----------|----------------|
| Pass@1 | Baseline | +15-20% | +25-30% | +35-45% |
| Code Quality | Baseline | +10% | +20% | +25% |
| Test Coverage | Baseline | +25% | +30% | +40% |

*Note: Projections based on extrapolating CodeCoR (multi-agent) and SOAR (evolution) results to FunctionGemma scale.*

---

## 14. Future Directions

1. **Speculative Execution:** Draft node generates, verify node checks in parallel
2. **Federated Experts:** Share trained adapters across Doppler instances via WebRTC
3. **Continuous Learning:** Update adapter selection based on production outcomes
4. **Hierarchical LoRA:** Stack multiple small LoRAs instead of one large one
5. **Quantization-Aware Routing:** Route complex tasks to fp16 nodes, simple to q4
6. **Topology Search:** Use [GTD](https://arxiv.org/html/2510.07799) to learn optimal FnG network structure per task type
7. **LLM-Guided Evolution:** Use main brain to guide mutations rather than random ([GECCO 2024](https://dl.acm.org/doi/10.1145/3638529.3654178))

---

## 15. References

### Internal Documentation
- [REPLOID: FunctionGemma Integration](../../../../../reploid/doppler/reploid/docs/plans/FUNCTIONGEMMA_INTEGRATION.md)

### Architecture & Infrastructure
- [Guided Topology Diffusion (GTD)](https://arxiv.org/html/2510.07799)
- [Multi-Agent Collaboration Survey](https://arxiv.org/html/2501.06322v1)
- [Multi-Agent Design Topologies](https://arxiv.org/html/2502.02533v1)
- [Mixture of Experts in LLMs (NVIDIA)](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/)
- [Neptune: Mixture of Experts Guide](https://neptune.ai/blog/mixture-of-experts-llms)
- [FunctionGemma for Multi-Agent Orchestration](https://dev.to/saikumaryava/beyond-mobile-actions-exploring-functiongemma-for-intelligent-multi-agent-orchestration-4jlf)
- [FunctionGemma Model Card](https://ai.google.dev/gemma/docs/functiongemma/model_card)

### Multi-Agent Code Generation
- [CodeCoR: LLM-Based Self-Reflective Multi-Agent Framework](https://www.researchgate.net/publication/388029025)
- [Self-Planning Code Generation (ASE 2024)](https://dl.acm.org/doi/10.1145/3672456)

### Evolutionary & Self-Improvement
- [SOAR: Self-Improving Program Synthesis](https://arxiv.org/pdf/2507.14172)
- [Genetic Improvement of LLM Code (EuroGP 2024)](https://link.springer.com/chapter/10.1007/978-3-031-56957-9_7)
- [LLaMEA: LLM Evolutionary Algorithm](https://arxiv.org/html/2405.20132v1)
- [Algorithm Discovery for RSI](https://arxiv.org/html/2410.15639)
- [Evolving Code with LLMs](https://arxiv.org/html/2401.07102v1)
- [GECCO 2024: LLM Evolution](https://dl.acm.org/doi/10.1145/3638529.3654017)
- [LLM-Guided Evolution](https://dl.acm.org/doi/10.1145/3638529.3654178)

### Caching & Performance
- [LMCache: KV Cache Management](https://arxiv.org/pdf/2510.09665)
- [Prompt Cache](https://arxiv.org/html/2412.19442v3)

---

*Last updated: December 2025*
