# FunctionGemma Integration & Hierarchical Memory Architecture

> Research compiled December 2025

This document explores integrating Google's FunctionGemma model into Doppler/Reploid and implementing hierarchical memory systems for effectively infinite context.

---

## Table of Contents

1. [FunctionGemma Overview](#functiongemma-overview)
2. [Integration Pathways](#integration-pathways)
3. [Hierarchical Memory Architectures](#hierarchical-memory-architectures)
4. [Proposed Implementation](#proposed-implementation)
5. [Alternatives Considered](#alternatives-considered)
6. [Sources](#sources)

---

## FunctionGemma Overview

Released December 18, 2025, FunctionGemma is a 270M parameter model specifically trained for function calling at the edge.

### Technical Specifications

| Attribute | Value |
|-----------|-------|
| Parameters | 270M (based on Gemma 3 270M) |
| Context Length | 32K tokens |
| Format | BF16 Safetensors, GGUF available |
| Size | ~288MB (int8), ~550MB RAM |
| Accuracy | 85% on Mobile Actions benchmark (vs 58% baseline) |
| Vocabulary | 256K tokens (efficient JSON tokenization) |

### Special Tokens

```
<start_function_declaration> / <end_function_declaration>  - Define a tool
<start_function_call> / <end_function_call>                - Model requests tool use
<start_function_response> / <end_function_response>        - Tool result to model
<escape>                                                   - String value delimiter
```

### Output Format

```
<start_function_call>call:function_name{param:<escape>value<escape>}<end_function_call>
```

### Function Schema Format

```json
{
  "type": "function",
  "function": {
    "name": "get_current_temperature",
    "description": "Gets the current temperature for a given location.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "The city name, e.g. San Francisco"
        }
      },
      "required": ["location"]
    }
  }
}
```

### Available Runtimes

| Runtime | Format | GPU | Browser Support |
|---------|--------|-----|-----------------|
| Transformers.js | ONNX | WebGPU (limited) | Yes |
| wllama | GGUF | No (WASM) | Yes |
| llama-cpp-wasm | GGUF | No (WASM) | Yes |
| Doppler (native) | RDRR | WebGPU | Yes |
| Ollama | GGUF | Metal/CUDA | No (server) |
| vLLM | Safetensors | CUDA | No (server) |

---

## Integration Pathways

### Path 1: Doppler Native (Recommended for Performance)

FunctionGemma is based on Gemma 3 270M - same architecture Doppler already supports.

**Implementation Steps:**

1. Convert weights to RDRR format:
   ```bash
   doppler convert --model functiongemma-270m-it --quant q4k
   ```

2. Add model config in `loader/doppler-loader.ts`:
   ```typescript
   'functiongemma-270m': {
     architecture: 'gemma3',
     hidden_size: 1024,
     num_heads: 8,
     num_layers: 18,
     vocab_size: 256000,
     // ... same as gemma3-270m
   }
   ```

3. Extend tokenizer for special tokens in `inference/tokenizer.ts`

4. Add output parser for function call extraction

**Pros:**
- Fastest inference (WebGPU acceleration)
- Unified runtime with other Doppler models
- Native browser support

**Cons:**
- Requires RDRR conversion tooling
- More implementation work

### Path 2: WASM Runtime (wllama)

Use existing GGUF weights with WebAssembly llama.cpp bindings.

**Implementation:**
```typescript
import { Wllama } from '@anthropic-ai/wllama';

const wllama = new Wllama();
await wllama.loadModel('functiongemma-270m-it-q4_k_m.gguf');
const output = await wllama.generate(prompt, { maxTokens: 128 });
```

**Pros:**
- GGUF weights already available
- Well-tested ecosystem
- Simpler integration

**Cons:**
- No GPU acceleration (~50 tok/s vs 125+ tok/s)
- WASM overhead
- Separate runtime from Doppler

### Path 3: Transformers.js (ONNX)

Google's official demo uses this approach.

**Pros:**
- Official support from Google
- ONNX ecosystem

**Cons:**
- Requires ONNX export
- Limited WebGPU support in ONNX Runtime Web

---

## When FunctionGemma Adds Value

### Valuable Use Cases

| Scenario | Value | Rationale |
|----------|-------|-----------|
| Fully offline mode | **High** | Replaces cloud LLM entirely |
| Browser automation tools | **High** | DOM/Web APIs stay local |
| Query routing/gatekeeper | **Medium** | Skip LLM for simple calls |
| Cost optimization (high volume) | **Medium** | Reduces API calls |
| Parallel tool execution | **Medium** | Mechanical translation while LLM reasons |

### NOT Valuable

| Scenario | Why Not |
|----------|---------|
| Tool calling intermediary | Main LLM already does this natively |
| General reasoning | Too small for complex tasks |
| Multi-step planning | Lacks reasoning capability |

### Recommended Architecture: Hybrid Two-Tier

```
User Query
    |
    v
+---------------------------+
| FunctionGemma (Doppler)   |  <-- Fast, local, private
| "Is this a simple tool?"  |
+---------------------------+
    |
    +---> YES: Execute tool directly (skip cloud LLM)
    |
    +---> NO: Escalate to main LLM for full reasoning
```

**Savings:** 80% of simple queries never hit expensive API.

---

## Hierarchical Memory Architectures

### The Problem

LLM context windows are finite. Naive approaches (truncation, sliding window) lose information. The solution: hierarchical storage + retrieval + summarization.

### Architecture 1: RAPTOR (Tree-Organized Retrieval)

```
                    [Global Summary]
                          |
            +-------------+-------------+
            v             v             v
      [Cluster A]   [Cluster B]   [Cluster C]
       Summary       Summary       Summary
          |             |             |
    +-----+-----+  +----+----+  +----+----+
    v     v     v  v    v    v  v    v    v
  [Chunk][Chunk]  [Chunk][Chunk] [Chunk][Chunk]
   Full   Full     Full   Full    Full   Full
```

**How it works:**
1. Embed all text chunks
2. Cluster similar chunks (UMAP + GMM)
3. Summarize each cluster
4. Recursively cluster and summarize summaries
5. At query time: retrieve from ANY level (collapsed tree search)

**Results:** 20% absolute accuracy improvement on QuALITY benchmark.

**Best for:** Document QA, static knowledge bases.

### Architecture 2: MemGPT (OS-Inspired Hierarchy)

```
+-----------------------------------------------------------+
|                 Main Context (RAM)                         |
|   Fixed window - what LLM "sees" during inference          |
|   +-------------+-------------+-----------------------+    |
|   | System      | Core Memory | Recent Messages       |    |
|   | Instructions| (Persona)   | (Working Memory)      |    |
|   +-------------+-------------+-----------------------+    |
+-----------------------------------------------------------+
                    ^                    |
                    | load               | evict + summarize
                    |                    v
+-----------------------------------------------------------+
|              External Context (Disk)                       |
|   +-----------------------+---------------------------+    |
|   |   Recall Memory       |    Archival Memory        |    |
|   |  (Conversation DB)    |   (Long-term Knowledge)   |    |
|   |   - Full messages     |   - Searchable facts      |    |
|   |   - Recursive sums    |   - User preferences      |    |
|   +-----------------------+---------------------------+    |
+-----------------------------------------------------------+
```

**Key mechanism - Recursive summarization on eviction:**
```python
evicted_messages = context.pop_oldest(n)
existing_summary = recall_memory.get_summary()
new_summary = LLM.summarize(existing_summary + evicted_messages)
recall_memory.store(evicted_messages, summary=new_summary)
```

**Best for:** Conversational agents, multi-session memory.

### Architecture 3: Cognitive Workspace (2025)

Most advanced approach with active memory management.

```
+--------------------------------------------------------------+
|                    ACTIVE MEMORY CONTROLLER                   |
|   (Metacognitive layer - predicts, consolidates, forgets)     |
+--------------------------------------------------------------+
         |              |              |              |
         v              v              v              v
+--------------+ +--------------+ +--------------+ +--------------+
|  Immediate   | |    Task      | |   Episodic   | |   Semantic   |
|  Scratchpad  | |   Buffer     | |    Cache     | |    Bridge    |
|    (8K)      | |   (64K)      | |   (256K)     | |    (1M+)     |
|              | |              | |              | |              |
| Active       | | Problem      | | Temporal     | | External     |
| reasoning    | | state        | | history      | | knowledge    |
+--------------+ +--------------+ +--------------+ +--------------+
```

**Three innovations:**

| Feature | RAG | MemGPT | Cognitive Workspace |
|---------|-----|--------|---------------------|
| Memory Reuse | 0% | 10-20% | **54-60%** |
| State Persistence | None | Session | **Continuous** |
| Retrieval | Passive | Reactive | **Anticipatory** |
| Forgetting | None | LRU | **Adaptive curves** |

**Key mechanisms:**
- **Anticipatory retrieval:** Predicts future needs, pre-fetches
- **Selective consolidation:** Compresses frequently accessed patterns
- **Adaptive forgetting:** Task-specific decay curves

**Best for:** Complex multi-session tasks, long-running agents.

### Architecture 4: EM-LLM (Human Episodic Memory)

```
Token Stream --> Bayesian Surprise Detection --> Event Boundaries
                                                    |
                                                    v
                                        +---------------------+
                                        |   Event Memory      |
                                        |   +-----+ +-----+   |
                                        |   | E1  | | E2  |   |
                                        |   +-----+ +-----+   |
                                        |   +-----+ +-----+   |
                                        |   | E3  | | E4  |   |
                                        |   +-----+ +-----+   |
                                        +---------------------+
                                                    |
                            +-----------------------+-----------------------+
                            v                       v                       v
                      Similarity              Temporal                 Causal
                      Retrieval              Contiguity               Linking
```

**How it differs:**
- No fixed chunk sizes
- Detects event boundaries using Bayesian surprise
- Retrieval mimics human free recall (temporal contiguity)
- No fine-tuning required

**Best for:** Narrative/temporal tasks, conversation history.

---

## Proposed Implementation

### For Doppler: FunctionGemma Provider

```typescript
// doppler/function-provider.ts

interface FunctionCall {
  name: string;
  args: Record<string, unknown>;
  confidence: number;
}

export class DopplerFunctionProvider {
  private pipeline: InferencePipeline;

  async init() {
    this.pipeline = await createPipeline({
      model: 'functiongemma-270m-q4k',
      maxTokens: 128,
    });
  }

  async call(
    query: string,
    tools: ToolSchema[]
  ): Promise<FunctionCall | null> {
    const prompt = this.buildPrompt(query, tools);
    const output = await this.pipeline.generate(prompt, {
      stopTokens: ['<end_function_call>'],
    });
    return this.parseOutput(output);
  }

  private buildPrompt(query: string, tools: ToolSchema[]): string {
    const toolDefs = tools.map(t =>
      `<start_function_declaration>${JSON.stringify(t)}<end_function_declaration>`
    ).join('\n');

    return `<start_of_turn>developer
You are a model that can do function calling with the following functions:
${toolDefs}<end_of_turn>
<start_of_turn>user
${query}<end_of_turn>
<start_of_turn>model
`;
  }

  private parseOutput(output: string): FunctionCall | null {
    const match = output.match(
      /<start_function_call>call:(\w+)\{(.+)\}<end_function_call>/
    );
    if (!match) return null;

    const name = match[1];
    const argsStr = match[2];
    const args = this.parseEscapedArgs(argsStr);

    return { name, args, confidence: 0.85 };
  }

  private parseEscapedArgs(argsStr: string): Record<string, unknown> {
    // Parse param:<escape>value<escape> format
    const args: Record<string, unknown> = {};
    const regex = /(\w+):<escape>([^<]*)<escape>/g;
    let match;
    while ((match = regex.exec(argsStr)) !== null) {
      args[match[1]] = match[2];
    }
    return args;
  }
}
```

### For Doppler: Hierarchical Memory Manager

```typescript
// doppler/memory/manager.ts

interface MemoryTier {
  name: string;
  maxTokens: number;
  storage: 'context' | 'indexeddb' | 'opfs';
}

const TIERS: MemoryTier[] = [
  { name: 'working', maxTokens: 8000, storage: 'context' },
  { name: 'episodic', maxTokens: 64000, storage: 'indexeddb' },
  { name: 'semantic', maxTokens: 1000000, storage: 'opfs' },
];

export class HierarchicalMemoryManager {
  private working: Message[] = [];
  private episodicSummary: string = '';
  private embeddingIndex: EmbeddingIndex;

  async evict(count: number): Promise<void> {
    const evicted = this.working.splice(0, count);

    // Recursive summarization
    const newSummary = await this.summarize(
      this.episodicSummary,
      evicted
    );
    this.episodicSummary = newSummary;

    // Store full messages + embeddings
    await this.storeEpisodic(evicted);
    await this.embeddingIndex.add(evicted);
  }

  async retrieve(query: string, maxTokens: number): Promise<Context> {
    const context: Context = { summary: '', episodes: [] };
    let tokens = 0;

    // Always include summary
    context.summary = this.episodicSummary;
    tokens += estimateTokens(this.episodicSummary);

    // Semantic search for relevant episodes
    const relevant = await this.embeddingIndex.search(query, 20);

    for (const result of relevant) {
      const episodeTokens = estimateTokens(result.content);
      if (tokens + episodeTokens > maxTokens) break;
      context.episodes.push(result);
      tokens += episodeTokens;
    }

    return context;
  }

  private async summarize(
    existing: string,
    messages: Message[]
  ): Promise<string> {
    // Use FunctionGemma or main LLM for summarization
    const prompt = `Previous summary:
${existing}

New messages:
${messages.map(m => `${m.role}: ${m.content}`).join('\n')}

Update the summary to include new information concisely:`;

    return await this.llm.generate(prompt, { temperature: 0 });
  }
}
```

### For Reploid: Integration Layer

```javascript
// reploid/core/memory-manager.js

const MemoryManager = {
  metadata: {
    id: 'MemoryManager',
    dependencies: ['VFS', 'EmbeddingStore', 'LLMClient', 'DopplerProvider?']
  },

  factory: (deps) => {
    const { VFS, EmbeddingStore, LLMClient, DopplerProvider } = deps;

    const WORKING_LIMIT = 8000; // tokens
    let workingMemory = [];
    let episodicSummary = '';

    const evictOldest = async (count) => {
      const evicted = workingMemory.splice(0, count);

      // Use local model if available, else cloud
      const summarizer = DopplerProvider || LLMClient;

      const newSummary = await summarizer.generate({
        prompt: buildSummaryPrompt(episodicSummary, evicted),
        temperature: 0
      });

      episodicSummary = newSummary;

      // Persist
      await VFS.write('/memory/episodes/summary.md', newSummary);
      await VFS.append('/memory/episodes/full.jsonl',
        evicted.map(JSON.stringify).join('\n'));

      // Index for retrieval
      await EmbeddingStore.add(evicted.map(m => ({
        text: m.content,
        metadata: { timestamp: Date.now(), role: m.role }
      })));
    };

    const retrieve = async (query, options = {}) => {
      const { maxTokens = 4000 } = options;

      let context = [];
      let tokens = 0;

      // 1. Include summary
      if (episodicSummary) {
        context.push({ type: 'summary', content: episodicSummary });
        tokens += estimateTokens(episodicSummary);
      }

      // 2. Semantic search
      const relevant = await EmbeddingStore.search(query, { limit: 20 });

      for (const result of relevant) {
        const t = estimateTokens(result.content);
        if (tokens + t > maxTokens) break;
        context.push({ type: 'episode', ...result });
        tokens += t;
      }

      return context;
    };

    return {
      add: (message) => workingMemory.push(message),
      evictOldest,
      retrieve,
      getWorking: () => [...workingMemory],
      getSummary: () => episodicSummary,
    };
  }
};
```

---

## Alternatives Considered

### Alternative 1: Reversible Compression via Temperature 0

**Hypothesis:** Summarize with temperature 0 so output is deterministic, then "reverse engineer" to recover full context.

**Why it doesn't work:**
- Temperature 0 = deterministic forward mapping, NOT reversible
- Many inputs can produce the same summary (many-to-one)
- Information theory: cannot losslessly compress below entropy
- Analogy: SHA-256 is deterministic but not reversible

**Verdict:** Rejected. Use storage + retrieval instead.

### Alternative 2: FunctionGemma as Tool-Calling Intermediary

**Hypothesis:** Route all tool calls through FunctionGemma even when using a capable main LLM.

**Why it's problematic:**
- Main LLM (Gemini, Claude, GPT-4) already does tool calling natively
- Adding FunctionGemma as middleman just adds latency
- No benefit if main LLM already decided the tool call

**When it IS useful:**
- Fully offline mode (no main LLM)
- Privacy-sensitive browser tools
- Cost optimization (skip API for simple calls)
- Query routing (decide if LLM needed at all)

**Verdict:** Use selectively, not universally.

### Alternative 3: Pure RAG (No Summaries)

**Hypothesis:** Just store everything and retrieve relevant chunks.

**Why summaries help:**
- RAG has 0% memory reuse (stateless)
- No narrative coherence across retrievals
- Missing high-level context
- Each query processed independently

**Hybrid is better:** Summary provides global context, RAG provides specific details.

**Verdict:** Use hierarchical (summary + retrieval), not pure RAG.

### Alternative 4: Sliding Window Only

**Hypothesis:** Just keep last N tokens, drop the rest.

**Problems:**
- Loses all historical context
- No way to reference earlier conversation
- Breaks multi-session continuity

**Verdict:** Rejected for agents. Maybe OK for single-turn tasks.

### Alternative 5: Infini-Attention (Google 2024)

**What it is:** Compressive memory integrated into attention mechanism.

**Pros:**
- 1B model scales to 1M context
- No external retrieval needed
- Trained end-to-end

**Cons:**
- Requires model modification
- Not available for arbitrary models
- Training overhead

**Verdict:** Interesting for future Doppler models, not for FunctionGemma integration.

---

## Comparison Matrix

| Approach | Complexity | Performance | Memory Reuse | Best For |
|----------|------------|-------------|--------------|----------|
| RAPTOR | Medium | High | N/A (static) | Document QA |
| MemGPT | Medium | Good | 10-20% | Conversational agents |
| Cognitive Workspace | High | Excellent | 54-60% | Complex multi-session |
| EM-LLM | Medium | Good | Variable | Temporal/narrative |
| Pure RAG | Low | Moderate | 0% | Simple retrieval |
| Sliding Window | Trivial | Poor | 0% | Single-turn only |

---

## Implementation Roadmap

### Phase 1: FunctionGemma in Doppler
1. Add RDRR conversion for functiongemma-270m-it
2. Extend tokenizer for special tokens
3. Create DopplerFunctionProvider with output parsing
4. Benchmark: target <20ms for simple tool calls

### Phase 2: Hierarchical Memory
1. Implement working memory eviction with recursive summarization
2. Add episodic storage in IndexedDB/OPFS
3. Integrate with existing EmbeddingStore for retrieval
4. Create unified MemoryManager API

### Phase 3: Reploid Integration
1. Connect MemoryManager to agent-loop
2. Add FunctionRouter for hybrid routing
3. Create offline mode using FunctionGemma
4. Benchmark end-to-end latency and accuracy

### Phase 4: Advanced Features
1. Anticipatory retrieval (predict future needs)
2. Adaptive forgetting curves
3. RAPTOR-style knowledge tree for semantic memory
4. Cross-session continuity

---

## Sources

### FunctionGemma
- [Google Blog: FunctionGemma](https://blog.google/technology/developers/functiongemma/)
- [FunctionGemma Model Card](https://ai.google.dev/gemma/docs/functiongemma/model_card)
- [FunctionGemma Formatting & Best Practices](https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices)
- [Function Calling with HuggingFace](https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf)
- [HuggingFace: google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)
- [GGUF: ggml-org/functiongemma-270m-it-GGUF](https://huggingface.co/ggml-org/functiongemma-270m-it-GGUF)
- [VentureBeat: Google releases FunctionGemma](https://venturebeat.com/technology/google-releases-functiongemma-a-tiny-edge-model-that-can-control-mobile)
- [Edge AI Vision: FunctionGemma](https://www.edge-ai-vision.com/2025/12/google-releases-functiongemma-lightweight-function-calling-model-aimed-at-on-device-agents/)

### Hierarchical Memory & Context Management
- [RAPTOR Paper (ICLR 2024)](https://arxiv.org/abs/2401.18059)
- [RAPTOR GitHub](https://github.com/parthsarthi03/raptor)
- [MemGPT Paper](https://arxiv.org/abs/2310.08560)
- [Letta (MemGPT) Documentation](https://docs.letta.com/)
- [Cognitive Workspace Paper (2025)](https://arxiv.org/abs/2508.13171)
- [EM-LLM Paper](https://arxiv.org/abs/2407.09450)
- [EM-LLM GitHub](https://github.com/em-llm/EM-LLM-model)
- [HMT: Hierarchical Memory Transformer](https://arxiv.org/abs/2405.06067)
- [MemTree: Dynamic Tree Memory](https://arxiv.org/abs/2410.14052)

### Context Window Research
- [Awesome LLM Long Context Modeling](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling)
- [Google Infini-Attention Paper](https://arxiv.org/abs/2404.07143)
- [GraphRAG Paper (Microsoft)](https://arxiv.org/abs/2404.16130)
- [JetBrains: Context Management Research](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)
- [Letta Blog: Agent Memory](https://www.letta.com/blog/agent-memory)

### Small Model Routing & Orchestration
- [NVIDIA: Train Small Orchestration Agents](https://developer.nvidia.com/blog/train-small-orchestration-agents-to-solve-big-problems)
- [Microsoft: Function Calling with SLMs](https://techcommunity.microsoft.com/blog/educatordeveloperblog/function-calling-with-small-language-models/4472720)
- [LogRocket: Small Language Models Future](https://blog.logrocket.com/small-language-models/)
- [RouterArena Paper](https://arxiv.org/abs/2510.00202)

### Browser Runtimes
- [wllama GitHub](https://github.com/ngxson/wllama)
- [llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- [Transformers.js](https://huggingface.co/docs/transformers.js)

---

*Last updated: December 2025*
