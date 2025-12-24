# FunctionGemma Integration Plan

> Research compiled December 2025

This document covers integrating Google's FunctionGemma model into Doppler for local function calling and summarization inference.

---

## Table of Contents

1. [FunctionGemma Overview](#functiongemma-overview)
2. [Integration Pathways](#integration-pathways)
3. [Doppler Implementation](#doppler-implementation)
4. [Use Cases](#use-cases)
5. [Sources](#sources)

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

---

## Integration Pathways

### Path 1: Doppler Native (Recommended)

FunctionGemma is based on Gemma 3 270M - same architecture Doppler already supports.

**Implementation Steps:**

1. **Convert weights** to RDRR format:
   ```bash
   doppler convert --model functiongemma-270m-it --quant q4k
   ```

2. **Add model config** in `loader/doppler-loader.ts`:
   ```typescript
   'functiongemma-270m': {
     architecture: 'gemma3',
     hidden_size: 1024,
     num_heads: 8,
     num_layers: 18,
     vocab_size: 256000,
   }
   ```

3. **Extend tokenizer** for special tokens in `inference/tokenizer.ts`

4. **Add output parser** for function call extraction

**Pros:**
- Fastest inference (WebGPU acceleration)
- Unified runtime with other Doppler models
- Native browser support

**Cons:**
- Requires RDRR conversion
- More implementation work

### Path 2: WASM Runtime (wllama)

Use existing GGUF weights with WebAssembly llama.cpp bindings.

```typescript
import { Wllama } from '@anthropic-ai/wllama';

const wllama = new Wllama();
await wllama.loadModel('functiongemma-270m-it-q4_k_m.gguf');
const output = await wllama.generate(prompt, { maxTokens: 128 });
```

**Pros:** GGUF weights available, simpler integration
**Cons:** No GPU acceleration (~50 tok/s vs 125+ tok/s)

### Path 3: Transformers.js (ONNX)

Google's official demo uses this approach.

**Pros:** Official support
**Cons:** Requires ONNX export, limited WebGPU

---

## Doppler Implementation

### DopplerFunctionProvider

```typescript
// doppler/providers/function-provider.ts

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

### Tokenizer Extension

```typescript
// inference/tokenizer.ts - additions

const FUNCTION_TOKENS = {
  START_FUNC_CALL: '<start_function_call>',
  END_FUNC_CALL: '<end_function_call>',
  START_FUNC_DECL: '<start_function_declaration>',
  END_FUNC_DECL: '<end_function_declaration>',
  START_FUNC_RESP: '<start_function_response>',
  END_FUNC_RESP: '<end_function_response>',
  ESCAPE: '<escape>',
};

// Add to vocabulary during tokenizer init
const addFunctionTokens = (vocab: Map<string, number>) => {
  let nextId = vocab.size;
  for (const token of Object.values(FUNCTION_TOKENS)) {
    if (!vocab.has(token)) {
      vocab.set(token, nextId++);
    }
  }
};
```

---

## Use Cases

### 1. Local Summarization for Memory Management

FunctionGemma can power the summarization step in Reploid's hierarchical memory:

```typescript
// When evicting messages from working memory
const summary = await DopplerFunctionProvider.generate({
  prompt: `Summarize: ${messages.join('\n')}`,
  temperature: 0
});
// No cloud API call needed - fully local
```

### 2. Offline Tool Routing

Decide which tool to call without cloud LLM:

```
User: "List files in /src"
  |
  v
FunctionGemma (local, <20ms)
  |
  v
call:ListFiles{path:<escape>/src<escape>}
  |
  v
Execute tool directly (skip cloud LLM)
```

### 3. Browser Automation

DOM/Web API function calling:

```typescript
const browserTools = [
  { name: 'click_element', description: 'Click element by selector', ... },
  { name: 'fill_input', description: 'Fill input field', ... },
  { name: 'navigate', description: 'Navigate to URL', ... },
];

const action = await DopplerFunctionProvider.call(
  "Click the submit button",
  browserTools
);
// Returns: { name: 'click_element', args: { selector: 'button[type=submit]' } }
```

### 4. Privacy-Sensitive Operations

Keep sensitive tool calls local:

```
Cloud LLM: General reasoning, planning
FunctionGemma (local): Clipboard, localStorage, DOM state, credentials
```

---

## When FunctionGemma Adds Value

| Scenario | Value | Rationale |
|----------|-------|-----------|
| Fully offline mode | **High** | Replaces cloud LLM entirely |
| Local summarization | **High** | Memory eviction without API calls |
| Browser automation | **High** | DOM/Web APIs stay local |
| Query routing | **Medium** | Skip LLM for simple calls |
| Privacy-sensitive tools | **High** | Data never leaves browser |

### NOT Valuable

| Scenario | Why Not |
|----------|---------|
| Tool calling intermediary | Main LLM already does this |
| General reasoning | Too small for complex tasks |

---

## Implementation Roadmap

### Phase 1: Basic Integration
- [ ] Add RDRR conversion for functiongemma-270m-it
- [ ] Extend tokenizer for special tokens
- [ ] Create DopplerFunctionProvider
- [ ] Benchmark: target <20ms for simple tool calls

### Phase 2: Summarization Support
- [ ] Add summarization mode (no function schema)
- [ ] Integrate with Reploid MemoryManager for eviction
- [ ] Benchmark summarization quality vs cloud models

### Phase 3: Web Agent
- [ ] Define browser tool schemas
- [ ] DOM inspection integration
- [ ] Action execution loop demo

---

## Sources

### FunctionGemma
- [Google Blog: FunctionGemma](https://blog.google/technology/developers/functiongemma/)
- [FunctionGemma Model Card](https://ai.google.dev/gemma/docs/functiongemma/model_card)
- [Formatting & Best Practices](https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices)
- [HuggingFace: google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)
- [GGUF: ggml-org/functiongemma-270m-it-GGUF](https://huggingface.co/ggml-org/functiongemma-270m-it-GGUF)

### Browser Runtimes
- [wllama GitHub](https://github.com/ngxson/wllama)
- [llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- [Transformers.js](https://huggingface.co/docs/transformers.js)

### Related (Sibling Repo)
- Reploid: `docs/MEMORY_ARCHITECTURE.md` - Hierarchical memory architecture (RAPTOR/MemGPT/Cognitive Workspace)

---

*Last updated: December 2025*
