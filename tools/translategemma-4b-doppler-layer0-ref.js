#!/usr/bin/env node
/**
 * TranslateGemma 4B — Doppler layer-0 activation dump.
 * Feeds the exact same 44-token prompt as the HF reference script,
 * bypassing the harness to avoid prompt contract checks.
 *
 * Runs in headless browser (WebGPU) via Puppeteer.
 */

import { runBrowserCommandInNode } from '../src/tooling/node-browser-command-runner.js';

// The exact formatted prompt that produces 44 tokens in both HF and Doppler tokenizers
const PROMPT = '<start_of_turn>user\nInstructions: translate the following to French.\nProduce only the French translation, without any additional explanations or commentary. Please translate the following English text into French:\n\n\nHello world.<end_of_turn>\n<start_of_turn>model\n';

const result = await runBrowserCommandInNode({
  command: 'debug',
  request: {
    workload: 'inference',
    modelId: 'translategemma-4b-it-q4k-ehf16-af32',
  },
  runtimeConfig: {
    extends: 'profiles/throughput',
    model: 'translategemma-4b-it-q4k-ehf16-af32',
    runtime: {
      shared: {
        tooling: { intent: 'verify' },
        debug: {
          logLevel: { defaultLogLevel: 'debug' },
          trace: { enabled: true, categories: ['attn', 'ffn', 'embed', 'logits'] },
          probes: [
            { id: 'embed_t0', stage: 'embed_out', tokens: [0], dims: [0,1,2,3,4,5,6,7] },
            { id: 'embed_last', stage: 'embed_out', tokens: [-1], dims: [0,1,2,3,4,5,6,7] },
            { id: 'L0_qproj', stage: 'q_proj', layers: [0], tokens: [-1], dims: [0,1,2,3,4,5,6,7] },
            { id: 'L0_kproj', stage: 'k_proj', layers: [0], tokens: [-1], dims: [0,1,2,3,4,5,6,7] },
            { id: 'L0_vproj', stage: 'v_proj', layers: [0], tokens: [-1], dims: [0,1,2,3,4,5,6,7] },
            { id: 'L0_qrope', stage: 'q_rope', layers: [0], tokens: [-1], dims: [0,1,2,3,4,5,6,7] },
            { id: 'L0_krope', stage: 'k_rope', layers: [0], tokens: [-1], dims: [0,1,2,3,4,5,6,7] },
            { id: 'L0_attn', stage: 'attn_out', layers: [0], tokens: [-1], dims: [0,1,2,3,4,5,6,7] },
            { id: 'L0_post', stage: 'post_attn', layers: [0], tokens: [-1], dims: [0,1,2,3,4,5,6,7] },
            { id: 'L0_out', stage: 'layer_out', layers: [0], tokens: [-1], dims: [0,1,2,3,4,5,6,7] },
          ],
        },
      },
      inference: {
        prompt: PROMPT,
        chatTemplate: { enabled: false },
        generation: { maxTokens: 4 },
        session: {
          decodeLoop: {
            disableCommandBatching: true,
          },
        },
        sampling: { temperature: 0, topK: 1, topP: 1 },
      },
    },
  },
});

console.log(JSON.stringify(result, null, 2));
