#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { SVG_FONTS, SVG_THEME } from './svg-theme.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const OUTPUT_PATH = path.join(__dirname, '..', '..', 'docs', 'architecture-overview.svg');

const {
  palette,
} = SVG_THEME;
const {
  uiCss: architectureUIFont,
  monoCss: architectureMonoFont,
} = SVG_FONTS;
const architecture = palette.architecture;
const textPaint = `paint-order: stroke fill; stroke: ${SVG_THEME.textStroke.color}; stroke-width: ${SVG_THEME.textStroke.width}; stroke-linejoin: ${SVG_THEME.textStroke.lineJoin};`;

const body = `  <defs>
  <style>
    .title { font: 700 32px ${architectureUIFont}; fill: ${palette.text}; ${textPaint} }
    .section { font: 700 24px ${architectureUIFont}; fill: ${palette.text}; ${textPaint} }
    .node-title { font: 700 20px ${architectureUIFont}; fill: ${palette.text}; ${textPaint} }
    .node-text { font: 500 17px ${architectureUIFont}; fill: ${palette.text}; ${textPaint} }
    .code-text { font: 500 17px ${architectureMonoFont}; fill: ${palette.text}; ${textPaint} }
    .edge-label { font: 500 15px ${architectureUIFont}; fill: ${palette.text}; ${textPaint} }
    .column-left { fill: url(#col-left-grad); stroke: ${architecture.loadBorder}; stroke-width: 4.4; }
    .column-right { fill: url(#col-right-grad); stroke: ${architecture.inferBorder}; stroke-width: 4.4; }
    .node { fill: transparent; stroke-width: 2; }
    .node-load { stroke: ${architecture.nodeLoad}; }
    .node-infer { stroke: ${architecture.nodeInfer}; }
    .edge { stroke: ${architecture.edge}; stroke-width: 2.5; fill: none; }
    .edge-dashed { stroke: ${architecture.inferBorder}; stroke-width: 2.5; fill: none; stroke-dasharray: 8 8; }
  </style>

  <linearGradient id="col-left-grad" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="${architecture.columnLeftStart}" />
    <stop offset="60%" stop-color="${architecture.columnLeftMid}" />
    <stop offset="100%" stop-color="${architecture.columnLeftEnd}" />
  </linearGradient>

  <linearGradient id="col-right-grad" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="${architecture.columnRightStart}" />
    <stop offset="60%" stop-color="${architecture.columnRightMid}" />
    <stop offset="100%" stop-color="${architecture.columnRightEnd}" />
  </linearGradient>

  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
    <path d="M 0 0 L 12 6 L 0 12 z" fill="${architecture.arrow}" />
  </marker>

  <marker id="arrow-muted" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
    <path d="M 0 0 L 12 6 L 0 12 z" fill="${architecture.arrowMuted}" />
  </marker>
</defs>`;

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="700" viewBox="0 0 1400 700" role="img" aria-labelledby="title desc">
  <title id="title">Doppler Architecture Overview</title>
  <desc id="desc">Two-column architecture: internal load-model orchestration on the left and public inference calls on the right, each with its own Config as Code node and a shared WebGPU lifecycle link.</desc>

${body}

  <text x="700" y="52" text-anchor="middle" class="title">Doppler Architecture Overview</text>

  <rect x="70" y="70" width="610" height="560" rx="0" class="column-left" />
  <rect x="720" y="70" width="610" height="560" rx="0" class="column-right" />

  <text x="375" y="120" text-anchor="middle" class="section">Load Model (one-time)</text>
  <text x="1025" y="120" text-anchor="middle" class="section">Inference (per-session)</text>

  <rect x="125" y="150" width="500" height="82" rx="12" class="node node-load" />
  <text x="375" y="184" text-anchor="middle" class="node-title">Config</text>
  <text x="375" y="210" text-anchor="middle" class="node-text">weight/attention dtypes, kernels</text>

  <rect x="125" y="270" width="500" height="72" rx="12" class="node node-load" />
  <text x="375" y="302" text-anchor="middle" class="node-title">Internal Call</text>
  <text x="375" y="326" text-anchor="middle" class="code-text">&#96;createInitializedPipeline() -> pipeline.loadModel()&#96;</text>

  <rect x="125" y="380" width="500" height="82" rx="12" class="node node-load" />
  <text x="375" y="414" text-anchor="middle" class="node-title">Model Loader</text>
  <text x="375" y="440" text-anchor="middle" class="node-text">shard cache + dequant + upload</text>

  <rect x="125" y="520" width="240" height="82" rx="12" class="node node-load" />
  <text x="245" y="554" text-anchor="middle" class="node-title">Storage</text>
  <text x="245" y="580" text-anchor="middle" class="node-text">RDRR / OPFS / Downloader</text>

  <rect x="385" y="520" width="240" height="82" rx="12" class="node node-load" />
  <text x="505" y="554" text-anchor="middle" class="node-title">WebGPU</text>
  <text x="505" y="580" text-anchor="middle" class="node-text">upload + buffers</text>

  <path d="M 375 232 L 375 270" class="edge" marker-end="url(#arrow)" />
  <path d="M 375 342 L 375 380" class="edge" marker-end="url(#arrow)" />

  <path d="M 375 462 L 375 488 L 245 488 L 245 520" class="edge" marker-end="url(#arrow)" />
  <path d="M 505 462 L 505 520" class="edge" marker-end="url(#arrow)" />

  <rect x="775" y="150" width="500" height="82" rx="12" class="node node-infer" />
  <text x="1025" y="184" text-anchor="middle" class="node-title">Config</text>
  <text x="1025" y="210" text-anchor="middle" class="node-text">attention mechanism, batching, kernel overrides</text>

  <rect x="775" y="270" width="500" height="72" rx="12" class="node node-infer" />
  <text x="1025" y="302" text-anchor="middle" class="node-title">Public API Call</text>
  <text x="1025" y="326" text-anchor="middle" class="code-text">&#96;generate() / dopplerChat()&#96;</text>

  <rect x="775" y="380" width="500" height="104" rx="12" class="node node-infer" />
  <text x="1025" y="415" text-anchor="middle" class="node-title">Inference Pipeline</text>
  <text x="1025" y="442" text-anchor="middle" class="node-text">tokenizer -&gt; KV cache -&gt; attention/FFN -&gt; sampling</text>

  <rect x="775" y="530" width="500" height="82" rx="12" class="node node-infer" />
  <text x="1025" y="564" text-anchor="middle" class="node-title">WebGPU</text>
  <text x="1025" y="590" text-anchor="middle" class="node-text">WGSL kernels + buffer pools</text>

  <path d="M 1025 232 L 1025 270" class="edge" marker-end="url(#arrow)" />
  <path d="M 1025 342 L 1025 380" class="edge" marker-end="url(#arrow)" />
  <path d="M 1025 484 L 1025 530" class="edge" marker-end="url(#arrow)" />

  <path d="M 625 561 L 775 561" class="edge-dashed" marker-end="url(#arrow-muted)" />
  <text x="700" y="545" text-anchor="middle" class="edge-label">shared WebGPU lifecycle</text>
</svg>`;

fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });
fs.writeFileSync(OUTPUT_PATH, `${svg}
`, 'utf-8');
console.log(`wrote ${OUTPUT_PATH}`);
