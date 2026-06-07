#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { SVG_FONTS, SVG_THEME } from './svg-theme.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const OUTPUT_PATH = path.join(__dirname, '..', '..', 'docs', 'architecture-overview.svg');

const {
  palette,
  radius,
  stroke,
} = SVG_THEME;
const {
  uiCss: architectureUIFont,
  monoCss: architectureMonoFont,
} = SVG_FONTS;
const architecture = palette.architecture;

const body = `  <defs>
  <style>
    text { fill: ${palette.text}; font-family: ${architectureUIFont}; letter-spacing: 0; }
    .title { font-size: 32px; font-weight: 700; font-family: ${architectureUIFont}; }
    .section { font-size: 24px; font-weight: 700; font-family: ${architectureUIFont}; }
    .node-title { font-size: 20px; font-weight: 700; font-family: ${architectureUIFont}; }
    .node-text { font-size: 17px; font-weight: 500; font-family: ${architectureUIFont}; }
    .code-text { font-size: 15px; font-weight: 500; font-family: ${architectureMonoFont}; }
    .edge-label { font-size: 15px; font-weight: 500; font-family: ${architectureUIFont}; fill: ${palette.muted}; }
    .column-left { fill: ${palette.panel}; stroke: ${architecture.loadBorder}; stroke-width: ${stroke.normal}; }
    .column-right { fill: ${palette.panel}; stroke: ${architecture.inferBorder}; stroke-width: ${stroke.normal}; }
    .node { fill: ${palette.panelAlt}; stroke-width: ${stroke.thin}; }
    .node-load { stroke: ${architecture.nodeLoad}; }
    .node-infer { stroke: ${architecture.nodeInfer}; }
    .edge { stroke: ${architecture.arrow}; stroke-width: ${stroke.normal}; fill: none; }
    .edge-dashed { stroke: ${architecture.arrowMuted}; stroke-width: ${stroke.normal}; fill: none; stroke-dasharray: 8 8; }
  </style>

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

  <rect x="0" y="0" width="1400" height="700" fill="${palette.bg}" />

  <text x="700" y="52" text-anchor="middle" class="title">Doppler Architecture Overview</text>

  <rect x="70" y="70" width="610" height="560" rx="${radius.panel}" fill="${palette.panel}" stroke="${architecture.loadBorder}" stroke-width="${stroke.normal}" />
  <rect x="720" y="70" width="610" height="560" rx="${radius.panel}" fill="${palette.panel}" stroke="${architecture.inferBorder}" stroke-width="${stroke.normal}" />

  <text x="375" y="120" text-anchor="middle" class="section">Load Model (one-time)</text>
  <text x="1025" y="120" text-anchor="middle" class="section">Inference (per-session)</text>

  <rect x="125" y="150" width="500" height="82" rx="${radius.panel}" fill="${palette.panelAlt}" stroke="${architecture.nodeLoad}" stroke-width="${stroke.thin}" />
  <text x="375" y="184" text-anchor="middle" class="node-title">Config</text>
  <text x="375" y="210" text-anchor="middle" class="node-text">weight/attention dtypes, kernels</text>

  <rect x="125" y="270" width="500" height="72" rx="${radius.panel}" fill="${palette.panelAlt}" stroke="${architecture.nodeLoad}" stroke-width="${stroke.thin}" />
  <text x="375" y="302" text-anchor="middle" class="node-title">Internal Call</text>
  <text x="375" y="322" text-anchor="middle" class="code-text">&#96;createInitializedPipeline()&#96;</text>
  <text x="375" y="338" text-anchor="middle" class="code-text">&#96;pipeline.loadModel()&#96;</text>

  <rect x="125" y="380" width="500" height="82" rx="${radius.panel}" fill="${palette.panelAlt}" stroke="${architecture.nodeLoad}" stroke-width="${stroke.thin}" />
  <text x="375" y="414" text-anchor="middle" class="node-title">Model Loader</text>
  <text x="375" y="440" text-anchor="middle" class="node-text">shard cache + dequant + upload</text>

  <rect x="125" y="520" width="240" height="82" rx="${radius.panel}" fill="${palette.panelAlt}" stroke="${architecture.nodeLoad}" stroke-width="${stroke.thin}" />
  <text x="245" y="554" text-anchor="middle" class="node-title">Storage</text>
  <text x="245" y="580" text-anchor="middle" class="node-text">RDRR / OPFS / Downloader</text>

  <rect x="385" y="520" width="240" height="82" rx="${radius.panel}" fill="${palette.panelAlt}" stroke="${architecture.nodeLoad}" stroke-width="${stroke.thin}" />
  <text x="505" y="554" text-anchor="middle" class="node-title">WebGPU</text>
  <text x="505" y="580" text-anchor="middle" class="node-text">upload + buffers</text>

  <line x1="375" y1="232" x2="375" y2="270" stroke="${architecture.arrow}" stroke-width="${stroke.heavy}" />
  <line x1="375" y1="342" x2="375" y2="380" stroke="${architecture.arrow}" stroke-width="${stroke.heavy}" />

  <polyline points="375,462 375,488 245,488 245,520" stroke="${architecture.arrow}" stroke-width="${stroke.heavy}" fill="none" />
  <line x1="505" y1="462" x2="505" y2="520" stroke="${architecture.arrow}" stroke-width="${stroke.heavy}" />

  <rect x="775" y="150" width="500" height="82" rx="${radius.panel}" fill="${palette.panelAlt}" stroke="${architecture.nodeInfer}" stroke-width="${stroke.thin}" />
  <text x="1025" y="184" text-anchor="middle" class="node-title">Config</text>
  <text x="1025" y="210" text-anchor="middle" class="node-text">attention mechanism, batching, kernel overrides</text>

  <rect x="775" y="270" width="500" height="72" rx="${radius.panel}" fill="${palette.panelAlt}" stroke="${architecture.nodeInfer}" stroke-width="${stroke.thin}" />
  <text x="1025" y="302" text-anchor="middle" class="node-title">Public API Call</text>
  <text x="1025" y="326" text-anchor="middle" class="code-text">&#96;generate() / dopplerChat()&#96;</text>

  <rect x="775" y="380" width="500" height="104" rx="${radius.panel}" fill="${palette.panelAlt}" stroke="${architecture.nodeInfer}" stroke-width="${stroke.thin}" />
  <text x="1025" y="415" text-anchor="middle" class="node-title">Inference Pipeline</text>
  <text x="1025" y="442" text-anchor="middle" class="node-text">tokenizer -&gt; KV cache -&gt; attention/FFN -&gt; sampling</text>

  <rect x="775" y="530" width="500" height="82" rx="${radius.panel}" fill="${palette.panelAlt}" stroke="${architecture.nodeInfer}" stroke-width="${stroke.thin}" />
  <text x="1025" y="564" text-anchor="middle" class="node-title">WebGPU</text>
  <text x="1025" y="590" text-anchor="middle" class="node-text">WGSL kernels + buffer pools</text>

  <line x1="1025" y1="232" x2="1025" y2="270" stroke="${architecture.arrow}" stroke-width="${stroke.heavy}" />
  <line x1="1025" y1="342" x2="1025" y2="380" stroke="${architecture.arrow}" stroke-width="${stroke.heavy}" />
  <line x1="1025" y1="484" x2="1025" y2="530" stroke="${architecture.arrow}" stroke-width="${stroke.heavy}" />

  <line x1="625" y1="561" x2="775" y2="561" stroke="${architecture.arrowMuted}" stroke-width="${stroke.heavy}" stroke-dasharray="8 8" />
  <text x="700" y="545" text-anchor="middle" class="edge-label">shared WebGPU lifecycle</text>
</svg>`;

fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });
fs.writeFileSync(OUTPUT_PATH, `${svg}
`, 'utf-8');
console.log(`wrote ${OUTPUT_PATH}`);
