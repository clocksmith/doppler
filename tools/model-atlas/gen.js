// Model Atlas renderer: turns the retained inspection inputs (data.json,
// stages.json) plus current catalog lifecycle labels into the matrix and card
// HTML fragments that tools/build-model-atlas.js places into template.html.
// The rendering rules mirror the original renderer, so output bytes are stable.

const BPW = { F32: 32, F16: 16, BF16: 16, Q4_K_M: 4.5, Q4_0: 4.5, Q4_1: 5, Q5_0: 5.5, Q8_0: 8.5, Q6_K: 6.5625, W4A16: 4.5 };
const QUICKSTART = new Set([
  'gemma-3-270m-it-q4k-ehf16-af32', 'gemma-3-1b-it-q4k-ehf16-af32', 'google-embeddinggemma-300m-q4k-ehf16-af32',
  'gemma-4-e2b-it-q4k-ehf16-af32', 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple', 'qwen-3-5-0-8b-q4k-ehaf16', 'qwen-3-5-2b-q4k-ehaf16',
]);

// Model groups: one architecture each; lanes are stored variants.
const GROUPS = [
  {
    key: 'gemma3-270m',
    name: 'Gemma 3 270M IT',
    lanes: ['gemma-3-270m-it-q4k-ehf16-af32', 'gemma-3-270m-it-f16-af32'],
    pipes: ['gen'],
    notes: [
      'Smallest text model; both lanes verified. The Q4_K lane graph has no fused Q4_K kernel, so the loader expands Q4_K to F16 at load.',
    ],
  },
  {
    key: 'gemma3-1b',
    name: 'Gemma 3 1B IT',
    lanes: ['gemma-3-1b-it-q4k-ehf16-af32', 'gemma-3-1b-it-f16-af32', 'create-gemma-3-1b-qualification'],
    pipes: ['gen'],
    notes: [
      'create-gemma-3-1b-qualification is a local copy of the Q4_K lane used by Create qualification, not a separate model.',
      'The F16 lane exists on amd-host only and is not in the catalog.',
    ],
  },
  {
    key: 'translategemma-1b',
    name: 'TranslateGemma 4B→1B en-es',
    lanes: ['translategemma-4b-1b-enes-q4k-ehf16-af32'],
    pipes: ['gen', 'tr'],
    notes: [
      'Same 1B geometry as Gemma 3 1B. Its manifest uses the plain gemma chat template, so translation is driven by the prompt, not the structured translategemma template.',
    ],
  },
  {
    key: 'translategemma-4b',
    name: 'TranslateGemma 4B IT',
    lanes: ['translategemma-4b-it-q4k-ehf16-af32'],
    pipes: ['gen', 'tr'],
    notes: [
      'Uses the translategemma template: string prompts are rejected; each message must carry source_lang_code and target_lang_code.',
    ],
  },
  {
    key: 'embeddinggemma',
    name: 'EmbeddingGemma 300M',
    lanes: ['google-embeddinggemma-300m-q4k-ehf16-af32'],
    pipes: ['emb'],
    notes: [
      'Output: mean pool → dense 768→3072 → dense 3072→768 → L2 normalize, giving a 768-value vector.',
    ],
  },
  {
    key: 'gemma4-e2b',
    name: 'Gemma 4 E2B IT',
    lanes: ['gemma-4-e2b-it-q4k-ehf16-af32', 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple', 'gemma-4-e2b-it-q4k-ehf16-af16-int4ple'],
    pipes: ['gen', 'img', 'aud', 'vid'],
    notes: [
      'Per-layer inputs (PLE): every token also gathers 35 × 256 = 8,960 values from a 262,144-row table, one 256-value slice per layer.',
      'The last 20 layers reuse earlier layers’ keys and values (shared KV) and the last 20 layers use a double-width MLP (12,288).',
      'int4ple lanes store the PLE table as 4-bit rows with one F32 scale each: 6.01 GiB of F16-labelled data becomes 2.73 GiB (7.52 GB → 4.00 GB on disk).',
      'The manifest enables vision and audio. The audio upload and layout code was repaired in a76f1add; full audio inference is not qualified; the video width bug is fixed in 3a58230f but no GPU video run has confirmed it.',
    ],
  },
  {
    key: 'gemma4-12b',
    name: 'Gemma 4 12B IT (text)',
    lanes: ['gemma-4-12b-it-text-q4k-ehf16-af32', 'gemma-4-12b-it-text-q4k-ehf16-af16', 'gemma-4-12b-it-text-w4a16-ct-ehf16-af16'],
    pipes: ['gen'],
    notes: [
      'Global layers use 512-wide heads, sliding layers 256-wide, so the KV cache uses mixed geometry.',
      'The W4A16 lane imports compressed-tensors 4-bit weights (group 32, one scale per row per group); its BF16 tensors are the scales.',
    ],
  },
  {
    key: 'gemma4-31b',
    name: 'Gemma 4 31B IT (text)',
    lanes: ['gemma-4-31b-it-text-q4k-ehf16-af32', 'gemma-4-31b-it-text-q4k-ehf16-af16'],
    pipes: ['gen'],
    notes: [
      '16 KV heads on sliding layers and 4 on global layers; the largest KV cache per token among the Gemma models.',
    ],
  },
  {
    key: 'diffusiongemma',
    name: 'DiffusionGemma 26B-A4B',
    lanes: ['diffusiongemma-26b-a4b-it-q4k-ehf16-af16'],
    pipes: ['dg'],
    notes: [
      'Writes 256-token canvases by denoising instead of one token at a time.',
      'Each layer runs a dense FFN and a mixture-of-experts branch (128 experts, 8 per token) and sums them.',
      'Logits per denoising step are 256 × 262,144 F32 values = 256 MiB.',
    ],
  },
  {
    key: 'qwen3-emb',
    name: 'Qwen3 Embedding 0.6B',
    lanes: ['qwen-3-embedding-0-6b-q4k-ehf16-af32'],
    pipes: ['emb'],
    notes: [
      'Output: last-token pool → L2 normalize, giving a 1,024-value vector.',
    ],
  },
  {
    key: 'qwen3-rr',
    name: 'Qwen3 Reranker 0.6B',
    lanes: ['qwen-3-reranker-0-6b-f16-af32', 'qwen-3-reranker-0-6b-q4k-ehf16-af32'],
    pipes: ['rr'],
    notes: [
      'Scores a query–document pair from two logits only: "yes" minus "no"; probability = sigmoid of that difference.',
    ],
  },
  {
    key: 'qwen35-08b',
    name: 'Qwen 3.5 0.8B',
    lanes: ['qwen-3-5-0-8b-q4k-ehaf16'],
    pipes: ['gen', 'img'],
    notes: [
      'Hybrid: 18 linear-attention layers with fixed state, 6 full-attention layers with KV cache.',
      'Its manifest carries a Qwen3-VL vision encoder, reachable only through embedImage (not transcription), with open gaps.',
      'The name says -ehaf16, but the head is Q4_K and activations are F32.',
    ],
  },
  {
    key: 'qwen35-2b',
    name: 'Qwen 3.5 2B',
    lanes: ['qwen-3-5-2b-q4k-ehaf16'],
    pipes: ['gen'],
    notes: [
      'Same layer mix and state size as 0.8B with a 2,048-wide hidden state.',
      'The name says -ehaf16, but the head is Q4_K and activations are F32.',
    ],
  },
  {
    key: 'qwen36-27b',
    name: 'Qwen 3.6 27B',
    lanes: ['qwen-3-6-27b-q4k-ehaf16', 'qwen-3-6-27b-q4k-eaf16'],
    pipes: ['gen'],
    notes: [
      'The eaf16 lane is a manifest that borrows the ehaf16 lane’s shards (weightsRef); only activations and math change to F16.',
      'The ehaf16 name is wrong in the same way as 0.8B: Q4_K head, F32 activations.',
    ],
  },
  {
    key: 'qwen36-35b',
    name: 'Qwen 3.6 35B-A3B (MoE)',
    lanes: ['qwen-3-6-35b-a3b-text-source-q4_0-af32'],
    pipes: ['gen'],
    notes: [
      'Counts come from the GGUF header on amd-host: 40 layers (plus 1 multi-token-prediction layer), 10 full-attention, 30 linear.',
      '256 experts with 8 used per token, plus one shared expert. Experts hold 33.0 B of the 35.5 B weights, so a token touches about 1.0 B expert weights plus 2.5 B shared weights.',
      'The "Q4_0" file is a mix: Q4_0 for most weights, Q4_1 and Q8_0 in expert down-projections, Q8_0 shared experts and K/V, Q6_K output head.',
      'Runs through the opt-in profile qwen3-6-35b-a3b-q4_0-candidate. A Q4_K conversion config exists, but no converted artifact was found.',
    ],
  },
  {
    key: 'qwen38-27b',
    name: 'Qwen 3.8 27B',
    lanes: ['qwen3-8-27b-text-f16-af32', 'qwen3-8-27b-text-q4k-ehaf16'],
    pipes: ['gen'],
    notes: [
      'Same geometry as Qwen 3.6 27B. The F16 lane passes source-token parity in the catalog; the Q4_K artifact on mac-host is not in the catalog.',
      'A Q4_0 passthrough config exists (qwen3.8-27b-q4_0-passthrough.json); no artifact found.',
    ],
  },
  {
    key: 'flash-next',
    name: 'Qwen 3.8 Flash-Next',
    lanes: ['qwen-3-8-flash-next-text-q4k-ehf16-af32'],
    pipes: ['gen'],
    notes: [
      '177 billion weights: 512 experts with 10 used per token, plus a shared expert of width 640.',
      'Hyper-connection layers with indexed (QSA) attention over selected KV rows; requires F32, GPU, and no LoRA, embedding, prefix-KV or batch decode.',
      'The 51.2 B BF16 weights are n-gram embedding tables (128 tensors).',
      'A capability rule widens its KV cache to F32 even though the manifest says F16.',
      'On amd-host only; not in the catalog.',
    ],
  },
  {
    key: 'lfm2',
    name: 'LFM2.5 1.2B Instruct',
    lanes: ['lfm2-5-1-2b-instruct-q4k-ehf16-af32'],
    pipes: ['gen'],
    notes: [
      '10 short-convolution layers and 6 attention layers. Batched GPU decode refuses conv models, so it decodes one token per step.',
      'Before 8815443c the conv layers also got KV buffers (128 MiB instead of 48 MiB at 4,096 tokens).',
      'Catalog: experimental, not tested.',
    ],
  },
  {
    key: 'minicpm',
    name: 'MiniCPM4 0.5B',
    lanes: ['minicpm4-0-5b-f16-af32'],
    pipes: ['gen'],
    notes: [
      'LongRoPE position scaling and an explicit ×12 embedding scale. Catalog: experimental, tests failed.',
    ],
  },
  {
    key: 'muse',
    name: 'Muse Glimmer 30B (text)',
    lanes: ['muse-glimmer-30b-text-f16-af32', 'muse-glimmer-30b-text-bf16-af32'],
    pipes: ['gen'],
    notes: [
      'Dense 27.9 B weights stored at 16 bits (55.7 GB). Its manifest caps the KV cache at 131,072 tokens, which allocates 6.5 GiB up front.',
      'Catalog: experimental, tests failed on both lanes.',
    ],
  },
  {
    key: 'glmocr',
    name: 'GLM-OCR',
    lanes: ['glm-ocr-f16-af32', 'glm-ocr-f16-af32-pad-token-v1'],
    pipes: ['img', 'gen'],
    notes: [
      'Image-to-text OCR: GLM vision encoder (14-pixel patches, 2×2 merge) feeding a 16-layer text decoder with 3-axis rotary positions.',
      'Two local artifacts; not in the catalog.',
    ],
  },
  {
    key: 'amplify',
    name: 'AMPLIFY 120M (protein)',
    lanes: ['amplify-120m-f16-af32'],
    pipes: ['seq', 'emb'],
    notes: [
      'Protein language model: one bidirectional pass returns per-residue embeddings, a mean-pooled embedding, and per-position logits (masked-token prediction).',
    ],
  },
  {
    key: 'esm2',
    name: 'ESM-2 t12 35M (protein)',
    lanes: ['esm2-t12-35m-ur50d-f32-af32'],
    pipes: ['seq', 'emb'],
    notes: [
      'Per-residue and mean-pooled embeddings; no logits. Embedding scale 0.88; LayerNorm.',
    ],
  },
  {
    key: 'esmc',
    name: 'ESM C 300M (protein)',
    lanes: ['esmc-300m-f32-af32'],
    pipes: ['seq', 'emb'],
    notes: [
      'Per-residue and mean-pooled embeddings; no logits.',
    ],
  },
  {
    key: 'nt',
    name: 'Nucleotide Transformer v2 50M (DNA)',
    lanes: ['nucleotide-transformer-v2-50m-f32-af32'],
    pipes: ['seq', 'emb'],
    notes: [
      'DNA sequence encoder: per-token and mean-pooled embeddings; no logits.',
    ],
  },
  {
    key: 'sana',
    name: 'Sana Sprint 0.6B (image)',
    lanes: ['sana-sprint-0.6b-wf16-ef16-hf16-f16'],
    pipes: ['imgdiff'],
    notes: [
      'Text-to-image model (Gemma 2 2B text encoder, 28-layer linear-attention transformer, DC-AE decoder). The artifact exists on amd-host, but the runtime rejects layout "sana" at load.',
    ],
  },
];

const PIPES = [
  ['gen', 'Text generation'], ['tr', 'Translation'], ['emb', 'Embedding'], ['seq', 'Sequence encoding'], ['rr', 'Rerank'],
  ['img', 'Image → text'], ['aud', 'Audio → text'], ['vid', 'Video → text'], ['dg', 'Diffusion text'], ['imgdiff', 'Image generation'],
];
const PIPE_NAMES = Object.fromEntries(PIPES);
// "group|pipe" -> label overrides
const PIPE_STATE = new Map([
  ['gemma4-e2b|aud', 'fixed, not run'], ['gemma4-e2b|vid', 'fixed, not run'], ['qwen35-08b|img', 'partial'], ['sana|imgdiff', 'rejected'],
]);

// Missing keys and JSON null both read as null.
function get(object, key, fallback = null) {
  if (object == null || !Object.hasOwn(object, key)) return fallback;
  return object[key] === undefined ? null : object[key];
}

// Truthiness of a JSON value where empty arrays and objects count as false.
function truthy(value) {
  if (Array.isArray(value)) return value.length > 0;
  if (value && typeof value === 'object') return Object.keys(value).length > 0;
  return Boolean(value);
}

// Text form of a JSON value as the original renderer printed it.
function text(value) {
  if (value === null || value === undefined) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  if (Array.isArray(value)) return '[' + value.map((item) => (typeof item === 'string' ? `'${item}'` : text(item))).join(', ') + ']';
  return String(value);
}

function escapeHtml(value) {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#x27;');
}
const E = escapeHtml;

function n(x) {
  if (typeof x === 'boolean') return x ? '1' : '0';
  if (typeof x === 'number' && Number.isInteger(x)) return String(x).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return x === null || x === undefined ? '—' : text(x);
}

// bytes / 2**shift with `digits` decimals, rounding exact ties to even.
function fixedPowerOfTwo(bytes, shift, digits) {
  if (!Number.isInteger(bytes)) return (bytes / 2 ** shift).toFixed(digits);
  const negative = bytes < 0;
  const divisor = 1n << BigInt(shift);
  const scaled = BigInt(Math.abs(bytes)) * 10n ** BigInt(digits);
  let quotient = scaled / divisor;
  const twiceRemainder = (scaled % divisor) * 2n;
  if (twiceRemainder > divisor || (twiceRemainder === divisor && quotient % 2n === 1n)) quotient += 1n;
  let digitsText = quotient.toString().padStart(digits + 1, '0');
  if (digits > 0) digitsText = digitsText.slice(0, -digits) + '.' + digitsText.slice(-digits);
  return (negative && quotient !== 0n ? '-' : '') + digitsText;
}

function size(b) {
  if (b === null || b === undefined) return '—';
  if (b >= 2 ** 30) return `${fixedPowerOfTwo(b, 30, 2)} GiB`;
  if (b >= 2 ** 20) return `${fixedPowerOfTwo(b, 20, 1)} MiB`;
  return `${fixedPowerOfTwo(b, 10, 0)} KiB`;
}

function fmtname(d) {
  return d === 'Q4_K_M' ? 'Q4_K' : d;
}

function kvBytesPerValue(r) {
  return get(r, 'kvDtype') === 'f32' ? 4 : 2;
}

function archOf(r) {
  const a = get(r, 'arch');
  return a && typeof a === 'object' && !Array.isArray(a) ? a : {};
}

// KV recompute with the exact rules (sliding ring only for mixed geometry outside DiffusionGemma).
function kvInfo(r, s) {
  if (truthy(get(s, 'sequence'))) return null;
  const a = get(r, 'arch');
  if (!a || typeof a !== 'object' || Array.isArray(a) || !truthy(get(a, 'numKeyValueHeads'))) return null;
  const lt = truthy(get(r, 'layerTypes')) ? r.layerTypes : {};
  const L = a.numLayers; const kvh = a.numKeyValueHeads; const hd = a.headDim;
  const ghd = get(a, 'globalHeadDim') || hd; const gkvh = get(a, 'numGlobalKeyValueHeads') || kvh;
  const shared = get(a, 'numKvSharedLayers') || 0;
  const cap = get(r, 'kvCapacityTokens');
  const sw = get(s, 'slidingWindow');
  const mixed = truthy(lt) && ((ghd !== hd) || shared > 0);
  const slide = get(lt, 'sliding_attention', 0); let full = get(lt, 'full_attention', 0);
  const lin = get(lt, 'linear_attention', 0); const conv = get(lt, 'conv', 0);
  if (!truthy(lt)) full = L;
  const out = {};
  const bpv = kvBytesPerValue(r);
  if (mixed) {
    const ring = !truthy(get(s, 'diffusionGemma'));
    const slPer = slide * 2 * kvh * hd; const glPer = full * 2 * gkvh * ghd;
    const slCap = ring && truthy(sw) ? Math.min(sw, cap) : cap;
    out.layout = 'mixed geometry, ' + (ring && truthy(sw) ? `sliding layers in a ring of ${n(slCap)} tokens` : 'all layers full length');
    out.perToken = slPer + glPer;
    out.bytes = (slPer * slCap + glPer * cap) * bpv;
    out.detail = `${text(slide)} sliding × 2 × ${text(kvh)}×${text(hd)} + ${text(full)} global × 2 × ${text(gkvh)}×${text(ghd)}`;
  } else {
    const layers = L - lin - conv;
    const per = layers * 2 * kvh * hd;
    out.layout = 'contiguous, every attention layer full length';
    out.perToken = per;
    out.bytes = per * cap * bpv;
    out.detail = `${text(layers)} layers × 2 × ${text(kvh)}×${text(hd)}`
      + (truthy(lin) ? ` (${text(lin)} linear layers excluded)` : '')
      + (truthy(conv) ? ` (${text(conv)} conv layers excluded since 8815443c)` : '');
  }
  out.cap = cap; out.capSource = get(r, 'kvCapSource', ''); out.dtype = get(r, 'kvDtype');
  return out;
}

export function generateAtlasParts({ data: records, stages, catalog }) {
  const data = Object.fromEntries(records.map((record) => [record.id, record]));
  const st = stages;
  const cat = Object.fromEntries(catalog.models.map((model) => [model.modelId, model]));

  function laneStatus(lid) {
    const c = get(cat, lid);
    if (lid === 'qwen-3-6-35b-a3b-text-source-q4_0-af32') return ['candidate', 'opt-in profile; qualification open'];
    if (!truthy(c)) return ['local', 'not in catalog'];
    const lifecycle = truthy(get(c, 'lifecycle')) ? c.lifecycle : {};
    const lc = truthy(get(lifecycle, 'status')) ? lifecycle.status : {};
    const rt = get(lc, 'runtime'); const te = get(lc, 'tested');
    if (te === 'verified') return ['ok', `${text(rt)}, verified`];
    if (te === 'failed') return ['bad', `${text(rt)}, tests failed`];
    return ['warn', `${text(rt)}, tests ${text(te)}`];
  }

  function formatsLine(r) {
    const parts = [];
    const entries = Object.entries(r.dtypes).sort((x, y) => y[1].elements - x[1].elements);
    for (const [d, v] of entries) {
      if (v.elements === 0) continue;
      const b = get(v, 'bytes') || Math.trunc(v.elements * (Object.hasOwn(BPW, d) ? BPW[d] : 16) / 8);
      parts.push(`<span class="fmt"><b>${E(fmtname(d))}</b> ${n(v.elements)} <i>${size(b)}</i></span>`);
    }
    return parts.join(' ');
  }

  function where(r) {
    const w = truthy(get(r, 'where')) ? [...r.where] : [];
    return w.length ? w.join(', ') : '—';
  }

  function q4kHandling(lid) {
    const s = truthy(get(st, lid)) ? st[lid] : {};
    const files = (truthy(get(s, 'kernelFiles')) ? s.kernelFiles : []).join(' ');
    const r = data[lid];
    const hasQ4k = Object.hasOwn(r.dtypes, 'Q4_K_M');
    if (lid.endsWith('q4_0-af32')) return 'GGUF blocks kept packed; native Q4_0/Q4_1/Q5_0/Q8_0/Q6_K kernels decode them on the fly.';
    if (Object.hasOwn(r.dtypes, 'W4A16')) return 'W4A16 kept packed for a fused kernel when the GPU has shader-f16; otherwise expanded to F16 on the CPU at load.';
    if (!hasQ4k) return '16/32-bit weights; BF16 is cast to F16 or F32 on the GPU unless a kernel needs BF16.';
    if (files.includes('fused_matmul_q4')) return 'Q4_K kept packed and decoded inside fused matmul kernels; a tensor that an F16 step also reads gets a dense copy too.';
    return 'Graph has no fused Q4_K kernel, so the loader expands Q4_K to F16 at load (more GPU memory than the file size).';
  }

  function lanesTable(g) {
    const rows = [];
    for (const lid of g.lanes) {
      const r = data[lid];
      const [cls, txt] = laneStatus(lid);
      const qs = QUICKSTART.has(lid) ? ' · quickstart' : '';
      const ref = truthy(get(r, 'weightsRef')) ? ' · borrows shards (weightsRef)' : '';
      rows.push(`<tr><td id="${E(lid)}"><code>${E(lid)}</code><div class="sub">${formatsLine(r)}</div></td>
<td class="num">${n(r.weights)}</td><td class="num">${size(get(r, 'bytes'))}</td>
<td>${E(text(get(r, 'activationDtype') || '—'))} / ${E(text(get(r, 'kvDtype') || '—'))}</td>
<td><span class="chip ${cls}">${E(txt)}</span><div class="sub">${E(where(r))}${qs}${ref}</div></td></tr>`);
    }
    return `<div class="scroll"><table class="lanes"><thead><tr><th>Stored lane and its number formats</th><th class="num">Weights</th><th class="num">On disk</th><th>Activations / KV</th><th>Status · location</th></tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
  }

  function walk(g) {
    const lid = g.lanes[0]; const r = data[lid]; const s = get(st, lid, {}); const a = archOf(r);
    const kv = kvInfo(r, s);
    const lt = truthy(get(r, 'layerTypes')) ? r.layerTypes : {};
    const pipes = g.pipes;
    const pipesSet = new Set(pipes);
    const rows = [];
    const row = (label, body) => rows.push(`<tr><th>${label}</th><td>${body}</td></tr>`);
    const pipeNames = pipes.map((p) => {
      const state = PIPE_STATE.get(`${g.key}|${p}`);
      const chip = state === undefined ? '' : ` <span class="chip ${state === 'broken' || state === 'rejected' ? 'bad' : 'warn'}">${state}</span>`;
      return `<a href="#p-${p}">${PIPE_NAMES[p]}</a>` + chip;
    }).join(', ');
    const mt = get(r, 'modelType');
    const routes = { transformer: 'InferencePipeline', gemma4: 'InferencePipeline', embedding: 'EmbeddingPipeline', diffusion_gemma: 'DiffusionGemmaPipeline', diffusion: 'DiffusionPipeline (experimental)' };
    const route = Object.hasOwn(routes, mt) ? routes[mt] : mt;
    row('Pipelines', pipeNames);
    row('1 · Route', `modelType <code>${E(text(mt))}</code> → ${E(route)}`);
    let path;
    if (lid.endsWith('q4_0-af32')) {
      path = '<a href="#path-gguf">GGUF passthrough</a>: llama.cpp blocks copied byte-for-byte into the manifest’s shards';
    } else if (mt === 'diffusion') {
      path = '<a href="#path-rdrr">Converted RDRR</a> from a diffusers directory';
    } else {
      path = '<a href="#path-rdrr">Converted RDRR</a> from SafeTensors';
    }
    if (g.lanes.some((x) => truthy(get(data[x], 'weightsRef')))) path += '; one lane is a <a href="#path-ref">weightsRef</a> variant';
    if (g.lanes.some((x) => QUICKSTART.has(x))) path += '; quickstart ID downloads a pinned Hugging Face revision';
    row('2 · Weights in', path + '. ' + E(q4kHandling(lid)));
    if (mt === 'diffusion') {
      row('3 · Stages', 'Text encoder → 28-layer transformer → DC-AE decoder. Load stops at the layout check, so no stage runs.');
      return '<table class="walk">' + rows.join('') + '</table>';
    }
    const vocab = get(r, 'vocab') || get(a, 'vocabSize');
    const tmpl = get(s, 'chatTemplate');
    const seqcfg = get(s, 'sequence');
    let tok;
    if (truthy(seqcfg)) {
      tok = `${E(text(get(seqcfg, 'alphabet', '')).replace(/[_-]/g, ' '))} alphabet, ${n(vocab)} token IDs; no chat template`;
    } else if (truthy(tmpl)) {
      tok = `${n(vocab)} token IDs; chat template <code>${E(text(tmpl))}</code>` + (truthy(get(s, 'thinking')) ? ' with thinking prefix' : '');
    } else {
      tok = `${n(vocab)} token IDs; no chat template`;
    }
    row('3 · Tokenize', tok);
    const H = get(r, 'hidden') || get(a, 'hiddenSize');
    const qi = lid.endsWith('q4_0-af32') ? { embeddings: 'q4_0', lmHead: 'q6k' } : (truthy(get(r, 'quant')) ? r.quant : {});
    const embeddingScale = get(s, 'embeddingScale');
    const scale = truthy(get(s, 'scaleEmbeddings')) ? '× √hidden'
      : (embeddingScale !== null && embeddingScale !== 1 && embeddingScale !== true ? `× ${text(embeddingScale)}` : 'unscaled');
    row('4 · Embed', `Gather ${n(H)} values per token from a ${n(vocab)} × ${n(H)} ${E(text(get(qi, 'embeddings', '')).toUpperCase())} table, ${scale} → [T, ${n(H)}]`);
    // mixer
    const nH = get(a, 'numAttentionHeads'); const nKV = get(a, 'numKeyValueHeads'); const D = get(a, 'headDim');
    const parts = [];
    const L = get(a, 'numLayers');
    if (truthy(lt)) {
      parts.push(Object.entries(lt).map(([k, v]) => `${text(v)} ${k.replaceAll('_attention', ' attention').replaceAll('conv', 'short-conv')}`).join(' + ') + ` = ${text(L)} layers`);
    } else {
      parts.push(`${text(L)} layers, all full attention` + (truthy(get(s, 'causal')) ? '' : ' (bidirectional)'));
    }
    const globalHeadDim = get(a, 'globalHeadDim');
    const mixedGeometry = truthy(globalHeadDim) && globalHeadDim !== D;
    let att = `Attention: ${text(nH)} query heads, ${text(nKV)} KV heads × ${text(D)}`;
    if (mixedGeometry) att += `; global layers ${text(get(a, 'numGlobalKeyValueHeads') || nKV)} KV heads × ${text(globalHeadDim)}`;
    if (truthy(get(s, 'slidingWindow')) && truthy(get(lt, 'sliding_attention'))) att += `; sliding window ${n(s.slidingWindow)}`;
    const extras = [];
    if (truthy(get(s, 'qkNorm'))) extras.push('Q/K RMSNorm');
    if (truthy(get(s, 'outputGate'))) extras.push('sigmoid output gate');
    if (truthy(get(s, 'mrope'))) extras.push('multi-axis RoPE');
    if (truthy(get(s, 'partialRotary')) && s.partialRotary !== 1) extras.push(`RoPE on ${Math.trunc(s.partialRotary * 100)}% of dims`);
    if (truthy(get(s, 'ropeLocalTheta'))) extras.push('separate local RoPE table');
    if (truthy(get(s, 'ropeScaling'))) extras.push(`${text(s.ropeScaling)} scaling`);
    if (truthy(get(s, 'attnSoftcap'))) extras.push(`attention softcap ${text(s.attnSoftcap)}`);
    if (extras.length) att += ' · ' + extras.join(', ');
    parts.push(att);
    if (truthy(get(lt, 'linear_attention'))) {
      const nk = get(a, 'linearNumKeyHeads'); const nv = get(a, 'linearNumValueHeads'); const dk = get(a, 'linearKeyHeadDim');
      const conv = 2 * nk * dk + nv * dk;
      parts.push(`Linear attention: ${text(nk)} key heads, ${text(nv)} value heads × ${text(dk)}; conv width ${n(conv)} (kernel 4); gated-delta recurrence`);
    }
    row('5 · Mixer', parts.map((p) => E(p)).join('<br>'));
    const I = get(r, 'ffn') || get(a, 'intermediateSize');
    const moe = get(r, 'moe');
    const act = get(s, 'activation');
    let f;
    if (truthy(moe)) {
      f = `Mixture of experts: router [T, ${n(moe.numExperts)}] → top ${text(moe.numExpertsPerToken)} → experts of width ${n(get(moe, 'expertIntermediateSize'))}`;
      if (truthy(get(moe, 'sharedExpertIntermediateSize'))) f += ` + shared expert ${n(moe.sharedExpertIntermediateSize)}`;
      if (get(s, 'ffnBranch') === 'dense-plus-moe') f += `; plus a dense ${text(act)} FFN of width ${n(I)}, summed`;
    } else {
      f = `Dense ${truthy(get(s, 'gated')) ? 'gated ' : ''}${text(act)}: [T, ${n(H)}] → [T, ${n(I)}] → [T, ${n(H)}]`;
      if (truthy(get(s, 'doubleWide'))) f += `; last ${text(get(a, 'numKvSharedLayers'))} layers double width (${n(2 * I)})`;
    }
    let norms = get(s, 'normType') === 'layernorm' ? 'LayerNorm' : (truthy(get(s, 'weightOffset')) ? 'RMSNorm (1 + w)' : 'RMSNorm');
    if (truthy(get(s, 'sandwich'))) norms += ', sandwich norms around attention and FFN';
    row('6 · FFN and norms', E(f) + '<br>' + E(norms));
    if (truthy(get(r, 'perLayerInputPerToken'))) {
      row('7 · Per-layer extras', 'Per-layer inputs: ' + `${n(r.perLayerInputPerToken)} values per token (${text(L)} × ${text(get(a, 'hiddenSizePerLayerInput'))}): gather + projection, gated into each layer’s residual`);
    } else if (g.key === 'flash-next') {
      row('7 · Per-layer extras', 'Hyper-connections: a root mixer replaces the final norm; the QSA indexer selects KV rows per query');
    } else if (truthy(get(lt, 'conv'))) {
      row('7 · Per-layer extras', 'Short-conv layers keep a persistent convolution state');
    } else {
      row('7 · Per-layer extras', 'None');
    }
    // output
    let out;
    if (pipesSet.has('emb') && !truthy(seqcfg)) {
      const ep = truthy(get(s, 'embPost')) ? s.embPost : {};
      const projs = truthy(get(ep, 'projections')) ? ep.projections : [];
      const dim = projs.length ? projs[projs.length - 1].outputSize : H;
      out = `Final norm → ${text(get(ep, 'poolingMode', 'mean'))} pool → `
        + (projs.length ? projs.map((p) => `dense ${text(p.inputSize)}→${text(p.outputSize)}`).join(' → ') + ' → ' : '')
        + `${text(get(ep, 'normalize', 'none'))} → ${n(dim)} values`;
    } else if (truthy(seqcfg)) {
      out = `Final norm → per-token embeddings [T, ${n(H)}] + mean pool (excluding IDs ${text(seqcfg.pooledEmbedding.excludeTokenIds)})`
        + (truthy(get(seqcfg, 'logits')) ? ` + per-position logits [T, ${n(vocab)}]` : '');
    } else if (pipesSet.has('rr')) {
      out = 'Final norm → LM head for 2 token IDs only ("yes", "no") → score = difference → sigmoid';
    } else {
      const head = text(get(qi, 'lmHead', '')).toUpperCase();
      out = `Final norm → LM head ${truthy(get(s, 'tied')) ? '(tied to embedding) ' : ''}${E(head)} [${n(vocab)} × ${n(H)}] → ${n(vocab)} F32 logits`;
      if (truthy(get(s, 'finalSoftcap'))) out += `, softcap ${text(s.finalSoftcap)}`;
      if (pipesSet.has('dg')) out += '; per denoising step [256, 262,144]';
    }
    row('8 · Output', out);
    // state
    const stParts = [];
    if (kv) {
      stParts.push(`KV cache (${E(text(kv.dtype))}): ${n(kv.perToken)} values per token = ${E(kv.detail)}; ${E(kv.layout)}; sized for ${n(kv.cap)} tokens (${E(kv.capSource)}) → <b>${size(kv.bytes)}</b> allocated at load`);
    }
    if (truthy(get(r, 'linearStateValues'))) {
      stParts.push(`Linear-attention state (F32, fixed): ${n(r.linearStateValues)} values = <b>${size(r.linearStateValues * 4)}</b> across ${text(get(r, 'linearLayers'))} layers`);
    }
    if (truthy(get(lt, 'conv'))) stParts.push('Short-conv state per conv layer (persistent)');
    if (truthy(seqcfg)) {
      stParts.push('No KV allocation: the loader honors skipEmbeddingKVCacheWrites for sequence encoders (a76f1add)');
    } else if (pipesSet.has('emb')) {
      stParts.push('Used as the attention store for the single pass, but sized for the full token cap, not the input');
    } else if (pipesSet.has('rr')) {
      stParts.push('Used: the rerank harness prefills the shared prompt prefix once and rewinds to it per document');
    }
    row('9 · State', stParts.join('<br>') || '—');
    const decodes = pipesSet.has('gen') || pipesSet.has('dg');
    if (!decodes) row('10 · Decode', 'No decode loop: one pass per input');
    if (decodes) {
      const bs = get(s, 'batchSize'); const ri = get(s, 'readbackInterval');
      let dec = truthy(bs) ? `Batch ${text(bs)} tokens per command buffer, read back every ${text(ri)} batch(es)` : 'Decode-loop settings come from the runtime profile';
      if (truthy(get(lt, 'conv'))) dec = 'Batched decode refuses conv models: one token per step';
      if (pipesSet.has('dg')) dec = 'No token loop: up to 48 denoising steps per 256-token canvas';
      row('10 · Decode', dec);
    }
    // partition rule
    const reasons = [];
    if (truthy(moe)) reasons.push('MoE');
    if (truthy(get(a, 'numKvSharedLayers'))) reasons.push('shared KV');
    if (truthy(get(r, 'perLayerInputPerToken'))) reasons.push('per-layer inputs');
    if (mixedGeometry) reasons.push('mixed KV geometry');
    if (truthy(get(lt, 'conv'))) reasons.push('conv layers');
    if (!pipesSet.has('gen') && !pipesSet.has('tr')) reasons.push('not a token-generation model');
    if (g.key === 'flash-next') reasons.push('hyper-connections');
    row('Partitions', reasons.length ? 'Blocked by ' + reasons.join(', ') : 'Allowed by the code rules (token-only causal layers); a rule check, not a run');
    return '<table class="walk">' + rows.join('') + '</table>';
  }

  function counts(g) {
    const r = data[g.lanes[0]]; const s = get(st, g.lanes[0], {}); const a = archOf(r);
    const kv = kvInfo(r, s);
    const vlabel = g.pipes.some((p) => p === 'gen' || p === 'tr' || p === 'dg') ? 'Logits per token' : 'Vocabulary';
    const vocab = get(r, 'vocab') || get(a, 'vocabSize');
    const items = [['Weights', n(r.weights)], ['Hidden per token', n(get(r, 'hidden') || get(a, 'hiddenSize'))], [vlabel, truthy(vocab) ? n(vocab) : '—']];
    if (kv) items.push(['KV values per token', n(kv.perToken)]);
    if (truthy(get(r, 'linearStateValues'))) items.push(['Fixed state values', n(r.linearStateValues)]);
    return '<dl class="counts">' + items.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('') + '</dl>';
  }

  const cards = GROUPS.map((g) => {
    const notes = g.notes.map((x) => `<li>${E(x)}</li>`).join('');
    return `<article class="model" id="m-${g.key}"><header><h3>${E(g.name)}</h3>${counts(g)}</header>
<ul class="notes">${notes}</ul>${lanesTable(g)}<h4>Pipeline walk</h4>${walk(g)}</article>`;
  });

  const matrixRows = GROUPS.map((g) => {
    const cells = PIPES.map(([p]) => {
      if (!g.pipes.includes(p)) return '<td></td>';
      const state = PIPE_STATE.get(`${g.key}|${p}`) ?? 'yes';
      const cls = { yes: 'ok', partial: 'warn', broken: 'bad', rejected: 'bad', 'fixed, not run': 'warn' }[state];
      return `<td><span class="dot ${cls}">${state}</span></td>`;
    });
    const r = data[g.lanes[0]];
    return `<tr><th><a href="#m-${g.key}">${E(g.name)}</a></th><td class="num">${n(r.weights)}</td>${cells.join('')}</tr>`;
  });
  const matrix = `<div class="scroll"><table class="matrix"><thead><tr><th>Model</th><th class="num">Weights</th>${PIPES.map(([, t]) => `<th>${E(t)}</th>`).join('')}</tr></thead><tbody>${matrixRows.join('')}</tbody></table></div>`;

  return { matrix, cards: cards.join('\n'), cardCount: cards.length };
}
