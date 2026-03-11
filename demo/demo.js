import {
  log,
  listPresets,
  createConverterConfig,
  detectPreset,
  resolvePreset,
  getRuntimeConfig,
  setRuntimeConfig,
  DEFAULT_MANIFEST_INFERENCE,
  formatBytes,
  listRegisteredModels,
  registerModel,
  openModelStore,
  writeShard,
  loadManifestFromStore,
  loadTensorsFromStore,
  saveManifest,
  saveTensorsToStore,
  saveTokenizer,
  saveTokenizerModel,
  saveAuxFile,
  loadTokenizerFromStore,
  loadTokenizerModelFromStore,
  parseManifest,
  getManifest,
  setManifest,
  clearManifest,
  classifyTensorRole,
  convertModel,
  createRemoteModelSources,
  isConversionSupported,
  buildManifestInference,
  inferEmbeddingOutputConfig,
  pickModelDirectory,
  pickModelFiles,
  createPipeline,
  initDevice,
  getDevice,
  getKernelCapabilities,
  getPlatformConfig,
  isWebGPUAvailable,
  captureMemorySnapshot,
  destroyBufferPool,
} from '@simulatte/doppler';
import { DiagnosticsController } from './diagnostics-controller.js';
import { state } from './ui/state.js';
import { $, setText, setHidden } from './ui/dom.js';
import { formatAutoValue } from './ui/format.js';
import { readOptionalNumber } from './ui/input.js';
import {
  showProgressOverlay,
  hideProgressOverlay,
  updateProgressFromLoader,
} from './ui/progress.js';
import {
  setStatusIndicator,
  updateStatusIndicator,
  clampPercent,
  showErrorModal,
  hideErrorModal,
} from './ui/ui.js';
import {
  updatePerformancePanel,
  updateMemoryPanel,
  updateMemoryControls,
  renderRunLog,
  recordRunLog,
} from './ui/stats.js';
import {
  ENERGY_DEMOS,
  DEFAULT_ENERGY_DEMO_ID,
  DEFAULT_RUNTIME_PRESET,
  RUNTIME_PRESET_REGISTRY,
} from './ui/constants.js';
import {
  updateEnergyStatus,
  getEnergyDemoById,
  setEnergyMetricLabels,
  toggleEnergyProblemControls,
  syncEnergyDemoSelection,
  populateEnergyDemoSelect,
  applyEnergyDemoDefaults,
} from './ui/energy/controls.js';
import {
  clearEnergyBoard,
  clearEnergyChart,
  renderEnergyBoard,
  renderEnergyVector,
  renderEnergyIntensityBoard,
  drawEnergyChart,
  updateEnergyStats,
} from './ui/energy/render.js';
import {
  storeDiagnosticsSelection,
  syncDiagnosticsModeUI,
  getDiagnosticsDefaultSuite,
  getDiagnosticsRuntimeConfig,
  refreshDiagnosticsRuntimeConfig,
  syncDiagnosticsDefaultsForMode,
  clearDiagnosticsOutput,
  renderDiagnosticsOutput,
  updateDiagnosticsStatus,
  updateDiagnosticsReport,
  updateDiagnosticsGuidance,
  decodeDiagnosticsProfileId,
  selectDiagnosticsModel,
  handleRuntimeConfigFile,
  applyRuntimeConfigPreset,
  applySelectedRuntimePreset,
} from './ui/diagnostics/index.js';
import {
  normalizeModelType,
  isCompatibleModelType,
  isModeModelSelectable,
  getModeModelLabel,
  getModelTypeForId,
} from './ui/models/utils.js';
import { updateStorageInfo, refreshStorageInspector, deleteStorageModel } from './ui/storage/inspector.js';
import {
  configureDownloadCallbacks,
  refreshDownloads,
  startDownload,
  startDownloadFromBaseUrl,
  pauseActiveDownload,
  resumeActiveDownload,
  cancelActiveDownload,
} from './ui/downloads/index.js';
import {
  createTranslateTextRequest,
} from './ui/translate/request.js';
import { runDistillReplay } from './ui/distill/replay.js';
import { formatChatMessages } from '../src/inference/pipelines/text/chat-format.js';

const controller = new DiagnosticsController({ log });

const PRIMARY_MODES = new Set(['run', 'translate', 'embedding', 'diffusion', 'energy']);
let modelListRefreshVersion = 0;
const DEFAULT_MODEL_AVAILABILITY = Object.freeze({
  total: 0,
  run: 0,
  translate: 0,
  embedding: 0,
  diffusion: 0,
  energy: 0,
});
const QUICK_MODEL_CATALOG_LOCAL_BASE_URL = typeof window === 'object' && window.location?.origin
  ? new URL('/models/catalog.json', window.location.origin).toString()
  : new URL('../models/catalog.json', import.meta.url).toString();
const QUICK_MODEL_CATALOG_CACHE_BUST = 'catalog-v2';
const QUICK_MODEL_CATALOG_LOCAL_URL = `${QUICK_MODEL_CATALOG_LOCAL_BASE_URL}?cacheBust=${QUICK_MODEL_CATALOG_CACHE_BUST}`;
const QUICK_MODEL_CATALOG_DEFAULT_HF_REPO_ID = 'Clocksmith/rdrr';
const QUICK_MODEL_CATALOG_DEFAULT_HF_REVISION = 'main';
const QUICK_MODEL_CATALOG_DEFAULT_HF_PATH = 'registry/catalog.json';
const QUICK_MODEL_CATALOG_OVERRIDE_URL = readGlobalString('__DOPPLER_QUICK_MODEL_CATALOG_URL');
const QUICK_MODEL_CATALOG_HF_REPO_ID = readGlobalString('__DOPPLER_HF_REGISTRY_REPO_ID') || QUICK_MODEL_CATALOG_DEFAULT_HF_REPO_ID;
const QUICK_MODEL_CATALOG_HF_REVISION = readGlobalString('__DOPPLER_HF_REGISTRY_REVISION') || QUICK_MODEL_CATALOG_DEFAULT_HF_REVISION;
const QUICK_MODEL_CATALOG_HF_PATH = readGlobalString('__DOPPLER_HF_REGISTRY_CATALOG_PATH') || QUICK_MODEL_CATALOG_DEFAULT_HF_PATH;
const QUICK_MODEL_CATALOG_URLS = buildQuickCatalogCandidateUrls();
const QUICK_MODEL_HF_HOST = 'huggingface.co';
const QUICK_MODEL_HF_COMMIT_PATTERN = /^[a-f0-9]{7,64}$/i;
const DISTILL_WORKLOAD_REGISTRY_URL = typeof window === 'object' && window.location?.origin
  ? new URL('/tools/configs/training-workloads/registry.json', window.location.origin).toString()
  : new URL('../tools/configs/training-workloads/registry.json', import.meta.url).toString();
const RUN_STARTER_PROMPTS = Object.freeze([
  'is potential energy real?',
  'compare zig to rust in elvish',
  'eat your cake and have it too',
  'pivot to neurosymbolic reasoning',
  'write a poem about an elephant that is bullish on QQQ',
  'explain why a toddler is exactly like a neural network',
  'explain the difference between the star trek migratation and star wars trek',
  'prove termination for a recursive functional agent using lean four and inductive types',
  'describe a toy store where the shelves are sorted by cognitive development stages and every single game has a proof of educational value attached',
  'is human intuition just a fast, low-energy heuristic that our biological hardware runs when the cost of slow, symbolic reasoning is too high for survival',
  'write a technical fable about an agent tasked with solving a paradox, forever rolling a high-energy gradient up a hill only for it to reset at every epoch',
  'is interpretability mostly archaeology on activations, or can it become a design-time discipline',
  'prove or disprove that benchmark parity without workload parity is a category error',
  'prove or disprove that deterministic failure modes outperform probabilistic success in trust-critical workflows',
]);
const TRANSLATE_STARTER_PROMPTS = Object.freeze([
  'Good software should fail loudly and explain why.',
  'Never silently fall back when model capabilities are not supported.',
  'Please translate this release note into clear, natural language.',
  'On-device inference keeps sensitive data local to the machine.',
  'A deterministic benchmark needs fixed prompts, seeds, and token budgets.',
]);
const TRANSLATE_LANGUAGE_OPTIONS = Object.freeze([
  Object.freeze({ code: 'ar_EG', name: 'Arabic (Egypt)' }),
  Object.freeze({ code: 'ar_SA', name: 'Arabic (Saudi Arabia)' }),
  Object.freeze({ code: 'bg_BG', name: 'Bulgarian' }),
  Object.freeze({ code: 'bn_IN', name: 'Bengali' }),
  Object.freeze({ code: 'ca_ES', name: 'Catalan' }),
  Object.freeze({ code: 'cs_CZ', name: 'Czech' }),
  Object.freeze({ code: 'da_DK', name: 'Danish' }),
  Object.freeze({ code: 'de_DE', name: 'German' }),
  Object.freeze({ code: 'el_GR', name: 'Greek' }),
  Object.freeze({ code: 'en', name: 'English' }),
  Object.freeze({ code: 'es', name: 'Spanish' }),
  Object.freeze({ code: 'et_EE', name: 'Estonian' }),
  Object.freeze({ code: 'fa_IR', name: 'Persian' }),
  Object.freeze({ code: 'fi_FI', name: 'Finnish' }),
  Object.freeze({ code: 'fil_PH', name: 'Filipino' }),
  Object.freeze({ code: 'fr_CA', name: 'French (Canada)' }),
  Object.freeze({ code: 'fr_FR', name: 'French' }),
  Object.freeze({ code: 'gu_IN', name: 'Gujarati' }),
  Object.freeze({ code: 'he_IL', name: 'Hebrew' }),
  Object.freeze({ code: 'hi_IN', name: 'Hindi' }),
  Object.freeze({ code: 'hr_HR', name: 'Croatian' }),
  Object.freeze({ code: 'hu_HU', name: 'Hungarian' }),
  Object.freeze({ code: 'id_ID', name: 'Indonesian' }),
  Object.freeze({ code: 'is_IS', name: 'Icelandic' }),
  Object.freeze({ code: 'it_IT', name: 'Italian' }),
  Object.freeze({ code: 'ja_JP', name: 'Japanese' }),
  Object.freeze({ code: 'kn_IN', name: 'Kannada' }),
  Object.freeze({ code: 'ko_KR', name: 'Korean' }),
  Object.freeze({ code: 'lt_LT', name: 'Lithuanian' }),
  Object.freeze({ code: 'lv_LV', name: 'Latvian' }),
  Object.freeze({ code: 'ml_IN', name: 'Malayalam' }),
  Object.freeze({ code: 'mr_IN', name: 'Marathi' }),
  Object.freeze({ code: 'nl_NL', name: 'Dutch' }),
  Object.freeze({ code: 'no_NO', name: 'Norwegian' }),
  Object.freeze({ code: 'pa_IN', name: 'Punjabi' }),
  Object.freeze({ code: 'pl_PL', name: 'Polish' }),
  Object.freeze({ code: 'pt_BR', name: 'Portuguese (Brazil)' }),
  Object.freeze({ code: 'pt_PT', name: 'Portuguese (Portugal)' }),
  Object.freeze({ code: 'ro_RO', name: 'Romanian' }),
  Object.freeze({ code: 'ru_RU', name: 'Russian' }),
  Object.freeze({ code: 'sk_SK', name: 'Slovak' }),
  Object.freeze({ code: 'sl_SI', name: 'Slovenian' }),
  Object.freeze({ code: 'sr_RS', name: 'Serbian' }),
  Object.freeze({ code: 'sv_SE', name: 'Swedish' }),
  Object.freeze({ code: 'sw_KE', name: 'Swahili' }),
  Object.freeze({ code: 'sw_TZ', name: 'Swahili (Tanzania)' }),
  Object.freeze({ code: 'ta_IN', name: 'Tamil' }),
  Object.freeze({ code: 'te_IN', name: 'Telugu' }),
  Object.freeze({ code: 'th_TH', name: 'Thai' }),
  Object.freeze({ code: 'tr_TR', name: 'Turkish' }),
  Object.freeze({ code: 'uk_UA', name: 'Ukrainian' }),
  Object.freeze({ code: 'ur_PK', name: 'Urdu' }),
  Object.freeze({ code: 'vi_VN', name: 'Vietnamese' }),
  Object.freeze({ code: 'zh_TW', name: 'Chinese (Traditional)' }),
  Object.freeze({ code: 'zu_ZA', name: 'Zulu' }),
]);
const DEFAULT_TRANSLATE_SOURCE = 'en';
const DEFAULT_TRANSLATE_TARGET = 'es';
const DEFAULT_TRANSLATE_TEMPERATURE = 1.0;
const DEFAULT_TRANSLATE_TOP_P = 0.95;
const DEFAULT_TRANSLATE_TOP_K = 64;
const DEFAULT_TRANSLATE_MAX_TOKENS = 1024;
const TRANSLATE_COMPARE_DEFAULT_MAX_TOKENS = 192;
const TRANSLATE_COMPARE_HISTORY_STORAGE_KEY = 'doppler.translate.compare.history.v1';
const TRANSLATE_COMPARE_CONFIG_VERSION = 2;
const TRANSLATE_COMPARE_ARTIFACT_KIND = 'doppler.translate.compare/v1';
const TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID = 'translategemma-4b-it-q4k-ehf16-af32';
const TRANSLATE_COMPARE_DEFAULT_TJS_DTYPE = 'q4';
const TRANSLATE_COMPARE_MAX_HISTORY = 12;
const TRANSLATE_COMPARE_ENGINES_CONFIG_URL = typeof window === 'object' && window.location?.origin
  ? new URL('/benchmarks/vendors/compare-engines.config.json', window.location.origin).toString()
  : new URL('../benchmarks/vendors/compare-engines.config.json', import.meta.url).toString();
const TRANSLATE_COMPARE_ENGINE_OPTIONS = Object.freeze([
  Object.freeze({ id: 'doppler', label: 'Doppler.js' }),
  Object.freeze({ id: 'transformersjs', label: 'Transformers.js v4' }),
]);
const TRANSLATE_COMPARE_PRESETS = Object.freeze([
  Object.freeze({
    id: 'proof',
    label: 'Proof preset',
    description: 'Public Transformers.js q4 baseline versus Doppler student on the same proof console.',
    lanes: Object.freeze({
      left: Object.freeze({ engine: 'transformersjs', role: 'mapped-baseline' }),
      right: Object.freeze({ engine: 'doppler', role: 'student' }),
    }),
  }),
  Object.freeze({
    id: 'engine-parity',
    label: 'Engine parity',
    description: 'Same TranslateGemma baseline across engines using the public ONNX q4 stack on the left lane.',
    lanes: Object.freeze({
      left: Object.freeze({ engine: 'transformersjs', role: 'mapped-baseline' }),
      right: Object.freeze({ engine: 'doppler', role: 'mapped-baseline' }),
    }),
  }),
  Object.freeze({
    id: 'student-parity',
    label: 'Student parity',
    description: 'Same student slot across engines once the student mapping is configured.',
    lanes: Object.freeze({
      left: Object.freeze({ engine: 'transformersjs', role: 'student-mapped' }),
      right: Object.freeze({ engine: 'doppler', role: 'student' }),
    }),
  }),
  Object.freeze({
    id: 'model-parity',
    label: 'Model parity',
    description: 'Doppler baseline versus Doppler student slot.',
    lanes: Object.freeze({
      left: Object.freeze({ engine: 'doppler', role: 'baseline' }),
      right: Object.freeze({ engine: 'doppler', role: 'student' }),
    }),
  }),
  Object.freeze({
    id: 'custom',
    label: 'Custom compare',
    description: 'Edit each lane directly.',
    lanes: Object.freeze({
      left: Object.freeze({ engine: 'doppler', role: 'baseline' }),
      right: Object.freeze({ engine: 'doppler', role: 'baseline' }),
    }),
  }),
]);
const TRANSLATE_COMPARE_HISTORY_FILTERS = Object.freeze([
  Object.freeze({ id: 'all', label: 'All' }),
  Object.freeze({ id: 'same-model', label: 'Same model' }),
  Object.freeze({ id: 'same-engine', label: 'Same engine' }),
  Object.freeze({ id: 'proof', label: 'Proof preset' }),
]);
const TRANSLATE_COMPARE_SMOKE_SAMPLES = Object.freeze([
  Object.freeze({
    id: 'easy-release-note',
    bucket: 'easy',
    label: 'Release note',
    sourceCode: 'en',
    targetCode: 'es',
    text: 'The patch fixes a memory leak and reduces startup time on low-end laptops.',
    note: 'Straightforward product language.',
  }),
  Object.freeze({
    id: 'easy-travel',
    bucket: 'easy',
    label: 'Travel update',
    sourceCode: 'en',
    targetCode: 'es',
    text: 'Our train leaves at seven, so please arrive at the station fifteen minutes early.',
    note: 'Simple scheduling language.',
  }),
  Object.freeze({
    id: 'easy-support',
    bucket: 'easy',
    label: 'Support policy',
    sourceCode: 'en',
    targetCode: 'es',
    text: 'If your order arrives damaged, send two photos and we will replace it within three business days.',
    note: 'Operational support copy.',
  }),
  Object.freeze({
    id: 'nuanced-idiom',
    bucket: 'nuanced',
    label: 'Idiom',
    sourceCode: 'en',
    targetCode: 'es',
    text: 'We are not trying to boil the ocean; we just need the first release to stop surprising users.',
    note: 'Idiom plus product nuance.',
  }),
  Object.freeze({
    id: 'nuanced-tone',
    bucket: 'nuanced',
    label: 'Tone',
    sourceCode: 'en',
    targetCode: 'es',
    text: 'The proposal is technically correct, but the timing makes it feel more reactive than deliberate.',
    note: 'Subtle stance and tone.',
  }),
  Object.freeze({
    id: 'nuanced-ambiguity',
    bucket: 'nuanced',
    label: 'Ambiguity',
    sourceCode: 'en',
    targetCode: 'es',
    text: 'Jordan told Alex that they should slow down before the review became hostile.',
    note: 'Pronoun ambiguity under pressure.',
  }),
  Object.freeze({
    id: 'domain-runtime',
    bucket: 'domain',
    label: 'Runtime contract',
    sourceCode: 'en',
    targetCode: 'es',
    text: 'Fail closed when the manifest and runtime config disagree about the kernel path.',
    note: 'Runtime-policy vocabulary.',
  }),
  Object.freeze({
    id: 'domain-privacy',
    bucket: 'domain',
    label: 'Privacy copy',
    sourceCode: 'en',
    targetCode: 'es',
    text: 'On-device inference keeps customer messages on the machine instead of routing them through a hosted API.',
    note: 'Privacy and deployment framing.',
  }),
  Object.freeze({
    id: 'edge-awkward-register',
    bucket: 'edge',
    label: 'Register shift',
    sourceCode: 'en',
    targetCode: 'es',
    text: 'Could you make it less corporate but still safe enough for legal to sign off on?',
    note: 'Casual tone mixed with formal constraint.',
  }),
  Object.freeze({
    id: 'edge-double-negative',
    bucket: 'edge',
    label: 'Double negative',
    sourceCode: 'en',
    targetCode: 'es',
    text: 'I am not saying the rollout did not help; I am saying it did not help enough yet.',
    note: 'Negation and contrast.',
  }),
]);
const TRANSLATE_COMPARE_EVIDENCE_FALLBACK = Object.freeze({
  updatedAt: null,
  summary: 'Awaiting frozen Gamma evidence bundle.',
  caution: 'Student metrics and receipts are intentionally blank until the selected checkpoint is frozen.',
  teacher: Object.freeze({
    label: 'Teacher',
    modelId: TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID,
    bleu: null,
    chrf: null,
    sizeBytes: 3167327178,
  }),
  student: Object.freeze({
    label: 'Student',
    modelId: null,
    bleu: null,
    chrf: null,
    sizeBytes: null,
  }),
  receipts: Object.freeze([]),
});
const TRANSFORMERSJS_IMPORT_CANDIDATES = Object.freeze([
  '/node_modules/@huggingface/transformers/dist/transformers.web.min.js',
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1',
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers',
]);
const DEEP_LINK_MODES = new Set([
  'run',
  'translate',
  'embedding',
  'diffusion',
  'energy',
  'distill',
  'models',
  'diagnostics',
]);
const TASK_SET = new Set(['run', 'evaluate']);
const TASK_MODE_ALLOWLIST = Object.freeze({
  run: Object.freeze(['run', 'translate', 'embedding', 'diffusion']),
  evaluate: Object.freeze(['diagnostics', 'distill', 'energy']),
});
const MODE_TASK_MAP = Object.freeze({
  run: 'run',
  translate: 'run',
  embedding: 'run',
  diffusion: 'run',
  diagnostics: 'evaluate',
  distill: 'evaluate',
  energy: 'evaluate',
});
const DEFAULT_TASK_MODE = Object.freeze({
  run: 'run',
  evaluate: 'diagnostics',
});
const SURFACE_SET = new Set(['demo']);
const SURFACE_MODE_ALLOWLIST = Object.freeze({
  demo: new Set([...DEEP_LINK_MODES]),
});
const EMBEDDING_DEMO_DOCUMENT_CATALOG = Object.freeze([
  Object.freeze({
    id: 'doc_webgpu_local',
    title: 'Local-First WebGPU',
    text: 'Local-first AI apps run inference in the browser using WebGPU and store model shards in OPFS for offline performance.',
  }),
  Object.freeze({
    id: 'doc_formal_methods',
    title: 'Formal Methods',
    text: 'Lean proofs can verify termination and memory-safety properties for recursive systems code with clear inductive structure.',
  }),
  Object.freeze({
    id: 'doc_market_qqq',
    title: 'Market Commentary',
    text: 'QQQ reflects large-cap technology exposure; risk management depends on volatility, drawdown tolerance, and rebalance discipline.',
  }),
  Object.freeze({
    id: 'doc_kv_cache',
    title: 'KV Cache Behavior',
    text: 'Transformer decoding reuses key/value cache state; resetting context between runs prevents prompt leakage and keeps measurements independent.',
  }),
  Object.freeze({
    id: 'doc_pkg_delivery',
    title: 'Support Delivery Case',
    text: 'Customers reporting damaged packages need replacement workflows, photo evidence handling, and clear refund timelines in support tooling.',
  }),
  Object.freeze({
    id: 'doc_formal_agent',
    title: 'Verified Agents',
    text: 'Recursive agents can be modeled with inductive types, then proven terminating so orchestration loops do not run forever in production.',
  }),
  Object.freeze({
    id: 'doc_diffusion',
    title: 'Image Generation',
    text: 'Diffusion inference denoises latent tensors over multiple steps, then decodes through a VAE to produce a final image.',
  }),
  Object.freeze({
    id: 'doc_energy_model',
    title: 'Energy Optimization',
    text: 'Energy-based solvers iteratively minimize objective functions and can visualize convergence as energy drops over time.',
  }),
  Object.freeze({
    id: 'doc_data_governance',
    title: 'Data Governance',
    text: 'Local storage policies should track model provenance, hash integrity, and retention windows for reproducible deployments.',
  }),
]);
const EMBEDDING_DEMO_DOCUMENT_COUNT = 3;
const DIFFUSION_STARTER_PROMPTS = Object.freeze([
  'A photo-realistic architectural render of a boutique toy store in Williamsburg, Brooklyn; matte black metal frame, floor-to-ceiling glass, warm wooden shelves with minimalist board games and wooden toys, soft morning sidewalk light.',
  "A vector logo for a software project named Doppler, cyber-industrial and minimalist, deep charcoal and neon teal palette, 90s arcade energy meets modern developer tooling.",
  'A top-down cinematic shot of a disassembled Framework DIY laptop next to a custom mechanical keyboard with translucent keycaps and coiled cables, shallow bokeh, texture-rich PCB details.',
  'A digital artwork of an ouroboros made from glowing fiber-optic cables and circuit board traces, dark background, high contrast, precise luminous edges.',
  'A candid documentary-style photo of a museum visitor looking up at a massive dinosaur skeleton in the American Museum of Natural History, soft natural light, slight desaturation.',
  'A 2022 Audi Q3 with honeycomb mesh grille and low-profile roof racks parked on a cobblestone street in DUMBO, cinematic automotive lighting, crisp reflections.',
  'A macro shot of a complex tabletop strategy game in progress, wooden pieces, intricate cards, polyhedral dice on dark walnut, warm cozy lighting.',
  'A macro, high-contrast black-and-white photo of a Somalia Elephant silver coin, emphasis on skin texture engraving and metallic edge luster.',
  'A surreal editorial scene of castles built inside a browser sandbox, translucent walls, strict geometric boundaries, glowing checker lines, dramatic side lighting.',
  'A futuristic operations room visualizing proofware: deterministic acceptance and rejection traces projected as layered HUD panels over a dark grid.',
  'A cinematic concept art frame of a local-first AI workstation with WebGPU kernels flowing into verification checkpoints, neon teal accents, restrained composition.',
  'An abstract infographic-style artwork showing interface to reasoning to checker flow as three distinct luminous channels converging into a green accept gate.',
  'A moody Brooklyn night street with wet pavement reflections, matte storefronts, minimal signage, and subtle cyber-industrial atmosphere.',
  'A technical poster aesthetic featuring Lean theorem symbols and circuit motifs, monochrome base with sharp teal highlights, clean negative space.',
]);
const DIFFUSION_NEGATIVE_STARTER_PROMPTS = Object.freeze([
  'blurry, lowres, jpeg artifacts, noisy, text, watermark',
  'deformed anatomy, extra fingers, duplicated limbs, bad hands',
  'overexposed, underexposed, washed colors, poor contrast',
  'cropped subject, out of frame, tilted horizon',
  'cartoonish proportions, unrealistic shadows, flat lighting',
  'muddy details, over-smoothing, plastic skin',
  'logo, signature, timestamp, subtitles',
  'distorted perspective, warped geometry, stretched objects',
  'banding, posterization, chromatic aberration',
  'cluttered background, messy composition, visual noise',
  'unreadable typography, gibberish text, malformed letters',
  'double pupils, asymmetrical eyes, broken facial structure',
  'incorrect limb count, fused fingers, disconnected joints',
  'overprocessed HDR, halo edges, ringing artifacts',
  'flat depth, no focal separation, poor subject isolation',
  'compression blocks, aliasing, moire patterns, scan lines',
]);

function resolveText(value, defaultValue = '') {
  if (value == null) return defaultValue;
  return String(value).trim();
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function normalizeTranslateLanguageCode(code, defaultLanguageCode = DEFAULT_TRANSLATE_SOURCE) {
  const requested = resolveText(code, '');
  if (TRANSLATE_LANGUAGE_OPTIONS.some((entry) => entry.code === requested)) {
    return requested;
  }
  return defaultLanguageCode;
}

function populateTranslateLanguageSelect(selectEl, selectedCode) {
  if (!(selectEl instanceof HTMLSelectElement)) return;
  const previousCode = resolveText(selectedCode, selectEl.value || DEFAULT_TRANSLATE_SOURCE);
  selectEl.innerHTML = '';
  for (const entry of TRANSLATE_LANGUAGE_OPTIONS) {
    const option = document.createElement('option');
    option.value = entry.code;
    option.textContent = `${entry.name} (${entry.code})`;
    selectEl.appendChild(option);
  }
  selectEl.value = normalizeTranslateLanguageCode(previousCode, DEFAULT_TRANSLATE_SOURCE);
}

function populateTranslateLanguageControls() {
  const sourceSelect = $('translate-source-language');
  const targetSelect = $('translate-target-language');
  populateTranslateLanguageSelect(sourceSelect, DEFAULT_TRANSLATE_SOURCE);
  populateTranslateLanguageSelect(targetSelect, DEFAULT_TRANSLATE_TARGET);
  if (sourceSelect && targetSelect && sourceSelect.value === targetSelect.value) {
    targetSelect.value = DEFAULT_TRANSLATE_TARGET;
  }
}

function swapTranslateLanguages() {
  const sourceSelect = $('translate-source-language');
  const targetSelect = $('translate-target-language');
  if (!(sourceSelect instanceof HTMLSelectElement) || !(targetSelect instanceof HTMLSelectElement)) {
    return;
  }
  const sourceCode = normalizeTranslateLanguageCode(sourceSelect.value, DEFAULT_TRANSLATE_SOURCE);
  const targetCode = normalizeTranslateLanguageCode(targetSelect.value, DEFAULT_TRANSLATE_TARGET);
  sourceSelect.value = targetCode;
  targetSelect.value = sourceCode;
}

function getTranslateLanguageSelection() {
  const sourceSelect = $('translate-source-language');
  const targetSelect = $('translate-target-language');
  const sourceCode = normalizeTranslateLanguageCode(sourceSelect?.value, DEFAULT_TRANSLATE_SOURCE);
  let targetCode = normalizeTranslateLanguageCode(targetSelect?.value, DEFAULT_TRANSLATE_TARGET);
  if (targetCode === sourceCode) {
    targetCode = sourceCode === DEFAULT_TRANSLATE_TARGET
      ? DEFAULT_TRANSLATE_SOURCE
      : DEFAULT_TRANSLATE_TARGET;
    if (targetSelect instanceof HTMLSelectElement) {
      targetSelect.value = targetCode;
    }
  }
  return { sourceCode, targetCode };
}

function getTranslateLanguageName(code) {
  const normalized = normalizeTranslateLanguageCode(code, DEFAULT_TRANSLATE_SOURCE);
  const match = TRANSLATE_LANGUAGE_OPTIONS.find((entry) => entry.code === normalized);
  return match?.name || normalized;
}

function getCompareLaneIds() {
  return ['left', 'right'];
}

function getCompareLane(laneId) {
  if (!laneId || !state.compareLanes || typeof state.compareLanes !== 'object') {
    return null;
  }
  return state.compareLanes[laneId] || null;
}

function createCompareRuntimeLane(label) {
  return {
    engine: 'doppler',
    modelId: null,
    tjsModelId: '',
    tjsDtype: TRANSLATE_COMPARE_DEFAULT_TJS_DTYPE,
    label,
    status: 'Idle',
    statusTone: 'info',
    output: '',
    metrics: null,
    error: null,
    pipeline: null,
    pipelineModelId: null,
    tjsGenerator: null,
    tjsGeneratorKey: null,
  };
}

function ensureTranslateCompareRuntimeState() {
  if (!state.compareLanes || typeof state.compareLanes !== 'object') {
    state.compareLanes = {
      left: createCompareRuntimeLane('Left Lane'),
      right: createCompareRuntimeLane('Right Lane'),
    };
  }
  if (!state.compareLanes.left) {
    state.compareLanes.left = createCompareRuntimeLane('Left Lane');
  }
  if (!state.compareLanes.right) {
    state.compareLanes.right = createCompareRuntimeLane('Right Lane');
  }
  if (!Array.isArray(state.compareHistory)) {
    state.compareHistory = [];
  }
  if (!resolveText(state.compareHistoryFilter, '')) {
    state.compareHistoryFilter = 'all';
  }
  if (!Array.isArray(state.compareProfiles)) {
    state.compareProfiles = [];
  }
  if (!state.compareEvidence || typeof state.compareEvidence !== 'object') {
    state.compareEvidence = cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK);
  }
  if (state.activeCompareSmokeSampleId != null && !resolveText(state.activeCompareSmokeSampleId, '')) {
    state.activeCompareSmokeSampleId = null;
  }
  if (state.lastCompareArtifact != null && typeof state.lastCompareArtifact !== 'object') {
    state.lastCompareArtifact = null;
  }
}

function setCompareLaneStatus(laneId, message, tone = 'info') {
  const lane = getCompareLane(laneId);
  if (!lane) return;
  lane.status = resolveText(message, 'Idle');
  lane.statusTone = tone || 'info';
}

function clearCompareLaneResult(laneId) {
  const lane = getCompareLane(laneId);
  if (!lane) return;
  lane.output = '';
  lane.metrics = null;
  lane.error = null;
  setCompareLaneStatus(laneId, 'Idle', 'info');
}

function isTranslateCompareEnabled() {
  return state.uiMode === 'translate' && state.compareEnabled === true;
}

function getTranslateCompareStudentModelId() {
  const explicit = readGlobalString('__DOPPLER_TRANSLATE_COMPARE_STUDENT_MODEL_ID');
  if (explicit) {
    return explicit;
  }
  const evidenceModelId = resolveText(state.compareEvidence?.student?.modelId, '');
  if (evidenceModelId) {
    return evidenceModelId;
  }
  const activeTranslateModelId = resolveText(state.modeModelId?.translate, '');
  if (activeTranslateModelId && activeTranslateModelId !== TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID) {
    return activeTranslateModelId;
  }
  for (const modelId of state.registeredModelIds || []) {
    if (!modelId || modelId === TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID) continue;
    const normalizedType = normalizeModelType(state.modelTypeCache?.[modelId]);
    if (isCompatibleModelType(normalizedType, 'translate')) {
      return modelId;
    }
  }
  return null;
}

function getMappedCompareBaselineProfile() {
  const profiles = Array.isArray(state.compareProfiles) ? state.compareProfiles : [];
  return profiles.find((entry) => (
    entry?.dopplerModelId === TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID
    && resolveText(entry?.defaultTjsModelId, '')
  )) || null;
}

function findCompareProfileByDopplerModelId(modelId) {
  const normalizedModelId = resolveText(modelId, '');
  if (!normalizedModelId) return null;
  return (state.compareProfiles || []).find((entry) => entry?.dopplerModelId === normalizedModelId) || null;
}

function resolveTranslateCompareRole(role) {
  const mappedBaseline = getMappedCompareBaselineProfile();
  const studentModelId = getTranslateCompareStudentModelId();
  if (role === 'baseline') {
    return {
      modelId: TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID,
      tjsModelId: resolveText(findCompareProfileByDopplerModelId(TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID)?.defaultTjsModelId, ''),
    };
  }
  if (role === 'student') {
    return {
      modelId: studentModelId,
      tjsModelId: resolveText(findCompareProfileByDopplerModelId(studentModelId)?.defaultTjsModelId, ''),
    };
  }
  if (role === 'mapped-baseline') {
    return {
      modelId: mappedBaseline?.dopplerModelId || TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID,
      tjsModelId: resolveText(mappedBaseline?.defaultTjsModelId, ''),
    };
  }
  if (role === 'student-mapped') {
    return {
      modelId: studentModelId,
      tjsModelId: resolveText(findCompareProfileByDopplerModelId(studentModelId)?.defaultTjsModelId, ''),
    };
  }
  return {
    modelId: null,
    tjsModelId: '',
  };
}

function serializeTranslateCompareArtifactPayload(artifact) {
  if (!artifact || typeof artifact !== 'object') {
    return null;
  }
  const lanes = {};
  for (const laneId of getCompareLaneIds()) {
    const lane = artifact?.lanes?.[laneId] || {};
    lanes[laneId] = {
      engine: resolveText(lane.engine, ''),
      modelId: resolveText(lane.modelId, ''),
      modelLabel: resolveText(lane.modelLabel, laneId),
      tjsModelId: resolveText(lane.tjsModelId, ''),
      roleLabel: resolveText(lane.roleLabel, 'custom'),
      status: resolveText(lane.status, ''),
      output: String(lane.output || ''),
      metrics: lane.metrics || null,
      error: lane.error || null,
    };
  }
  return {
    schemaVersion: Number.isFinite(Number(artifact.schemaVersion))
      ? Number(artifact.schemaVersion)
      : TRANSLATE_COMPARE_CONFIG_VERSION,
    kind: resolveText(artifact.kind, TRANSLATE_COMPARE_ARTIFACT_KIND),
    artifactId: resolveText(artifact.artifactId, ''),
    createdAt: resolveText(artifact.createdAt, ''),
    shareUrl: resolveText(artifact.shareUrl, '') || null,
    request: artifact.request && typeof artifact.request === 'object'
      ? {
        prompt: String(artifact.request.prompt || ''),
        sourceCode: resolveText(artifact.request.sourceCode, DEFAULT_TRANSLATE_SOURCE),
        sourceName: resolveText(artifact.request.sourceName, DEFAULT_TRANSLATE_SOURCE),
        targetCode: resolveText(artifact.request.targetCode, DEFAULT_TRANSLATE_TARGET),
        targetName: resolveText(artifact.request.targetName, DEFAULT_TRANSLATE_TARGET),
        options: artifact.request.options && typeof artifact.request.options === 'object'
          ? artifact.request.options
          : {},
        presetId: resolveText(artifact.request.presetId, 'custom'),
      }
      : {
        prompt: '',
        sourceCode: DEFAULT_TRANSLATE_SOURCE,
        sourceName: DEFAULT_TRANSLATE_SOURCE,
        targetCode: DEFAULT_TRANSLATE_TARGET,
        targetName: DEFAULT_TRANSLATE_TARGET,
        options: {},
        presetId: 'custom',
      },
    environment: artifact.environment && typeof artifact.environment === 'object'
      ? artifact.environment
      : {},
    evidence: artifact.evidence && typeof artifact.evidence === 'object'
      ? {
        updatedAt: resolveText(artifact.evidence.updatedAt, '') || null,
        summary: resolveText(artifact.evidence.summary, ''),
        receipts: Array.isArray(artifact.evidence.receipts) ? artifact.evidence.receipts : [],
      }
      : {
        updatedAt: null,
        summary: '',
        receipts: [],
      },
    summary: artifact.summary && typeof artifact.summary === 'object' ? artifact.summary : {},
    lanes,
  };
}

function serializeTranslateCompareHistoryEntry(entry) {
  const lanes = {};
  for (const laneId of getCompareLaneIds()) {
    const lane = entry?.lanes?.[laneId] || {};
    lanes[laneId] = {
      engine: resolveText(lane.engine, ''),
      modelId: resolveText(lane.modelId, ''),
      tjsModelId: resolveText(lane.tjsModelId, ''),
      label: resolveText(lane.label, laneId),
      status: resolveText(lane.status, ''),
      output: String(lane.output || ''),
      metrics: lane.metrics || null,
      error: lane.error || null,
    };
  }
  return {
    id: resolveText(entry?.id, ''),
    createdAt: resolveText(entry?.createdAt, ''),
    sourceCode: resolveText(entry?.sourceCode, DEFAULT_TRANSLATE_SOURCE),
    targetCode: resolveText(entry?.targetCode, DEFAULT_TRANSLATE_TARGET),
    prompt: String(entry?.prompt || ''),
    presetId: resolveText(entry?.presetId, 'custom'),
    artifact: serializeTranslateCompareArtifactPayload(entry?.artifact),
    lanes,
  };
}

function persistTranslateCompareHistory() {
  if (typeof localStorage === 'undefined') return;
  try {
    const payload = {
      schemaVersion: TRANSLATE_COMPARE_CONFIG_VERSION,
      history: (state.compareHistory || []).slice(0, TRANSLATE_COMPARE_MAX_HISTORY).map(serializeTranslateCompareHistoryEntry),
    };
    localStorage.setItem(TRANSLATE_COMPARE_HISTORY_STORAGE_KEY, JSON.stringify(payload));
  } catch (error) {
    log.warn('DopplerDemo', `Compare history persistence skipped: ${error.message}`);
  }
}

function hydrateTranslateCompareHistory() {
  if (typeof localStorage === 'undefined') return;
  try {
    const raw = localStorage.getItem(TRANSLATE_COMPARE_HISTORY_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    const rows = Array.isArray(parsed?.history) ? parsed.history : [];
    state.compareHistory = rows.map(serializeTranslateCompareHistoryEntry).slice(0, TRANSLATE_COMPARE_MAX_HISTORY);
    state.lastCompareArtifact = state.compareHistory[0]?.artifact || null;
  } catch (error) {
    log.warn('DopplerDemo', `Compare history restore skipped: ${error.message}`);
  }
}

async function loadTranslateCompareProfiles() {
  try {
    const response = await fetch(TRANSLATE_COMPARE_ENGINES_CONFIG_URL, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    const rows = Array.isArray(payload?.modelProfiles) ? payload.modelProfiles : [];
    state.compareProfiles = rows.map((entry) => ({
      dopplerModelId: resolveText(entry?.dopplerModelId, ''),
      defaultTjsModelId: resolveText(entry?.defaultTjsModelId, ''),
      defaultKernelPath: resolveText(entry?.defaultKernelPath, ''),
      modelBaseDir: resolveText(entry?.modelBaseDir, ''),
      defaultDopplerSurface: resolveText(entry?.defaultDopplerSurface, 'auto'),
    }));
  } catch (error) {
    state.compareProfiles = [];
    log.warn('DopplerDemo', `Compare profiles unavailable: ${error.message}`);
  }
}

function normalizeTranslateCompareEvidence(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK);
  }
  return {
    ...cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK),
    ...payload,
    teacher: {
      ...cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK.teacher),
      ...(payload.teacher && typeof payload.teacher === 'object' ? payload.teacher : {}),
    },
    student: {
      ...cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK.student),
      ...(payload.student && typeof payload.student === 'object' ? payload.student : {}),
    },
    receipts: Array.isArray(payload.receipts) ? payload.receipts : cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK.receipts),
  };
}

async function loadTranslateCompareEvidence() {
  const embedded = globalThis?.__DOPPLER_TRANSLATE_COMPARE_EVIDENCE__;
  if (embedded && typeof embedded === 'object' && !Array.isArray(embedded)) {
    state.compareEvidence = normalizeTranslateCompareEvidence(embedded);
    return;
  }

  const evidenceUrl = readGlobalString('__DOPPLER_TRANSLATE_COMPARE_EVIDENCE_URL');
  if (!evidenceUrl) {
    state.compareEvidence = cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK);
    return;
  }

  try {
    const response = await fetch(evidenceUrl, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    state.compareEvidence = normalizeTranslateCompareEvidence(payload);
  } catch (error) {
    state.compareEvidence = cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK);
    log.warn('DopplerDemo', `Compare evidence unavailable: ${error.message}`);
  }
}

function buildTranslateInstructionPrompt(prompt, sourceCode, targetCode) {
  return `Translate the following from ${sourceCode} to ${targetCode}. Output only the translation, no explanation.\n\n${prompt}`;
}

function buildTransformersTranslatePrompt(prompt, sourceCode, targetCode, modelHint = '') {
  const normalizedHint = resolveText(modelHint, '').toLowerCase();
  if (normalizedHint.includes('translategemma')) {
    return formatChatMessages(
      createTranslateTextRequest(prompt, sourceCode, targetCode).messages,
      'translategemma'
    );
  }
  return buildTranslateInstructionPrompt(prompt, sourceCode, targetCode);
}

function getTranslateCompatibleRegisteredModelIds() {
  const ids = [];
  for (const modelId of state.registeredModelIds || []) {
    const normalizedType = normalizeModelType(state.modelTypeCache?.[modelId]);
    if (isCompatibleModelType(normalizedType, 'translate')) {
      ids.push(modelId);
    }
  }
  return ids;
}

function getTranslateComparePreset(presetId) {
  const normalizedId = resolveText(presetId, 'proof');
  return TRANSLATE_COMPARE_PRESETS.find((entry) => entry.id === normalizedId)
    || TRANSLATE_COMPARE_PRESETS[0];
}

function getTranslateComparePresetNote(presetId) {
  const preset = getTranslateComparePreset(presetId);
  if (preset.id === 'engine-parity' && !getMappedCompareBaselineProfile()) {
    return `${preset.description} The UI will fail closed until a baseline mapping is configured.`;
  }
  const usesStudentRole = ['left', 'right'].some((laneId) => {
    const role = resolveText(preset?.lanes?.[laneId]?.role, '');
    return role === 'student' || role === 'student-mapped';
  });
  if (usesStudentRole && !resolveText(getTranslateCompareStudentModelId(), '')) {
    return `${preset.description} Waiting for the student artifact and any TJS mapping.`;
  }
  return preset.description;
}

function getTranslateCompareModelLabel(modelId) {
  const normalizedModelId = resolveText(modelId, '');
  if (!normalizedModelId) return 'Select model';
  const quickEntry = (state.quickModelCatalog || []).find((entry) => entry?.modelId === normalizedModelId);
  return resolveText(quickEntry?.label, formatModelIdLabel(normalizedModelId));
}

function resolveCompareModelSizeBytes(modelId) {
  const normalizedModelId = resolveText(modelId, '');
  if (!normalizedModelId) return null;
  const quickEntry = (state.quickModelCatalog || []).find((entry) => entry?.modelId === normalizedModelId);
  if (Number.isFinite(quickEntry?.sizeBytes)) {
    return quickEntry.sizeBytes;
  }
  const evidenceTeacherModelId = resolveText(state.compareEvidence?.teacher?.modelId, '');
  if (normalizedModelId === evidenceTeacherModelId && Number.isFinite(state.compareEvidence?.teacher?.sizeBytes)) {
    return state.compareEvidence.teacher.sizeBytes;
  }
  const evidenceStudentModelId = resolveText(state.compareEvidence?.student?.modelId, '');
  if (normalizedModelId === evidenceStudentModelId && Number.isFinite(state.compareEvidence?.student?.sizeBytes)) {
    return state.compareEvidence.student.sizeBytes;
  }
  return null;
}

function resolveTransformersModelIdForLane(lane) {
  if (!lane) return '';
  const explicit = resolveText(lane.tjsModelId, '');
  if (explicit) return explicit;
  return resolveText(findCompareProfileByDopplerModelId(lane.modelId)?.defaultTjsModelId, '');
}

function formatCompareMetricMs(value) {
  return Number.isFinite(value) ? `${Math.round(value)} ms` : '--';
}

function formatCompareMetricRate(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} tok/s` : '--';
}

function formatCompareMetricBytes(value) {
  return Number.isFinite(value) ? formatBytes(value) : '--';
}

function formatCompareTimestamp(isoString) {
  if (!isoString) return '--';
  const timestamp = Date.parse(isoString);
  if (!Number.isFinite(timestamp)) return '--';
  try {
    return new Intl.DateTimeFormat(undefined, {
      hour: 'numeric',
      minute: '2-digit',
      month: 'short',
      day: 'numeric',
    }).format(new Date(timestamp));
  } catch {
    return new Date(timestamp).toISOString();
  }
}

let compareAdapterLabelPromise = null;

async function resolveWebGpuDeviceLabel(prefix = 'WebGPU') {
  if (compareAdapterLabelPromise) {
    const resolved = await compareAdapterLabelPromise;
    return resolved || prefix;
  }
  compareAdapterLabelPromise = (async () => {
    try {
      const adapter = await globalThis?.navigator?.gpu?.requestAdapter?.();
      if (!adapter) {
        return prefix;
      }
      const info = adapter.info || (await adapter.requestAdapterInfo?.()) || {};
      const deviceName = resolveText(info.description || info.device, '');
      const vendorName = resolveText(info.vendor, '');
      const suffix = [vendorName, deviceName].filter(Boolean).join(' ');
      return suffix ? `${prefix} · ${suffix}` : prefix;
    } catch {
      return prefix;
    }
  })();
  const resolved = await compareAdapterLabelPromise;
  return resolved || prefix;
}

function getCompareLaneSelectId(laneId) {
  return laneId === 'left' ? 'translate-left-model' : 'translate-right-model';
}

function getCompareLaneEngineSelectId(laneId) {
  return laneId === 'left' ? 'translate-left-engine' : 'translate-right-engine';
}

function populateCompareLaneModelSelect(laneId) {
  const lane = getCompareLane(laneId);
  const select = $(getCompareLaneSelectId(laneId));
  if (!(select instanceof HTMLSelectElement) || !lane) return;
  const previousValue = resolveText(lane.modelId, '');
  select.innerHTML = '';
  const options = [];

  if (lane.engine === 'transformersjs') {
    for (const profile of state.compareProfiles || []) {
      const dopplerModelId = resolveText(profile?.dopplerModelId, '');
      const tjsModelId = resolveText(profile?.defaultTjsModelId, '');
      if (!dopplerModelId || !tjsModelId) continue;
      options.push({
        value: dopplerModelId,
        label: `${getTranslateCompareModelLabel(dopplerModelId)} · ${tjsModelId}`,
      });
    }
  } else {
    for (const modelId of getTranslateCompatibleRegisteredModelIds()) {
      options.push({
        value: modelId,
        label: getTranslateCompareModelLabel(modelId),
      });
    }
  }

  if (options.length === 0) {
    const emptyOption = document.createElement('option');
    emptyOption.value = '';
    emptyOption.textContent = lane.engine === 'transformersjs'
      ? 'No mapped Transformers.js profiles'
      : 'No translate models imported';
    select.appendChild(emptyOption);
    lane.modelId = '';
    return;
  }

  for (const optionInfo of options) {
    const option = document.createElement('option');
    option.value = optionInfo.value;
    option.textContent = optionInfo.label;
    option.title = optionInfo.label;
    select.appendChild(option);
  }
  const nextValue = options.some((entry) => entry.value === previousValue)
    ? previousValue
    : options[0].value;
  select.value = nextValue;
  lane.modelId = nextValue;
  if (lane.engine === 'transformersjs') {
    lane.tjsModelId = resolveTransformersModelIdForLane(lane);
  }
}

function renderTranslateCompareSelectors() {
  for (const laneId of getCompareLaneIds()) {
    const lane = getCompareLane(laneId);
    const engineSelect = $(getCompareLaneEngineSelectId(laneId));
    if (engineSelect instanceof HTMLSelectElement && lane) {
      engineSelect.value = lane.engine;
    }
    populateCompareLaneModelSelect(laneId);
  }
}

function renderTranslateCompareEvidence() {
  const evidence = state.compareEvidence || TRANSLATE_COMPARE_EVIDENCE_FALLBACK;
  setText($('translate-proof-bundle'), evidence.updatedAt ? `Frozen ${evidence.updatedAt}` : 'Awaiting frozen scoreboard');
  setText(
    $('translate-proof-delta'),
    Number.isFinite(evidence?.teacher?.bleu) && Number.isFinite(evidence?.student?.bleu)
      ? `${(evidence.student.bleu - evidence.teacher.bleu).toFixed(2)} BLEU`
      : 'Delta pending'
  );
  setText($('translate-proof-claim'), evidence.summary || TRANSLATE_COMPARE_EVIDENCE_FALLBACK.summary);
  setText($('translate-proof-source'), evidence.updatedAt ? `Updated ${evidence.updatedAt}` : 'No receipt loaded');
  setText($('translate-evidence-teacher-bleu'), Number.isFinite(evidence?.teacher?.bleu) ? evidence.teacher.bleu.toFixed(2) : '--');
  setText($('translate-evidence-student-bleu'), Number.isFinite(evidence?.student?.bleu) ? evidence.student.bleu.toFixed(2) : '--');
  setText($('translate-evidence-teacher-chrf'), Number.isFinite(evidence?.teacher?.chrf) ? evidence.teacher.chrf.toFixed(2) : '--');
  setText($('translate-evidence-student-chrf'), Number.isFinite(evidence?.student?.chrf) ? evidence.student.chrf.toFixed(2) : '--');
  setText(
    $('translate-evidence-size-delta'),
    Number.isFinite(evidence?.teacher?.sizeBytes) && Number.isFinite(evidence?.student?.sizeBytes)
      ? `${formatBytes(evidence.teacher.sizeBytes - evidence.student.sizeBytes)} saved`
      : '--'
  );
  setText(
    $('translate-evidence-artifact'),
    Array.isArray(evidence?.receipts) && evidence.receipts.length > 0
      ? String(evidence.receipts[0].label || evidence.receipts[0].href || 'Open receipt')
      : 'Pending'
  );
}

function normalizeTranslateCompareHistoryFilter(filterId) {
  const normalized = resolveText(filterId, 'all');
  return TRANSLATE_COMPARE_HISTORY_FILTERS.some((entry) => entry.id === normalized)
    ? normalized
    : 'all';
}

function getTranslateCompareLaneRoleLabel({ presetId, laneId, modelId }) {
  const presetRole = resolveText(getTranslateComparePreset(presetId)?.lanes?.[laneId]?.role, '');
  if (presetRole === 'baseline' || presetRole === 'mapped-baseline') {
    return 'baseline';
  }
  if (presetRole === 'student' || presetRole === 'student-mapped') {
    return resolveText(modelId, '') ? 'student' : 'student slot';
  }
  if (resolveText(modelId, '') === TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID) {
    return 'baseline';
  }
  const studentModelId = resolveText(getTranslateCompareStudentModelId(), '');
  if (studentModelId && resolveText(modelId, '') === studentModelId) {
    return 'student';
  }
  return 'custom';
}

function getTranslateCompareEntryLaneRoleLabel(entry, laneId) {
  const artifactRole = resolveText(entry?.artifact?.lanes?.[laneId]?.roleLabel, '');
  if (artifactRole) {
    return artifactRole;
  }
  return getTranslateCompareLaneRoleLabel({
    presetId: entry?.presetId,
    laneId,
    modelId: entry?.lanes?.[laneId]?.modelId,
  });
}

function summarizeTranslateCompareHistoryEntry(entry) {
  const left = entry?.lanes?.left || {};
  const right = entry?.lanes?.right || {};
  const leftTotalMs = Number(left?.metrics?.totalMs);
  const rightTotalMs = Number(right?.metrics?.totalMs);
  const leftSizeBytes = Number(left?.metrics?.sizeBytes);
  const rightSizeBytes = Number(right?.metrics?.sizeBytes);
  const errorLaneIds = getCompareLaneIds().filter((laneId) => {
    const lane = entry?.lanes?.[laneId];
    return resolveText(lane?.status, '').toLowerCase() === 'error' || lane?.error != null;
  });
  let fasterLaneId = null;
  if (Number.isFinite(leftTotalMs) && Number.isFinite(rightTotalMs) && leftTotalMs !== rightTotalMs) {
    fasterLaneId = leftTotalMs < rightTotalMs ? 'left' : 'right';
  }
  let smallerLaneId = null;
  if (Number.isFinite(leftSizeBytes) && Number.isFinite(rightSizeBytes) && leftSizeBytes !== rightSizeBytes) {
    smallerLaneId = leftSizeBytes < rightSizeBytes ? 'left' : 'right';
  }
  const sameModel = resolveText(left.modelId, '') !== '' && resolveText(left.modelId, '') === resolveText(right.modelId, '');
  const sameEngine = resolveText(left.engine, '') !== '' && resolveText(left.engine, '') === resolveText(right.engine, '');
  const roleLabels = getCompareLaneIds().map((laneId) => getTranslateCompareEntryLaneRoleLabel(entry, laneId));
  return {
    fasterLaneId,
    smallerLaneId,
    sameModel,
    sameEngine,
    hasBaseline: roleLabels.includes('baseline'),
    hasStudent: roleLabels.includes('student') || roleLabels.includes('student slot'),
    errorLaneIds,
  };
}

function matchesTranslateCompareHistoryFilter(entry, filterId) {
  const normalizedFilter = normalizeTranslateCompareHistoryFilter(filterId);
  const summary = summarizeTranslateCompareHistoryEntry(entry);
  if (normalizedFilter === 'same-model') {
    return summary.sameModel;
  }
  if (normalizedFilter === 'same-engine') {
    return summary.sameEngine;
  }
  if (normalizedFilter === 'proof') {
    return resolveText(entry?.presetId, '') === 'proof';
  }
  return true;
}

function getLatestTranslateCompareArtifact() {
  return state.lastCompareArtifact || state.compareHistory?.[0]?.artifact || null;
}

function buildTranslateCompareBrowserLabel() {
  const nav = typeof navigator === 'object' && navigator ? navigator : null;
  const brands = Array.isArray(nav?.userAgentData?.brands)
    ? nav.userAgentData.brands.map((entry) => resolveText(entry?.brand, '')).filter(Boolean)
    : [];
  if (brands.length > 0) {
    return brands.join(' / ');
  }
  const ua = resolveText(nav?.userAgent, '');
  if (!ua) return 'Browser';
  if (ua.includes('Chrome/')) return 'Chrome';
  if (ua.includes('Firefox/')) return 'Firefox';
  if (ua.includes('Safari/')) return 'Safari';
  return ua.split(' ').slice(0, 2).join(' ') || 'Browser';
}

function buildTranslateCompareEnvironmentMetadata() {
  const nav = typeof navigator === 'object' && navigator ? navigator : null;
  return {
    browserLabel: buildTranslateCompareBrowserLabel(),
    userAgent: resolveText(nav?.userAgent, ''),
    language: resolveText(nav?.language, ''),
    languages: Array.isArray(nav?.languages) ? nav.languages.slice() : [],
    platform: resolveText(nav?.platform, ''),
    hardwareConcurrency: Number.isFinite(Number(nav?.hardwareConcurrency))
      ? Number(nav.hardwareConcurrency)
      : null,
    deviceMemoryGb: Number.isFinite(Number(nav?.deviceMemory))
      ? Number(nav.deviceMemory)
      : null,
    devicePixelRatio: Number.isFinite(Number(globalThis?.devicePixelRatio))
      ? Number(globalThis.devicePixelRatio)
      : null,
    webgpuDeviceLabel: resolveText(state.compareDeviceLabel, ''),
    url: typeof window === 'object' ? window.location.href : '',
  };
}

function downloadJsonFile(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function exportTranslateCompareArtifactPayload(artifact) {
  if (!artifact || typeof artifact !== 'object') {
    updateRunStatus('No compare artifact available to export.');
    return;
  }
  const artifactId = resolveText(artifact.artifactId, new Date().toISOString().replace(/[:]/g, '-'));
  downloadJsonFile(`translate-compare-${artifactId}.json`, artifact);
  updateRunStatus(`Exported compare artifact ${artifactId}`);
}

function buildTranslateCompareArtifact(prompt, sourceCode, targetCode, options) {
  const createdAt = new Date().toISOString();
  const lanes = {};
  for (const laneId of getCompareLaneIds()) {
    const lane = getCompareLane(laneId) || {};
    lanes[laneId] = {
      engine: resolveText(lane.engine, ''),
      modelId: resolveText(lane.modelId, ''),
      modelLabel: getTranslateCompareModelLabel(lane.modelId),
      tjsModelId: resolveText(resolveTransformersModelIdForLane(lane), ''),
      roleLabel: getTranslateCompareLaneRoleLabel({
        presetId: state.comparePresetId,
        laneId,
        modelId: lane.modelId,
      }),
      status: resolveText(lane.status, ''),
      output: String(lane.output || ''),
      metrics: lane.metrics || null,
      error: lane.error || null,
    };
  }
  const snapshot = {
    presetId: state.comparePresetId,
    lanes,
  };
  const summary = summarizeTranslateCompareHistoryEntry(snapshot);
  const evidence = state.compareEvidence || TRANSLATE_COMPARE_EVIDENCE_FALLBACK;
  return {
    schemaVersion: TRANSLATE_COMPARE_CONFIG_VERSION,
    kind: TRANSLATE_COMPARE_ARTIFACT_KIND,
    artifactId: crypto?.randomUUID?.() || `${Date.now()}`,
    createdAt,
    shareUrl: buildTranslateDeepLinkUrl(),
    request: {
      prompt,
      sourceCode,
      sourceName: getTranslateLanguageName(sourceCode),
      targetCode,
      targetName: getTranslateLanguageName(targetCode),
      options: {
        temperature: options.temperature,
        topP: options.topP,
        topK: options.topK,
        maxTokens: options.maxTokens,
      },
      presetId: state.comparePresetId || 'custom',
    },
    environment: buildTranslateCompareEnvironmentMetadata(),
    evidence: {
      updatedAt: evidence.updatedAt,
      summary: evidence.summary,
      receipts: Array.isArray(evidence.receipts) ? evidence.receipts : [],
    },
    summary,
    lanes,
  };
}

function renderTranslateCompareReceipts() {
  const artifact = getLatestTranslateCompareArtifact();
  const exportButton = $('translate-compare-export-latest-btn');
  const linksWrap = $('translate-compare-receipts-links');
  setText($('translate-compare-artifact-id'), resolveText(artifact?.artifactId, '') || 'Pending');
  setText(
    $('translate-compare-artifact-env'),
    artifact?.environment?.browserLabel
      ? `${artifact.environment.browserLabel} · ${resolveText(artifact.environment.webgpuDeviceLabel, 'WebGPU')}`
      : 'Awaiting run'
  );
  const receiptCount = Array.isArray(artifact?.evidence?.receipts) ? artifact.evidence.receipts.length : 0;
  setText($('translate-compare-receipt-count'), String(receiptCount));
  if (exportButton instanceof HTMLButtonElement) {
    exportButton.disabled = !artifact;
  }
  if (!linksWrap) return;
  linksWrap.textContent = '';
  if (!artifact || receiptCount === 0) {
    const empty = document.createElement('span');
    empty.className = 'type-caption';
    empty.textContent = 'No benchmark receipts linked yet.';
    linksWrap.appendChild(empty);
    return;
  }
  let appended = 0;
  for (const receipt of artifact.evidence.receipts) {
    const href = resolveText(receipt?.href, '');
    if (!href) {
      const label = resolveText(receipt?.label, '');
      if (!label) continue;
      const span = document.createElement('span');
      span.className = 'type-caption';
      span.textContent = label;
      linksWrap.appendChild(span);
      appended += 1;
      continue;
    }
    const link = document.createElement('a');
    link.href = href;
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = resolveText(receipt?.label, href);
    linksWrap.appendChild(link);
    appended += 1;
  }
  if (appended === 0) {
    const empty = document.createElement('span');
    empty.className = 'type-caption';
    empty.textContent = 'No benchmark receipts linked yet.';
    linksWrap.appendChild(empty);
  }
}

function formatTranslateSmokeBucketLabel(bucket) {
  if (bucket === 'easy') return 'easy';
  if (bucket === 'nuanced') return 'nuanced';
  if (bucket === 'domain') return 'domain';
  if (bucket === 'edge') return 'edge';
  return 'sample';
}

function renderTranslateCompareSmokePanel() {
  const grid = $('translate-smoke-grid');
  if (!grid) return;
  grid.textContent = '';
  for (const sample of TRANSLATE_COMPARE_SMOKE_SAMPLES) {
    const card = document.createElement('article');
    card.className = state.activeCompareSmokeSampleId === sample.id
      ? 'translate-smoke-card is-active'
      : 'translate-smoke-card';
    card.innerHTML = `
      <div class="translate-smoke-card-top">
        <span class="translate-smoke-chip">${escapeHtml(formatTranslateSmokeBucketLabel(sample.bucket))}</span>
        <span class="type-caption">${escapeHtml(sample.label)}</span>
      </div>
      <p class="translate-history-snippet">${escapeHtml(sample.text)}</p>
      <div class="translate-smoke-meta">
        <span class="type-caption">${escapeHtml(getTranslateLanguageName(sample.sourceCode))} -> ${escapeHtml(getTranslateLanguageName(sample.targetCode))}</span>
        <span class="type-caption">${escapeHtml(sample.note)}</span>
      </div>
      <div class="translate-smoke-card-actions">
        <button class="btn btn-small" type="button" data-compare-smoke-sample="${escapeHtml(sample.id)}">Load sample</button>
      </div>
    `;
    grid.appendChild(card);
  }
}

function renderTranslateCompareLane(laneId) {
  const lane = getCompareLane(laneId);
  if (!lane) return;
  const prefix = laneId === 'left' ? 'translate-left' : 'translate-right';
  const metrics = lane.metrics || {};
  const statusEl = $(`${prefix}-status`);
  const badgeEl = $(`${prefix}-badge`);
  const sizeBytes = Number.isFinite(metrics.sizeBytes)
    ? metrics.sizeBytes
    : resolveCompareModelSizeBytes(lane.modelId);
  setText(statusEl, lane.status || 'Idle');
  setText($(`${prefix}-size`), formatCompareMetricBytes(sizeBytes));
  setText($(`${prefix}-load-ms`), formatCompareMetricMs(metrics.modelLoadMs));
  setText($(`${prefix}-ttft`), formatCompareMetricMs(metrics.ttftMs));
  setText($(`${prefix}-decode-rate`), formatCompareMetricRate(metrics.decodeTokensPerSec));
  setText($(`${prefix}-total-ms`), formatCompareMetricMs(metrics.totalMs));
  setText(
    $(`${prefix}-device`),
    resolveText(metrics.deviceLabel, state.compareDeviceLabel || (lane.engine === 'transformersjs' ? 'WebGPU / ORT' : 'WebGPU'))
  );
  setText($(`${prefix}-meta`), resolveText(metrics.metaLabel, lane.error ? 'Run failed' : 'No run yet'));
  setText($(`${prefix}-output`), lane.output || (lane.error ? String(lane.error) : 'Awaiting compare run.'));
  if (statusEl) {
    statusEl.dataset.tone = resolveText(lane.statusTone, 'info');
  }
  setText(
    badgeEl,
    getTranslateCompareLaneRoleLabel({
      presetId: state.comparePresetId,
      laneId,
      modelId: lane.modelId,
    })
  );
}

function renderTranslateCompareHistory() {
  const list = $('translate-history-list');
  if (!list) return;
  const rows = (Array.isArray(state.compareHistory) ? state.compareHistory : [])
    .filter((entry) => matchesTranslateCompareHistoryFilter(entry, state.compareHistoryFilter));
  list.innerHTML = '';

  if (rows.length === 0) {
    const filterLabel = TRANSLATE_COMPARE_HISTORY_FILTERS.find((entry) => entry.id === normalizeTranslateCompareHistoryFilter(state.compareHistoryFilter))?.label || 'All';
    list.innerHTML = `
      <article class="translate-history-card is-placeholder">
        <div class="translate-history-card-top">
          <span class="translate-history-time type-caption">Pending</span>
          <span class="translate-history-badge type-caption">${escapeHtml(filterLabel)}</span>
        </div>
        <p class="translate-history-snippet">Compare receipts will stack here with engine/model labels, timing, and expandable outputs.</p>
        <div class="translate-history-empty type-caption">No history entries match the current filter yet.</div>
      </article>
    `;
    return;
  }

  for (const entry of rows) {
    const summaryState = summarizeTranslateCompareHistoryEntry(entry);
    const card = document.createElement('details');
    card.className = 'translate-history-card';
    const left = entry?.lanes?.left || {};
    const right = entry?.lanes?.right || {};
    const badges = [];
    if (summaryState.fasterLaneId) {
      badges.push({ label: `${summaryState.fasterLaneId} faster`, tone: 'success' });
    }
    if (summaryState.smallerLaneId) {
      badges.push({ label: `${summaryState.smallerLaneId} smaller`, tone: 'success' });
    }
    if (summaryState.sameModel) {
      badges.push({ label: 'same model', tone: 'neutral' });
    }
    if (summaryState.sameEngine) {
      badges.push({ label: 'same engine', tone: 'neutral' });
    }
    if (summaryState.hasBaseline) {
      badges.push({ label: 'baseline', tone: 'neutral' });
    }
    if (summaryState.hasStudent) {
      badges.push({ label: 'student', tone: 'warning' });
    }
    if (summaryState.errorLaneIds.length > 0) {
      badges.push({ label: `error:${summaryState.errorLaneIds.join(',')}`, tone: 'warning' });
    }
    const badgeMarkup = badges.map((badge) => (
      `<span class="translate-history-badge type-caption is-${escapeHtml(badge.tone)}">${escapeHtml(badge.label)}</span>`
    )).join('');
    const receiptLinks = Array.isArray(entry?.artifact?.evidence?.receipts)
      ? entry.artifact.evidence.receipts.filter((receipt) => resolveText(receipt?.href, ''))
      : [];
    const receiptMarkup = receiptLinks.length > 0
      ? receiptLinks.map((receipt) => (
        `<a class="btn btn-small" href="${escapeHtml(receipt.href)}" target="_blank" rel="noopener">${escapeHtml(resolveText(receipt.label, 'Receipt'))}</a>`
      )).join('')
      : '<span class="type-caption">No linked receipts</span>';
    const rawTiming = JSON.stringify({
      left: left.metrics || null,
      right: right.metrics || null,
    }, null, 2);
    const summary = document.createElement('summary');
    summary.innerHTML = `
      <div class="translate-history-card-top">
        <span class="translate-history-time type-caption">${formatCompareTimestamp(entry.createdAt)}</span>
        <span class="translate-history-badge type-caption">${entry.presetId || 'custom'}</span>
      </div>
      <div class="translate-history-badges">${badgeMarkup}</div>
      <p class="translate-history-snippet">${escapeHtml(String(entry.prompt || '').slice(0, 180) || 'No prompt captured.')}</p>
      <div class="translate-history-meta">
        <span class="type-caption">${escapeHtml(getTranslateLanguageName(entry.sourceCode))} -> ${escapeHtml(getTranslateLanguageName(entry.targetCode))}</span>
        <span class="type-caption">${escapeHtml(left.engine || 'left')} vs ${escapeHtml(right.engine || 'right')}</span>
      </div>
    `;
    card.appendChild(summary);

    const body = document.createElement('div');
    body.className = 'translate-history-body';
    body.innerHTML = `
      <div class="translate-history-lane-block">
        <div class="translate-history-meta">
          <span class="type-caption">${escapeHtml(getTranslateCompareModelLabel(left.modelId))} · ${escapeHtml(left.status || 'idle')} · ${escapeHtml(getTranslateCompareEntryLaneRoleLabel(entry, 'left'))}</span>
          <span class="type-caption">load ${formatCompareMetricMs(left.metrics?.modelLoadMs)} · ttft ${formatCompareMetricMs(left.metrics?.ttftMs)} · total ${formatCompareMetricMs(left.metrics?.totalMs)} · ${formatCompareMetricRate(left.metrics?.decodeTokensPerSec)}</span>
        </div>
        <pre class="playground-output-box translate-lane-output-box">${escapeHtml(String(left.output || ''))}</pre>
      </div>
      <div class="translate-history-lane-block">
        <div class="translate-history-meta">
          <span class="type-caption">${escapeHtml(getTranslateCompareModelLabel(right.modelId))} · ${escapeHtml(right.status || 'idle')} · ${escapeHtml(getTranslateCompareEntryLaneRoleLabel(entry, 'right'))}</span>
          <span class="type-caption">load ${formatCompareMetricMs(right.metrics?.modelLoadMs)} · ttft ${formatCompareMetricMs(right.metrics?.ttftMs)} · total ${formatCompareMetricMs(right.metrics?.totalMs)} · ${formatCompareMetricRate(right.metrics?.decodeTokensPerSec)}</span>
        </div>
        <pre class="playground-output-box translate-lane-output-box">${escapeHtml(String(right.output || ''))}</pre>
      </div>
      <div class="translate-history-actions">
        <button class="btn btn-small" type="button" data-compare-history-export="${escapeHtml(entry.id)}">Export JSON</button>
        ${receiptMarkup}
      </div>
      <details class="diagnostics-output-json">
        <summary class="type-caption">Raw timing breakdown</summary>
        <pre class="playground-output-box translate-history-raw">${escapeHtml(rawTiming)}</pre>
      </details>
    `;
    card.appendChild(body);
    list.appendChild(card);
  }
}

function syncTranslateCompareToggleButtons() {
  const enabled = isTranslateCompareEnabled();
  document.querySelectorAll('[data-translate-view], [data-translate-layout]').forEach((button) => {
    const target = button?.dataset?.translateView || button?.dataset?.translateLayout;
    const isCompareTarget = target === 'compare';
    button.classList.toggle('is-active', enabled === isCompareTarget);
  });
}

function syncTranslateCompareUI() {
  const compareShell = $('translate-compare-shell');
  const singleOutputBox = $('run-output')?.closest('.playground-output');
  const presetSelect = $('translate-compare-preset');
  const presetNote = $('translate-compare-preset-note');
  const runButton = $('translate-compare-run-btn');
  const exportButton = $('translate-compare-export-btn');
  const shareButton = $('translate-compare-share-btn');
  const enabled = isTranslateCompareEnabled();
  setHidden(compareShell, !enabled);
  setHidden(singleOutputBox, enabled);
  if (presetSelect instanceof HTMLSelectElement) {
    presetSelect.value = state.comparePresetId || 'proof';
  }
  setText(presetNote, getTranslateComparePresetNote(state.comparePresetId || 'proof'));
  if (runButton instanceof HTMLButtonElement) {
    runButton.disabled = state.compareGenerating || state.compareLoading;
  }
  if (exportButton instanceof HTMLButtonElement) {
    exportButton.disabled = state.compareGenerating || state.compareLoading || !getLatestTranslateCompareArtifact();
  }
  if (shareButton instanceof HTMLButtonElement) {
    shareButton.disabled = state.compareGenerating || state.compareLoading;
  }
  document.querySelectorAll('[data-compare-history-filter]').forEach((button) => {
    const filterId = normalizeTranslateCompareHistoryFilter(button?.dataset?.compareHistoryFilter);
    button.classList.toggle('is-active', filterId === normalizeTranslateCompareHistoryFilter(state.compareHistoryFilter));
  });
  syncTranslateCompareToggleButtons();
  renderTranslateCompareEvidence();
  renderTranslateCompareReceipts();
  renderTranslateCompareSmokePanel();
  renderTranslateCompareSelectors();
  renderTranslateCompareHistory();
  for (const laneId of getCompareLaneIds()) {
    renderTranslateCompareLane(laneId);
  }
}

async function applyTranslateComparePreset(presetId, options = {}) {
  ensureTranslateCompareRuntimeState();
  const preset = getTranslateComparePreset(presetId);
  state.comparePresetId = preset.id;
  const { preserveExisting = false } = options;

  for (const laneId of getCompareLaneIds()) {
    const lane = getCompareLane(laneId);
    const lanePreset = preset.lanes?.[laneId] || {};
    const resolved = resolveTranslateCompareRole(lanePreset.role);
    if (!preserveExisting || !resolveText(lane.modelId, '')) {
      lane.engine = resolveText(lanePreset.engine, lane.engine || 'doppler');
      lane.modelId = resolveText(resolved.modelId, lane.modelId || '');
      lane.tjsModelId = resolveText(resolved.tjsModelId, lane.tjsModelId || '');
    }
    clearCompareLaneResult(laneId);
  }
  syncTranslateCompareUI();
}

let transformersRuntimePromise = null;

async function loadTransformersJsRuntime() {
  if (transformersRuntimePromise) {
    return transformersRuntimePromise;
  }
  transformersRuntimePromise = (async () => {
    let lastError = null;
    for (const candidate of TRANSFORMERSJS_IMPORT_CANDIDATES) {
      try {
        const runtime = await import(candidate);
        if (!runtime?.pipeline) {
          throw new Error('module did not expose pipeline()');
        }
        if (runtime.env && typeof runtime.env === 'object') {
          runtime.env.allowLocalModels = false;
          runtime.env.allowRemoteModels = true;
        }
        return runtime;
      } catch (error) {
        lastError = error;
      }
    }
    throw new Error(`Transformers.js runtime unavailable: ${lastError?.message || 'unknown import failure'}`);
  })();
  return transformersRuntimePromise;
}

async function unloadCompareLaneRuntime(laneId) {
  const lane = getCompareLane(laneId);
  if (!lane) return;
  if (lane.pipeline) {
    try {
      await lane.pipeline.unload?.();
    } catch (error) {
      log.warn('DopplerDemo', `Compare lane unload failed (${laneId}): ${error.message}`);
    }
  }
  lane.pipeline = null;
  lane.pipelineModelId = null;
  lane.tjsGenerator = null;
  lane.tjsGeneratorKey = null;
}

async function unloadAllCompareLaneRuntimes() {
  for (const laneId of getCompareLaneIds()) {
    await unloadCompareLaneRuntime(laneId);
  }
}

async function ensureCompareDopplerPipeline(laneId) {
  const lane = getCompareLane(laneId);
  if (!lane) {
    throw new Error(`Unknown compare lane "${laneId}".`);
  }
  const modelId = resolveText(lane.modelId, '');
  if (!modelId) {
    throw new Error(`Compare ${laneId}: select a Doppler model first.`);
  }
  if (lane.pipeline && lane.pipelineModelId === modelId) {
    return {
      pipeline: lane.pipeline,
      modelLoadMs: 0,
    };
  }

  await unloadCompareLaneRuntime(laneId);
  setCompareLaneStatus(laneId, 'Loading', 'info');
  renderTranslateCompareLane(laneId);
  const startedAt = performance.now();
  await openModelStore(modelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) {
    throw new Error(`Compare ${laneId}: manifest missing for "${modelId}".`);
  }
  const manifest = parseManifest(manifestText);
  await initDevice();
  const device = getDevice();
  const pipeline = await createPipeline(manifest, {
    gpu: { device },
    runtimeConfig: cloneRuntimeConfig(getRuntimeConfig()),
  });
  lane.pipeline = pipeline;
  lane.pipelineModelId = modelId;
  return {
    pipeline,
    modelLoadMs: Math.max(0, performance.now() - startedAt),
  };
}

async function ensureTransformersGeneratorForLane(laneId) {
  const lane = getCompareLane(laneId);
  if (!lane) {
    throw new Error(`Unknown compare lane "${laneId}".`);
  }
  const modelId = resolveTransformersModelIdForLane(lane);
  if (!modelId) {
    throw new Error(`Compare ${laneId}: no Transformers.js profile is configured for "${lane.modelId || 'this lane'}".`);
  }
  const generatorKey = `${modelId}::${lane.tjsDtype || TRANSLATE_COMPARE_DEFAULT_TJS_DTYPE}`;
  if (lane.tjsGenerator && lane.tjsGeneratorKey === generatorKey) {
    return {
      generator: lane.tjsGenerator,
      runtime: await loadTransformersJsRuntime(),
      modelLoadMs: 0,
    };
  }
  const runtime = await loadTransformersJsRuntime();
  setCompareLaneStatus(laneId, 'Loading', 'info');
  renderTranslateCompareLane(laneId);
  const startedAt = performance.now();
  const generator = await runtime.pipeline('text-generation', modelId, {
    device: 'webgpu',
    dtype: lane.tjsDtype || TRANSLATE_COMPARE_DEFAULT_TJS_DTYPE,
  });
  lane.tjsGenerator = generator;
  lane.tjsGeneratorKey = generatorKey;
  lane.tjsModelId = modelId;
  return {
    generator,
    runtime,
    modelLoadMs: Math.max(0, performance.now() - startedAt),
  };
}

function buildTranslateCompareOptions() {
  const temperature = readOptionalNumber($('temperature-input'));
  const topP = readOptionalNumber($('top-p-input'));
  const topK = readOptionalNumber($('top-k-input'), { integer: true });
  const maxTokens = readOptionalNumber($('max-tokens-input'), { integer: true });
  return {
    temperature: temperature != null ? Math.max(0, temperature) : 0,
    topP: topP != null ? Math.max(0, Math.min(1, topP)) : 1,
    topK: topK != null ? Math.max(1, topK) : 1,
    maxTokens: maxTokens != null && maxTokens > 0 ? maxTokens : TRANSLATE_COMPARE_DEFAULT_MAX_TOKENS,
  };
}

async function runDopplerCompareLane(laneId, context) {
  const lane = getCompareLane(laneId);
  const { prompt, sourceCode, targetCode, options } = context;
  const { pipeline, modelLoadMs } = await ensureCompareDopplerPipeline(laneId);
  const manifestModelId = resolveText(pipeline?.manifest?.modelId, lane.modelId);
  const modelType = normalizeModelType(pipeline?.manifest?.modelType);
  const translateRequest = createTranslateTextRequest(prompt, sourceCode, targetCode);
  let generationInput = translateRequest;
  if (pipeline?.manifest?.inference?.chatTemplate?.type !== 'translategemma') {
    generationInput = buildTranslateInstructionPrompt(prompt, sourceCode, targetCode);
  }
  if (modelType !== 'transformer' && modelType !== null) {
    throw new Error(`Compare ${laneId}: selected model "${manifestModelId}" is not a text model.`);
  }

  lane.output = '';
  lane.error = null;
  setCompareLaneStatus(laneId, modelLoadMs > 0 ? 'Warm' : 'Warm', 'info');
  renderTranslateCompareLane(laneId);
  pipeline.reset?.();
  setCompareLaneStatus(laneId, 'Translating', 'info');
  renderTranslateCompareLane(laneId);

  for await (const token of pipeline.generate(generationInput, {
    ...options,
    useChatTemplate: true,
  })) {
    lane.output += token;
    renderTranslateCompareLane(laneId);
  }

  const stats = pipeline.getStats?.() || {};
  const totalMs = Number.isFinite(stats.totalTimeMs) ? stats.totalTimeMs : null;
  const decodeTokens = Number.isFinite(stats.decodeTokens) ? stats.decodeTokens : null;
  const decodeMs = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : null;
  lane.metrics = {
    modelLoadMs,
    ttftMs: Number.isFinite(stats.ttftMs) ? stats.ttftMs : stats.prefillTimeMs,
    totalMs,
    decodeTokensPerSec: (decodeTokens != null && decodeMs && decodeMs > 0) ? decodeTokens / (decodeMs / 1000) : null,
    sizeBytes: resolveCompareModelSizeBytes(lane.modelId),
    deviceLabel: state.compareDeviceLabel || 'WebGPU',
    metaLabel: `${getTranslateCompareModelLabel(lane.modelId)} · ${lane.output.length} chars`,
  };
  setCompareLaneStatus(laneId, 'Complete', 'success');
  renderTranslateCompareLane(laneId);
}

async function runTransformersCompareLane(laneId, context) {
  const lane = getCompareLane(laneId);
  const { prompt, sourceCode, targetCode, options } = context;
  const { generator, runtime, modelLoadMs } = await ensureTransformersGeneratorForLane(laneId);
  const modelRef = resolveTransformersModelIdForLane(lane);
  const generationPrompt = buildTransformersTranslatePrompt(prompt, sourceCode, targetCode, modelRef);
  const doSample = Number(options.temperature) > 0 && Number(options.topK) > 1;

  lane.output = '';
  lane.error = null;
  setCompareLaneStatus(laneId, 'Warm', 'info');
  renderTranslateCompareLane(laneId);

  let firstTokenAt = null;
  let decodeTokens = 0;
  const tokenTimestamps = [];
  const chunks = [];
  const startedAt = performance.now();
  const TextStreamer = runtime.TextStreamer;
  const streamer = typeof TextStreamer === 'function'
    ? new TextStreamer(generator.tokenizer, {
      skip_prompt: true,
      callback_function: (text) => {
        chunks.push(text);
        lane.output = chunks.join('');
        renderTranslateCompareLane(laneId);
      },
      token_callback_function: (tokens) => {
        const now = performance.now();
        const count = Array.isArray(tokens) ? tokens.length : 1;
        decodeTokens += count;
        if (firstTokenAt === null) {
          firstTokenAt = now;
        }
        for (let index = 0; index < count; index += 1) {
          tokenTimestamps.push(now);
        }
      },
    })
    : null;

  setCompareLaneStatus(laneId, 'Translating', 'info');
  renderTranslateCompareLane(laneId);
  await generator(generationPrompt, {
    max_new_tokens: options.maxTokens,
    do_sample: doSample,
    temperature: doSample ? options.temperature : 1,
    top_k: doSample ? options.topK : 1,
    top_p: doSample ? options.topP : 1,
    num_beams: 1,
    num_beam_groups: 1,
    ...(streamer ? { streamer } : {}),
  });
  const endedAt = performance.now();
  const totalMs = Math.max(1, endedAt - startedAt);
  const ttftMs = firstTokenAt != null ? Math.max(1, firstTokenAt - startedAt) : totalMs;
  const decodeMs = Math.max(totalMs - ttftMs, 1);
  const effectiveDecodeTokens = Math.max(decodeTokens - 1, 0);
  lane.metrics = {
    modelLoadMs,
    ttftMs,
    totalMs,
    decodeTokensPerSec: effectiveDecodeTokens > 0 ? effectiveDecodeTokens / (decodeMs / 1000) : null,
    sizeBytes: resolveCompareModelSizeBytes(lane.modelId),
    deviceLabel: await resolveWebGpuDeviceLabel('WebGPU / ORT'),
    metaLabel: `${resolveText(modelRef, lane.modelId)} · ${lane.output.length} chars`,
  };
  setCompareLaneStatus(laneId, 'Complete', 'success');
  renderTranslateCompareLane(laneId);
}

function captureTranslateCompareHistoryEntry(prompt, sourceCode, targetCode, artifact) {
  return {
    id: crypto?.randomUUID?.() || `${Date.now()}`,
    createdAt: new Date().toISOString(),
    sourceCode,
    targetCode,
    prompt,
    presetId: state.comparePresetId,
    artifact: serializeTranslateCompareArtifactPayload(artifact),
    lanes: {
      left: serializeTranslateCompareHistoryEntry({ lanes: { left: state.compareLanes.left } }).lanes.left,
      right: serializeTranslateCompareHistoryEntry({ lanes: { right: state.compareLanes.right } }).lanes.right,
    },
  };
}

function applyTranslateCompareSmokeSample(sampleId) {
  const sample = TRANSLATE_COMPARE_SMOKE_SAMPLES.find((entry) => entry.id === resolveText(sampleId, ''));
  if (!sample) {
    updateRunStatus('Unknown smoke sample.');
    return;
  }
  const promptEl = $('run-prompt');
  const sourceEl = $('translate-source-language');
  const targetEl = $('translate-target-language');
  if (promptEl instanceof HTMLTextAreaElement || promptEl instanceof HTMLInputElement) {
    promptEl.value = sample.text;
    setStarterExampleInput(promptEl, false);
  }
  if (sourceEl instanceof HTMLSelectElement) {
    sourceEl.value = normalizeTranslateLanguageCode(sample.sourceCode, DEFAULT_TRANSLATE_SOURCE);
  }
  if (targetEl instanceof HTMLSelectElement) {
    targetEl.value = normalizeTranslateLanguageCode(sample.targetCode, DEFAULT_TRANSLATE_TARGET);
  }
  state.activeCompareSmokeSampleId = sample.id;
  renderTranslateCompareSmokePanel();
  syncDeepLinkFromUI();
  updateRunStatus(`Loaded smoke sample: ${sample.label}`);
}

function exportTranslateCompareHistoryArtifactById(entryId) {
  const entry = (state.compareHistory || []).find((row) => row?.id === entryId);
  if (!entry?.artifact) {
    updateRunStatus('Saved compare artifact not found.');
    return;
  }
  exportTranslateCompareArtifactPayload(entry.artifact);
}

async function handleTranslateCompareRun() {
  if (state.compareGenerating || state.compareLoading) return;
  ensureTranslateCompareRuntimeState();
  const prompt = $('run-prompt')?.value?.trim() || '';
  if (!prompt) {
    updateRunStatus('Enter text to translate.');
    return;
  }
  const { sourceCode, targetCode } = getTranslateLanguageSelection();
  const options = buildTranslateCompareOptions();

  state.compareGenerating = true;
  updateStatusIndicator();
  updateRunStatus('Running compare...');
  syncRunControls();
  syncTranslateCompareUI();
  for (const laneId of getCompareLaneIds()) {
    clearCompareLaneResult(laneId);
    renderTranslateCompareLane(laneId);
  }

  try {
    const laneErrors = [];
    for (const laneId of getCompareLaneIds()) {
      const lane = getCompareLane(laneId);
      if (!lane) continue;
      try {
        if (lane.engine === 'transformersjs') {
          await runTransformersCompareLane(laneId, { prompt, sourceCode, targetCode, options });
        } else {
          await runDopplerCompareLane(laneId, { prompt, sourceCode, targetCode, options });
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        lane.error = message;
        lane.output = '';
        lane.metrics = {
          modelLoadMs: lane.metrics?.modelLoadMs ?? null,
          ttftMs: lane.metrics?.ttftMs ?? null,
          totalMs: lane.metrics?.totalMs ?? null,
          decodeTokensPerSec: lane.metrics?.decodeTokensPerSec ?? null,
          sizeBytes: resolveCompareModelSizeBytes(lane.modelId),
          deviceLabel: resolveText(state.compareDeviceLabel, lane.engine === 'transformersjs' ? 'WebGPU / ORT' : 'WebGPU'),
          metaLabel: 'Run failed',
        };
        setCompareLaneStatus(laneId, 'Error', 'error');
        renderTranslateCompareLane(laneId);
        laneErrors.push({ laneId, message });
        log.error('DopplerDemo', `Translate compare lane failed (${laneId}): ${message}`);
      }
    }
    const artifact = buildTranslateCompareArtifact(prompt, sourceCode, targetCode, options);
    state.lastCompareArtifact = artifact;
    const entry = captureTranslateCompareHistoryEntry(prompt, sourceCode, targetCode, artifact);
    state.compareHistory.unshift(entry);
    state.compareHistory = state.compareHistory.slice(0, TRANSLATE_COMPARE_MAX_HISTORY);
    persistTranslateCompareHistory();
    renderTranslateCompareHistory();
    renderTranslateCompareReceipts();
    if (laneErrors.length > 0) {
      updateRunStatus(`Compare completed with ${laneErrors.length} lane error${laneErrors.length === 1 ? '' : 's'}.`);
    } else {
      updateRunStatus('Compare complete');
    }
  } finally {
    state.compareGenerating = false;
    updateStatusIndicator();
    syncRunControls();
    syncTranslateCompareUI();
  }
}

function normalizeDeepLinkMode(mode, defaultMode = null) {
  const normalized = resolveText(mode, '').toLowerCase();
  if (normalized === 'text') return 'run';
  if (normalized === 'translation') return 'translate';
  if (normalized === 'embed') return 'embedding';
  if (normalized === 'image') return 'diffusion';
  if (DEEP_LINK_MODES.has(normalized)) return normalized;
  return defaultMode;
}

function normalizeTask(value, fallback = null) {
  const normalized = resolveText(value, '').toLowerCase();
  if (TASK_SET.has(normalized)) return normalized;
  return fallback;
}

function getTaskModes(task) {
  const normalizedTask = normalizeTask(task, 'run');
  return TASK_MODE_ALLOWLIST[normalizedTask] || TASK_MODE_ALLOWLIST.run;
}

function getTaskForMode(mode, fallback = null) {
  const normalizedMode = normalizeDeepLinkMode(mode, null);
  if (normalizedMode && MODE_TASK_MAP[normalizedMode]) {
    return MODE_TASK_MAP[normalizedMode];
  }
  return normalizeTask(fallback, null);
}

function normalizeSurface(value, fallback = 'demo') {
  const normalized = resolveText(value, fallback).toLowerCase();
  if (SURFACE_SET.has(normalized)) return normalized;
  return fallback;
}

function getAllowedModesForSurface(surface) {
  return SURFACE_MODE_ALLOWLIST[normalizeSurface(surface, 'demo')] || SURFACE_MODE_ALLOWLIST.demo;
}

function getAllowedModesForTask(task, surface) {
  const modesForTask = getTaskModes(task);
  return modesForTask.filter((mode) => isModeAllowedForSurface(mode, surface));
}

function isModeAllowedForSurface(mode, surface) {
  return getAllowedModesForSurface(surface).has(mode);
}

function isTaskAllowedForSurface(task, surface) {
  return getAllowedModesForTask(task, surface).length > 0;
}

function resolveModeForSurface(mode, surface) {
  const normalizedMode = normalizeDeepLinkMode(mode, 'run');
  if (isModeAllowedForSurface(normalizedMode, surface)) {
    return normalizedMode;
  }
  const fallbackPrimary = normalizeDeepLinkMode(state.lastPrimaryMode, 'run');
  if (isModeAllowedForSurface(fallbackPrimary, surface)) {
    return fallbackPrimary;
  }
  return 'run';
}

function resolveTaskForSurface(task, surface, modeHint = null) {
  const normalizedSurface = normalizeSurface(surface, 'demo');
  const requestedTask = normalizeTask(task, null);
  if (requestedTask && isTaskAllowedForSurface(requestedTask, normalizedSurface)) {
    return requestedTask;
  }
  const modeTask = getTaskForMode(modeHint, null);
  if (modeTask && isTaskAllowedForSurface(modeTask, normalizedSurface)) {
    return modeTask;
  }
  const rememberedTask = normalizeTask(state.uiTask, null);
  if (rememberedTask && isTaskAllowedForSurface(rememberedTask, normalizedSurface)) {
    return rememberedTask;
  }
  for (const fallbackTask of ['run', 'evaluate']) {
    if (isTaskAllowedForSurface(fallbackTask, normalizedSurface)) {
      return fallbackTask;
    }
  }
  return 'run';
}

function resolveModeForTask(task, surface, preferredMode = null) {
  const normalizedSurface = normalizeSurface(surface, 'demo');
  const resolvedTask = resolveTaskForSurface(task, normalizedSurface, preferredMode);
  const allowedModes = getAllowedModesForTask(resolvedTask, normalizedSurface);
  if (allowedModes.length === 0) {
    return resolveModeForSurface(preferredMode || state.uiMode || 'run', normalizedSurface);
  }
  const normalizedPreferred = normalizeDeepLinkMode(preferredMode, null);
  if (normalizedPreferred && allowedModes.includes(normalizedPreferred)) {
    return normalizedPreferred;
  }
  const rememberedMode = normalizeDeepLinkMode(state.lastTaskMode?.[resolvedTask], null);
  if (rememberedMode && allowedModes.includes(rememberedMode)) {
    return rememberedMode;
  }
  const defaultTaskMode = normalizeDeepLinkMode(DEFAULT_TASK_MODE[resolvedTask], null);
  if (defaultTaskMode && allowedModes.includes(defaultTaskMode)) {
    return defaultTaskMode;
  }
  return allowedModes[0];
}

function parseAllowedTasks(rawTasks) {
  return resolveText(rawTasks, '')
    .split(/\s+/)
    .map((value) => normalizeTask(value, null))
    .filter(Boolean);
}

function syncSurfaceUI(surface) {
  const normalizedSurface = normalizeSurface(surface, 'demo');
  const normalizedTask = resolveTaskForSurface(
    getTaskForMode(state.uiMode, state.uiTask || 'run'),
    normalizedSurface,
    state.uiMode
  );
  const app = $('app');
  if (app) {
    app.dataset.surface = normalizedSurface;
  }
  document.querySelectorAll('.surface-tab').forEach((button) => {
    const isActive = button.dataset.surface === normalizedSurface;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });
  document.querySelectorAll('.task-tab').forEach((button) => {
    const task = normalizeTask(button.dataset.task, null);
    const taskAllowedForSurface = task != null && isTaskAllowedForSurface(task, normalizedSurface);
    const isVisible = true;
    if (button instanceof HTMLButtonElement) {
      button.hidden = !isVisible;
    }
    button.classList.toggle('is-unavailable', isVisible && !taskAllowedForSurface);
    button.setAttribute('aria-disabled', (isVisible && !taskAllowedForSurface) ? 'true' : 'false');
    button.setAttribute('aria-hidden', isVisible ? 'false' : 'true');
  });
  document.querySelectorAll('.mode-subtab').forEach((button) => {
    const mode = normalizeDeepLinkMode(button.dataset.mode, null);
    const modeAllowed = mode != null && isModeAllowedForSurface(mode, normalizedSurface);
    const isVisible = true;
    const isUnavailable = !modeAllowed;
    button.hidden = false;
    button.classList.toggle('is-unavailable', isUnavailable);
    button.setAttribute('aria-disabled', isUnavailable ? 'true' : 'false');
    button.setAttribute('aria-hidden', 'false');
  });
  if (typeof document !== 'undefined') {
    document.title = 'Doppler';
  }
}

function readDeepLinkValue(hashParams, queryParams, keys) {
  for (const key of keys) {
    const hashValue = hashParams.get(key);
    if (hashValue != null && hashValue !== '') return hashValue;
    const queryValue = queryParams.get(key);
    if (queryValue != null && queryValue !== '') return queryValue;
  }
  return null;
}

function decodeDeepLinkText(rawText) {
  const text = resolveText(rawText);
  if (!text) return '';
  try {
    return decodeURIComponent(text);
  } catch {
    return text;
  }
}

function readDeepLinkStateFromLocation() {
  if (typeof window === 'undefined') {
    return {
      surface: 'demo',
      task: null,
      mode: null,
      sourceCode: DEFAULT_TRANSLATE_SOURCE,
      targetCode: DEFAULT_TRANSLATE_TARGET,
      text: null,
      compareEnabled: false,
      comparePresetId: 'proof',
      lanes: null,
    };
  }

  const queryParams = new URLSearchParams(window.location.search);
  const hashRaw = resolveText(window.location.hash).replace(/^#/, '').replace(/^\?/, '');
  const hashParams = new URLSearchParams(hashRaw);

  const sourceRaw = readDeepLinkValue(hashParams, queryParams, ['sl', 'source', 'source_lang_code']);
  const targetRaw = readDeepLinkValue(hashParams, queryParams, ['tl', 'target', 'target_lang_code']);
  const textRaw = readDeepLinkValue(hashParams, queryParams, ['text', 'prompt', 'q']);
  const taskRaw = readDeepLinkValue(hashParams, queryParams, ['task', 't']);
  const modeRaw = readDeepLinkValue(hashParams, queryParams, ['mode', 'm']);
  const surfaceRaw = readDeepLinkValue(hashParams, queryParams, ['surface', 's']);
  const compareRaw = readDeepLinkValue(hashParams, queryParams, ['compare', 'cv']);
  const comparePresetRaw = readDeepLinkValue(hashParams, queryParams, ['compare_preset', 'cp']);
  const leftEngineRaw = readDeepLinkValue(hashParams, queryParams, ['left_engine', 'le']);
  const rightEngineRaw = readDeepLinkValue(hashParams, queryParams, ['right_engine', 're']);
  const leftModelRaw = readDeepLinkValue(hashParams, queryParams, ['left_model', 'lm']);
  const rightModelRaw = readDeepLinkValue(hashParams, queryParams, ['right_model', 'rm']);
  const surface = normalizeSurface(surfaceRaw, 'demo');

  let task = normalizeTask(taskRaw, null);
  let mode = normalizeDeepLinkMode(modeRaw, null);
  if (!mode && (sourceRaw != null || targetRaw != null || textRaw != null)) {
    mode = 'translate';
  }
  if (!mode && task) {
    mode = resolveModeForTask(task, surface, null);
  }
  if (mode && !isModeAllowedForSurface(mode, surface)) {
    mode = resolveModeForSurface(mode, surface);
  }
  if (mode) {
    task = resolveTaskForSurface(getTaskForMode(mode, task), surface, mode);
  } else if (task && !isTaskAllowedForSurface(task, surface)) {
    task = resolveTaskForSurface(task, surface, null);
  }

  const sourceCode = normalizeTranslateLanguageCode(sourceRaw, DEFAULT_TRANSLATE_SOURCE);
  let targetCode = normalizeTranslateLanguageCode(targetRaw, DEFAULT_TRANSLATE_TARGET);
  if (targetCode === sourceCode) {
    targetCode = sourceCode === DEFAULT_TRANSLATE_TARGET
      ? DEFAULT_TRANSLATE_SOURCE
      : DEFAULT_TRANSLATE_TARGET;
  }

  return {
    surface,
    task,
    mode,
    sourceCode,
    targetCode,
    text: textRaw == null ? null : decodeDeepLinkText(textRaw),
    compareEnabled: compareRaw === '1' || compareRaw === 'true' || compareRaw === 'compare',
    comparePresetId: resolveText(comparePresetRaw, 'proof'),
    lanes: {
      left: {
        engine: resolveText(leftEngineRaw, ''),
        modelId: resolveText(leftModelRaw, ''),
      },
      right: {
        engine: resolveText(rightEngineRaw, ''),
        modelId: resolveText(rightModelRaw, ''),
      },
    },
  };
}

function applyDeepLinkStateToUI(deepLinkState) {
  const sourceSelect = $('translate-source-language');
  const targetSelect = $('translate-target-language');
  if (sourceSelect instanceof HTMLSelectElement) {
    sourceSelect.value = normalizeTranslateLanguageCode(deepLinkState?.sourceCode, DEFAULT_TRANSLATE_SOURCE);
  }
  if (targetSelect instanceof HTMLSelectElement) {
    targetSelect.value = normalizeTranslateLanguageCode(deepLinkState?.targetCode, DEFAULT_TRANSLATE_TARGET);
  }
  if (sourceSelect instanceof HTMLSelectElement && targetSelect instanceof HTMLSelectElement) {
    const selected = getTranslateLanguageSelection();
    sourceSelect.value = selected.sourceCode;
    targetSelect.value = selected.targetCode;
  }

  if (typeof deepLinkState?.text === 'string') {
    const promptEl = $('run-prompt');
    if (promptEl instanceof HTMLTextAreaElement) {
      promptEl.value = deepLinkState.text;
      setStarterExampleInput(promptEl, false);
    }
  }
  ensureTranslateCompareRuntimeState();
  state.compareEnabled = deepLinkState?.compareEnabled === true;
  state.comparePresetId = resolveText(deepLinkState?.comparePresetId, state.comparePresetId || 'proof');
  for (const laneId of getCompareLaneIds()) {
    const laneState = deepLinkState?.lanes?.[laneId] || {};
    const lane = getCompareLane(laneId);
    if (!lane) continue;
    if (laneState.engine === 'doppler' || laneState.engine === 'transformersjs') {
      lane.engine = laneState.engine;
    }
    if (laneState.modelId) {
      lane.modelId = laneState.modelId;
    }
  }
}

function buildDeepLinkHash(modeOverride = null, taskOverride = null) {
  const surface = normalizeSurface(state.surface, 'demo');
  const mode = resolveModeForSurface(resolveText(modeOverride, state.uiMode || 'run'), surface);
  const modeTask = getTaskForMode(mode, 'run');
  const task = resolveTaskForSurface(
    resolveText(taskOverride, modeTask),
    surface,
    mode
  );
  const params = new URLSearchParams();

  if (task !== 'run') {
    params.set('task', task);
  }

  if (mode !== 'run') {
    params.set('mode', mode);
  }

  if (mode === 'translate') {
    const promptEl = $('run-prompt');
    const prompt = resolveText(promptEl?.value, '');
    const { sourceCode, targetCode } = getTranslateLanguageSelection();
    params.set('sl', sourceCode);
    params.set('tl', targetCode);
    const shouldIncludePrompt = prompt.length > 0 && !isStarterExampleInput(promptEl);
    if (shouldIncludePrompt) {
      params.set('text', prompt);
    }
    if (state.compareEnabled) {
      params.set('compare', '1');
      params.set('compare_preset', state.comparePresetId || 'proof');
      for (const laneId of getCompareLaneIds()) {
        const lane = getCompareLane(laneId);
        if (!lane) continue;
        const engineKey = laneId === 'left' ? 'le' : 're';
        const modelKey = laneId === 'left' ? 'lm' : 'rm';
        params.set(engineKey, resolveText(lane.engine, 'doppler'));
        if (lane.modelId) {
          params.set(modelKey, lane.modelId);
        }
      }
    }
  }

  return params.toString();
}

function syncDeepLinkFromUI() {
  if (typeof window === 'undefined' || typeof window.history?.replaceState !== 'function') {
    return;
  }
  const next = new URL(window.location.href);
  next.hash = buildDeepLinkHash();
  const nextPath = `${next.pathname}${next.search}${next.hash}`;
  const currentPath = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (nextPath === currentPath) return;
  window.history.replaceState(null, '', nextPath);
}

function buildTranslateDeepLinkUrl() {
  const next = new URL(window.location.href);
  next.hash = buildDeepLinkHash('translate');
  return next.toString();
}

function setTranslateCompareEnabled(enabled) {
  state.compareEnabled = enabled === true;
  syncTranslateCompareUI();
  syncDeepLinkFromUI();
}

async function copyTranslateCompareShareLink() {
  const url = buildTranslateDeepLinkUrl();
  if (navigator?.clipboard?.writeText) {
    await navigator.clipboard.writeText(url);
    updateRunStatus('Compare link copied');
    return;
  }
  updateRunStatus(url);
}

function getRunStarterPromptPool() {
  if (state.uiMode === 'translate') {
    return TRANSLATE_STARTER_PROMPTS;
  }
  return RUN_STARTER_PROMPTS;
}

function readGlobalString(key) {
  if (!key || typeof globalThis !== 'object' || !globalThis) return '';
  const value = globalThis[key];
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeUrlPathname(pathname) {
  return typeof pathname === 'string' ? pathname.replace(/\/+/g, '/') : '';
}

function isHuggingFaceHost(hostname) {
  if (typeof hostname !== 'string' || !hostname) return false;
  const lowered = hostname.toLowerCase();
  return lowered === QUICK_MODEL_HF_HOST || lowered.endsWith(`.${QUICK_MODEL_HF_HOST}`);
}

function buildHfResolveUrl(repoId, revision, path) {
  const normalizedRepoId = typeof repoId === 'string' ? repoId.trim().replace(/^\/+|\/+$/g, '') : '';
  const normalizedRevision = typeof revision === 'string' ? revision.trim() : '';
  const normalizedPath = typeof path === 'string' ? path.trim().replace(/^\/+/, '') : '';
  if (!normalizedRepoId || !normalizedRevision) return '';
  const pathSuffix = normalizedPath ? `/${normalizedPath}` : '';
  return `https://huggingface.co/${normalizedRepoId}/resolve/${encodeURIComponent(normalizedRevision)}${pathSuffix}`;
}

function extractHfResolveRevisionFromUrl(inputUrl) {
  try {
    const parsed = new URL(inputUrl);
    if (!isHuggingFaceHost(parsed.hostname)) return null;
    const parts = normalizeUrlPathname(parsed.pathname).split('/').filter(Boolean);
    const resolveIndex = parts.indexOf('resolve');
    if (resolveIndex < 0 || resolveIndex + 1 >= parts.length) return null;
    return decodeURIComponent(parts[resolveIndex + 1]);
  } catch {
    return null;
  }
}

function isImmutableHfResolveUrl(inputUrl) {
  const revision = extractHfResolveRevisionFromUrl(inputUrl);
  return !!(revision && QUICK_MODEL_HF_COMMIT_PATTERN.test(revision));
}

function resolveRemoteCacheMode(inputUrl) {
  return isImmutableHfResolveUrl(inputUrl) ? 'force-cache' : 'default';
}

function buildQuickCatalogCandidateUrls() {
  const candidates = [];
  if (QUICK_MODEL_CATALOG_OVERRIDE_URL) {
    candidates.push(QUICK_MODEL_CATALOG_OVERRIDE_URL);
  }
  const hfCatalogUrl = buildHfResolveUrl(
    QUICK_MODEL_CATALOG_HF_REPO_ID,
    QUICK_MODEL_CATALOG_HF_REVISION,
    QUICK_MODEL_CATALOG_HF_PATH
  );
  if (hfCatalogUrl) {
    candidates.push(hfCatalogUrl);
  }
  candidates.push(QUICK_MODEL_CATALOG_LOCAL_URL);
  const deduped = [];
  const seen = new Set();
  for (const candidate of candidates) {
    if (typeof candidate !== 'string') continue;
    const trimmed = candidate.trim();
    if (!trimmed || seen.has(trimmed)) continue;
    seen.add(trimmed);
    deduped.push(trimmed);
  }
  return deduped;
}

function normalizeQuickLookupToken(value) {
  return typeof value === 'string' ? value.trim().toLowerCase() : '';
}

function normalizeQuickCatalogAliases(rawAliases, modelId) {
  const aliases = [];
  if (Array.isArray(rawAliases)) {
    for (const alias of rawAliases) {
      if (typeof alias !== 'string') continue;
      const trimmed = alias.trim();
      if (trimmed) aliases.push(trimmed);
    }
  } else if (typeof rawAliases === 'string') {
    const trimmed = rawAliases.trim();
    if (trimmed) aliases.push(trimmed);
  }
  aliases.push(modelId);
  const deduped = [];
  const seen = new Set();
  for (const alias of aliases) {
    const token = normalizeQuickLookupToken(alias);
    if (!token || seen.has(token)) continue;
    seen.add(token);
    deduped.push(alias.trim());
  }
  return deduped;
}

function normalizeQuickCatalogHfSpec(rawHf) {
  if (!rawHf || typeof rawHf !== 'object' || Array.isArray(rawHf)) return null;
  const repoId = typeof rawHf.repoId === 'string' ? rawHf.repoId.trim() : '';
  const revision = typeof rawHf.revision === 'string' ? rawHf.revision.trim() : '';
  const path = typeof rawHf.path === 'string' ? rawHf.path.trim() : '';
  if (!repoId || !revision) return null;
  return {
    repoId,
    revision,
    path,
  };
}

function isQuickCatalogHfSourceUrl(catalogSourceUrl) {
  try {
    return isHuggingFaceHost(new URL(catalogSourceUrl).hostname);
  } catch {
    return false;
  }
}

function hasQuickCatalogExplicitBaseUrl(baseUrl) {
  return typeof baseUrl === 'string' && baseUrl.trim().length > 0;
}

function extractHfRepoIdFromInput(value) {
  const raw = typeof value === 'string' ? value.trim() : '';
  if (!raw) return '';
  if (raw.startsWith('hf://')) {
    const sliced = raw.slice(5).replace(/^\/+/, '');
    const [owner, repo] = sliced.split('/');
    if (owner && repo) {
      return `${owner}/${repo}`.toLowerCase();
    }
  }
  try {
    const parsed = new URL(raw);
    if (!isHuggingFaceHost(parsed.hostname)) return '';
    const [owner, repo] = normalizeUrlPathname(parsed.pathname).split('/').filter(Boolean);
    if (owner && repo) {
      return `${owner}/${repo}`.toLowerCase();
    }
    return '';
  } catch {
    const match = raw.match(/^([A-Za-z0-9._-]+)\/([A-Za-z0-9._-]+)$/);
    if (!match) return '';
    return `${match[1]}/${match[2]}`.toLowerCase();
  }
}

function collectQuickCatalogLookupTokens(values) {
  const tokens = new Set();
  for (const value of values || []) {
    const raw = typeof value === 'string' ? value.trim() : '';
    if (!raw) continue;
    tokens.add(normalizeQuickLookupToken(raw));
    const hfRepoId = extractHfRepoIdFromInput(raw);
    if (hfRepoId) {
      tokens.add(hfRepoId);
    }
  }
  return tokens;
}

function findQuickCatalogEntryForRegistryInput(values) {
  const lookup = collectQuickCatalogLookupTokens(values);
  if (lookup.size === 0) return null;
  for (const entry of getQuickCatalogEntries()) {
    const modelToken = normalizeQuickLookupToken(entry?.modelId);
    if (modelToken && lookup.has(modelToken)) return entry;
    const hfRepoToken = normalizeQuickLookupToken(entry?.hfRepoId);
    if (hfRepoToken && lookup.has(hfRepoToken)) return entry;
    const aliases = Array.isArray(entry?.aliases) ? entry.aliases : [];
    for (const alias of aliases) {
      const aliasToken = normalizeQuickLookupToken(alias);
      if (aliasToken && lookup.has(aliasToken)) {
        return entry;
      }
    }
  }
  return null;
}

function resolveDirectRdrrBaseUrlFromInput(value) {
  const raw = typeof value === 'string' ? value.trim() : '';
  if (!raw) return '';
  try {
    const parsed = new URL(raw);
    const normalizedPath = normalizeUrlPathname(parsed.pathname);
    if (!normalizedPath.endsWith('/manifest.json')) return '';
    parsed.pathname = normalizedPath.replace(/\/manifest\.json$/, '/');
    parsed.search = '';
    parsed.hash = '';
    return parsed.toString().replace(/\/+$/, '');
  } catch {
    return '';
  }
}

function normalizeQuickModeToken(value) {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'run' || normalized === 'text') return 'run';
  if (normalized === 'translate' || normalized === 'translation') return 'translate';
  if (normalized === 'embedding' || normalized === 'embed') return 'embedding';
  if (normalized === 'diffusion' || normalized === 'image') return 'diffusion';
  if (normalized === 'energy') return 'energy';
  return null;
}

function normalizeQuickModes(rawMode, rawModes) {
  const values = [];
  if (Array.isArray(rawModes)) values.push(...rawModes);
  if (rawMode !== undefined) values.push(rawMode);
  const tokens = new Set();
  for (const value of values) {
    if (typeof value === 'string') {
      const lowered = value.trim().toLowerCase();
      if (lowered === 'both' || lowered === 'all' || lowered === 'text+embedding') {
        tokens.add('run');
        tokens.add('translate');
        tokens.add('embedding');
        continue;
      }
      const splitValues = lowered.split(/[,\s+/]+/).filter(Boolean);
      for (const token of splitValues) {
        const normalized = normalizeQuickModeToken(token);
        if (normalized) tokens.add(normalized);
      }
      continue;
    }
    const normalized = normalizeQuickModeToken(value);
    if (normalized) tokens.add(normalized);
  }
  if (tokens.has('run')) tokens.add('translate');
  if (tokens.has('translate')) tokens.add('run');
  if (tokens.size === 0) {
    tokens.add('run');
    tokens.add('translate');
  }
  return [...tokens];
}

function resolveQuickModelBaseUrl(baseUrl, modelId, catalogSourceUrl, hfSpec = null) {
  if (hfSpec?.repoId && hfSpec?.revision) {
    const hfPath = hfSpec.path || `models/${encodeURIComponent(modelId)}`;
    const resolvedHfUrl = buildHfResolveUrl(hfSpec.repoId, hfSpec.revision, hfPath).replace(/\/+$/, '');
    return isQuickModelAllowedUrl(resolvedHfUrl) ? resolvedHfUrl : null;
  }

  if (!hasQuickCatalogExplicitBaseUrl(baseUrl)) {
    return null;
  }

  const resolved = new URL(baseUrl.trim(), catalogSourceUrl || QUICK_MODEL_CATALOG_LOCAL_BASE_URL).toString();
  return isQuickModelAllowedUrl(resolved) ? resolved : null;
}

function isQuickModelLocalUrl(resolvedUrl) {
  try {
    const resolved = new URL(resolvedUrl);
    const catalogUrl = new URL(QUICK_MODEL_CATALOG_LOCAL_BASE_URL);
    if (resolved.origin !== catalogUrl.origin) return false;
    const normalizedPath = normalizeUrlPathname(resolved.pathname);
    return normalizedPath.startsWith('/models/local/');
  } catch {
    return false;
  }
}

function isQuickModelHfResolveUrl(resolvedUrl) {
  try {
    const resolved = new URL(resolvedUrl);
    if (!isHuggingFaceHost(resolved.hostname)) return false;
    const normalizedPath = normalizeUrlPathname(resolved.pathname);
    return normalizedPath.includes('/resolve/');
  } catch {
    return false;
  }
}

function isQuickModelAllowedUrl(resolvedUrl) {
  return isQuickModelLocalUrl(resolvedUrl) || isQuickModelHfResolveUrl(resolvedUrl);
}

function normalizeQuickCatalogEntry(raw, index, catalogSourceUrl) {
  if (!raw || typeof raw !== 'object') return null;
  const modelId = typeof raw.modelId === 'string' ? raw.modelId.trim() : '';
  if (!modelId) return null;
  const hfSpec = normalizeQuickCatalogHfSpec(
    (raw.hf && typeof raw.hf === 'object' && !Array.isArray(raw.hf))
      ? raw.hf
      : {
        repoId: raw.hfRepoId,
        revision: raw.hfRevision,
        path: raw.hfPath,
      }
  );
  if (isQuickCatalogHfSourceUrl(catalogSourceUrl) && !hfSpec) {
    return null;
  }
  const resolvedBaseUrl = resolveQuickModelBaseUrl(raw.baseUrl, modelId, catalogSourceUrl, hfSpec);
  if (!resolvedBaseUrl) return null;
  const modes = normalizeQuickModes(raw.mode, raw.modes);
  const sizeBytes = Number(raw.sizeBytes);
  const aliases = normalizeQuickCatalogAliases(raw.aliases, modelId);
  return {
    id: modelId,
    modelId,
    aliases,
    label: typeof raw.label === 'string' && raw.label.trim() ? raw.label.trim() : modelId,
    description: typeof raw.description === 'string' ? raw.description.trim() : '',
    baseUrl: resolvedBaseUrl,
    hfRepoId: hfSpec?.repoId || null,
    hfRevision: hfSpec?.revision || null,
    modes,
    sizeBytes: Number.isFinite(sizeBytes) && sizeBytes > 0 ? Math.floor(sizeBytes) : null,
    recommended: raw.recommended === true,
    sortOrder: Number.isFinite(Number(raw.sortOrder)) ? Number(raw.sortOrder) : index,
  };
}

function parseQuickCatalogPayload(payload, catalogSourceUrl) {
  if (!payload || typeof payload !== 'object') {
    return [];
  }
  const entries = Array.isArray(payload.models) ? payload.models : [];
  const normalized = [];
  for (let i = 0; i < entries.length; i += 1) {
    const entry = normalizeQuickCatalogEntry(entries[i], i, catalogSourceUrl);
    if (!entry) continue;
    normalized.push(entry);
  }
  normalized.sort((a, b) => {
    if (a.recommended !== b.recommended) return a.recommended ? -1 : 1;
    if (a.sortOrder !== b.sortOrder) return a.sortOrder - b.sortOrder;
    return a.label.localeCompare(b.label);
  });
  return normalized;
}

function getQuickCatalogEntries() {
  return Array.isArray(state.quickModelCatalog) ? state.quickModelCatalog : [];
}

function getQuickCatalogEntriesForSurface(surface = state.surface) {
  const allowedModes = getAllowedModesForSurface(surface);
  return getQuickCatalogEntries().filter((entry) => (
    Array.isArray(entry?.modes) && entry.modes.some((modeToken) => allowedModes.has(modeToken))
  ));
}

function formatQuickModelBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return 'size unknown';
  return formatBytes(bytes);
}

function setDistillStatus(message, isError = false) {
  const statusEl = $('distill-status');
  if (!statusEl) return;
  statusEl.textContent = message;
  statusEl.dataset.state = isError ? 'error' : 'ready';
}

function setDistillOutput(payload) {
  const outputEl = $('distill-output');
  if (!outputEl) return;
  if (!payload || typeof payload !== 'object') {
    outputEl.textContent = 'No distill output yet.';
    return;
  }
  outputEl.textContent = JSON.stringify(payload, null, 2);
}

function getDistillWorkloads() {
  return Array.isArray(state.distillWorkloads) ? state.distillWorkloads : [];
}

function populateDistillWorkloadSelect() {
  const selectEl = $('distill-workload-select');
  if (!(selectEl instanceof HTMLSelectElement)) return;
  const previous = selectEl.value || '';
  selectEl.innerHTML = '';

  const noneOption = document.createElement('option');
  noneOption.value = '';
  noneOption.textContent = 'None';
  selectEl.appendChild(noneOption);

  for (const workload of getDistillWorkloads()) {
    const option = document.createElement('option');
    option.value = workload.id;
    const suffix = workload.workloadKind ? ` (${workload.workloadKind})` : '';
    option.textContent = `${workload.id}${suffix}`;
    selectEl.appendChild(option);
  }
  selectEl.value = Array.from(selectEl.options).some((option) => option.value === previous)
    ? previous
    : '';
}

function findDistillWorkloadById(workloadId) {
  if (!workloadId) return null;
  return getDistillWorkloads().find((entry) => entry.id === workloadId) || null;
}

async function loadDistillWorkloadRegistry() {
  state.distillWorkloadsLoading = true;
  state.distillWorkloadsError = null;
  try {
    const response = await fetch(DISTILL_WORKLOAD_REGISTRY_URL, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    const workloads = Array.isArray(payload?.workloads) ? payload.workloads : [];
    state.distillWorkloads = workloads
      .filter((entry) => entry && typeof entry === 'object' && typeof entry.id === 'string' && entry.id.trim())
      .map((entry) => ({
        id: entry.id.trim(),
        path: typeof entry.path === 'string' ? entry.path : null,
        sha256: typeof entry.sha256 === 'string' ? entry.sha256 : null,
        workloadKind: typeof entry.workloadKind === 'string' ? entry.workloadKind : null,
      }));
    populateDistillWorkloadSelect();
    if (state.distillWorkloads.length > 0) {
      setDistillStatus(`Loaded ${state.distillWorkloads.length} workload pack entries.`);
    } else {
      setDistillStatus('No workload packs found in registry.');
    }
  } catch (error) {
    state.distillWorkloads = [];
    state.distillWorkloadsError = error instanceof Error ? error.message : String(error);
    populateDistillWorkloadSelect();
    setDistillStatus(`Workload registry unavailable: ${state.distillWorkloadsError}`, true);
  } finally {
    state.distillWorkloadsLoading = false;
  }
}

async function readFileAsText(file) {
  if (!file) return '';
  return file.text();
}

async function handleDistillReplay() {
  const teacherJsonEl = $('distill-teacher-json');
  const workloadSelect = $('distill-workload-select');
  const teacherJsonText = teacherJsonEl?.value || '';
  const selectedWorkloadId = workloadSelect?.value || '';
  const workloadEntry = findDistillWorkloadById(selectedWorkloadId);

  setDistillStatus('Running replay...');
  const result = await runDistillReplay({
    teacherJsonText,
    workloadEntry,
  });
  state.distillLastReplay = result;
  setDistillOutput(result);
  const steps = Array.isArray(result.timeline) ? result.timeline.length : 0;
  const reportId = result.traceability?.teacherReportId || 'unknown';
  setDistillStatus(`Replay complete. Steps: ${steps}. teacherReportId: ${reportId}`);
}

function exportDistillReplay() {
  if (!state.distillLastReplay) {
    setDistillStatus('No replay result available to export.', true);
    return;
  }
  const payload = state.distillLastReplay;
  const timestamp = new Date().toISOString().replace(/[:]/g, '-');
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `distill-replay-${timestamp}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}


function resolveDownloadProgressForModel(modelId) {
  const progress = state.downloadProgress;
  if (!progress || typeof progress !== 'object') return null;
  const progressModelId = typeof progress.modelId === 'string' ? progress.modelId : '';
  if (modelId && progressModelId && progressModelId !== modelId) return null;

  const percent = Number(progress.percent);
  const downloadedBytes = Number(progress.downloadedBytes);
  const totalBytes = Number(progress.totalBytes);
  return {
    modelId: progressModelId || modelId || '',
    percent: Number.isFinite(percent) ? clampPercent(percent) : null,
    downloadedBytes: Number.isFinite(downloadedBytes) && downloadedBytes > 0 ? downloadedBytes : 0,
    totalBytes: Number.isFinite(totalBytes) && totalBytes > 0 ? totalBytes : 0,
  };
}

function findQuickModelEntry(modelId) {
  return getQuickCatalogEntries().find((entry) => entry.modelId === modelId) || null;
}

function formatQuickModelModeBadge(modes = []) {
  if (!Array.isArray(modes) || modes.length === 0) return 'text';
  const labels = [];
  if (modes.includes('run')) {
    labels.push('text');
  } else if (modes.includes('translate')) {
    labels.push('translate');
  }
  if (modes.includes('embedding')) labels.push('embedding');
  if (modes.includes('diffusion')) labels.push('diffusion');
  if (modes.includes('energy')) labels.push('energy');
  return labels.length > 0 ? labels.join('+') : 'text';
}

function getComparableQuickModelSize(entry) {
  const size = Number(entry?.sizeBytes);
  return Number.isFinite(size) && size > 0 ? size : Number.POSITIVE_INFINITY;
}

function getSmallestQuickModelForMode(modeToken) {
  if (!modeToken) return null;
  const candidates = getQuickCatalogEntries().filter((entry) => entry.modes.includes(modeToken));
  if (candidates.length === 0) return null;
  candidates.sort((a, b) => {
    const sizeDiff = getComparableQuickModelSize(a) - getComparableQuickModelSize(b);
    if (sizeDiff !== 0) return sizeDiff;
    if (a.sortOrder !== b.sortOrder) return a.sortOrder - b.sortOrder;
    return a.label.localeCompare(b.label);
  });
  return candidates[0] || null;
}

function getPreferredQuickModelForMode(modeToken) {
  if (!modeToken) return null;
  if (modeToken !== 'run' && modeToken !== 'translate') {
    return getSmallestQuickModelForMode(modeToken);
  }
  const candidates = getQuickCatalogEntries().filter((entry) => (
    Array.isArray(entry?.modes) && entry.modes.includes(modeToken)
  ));
  if (candidates.length === 0) return null;
  candidates.sort((a, b) => {
    const scoreDiff = getModelSelectionScore(modeToken, b.modelId) - getModelSelectionScore(modeToken, a.modelId);
    if (scoreDiff !== 0) return scoreDiff;
    const sizeDiff = getComparableQuickModelSize(a) - getComparableQuickModelSize(b);
    if (sizeDiff !== 0) return sizeDiff;
    if (a.sortOrder !== b.sortOrder) return a.sortOrder - b.sortOrder;
    return a.label.localeCompare(b.label);
  });
  return candidates[0] || null;
}

function getDiagnosticsRequiredQuickMode() {
  const selection = state.diagnosticsSelections?.diagnostics || {};
  const selectedProfile = decodeDiagnosticsProfileId(selection.profile || '');
  const suite = selectedProfile?.suite || selection.suite || getDiagnosticsDefaultSuite('diagnostics');
  if (suite === 'kernels') return null;
  if (suite === 'diffusion') return 'diffusion';
  if (suite === 'energy') return 'energy';
  const preset = String(selectedProfile?.preset || selection.preset || '').toLowerCase();
  if (preset.includes('embedding')) return 'embedding';
  return 'run';
}

function updateNavState(mode, task = null) {
  const normalizedMode = normalizeDeepLinkMode(mode, 'run');
  const normalizedTask = resolveTaskForSurface(
    getTaskForMode(normalizedMode, task || state.uiTask || 'run'),
    state.surface,
    normalizedMode
  );

  document.querySelectorAll('.task-tab').forEach((button) => {
    const buttonTask = normalizeTask(button.dataset.task, null);
    const isActive = buttonTask === normalizedTask && !button.hidden;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });

  document.querySelectorAll('.mode-subtab').forEach((button) => {
    const buttonMode = normalizeDeepLinkMode(button.dataset.mode, null);
    const isActive = buttonMode === normalizedMode && !button.hidden;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });
}


function cloneRuntimeConfig(config) {
  try {
    return structuredClone(config);
  } catch {
    return JSON.parse(JSON.stringify(config));
  }
}

function applyModeVisibility(mode) {
  const panels = document.querySelectorAll('[data-modes]');
  panels.forEach((panel) => {
    const modes = panel.dataset.modes?.split(/\s+/).filter(Boolean) || [];
    const modeVisible = modes.length === 0 || modes.includes(mode);
    panel.hidden = !modeVisible;
  });
}

function ensurePrimaryModeControlStack() {
  const panelGrid = $('panel-grid');
  if (!panelGrid) return;

  const railStack = panelGrid.querySelector('.panel-stack-rail');
  if (!railStack) return;

  let controlsStack = panelGrid.querySelector('.panel-stack-controls');
  if (!controlsStack) {
    controlsStack = document.createElement('div');
    controlsStack.className = 'panel-stack panel-stack-controls';
    controlsStack.dataset.modes = 'run translate embedding diffusion energy';
    panelGrid.insertBefore(controlsStack, railStack);
  }

  const controlSectionSelectors = [
    '.run-controls-panel',
    '.diffusion-controls-panel',
    '.energy-controls-panel',
  ];
  for (const selector of controlSectionSelectors) {
    const section = panelGrid.querySelector(selector);
    if (!section || section.parentElement === controlsStack) continue;
    controlsStack.appendChild(section);
  }
}

function syncRunModeUI(mode) {
  const isEmbeddingMode = mode === 'embedding';
  const isTranslateMode = mode === 'translate';
  setText(
    $('run-panel-title'),
    isEmbeddingMode ? 'Embeddings' : (isTranslateMode ? 'Translation' : 'Text Generation')
  );
  setText(
    $('run-controls-title'),
    isEmbeddingMode ? 'Embedding Controls' : (isTranslateMode ? 'Translation Controls' : 'Run Controls')
  );
  setText($('run-prompt-label'), isEmbeddingMode ? 'Input text' : (isTranslateMode ? 'Text to translate' : 'Prompt'));
  setText($('run-generate-btn'), isEmbeddingMode ? 'Embed' : (isTranslateMode ? 'Translate' : 'Generate'));
  const prompt = $('run-prompt');
  if (prompt) {
    prompt.placeholder = isEmbeddingMode
      ? 'Enter text to embed...'
      : (isTranslateMode
        ? 'Enter text to translate...'
        : 'Ask a question or provide a prompt...');
    if (isTranslateMode && isStarterExampleInput(prompt)) {
      applyStarterPrompt(prompt, TRANSLATE_STARTER_PROMPTS, { force: true });
    }
  }
  setHidden($('run-sampling-controls'), isEmbeddingMode);
  setHidden($('run-embedding-docs'), !isEmbeddingMode);
  setHidden($('translate-controls'), !isTranslateMode);
  if (isEmbeddingMode) {
    refreshEmbeddingDemoDocuments();
  }
  renderEmbeddingDocumentSet();
  updateRunAutoLabels();
  syncTranslateCompareUI();
}

async function setUiTask(task, modeHint = null) {
  const surface = normalizeSurface(state.surface, 'demo');
  const resolvedTask = resolveTaskForSurface(task, surface, modeHint || state.uiMode);
  const targetMode = resolveModeForTask(
    resolvedTask,
    surface,
    modeHint || state.lastTaskMode?.[resolvedTask] || state.uiMode || DEFAULT_TASK_MODE[resolvedTask]
  );
  await setUiMode(targetMode, { task: resolvedTask });
}

async function setUiMode(mode, options = {}) {
  const app = $('app');
  if (!app) return;
  const surface = normalizeSurface(state.surface, 'demo');
  const resolvedMode = resolveModeForSurface(mode, surface);
  const modeTask = getTaskForMode(resolvedMode, options?.task || state.uiTask || 'run');
  const resolvedTask = resolveTaskForSurface(
    modeTask,
    surface,
    resolvedMode
  );
  state.uiMode = resolvedMode;
  state.uiTask = resolvedTask;
  state.lastTaskMode[resolvedTask] = resolvedMode;
  if (PRIMARY_MODES.has(resolvedMode)) {
    state.lastPrimaryMode = resolvedMode;
  }
  app.dataset.mode = resolvedMode;
  app.dataset.task = resolvedTask;
  syncSurfaceUI(surface);
  updateNavState(resolvedMode, resolvedTask);
  applyModeVisibility(resolvedMode);
  syncRunModeUI(resolvedMode);
  syncDiagnosticsModeUI(resolvedMode);
  if (resolvedMode === 'models') {
    refreshStorageInspector({
      onSelectModel: selectDiagnosticsModel,
      onTryModel: handleStorageTryModel,
      onUnloadActiveModel: unloadActivePipeline,
      onStorageInventoryRefreshed: renderQuickModelPanels,
      onModelsUpdated: refreshModelList,
    }).catch((error) => {
      log.warn('DopplerDemo', `Storage inspector refresh failed: ${error.message}`);
    });
  }
  try {
    await refreshModelList();
  } catch (error) {
    log.warn('DopplerDemo', `Model list refresh failed: ${error.message}`);
  }
  updatePerformancePanel();
  renderRunLog();
  syncDiagnosticsDefaultsForMode(resolvedMode).catch((error) => {
    updateDiagnosticsStatus(`Diagnostics config error: ${error.message}`, true);
  });
  if (resolvedMode === 'energy') {
    syncEnergyDemoSelection();
  }
  updateModelEmptyStates();
  syncDeepLinkFromUI();
}

function getModelAvailability() {
  const availability = state.modelAvailability;
  if (!availability || typeof availability !== 'object') {
    return { ...DEFAULT_MODEL_AVAILABILITY };
  }
  return {
    total: Number.isFinite(availability.total) ? availability.total : 0,
    run: Number.isFinite(availability.run) ? availability.run : 0,
    translate: Number.isFinite(availability.translate) ? availability.translate : 0,
    embedding: Number.isFinite(availability.embedding) ? availability.embedding : 0,
    diffusion: Number.isFinite(availability.diffusion) ? availability.diffusion : 0,
    energy: Number.isFinite(availability.energy) ? availability.energy : 0,
  };
}

function setEmptyNotice(scope, message) {
  const notice = $(`${scope}-empty-notice`);
  const text = $(`${scope}-empty-notice-text`);
  const normalized = typeof message === 'string' ? message.trim() : '';
  setHidden(notice, normalized.length === 0);
  setText(text, normalized);
}

function setEmptyNoticeAction(scope, quickModelEntry) {
  const button = $(`${scope}-empty-notice-btn`);
  if (!button) return;
  const busyModelId = state.quickModelActionModelId;
  const hasBusyImport = typeof busyModelId === 'string' && busyModelId.length > 0;

  if (quickModelEntry?.modelId) {
    const isBusy = busyModelId === quickModelEntry.modelId;
    button.dataset.noticeAction = 'download';
    button.dataset.quickModelId = quickModelEntry.modelId;
    if (isBusy) {
      const progress = resolveDownloadProgressForModel(quickModelEntry.modelId);
      const pct = progress?.percent;
      button.textContent = Number.isFinite(pct) ? `Fetching ${Math.round(pct)}%` : 'Fetching...';
    } else {
      button.textContent = `Download ${quickModelEntry.label}`;
    }
    button.disabled = isBusy || (hasBusyImport && !isBusy);
    return;
  }

  button.dataset.noticeAction = 'models';
  delete button.dataset.quickModelId;
  button.textContent = 'Go to Models';
  button.disabled = hasBusyImport;
}

function getMissingModelMessage(mode, availability, quickModelEntry) {
  if (mode === 'energy') {
    return '';
  }
  const total = Number.isFinite(availability?.total) ? availability.total : 0;
  const hasQuickSuggestion = !!(quickModelEntry && typeof quickModelEntry.modelId === 'string' && quickModelEntry.modelId.length > 0);
  if (total <= 0) {
    return hasQuickSuggestion
      ? 'Import a model that supports this mode to continue.'
      : 'No models found in OPFS. Import a model from the Models tab.';
  }
  const compatible = Number.isFinite(availability?.[mode]) ? availability[mode] : 0;
  if (compatible > 0) return '';
  if (mode === 'embedding') {
    return 'No embedding model available in OPFS for this mode.';
  }
  if (mode === 'translate') {
    return 'No text translation model available in OPFS for this mode.';
  }
  if (mode === 'diffusion') {
    return 'No diffusion model available in OPFS for this mode.';
  }
  return 'No text model available in OPFS for this mode.';
}

function setQuickModelStatus(message) {
  const statusEl = $('models-quick-models-status');
  if (!statusEl) return;
  setText(statusEl, message || '');
}

function createQuickModelBadge(text) {
  const badge = document.createElement('span');
  badge.className = 'quick-model-badge';
  badge.textContent = text;
  return badge;
}

function createQuickModelActionButton({ label, action, modelId, disabled, title = '' }) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'btn btn-small';
  button.textContent = label;
  button.dataset.quickAction = action;
  button.dataset.quickModelId = modelId;
  if (title) {
    button.title = title;
  }
  button.disabled = disabled;
  return button;
}

function renderQuickModelList(listEl, catalogEntries) {
  if (!listEl) return;
  listEl.textContent = '';

  const busyId = state.quickModelActionModelId;
  const hasBusyAction = typeof busyId === 'string' && busyId.length > 0;
  const storageEntries = Array.isArray(state.storageEntriesData) ? state.storageEntriesData : [];
  const storageByModelId = new Map(storageEntries.map((e) => [e.modelId, e]));
  const catalogIds = new Set(catalogEntries.map((e) => e.modelId));

  const storageDeleteCallbacks = {
    onUnloadActiveModel: unloadActivePipeline,
    onModelsUpdated: async () => {
      await refreshModelList();
      await refreshStorageInspector({
        onTryModel: handleStorageTryModel,
        onUnloadActiveModel: unloadActivePipeline,
        onStorageInventoryRefreshed: renderQuickModelPanels,
        onModelsUpdated: refreshModelList,
      });
    },
  };

  // OPFS-only orphans first (in storage but not in catalog)
  const orphans = storageEntries.filter((e) => !e.missingStorage && !catalogIds.has(e.modelId));
  // Catalog entries: OPFS models first, then not-in-OPFS
  const catalogSorted = [
    ...catalogEntries.filter((e) => storageByModelId.has(e.modelId)),
    ...catalogEntries.filter((e) => !storageByModelId.has(e.modelId)),
  ];

  function appendCard(card) {
    listEl.appendChild(card);
  }

  // Render orphan OPFS cards
  for (const storageEntry of orphans) {
    const card = document.createElement('article');
    card.className = 'quick-model-card';

    const row = document.createElement('div');
    row.className = 'quick-model-row';

    const main = document.createElement('div');
    main.className = 'quick-model-main';

    const title = document.createElement('div');
    title.className = 'quick-model-title';
    title.textContent = storageEntry.modelId;
    main.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'quick-model-meta';
    meta.appendChild(createQuickModelBadge(storageEntry.backend === 'indexeddb' ? 'idb' : (storageEntry.backend || 'opfs')));
    if (Number.isFinite(storageEntry.totalBytes) && storageEntry.totalBytes > 0) {
      meta.appendChild(createQuickModelBadge(formatQuickModelBytes(storageEntry.totalBytes)));
    }
    if (storageEntry.modelId === state.activeModelId) {
      meta.appendChild(createQuickModelBadge('active'));
    }
    main.appendChild(meta);

    row.appendChild(main);

    const actions = document.createElement('div');
    actions.className = 'quick-model-actions';
    if (isRunnableStorageEntry(storageEntry)) {
      const tryBtn = document.createElement('button');
      tryBtn.type = 'button';
      tryBtn.className = 'btn btn-small btn-primary';
      tryBtn.textContent = 'Try It';
      tryBtn.addEventListener('click', () => handleStorageTryModel(storageEntry.modelId));
      actions.appendChild(tryBtn);
    }
    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.className = 'btn btn-small';
    deleteBtn.textContent = 'Delete';
    deleteBtn.addEventListener('click', () => deleteStorageModel(storageEntry, storageDeleteCallbacks));
    actions.appendChild(deleteBtn);
    row.appendChild(actions);

    card.appendChild(row);
    appendCard(card);
  }

  // Render catalog cards (OPFS first, then available)
  for (const entry of catalogSorted) {
    const isBusy = hasBusyAction && busyId === entry.modelId;
    const storageEntry = storageByModelId.get(entry.modelId);
    const isInOpfs = isRunnableStorageEntry(storageEntry);

    const card = document.createElement('article');
    card.className = entry.recommended ? 'quick-model-card is-recommended' : 'quick-model-card';

    const row = document.createElement('div');
    row.className = 'quick-model-row';

    const main = document.createElement('div');
    main.className = 'quick-model-main';

    const title = document.createElement('div');
    title.className = 'quick-model-title';
    title.textContent = entry.label;
    main.appendChild(title);

    const modelId = document.createElement('div');
    modelId.className = 'quick-model-id type-caption';
    modelId.textContent = entry.modelId;
    main.appendChild(modelId);

    const meta = document.createElement('div');
    meta.className = 'quick-model-meta';
    if (entry.recommended) {
      meta.appendChild(createQuickModelBadge('recommended'));
    }
    meta.appendChild(createQuickModelBadge(formatQuickModelModeBadge(entry.modes)));
    meta.appendChild(createQuickModelBadge(formatQuickModelBytes(entry.sizeBytes)));
    if (isInOpfs && storageEntry.modelId === state.activeModelId) {
      meta.appendChild(createQuickModelBadge('active'));
    }
    main.appendChild(meta);

    if (isBusy) {
      const dlProgress = resolveDownloadProgressForModel(entry.modelId);
      const pct = dlProgress?.percent ?? 0;
      const bar = document.createElement('div');
      bar.className = 'quick-model-progress';
      const fill = document.createElement('div');
      fill.className = 'quick-model-progress-fill';
      fill.style.width = `${pct}%`;
      bar.appendChild(fill);
      main.appendChild(bar);
    }

    row.appendChild(main);

    const actions = document.createElement('div');
    actions.className = 'quick-model-actions';
    if (isInOpfs) {
      const tryBtn = document.createElement('button');
      tryBtn.type = 'button';
      tryBtn.className = 'btn btn-small btn-primary';
      tryBtn.textContent = 'Try It';
      tryBtn.addEventListener('click', () => handleStorageTryModel(entry.modelId));
      actions.appendChild(tryBtn);
      const deleteBtn = document.createElement('button');
      deleteBtn.type = 'button';
      deleteBtn.className = 'btn btn-small';
      deleteBtn.textContent = 'Delete';
      deleteBtn.addEventListener('click', () => deleteStorageModel(storageEntry, storageDeleteCallbacks));
      actions.appendChild(deleteBtn);
    } else {
      actions.appendChild(createQuickModelActionButton({
        label: isBusy ? 'Fetching...' : 'Fetch',
        action: 'download',
        modelId: entry.modelId,
        disabled: isBusy || hasBusyAction,
      }));
    }

    row.appendChild(actions);
    card.appendChild(row);
    appendCard(card);
  }

  if (orphans.length === 0 && catalogEntries.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'type-caption';
    empty.textContent = 'No models configured yet.';
    listEl.appendChild(empty);
  }
}

function renderQuickModelPanels() {
  const catalog = getQuickCatalogEntriesForSurface();
  const rawCatalog = getQuickCatalogEntries();

  if (state.quickModelActionModelId) {
    const modelId = state.quickModelActionModelId;
    const progress = resolveDownloadProgressForModel(modelId);
    const pct = progress?.percent;
    setQuickModelStatus(Number.isFinite(pct) ? `Fetching ${modelId}: ${Math.round(pct)}%` : `Fetching ${modelId}...`);
  } else if (state.quickModelCatalogLoading) {
    setQuickModelStatus('Loading quick models...');
  } else if (state.quickModelCatalogError) {
    const message = `Quick model catalog unavailable: ${state.quickModelCatalogError}`;
    setQuickModelStatus(message);
  } else if (rawCatalog.length > 0 && catalog.length === 0) {
    setQuickModelStatus('No quick models are tagged for currently supported demo modes.');
  } else {
    setQuickModelStatus(
      catalog.length > 0
        ? ''
        : 'No quick models configured in catalog.json yet.'
    );
  }

  renderQuickModelList($('models-list'), catalog);
}

async function loadQuickModelCatalog() {
  state.quickModelCatalogLoading = true;
  state.quickModelCatalogError = null;
  renderQuickModelPanels();
  try {
    let lastError = null;
    let loaded = false;
    for (const catalogUrl of QUICK_MODEL_CATALOG_URLS) {
      try {
        const response = await fetch(catalogUrl, { cache: resolveRemoteCacheMode(catalogUrl) });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status} (${catalogUrl})`);
        }
        const payload = await response.json();
        state.quickModelCatalog = parseQuickCatalogPayload(payload, catalogUrl);
        loaded = true;
        break;
      } catch (error) {
        lastError = error;
      }
    }
    if (!loaded) {
      throw lastError || new Error('Quick model catalog fetch failed.');
    }
  } catch (error) {
    state.quickModelCatalog = [];
    state.quickModelCatalogError = error instanceof Error ? error.message : String(error);
  } finally {
    state.quickModelCatalogLoading = false;
    renderQuickModelPanels();
  }
}

async function applyImportedModelToCurrentMode(modelId) {
  if (!modelId) return;
  const mode = state.uiMode;
  if (mode === 'models') return;

  if (mode === 'diagnostics') {
    selectDiagnosticsModel(modelId);
    return;
  }

  if (!isModeModelSelectable(mode)) return;
  const modelType = await getModelTypeForId(modelId);
  if (!isCompatibleModelType(modelType, mode)) return;

  selectDiagnosticsModel(modelId);
  state.modeModelId[mode] = modelId;
}

async function handleEmptyNoticeAction(scope) {
  const button = $(`${scope}-empty-notice-btn`);
  if (!button) return;
  const action = button.dataset.noticeAction || 'models';
  if (action !== 'download') {
    setUiMode('models');
    return;
  }
  const modelId = button.dataset.quickModelId || '';
  if (!modelId) {
    setUiMode('models');
    return;
  }
  await runQuickModelAction('download', modelId);
}

function handleDownloadProgressEvent(progress) {
  const modelId = typeof progress?.modelId === 'string' && progress.modelId.trim()
    ? progress.modelId.trim()
    : (typeof state.activeDownloadId === 'string' ? state.activeDownloadId : '');
  const percent = Number(progress?.percent);
  const downloadedBytes = Number(progress?.downloadedBytes);
  const totalBytes = Number(progress?.totalBytes);

  state.downloadProgress = {
    modelId,
    percent: Number.isFinite(percent) ? clampPercent(percent) : null,
    downloadedBytes: Number.isFinite(downloadedBytes) && downloadedBytes > 0 ? downloadedBytes : 0,
    totalBytes: Number.isFinite(totalBytes) && totalBytes > 0 ? totalBytes : 0,
    status: typeof progress?.status === 'string' ? progress.status : '',
  };
  if (modelId) {
    state.activeDownloadId = modelId;
  }
  state.downloadActive = true;
  updateStatusIndicator();
  if (state.quickModelActionModelId && modelId && modelId === state.quickModelActionModelId) {
    updateModelEmptyStates();
  } else {
    renderQuickModelPanels();
  }
}

function handleDownloadStateChangeEvent(update) {
  if (!update || typeof update !== 'object') return;
  const modelId = typeof update.modelId === 'string' && update.modelId.trim() ? update.modelId.trim() : '';
  if (modelId) {
    state.activeDownloadId = modelId;
  }
  if (update.active === true) {
    state.downloadActive = true;
  } else if (update.active === false) {
    state.downloadActive = false;
    if (!modelId || state.downloadProgress?.modelId === modelId) {
      state.downloadProgress = null;
    }
  }
  updateStatusIndicator();
  if (state.quickModelActionModelId && (!modelId || modelId === state.quickModelActionModelId)) {
    updateModelEmptyStates();
  } else {
    renderQuickModelPanels();
  }
}

async function importRdrrFromBaseUrl(baseUrl, modelIdOverride = '') {
  const imported = await startDownloadFromBaseUrl(baseUrl, modelIdOverride);
  if (!imported) {
    throw new Error(`Could not import model ${modelIdOverride || baseUrl}.`);
  }
  await updateStorageInfo();
  await refreshModelList();
  if (state.uiMode === 'models') {
    await refreshStorageInspector({
      onSelectModel: selectDiagnosticsModel,
      onTryModel: handleStorageTryModel,
      onUnloadActiveModel: unloadActivePipeline,
      onStorageInventoryRefreshed: renderQuickModelPanels,
      onModelsUpdated: refreshModelList,
    });
  }
}

async function importQuickModelEntry(entry) {
  await importRdrrFromBaseUrl(entry.baseUrl, entry.modelId);
  await applyImportedModelToCurrentMode(entry.modelId);
}

function isRunnableStorageEntry(entry) {
  return Boolean(entry && !entry.missingStorage && entry.hasManifest);
}

function describeImportedStorage(modelId) {
  const entry = state.storageEntriesData.find((candidate) => candidate.modelId === modelId);
  if (!entry?.backend) {
    return 'local storage';
  }
  if (entry.backend === 'indexeddb') {
    return 'IndexedDB';
  }
  if (entry.backend === 'opfs') {
    return 'OPFS';
  }
  return entry.backend;
}

async function runQuickModelAction(action, modelId) {
  if (action !== 'download') return;
  const entry = findQuickModelEntry(modelId);
  if (!entry) {
    updateConvertStatus(`Quick model not found: ${modelId}`, 0);
    return;
  }
  if (state.quickModelActionModelId) return;

  let finalQuickStatus = '';
  state.quickModelActionModelId = modelId;
  state.downloadActive = true;
  state.activeDownloadId = modelId;
  state.downloadProgress = null;
  updateStatusIndicator();
  setQuickModelStatus(`Fetching ${modelId}...`);
  updateModelEmptyStates();
  renderQuickModelPanels();
  try {
    await importQuickModelEntry(entry);
    finalQuickStatus = `Fetched ${modelId} to ${describeImportedStorage(modelId)}.`;
    renderQuickModelPanels();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    finalQuickStatus = `Fetch failed: ${message}`;
    updateConvertStatus(`Quick model action failed: ${message}`, 0);
    updateDiagnosticsStatus(`Quick model action failed: ${message}`, true);
  } finally {
    if (!state.downloadProgress || state.downloadProgress.modelId === modelId) {
      state.downloadProgress = null;
    }
    state.quickModelActionModelId = null;
    state.downloadActive = false;
    state.activeDownloadId = null;
    updateStatusIndicator();
    updateModelEmptyStates();
    renderQuickModelPanels();
    if (finalQuickStatus) {
      setQuickModelStatus(finalQuickStatus);
    }
  }
}

function updateModelEmptyStates() {
  if (state.modelAvailabilityLoading) {
    const emptyMessage = '';
    setEmptyNotice('run', emptyMessage);
    setEmptyNotice('diffusion', emptyMessage);
    setEmptyNotice('energy', emptyMessage);
    setEmptyNotice('diagnostics', emptyMessage);
    setEmptyNoticeAction('run', null);
    setEmptyNoticeAction('diffusion', null);
    setEmptyNoticeAction('energy', null);
    setEmptyNoticeAction('diagnostics', null);
    renderQuickModelPanels();
    return;
  }

  const availability = getModelAvailability();
  const runTargetMode = state.uiMode === 'embedding'
    ? 'embedding'
    : (state.uiMode === 'translate' ? 'translate' : 'run');
  const runQuickSuggestion = getPreferredQuickModelForMode(runTargetMode);
  const diffusionQuickSuggestion = getPreferredQuickModelForMode('diffusion');
  const energyQuickSuggestion = getPreferredQuickModelForMode('energy');
  const diagnosticsQuickSuggestion = getPreferredQuickModelForMode(getDiagnosticsRequiredQuickMode());
  const runMessage = getMissingModelMessage(runTargetMode, availability, runQuickSuggestion);
  const diffusionMessage = getMissingModelMessage('diffusion', availability, diffusionQuickSuggestion);
  const energyMessage = getMissingModelMessage('energy', availability, energyQuickSuggestion);
  const diagnosticsTargetMode = getDiagnosticsRequiredQuickMode();
  const diagnosticsMessage = (
    state.uiMode === 'diagnostics'
      ? (diagnosticsTargetMode ? getMissingModelMessage(diagnosticsTargetMode, availability, diagnosticsQuickSuggestion) : '')
      : ''
  );

  setEmptyNotice('run', runMessage);
  setEmptyNotice('diffusion', diffusionMessage);
  setEmptyNotice('energy', energyMessage);
  setEmptyNotice('diagnostics', diagnosticsMessage);
  setEmptyNoticeAction('run', runMessage ? runQuickSuggestion : null);
  setEmptyNoticeAction('diffusion', diffusionMessage ? diffusionQuickSuggestion : null);
  setEmptyNoticeAction('energy', energyMessage ? energyQuickSuggestion : null);
  setEmptyNoticeAction('diagnostics', diagnosticsMessage ? diagnosticsQuickSuggestion : null);
  renderQuickModelPanels();

  const diffusionRun = $('diffusion-run-btn');
  if (diffusionRun) {
    diffusionRun.disabled = state.diffusionGenerating || state.diffusionLoading || diffusionMessage.length > 0;
  }
  const energyRun = $('energy-run-btn');
  if (energyRun) {
    energyRun.disabled = state.energyGenerating || state.energyLoading || energyMessage.length > 0;
  }
  syncRunControls();
}

function updateConvertStatus(message, percent) {
  const status = $('convert-status');
  const progress = $('convert-progress');
  const label = $('convert-message');
  if (!status || !progress || !label) return;
  setHidden(status, false);
  setText(label, message || '');
  if (Number.isFinite(percent)) {
    progress.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  }
}

function resetConvertStatus() {
  const status = $('convert-status');
  const progress = $('convert-progress');
  const label = $('convert-message');
  if (!status || !progress || !label) return;
  setHidden(status, false);
  progress.style.width = '0%';
  setText(label, 'Ready');
}

function updateRunStatus(message) {
  const status = $('run-output-status');
  if (!status) return;
  setText(status, message || 'Idle');
}

function updateDiffusionStatus(message) {
  const status = $('diffusion-output-status');
  if (!status) return;
  setText(status, message || 'Idle');
}

const AUX_IMPORT_FILENAMES = [
  'config.json',
  'generation_config.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'added_tokens.json',
  'preprocessor_config.json',
  'vocab.txt',
  'merges.txt',
];

function getPickedFilePath(file) {
  if (!file) return '';
  if (typeof file.relativePath === 'string' && file.relativePath.length > 0) {
    return file.relativePath;
  }
  if (typeof file.webkitRelativePath === 'string' && file.webkitRelativePath.length > 0) {
    return file.webkitRelativePath;
  }
  if (typeof file.name === 'string') return file.name;
  return '';
}

function normalizePickedPath(path) {
  return String(path || '')
    .replace(/\\/g, '/')
    .replace(/^\.?\//, '')
    .trim();
}

function getPathBaseName(path) {
  const normalized = normalizePickedPath(path);
  if (!normalized) return '';
  const parts = normalized.split('/');
  return parts[parts.length - 1] || '';
}

function findPickedFileByPath(files, path) {
  const targetPath = normalizePickedPath(path);
  if (!targetPath) return null;

  const exact = files.find((file) => normalizePickedPath(getPickedFilePath(file)) === targetPath);
  if (exact) return exact;

  const targetBase = getPathBaseName(targetPath);
  if (!targetBase) return null;
  const baseMatches = files.filter((file) => getPathBaseName(getPickedFilePath(file)) === targetBase);
  if (baseMatches.length === 1) return baseMatches[0];
  return null;
}

function findPickedFileByBaseName(files, name) {
  const target = String(name || '').trim();
  if (!target) return null;
  const matches = files.filter((file) => getPathBaseName(getPickedFilePath(file)) === target);
  if (matches.length === 0) return null;
  return matches[0];
}

const MODEL_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$/;
const MODEL_ID_LABEL_MAX = 56;

function normalizeModelIdInput(value) {
  return String(value || '').trim();
}

function isValidModelId(value) {
  return MODEL_ID_PATTERN.test(normalizeModelIdInput(value));
}

function assertValidModelId(value, sourceLabel = 'modelId') {
  const normalized = normalizeModelIdInput(value);
  if (!normalized) {
    throw new Error(`${sourceLabel} is required.`);
  }
  if (!isValidModelId(normalized)) {
    throw new Error(
      `${sourceLabel} must match ${MODEL_ID_PATTERN.source} (2-128 chars, alnum, dot, underscore, hyphen).`
    );
  }
  return normalized;
}

function formatModelIdLabel(modelId, maxLength = MODEL_ID_LABEL_MAX) {
  const normalized = normalizeModelIdInput(modelId).replace(/\s+/g, ' ');
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, Math.max(0, maxLength - 3))}...`;
}

function getRegisteredModelId(entry) {
  const candidate = typeof entry?.modelId === 'string' && entry.modelId
    ? entry.modelId
    : (typeof entry?.id === 'string' ? entry.id : '');
  const normalized = normalizeModelIdInput(candidate);
  if (!normalized) return '';
  if (!isValidModelId(normalized)) {
    log.warn('DopplerDemo', `Skipping invalid modelId from registry: ${formatModelIdLabel(normalized, 96)}`);
    return '';
  }
  return normalized;
}

const TRANSLATE_MODEL_HINTS = Object.freeze([
  'translate',
  'translation',
  'nllb',
  'm2m',
  'marian',
  'madlad',
  'seamless',
  'opus',
  'mt',
]);

function getModelSelectionScore(mode, modelId) {
  const normalizedMode = normalizeDeepLinkMode(mode, 'run');
  const id = String(modelId || '').toLowerCase();
  let score = 0;

  if (normalizedMode === 'run') {
    if (id.includes('gemma-3')) score += 50;
    else if (id.includes('gemma')) score += 25;
  }

  if (normalizedMode === 'translate') {
    let hasTranslateHint = false;
    for (const hint of TRANSLATE_MODEL_HINTS) {
      if (id.includes(hint)) {
        score += 24;
        hasTranslateHint = true;
      }
    }
    if (id.includes('gemma-3')) score -= 60;
    else if (id.includes('gemma')) score -= 30;
    if (!hasTranslateHint) score -= 10;
  }

  if (id.includes('embedding') || id.includes('diffusion') || id.includes('energy')) {
    score -= 80;
  }

  return score;
}

async function deriveModelIdFromFiles(files, fallbackLabel) {
  const fallback = normalizeModelIdInput(fallbackLabel);
  if (isValidModelId(fallback)) return fallback;

  const configFile = files.find((file) => file.name === 'config.json');
  if (configFile) {
    try {
      const text = await configFile.text();
      const json = JSON.parse(text);
      const rawName = json?._name_or_path || json?.model_id || json?.modelId || json?.name;
      if (typeof rawName === 'string' && rawName.trim()) {
        const parts = rawName.trim().split('/');
        const name = parts[parts.length - 1];
        if (isValidModelId(name)) return name;
      }
    } catch {
      // Ignore config parsing errors here; converter will handle validation.
    }
  }

  const weightFile = files.find((file) => {
    const name = file.name.toLowerCase();
    return name.endsWith('.safetensors') || name.endsWith('.gguf');
  });
  if (weightFile) {
    const base = weightFile.name.replace(/\.(safetensors|gguf)$/i, '');
    if (isValidModelId(base)) return base;
  }

  return null;
}

async function filterModelsForMode(models, mode) {
  if (!isModeModelSelectable(mode)) return models;
  const filtered = [];
  for (const model of models) {
    const modelId = getRegisteredModelId(model);
    if (!modelId) continue;
    const modelType = await getModelTypeForId(modelId);
    if (isCompatibleModelType(modelType, mode)) {
      filtered.push(model);
    }
  }
  return filtered;
}

async function registerDownloadedModel(modelId) {
  const normalizedModelId = assertValidModelId(modelId, 'Downloaded modelId');
  await openModelStore(normalizedModelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) return null;
  const manifest = parseManifest(manifestText);
  const entry = {
    modelId: normalizedModelId,
    totalSize: manifest.totalSize,
    quantization: manifest.quantization,
    hashAlgorithm: manifest.hashAlgorithm,
    modelType: manifest.modelType,
  };
  if (manifest.modelId && manifest.modelId !== normalizedModelId) {
    entry.sourceModelId = manifest.modelId;
  }
  return registerModel(entry);
}

async function resolveCompatibleModelId(mode) {
  if (!isModeModelSelectable(mode)) return null;
  const normalizedMode = normalizeDeepLinkMode(mode, 'run');
  let models = [];
  try {
    models = await listRegisteredModels();
  } catch (error) {
    log.warn('DopplerDemo', `Model registry unavailable: ${error.message}`);
  }
  const modelIds = models
    .map((entry) => getRegisteredModelId(entry))
    .filter(Boolean);
  if (!modelIds.length) return null;

  const preferred = state.modeModelId?.[normalizedMode] || null;
  if (preferred && modelIds.includes(preferred)) {
    const preferredType = await getModelTypeForId(preferred);
    if (isCompatibleModelType(preferredType, normalizedMode)) {
      return preferred;
    }
  }

  if (normalizedMode !== 'translate') {
    const pipelineId = state.activePipelineModelId;
    if (pipelineId && modelIds.includes(pipelineId)) {
      const pipelineType = normalizeModelType(state.activePipeline?.manifest?.modelType)
        || await getModelTypeForId(pipelineId);
      if (isCompatibleModelType(pipelineType, normalizedMode)) {
        return pipelineId;
      }
    }

    const current = state.activeModelId;
    if (current && modelIds.includes(current)) {
      const currentType = await getModelTypeForId(current);
      if (isCompatibleModelType(currentType, normalizedMode)) {
        return current;
      }
    }
  }

  let bestModelId = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const modelId of modelIds) {
    const modelType = await getModelTypeForId(modelId);
    if (!isCompatibleModelType(modelType, normalizedMode)) {
      continue;
    }
    const score = getModelSelectionScore(normalizedMode, modelId);
    if (bestModelId == null || score > bestScore) {
      bestModelId = modelId;
      bestScore = score;
    }
  }
  return bestModelId;
}

async function syncModelForMode(mode) {
  if (!isModeModelSelectable(mode)) return;
  const compatibleId = await resolveCompatibleModelId(mode);
  if (!compatibleId) {
    state.modeModelId[mode] = null;
    if (state.uiMode === mode) {
      state.activeModelId = null;
      const modelSelect = $('diagnostics-model');
      if (modelSelect) modelSelect.value = '';
    }
    return;
  }
  if (state.activeModelId !== compatibleId) {
    if (state.activePipeline && state.activePipelineModelId && state.activePipelineModelId !== compatibleId) {
      await unloadActivePipeline();
    }
    selectDiagnosticsModel(compatibleId);
  }
  state.modeModelId[mode] = compatibleId;
}

function getUiModeForModelType(modelType) {
  const normalizedType = normalizeModelType(modelType);
  if (normalizedType === 'embedding') return 'embedding';
  if (normalizedType === 'diffusion') return 'diffusion';
  if (normalizedType === 'energy') return 'energy';
  return 'run';
}

async function handleStorageTryModel(modelId) {
  if (!modelId) return;
  const modelType = await getModelTypeForId(modelId);
  const targetMode = getUiModeForModelType(modelType);
  await setUiMode(targetMode);
  selectDiagnosticsModel(modelId);
}

function updateSidebarLayout(models) {
  const panelGrid = $('panel-grid');
  if (!panelGrid) return;
  const hasModels = Array.isArray(models) && models.length > 0;
  panelGrid.dataset.layout = hasModels ? 'ready' : 'empty';
}

async function computeModelAvailability(models) {
  const availability = { ...DEFAULT_MODEL_AVAILABILITY };
  if (!Array.isArray(models)) return availability;
  const seenModelIds = new Set();
  for (const model of models) {
    const modelId = getRegisteredModelId(model);
    if (!modelId || seenModelIds.has(modelId)) continue;
    seenModelIds.add(modelId);
    availability.total += 1;

    let modelType = normalizeModelType(model?.modelType);
    if (!modelType) {
      modelType = normalizeModelType(await getModelTypeForId(modelId));
    }
    if (isCompatibleModelType(modelType, 'run')) availability.run += 1;
    if (isCompatibleModelType(modelType, 'translate')) availability.translate += 1;
    if (isCompatibleModelType(modelType, 'embedding')) availability.embedding += 1;
    if (isCompatibleModelType(modelType, 'diffusion')) availability.diffusion += 1;
    if (isCompatibleModelType(modelType, 'energy')) availability.energy += 1;
  }
  return availability;
}

async function refreshModelList() {
  const modelSelect = $('diagnostics-model');
  if (!modelSelect) return;
  const refreshVersion = ++modelListRefreshVersion;
  state.modelAvailabilityLoading = true;
  state.modelAvailability = { ...DEFAULT_MODEL_AVAILABILITY };
  updateStatusIndicator();
  let models = [];
  try {
    models = await listRegisteredModels();
  } catch (error) {
    log.warn('DopplerDemo', `Model registry unavailable: ${error.message}`);
  }
  try {
    state.registeredModelIds = [...new Set(models
      .map((entry) => getRegisteredModelId(entry))
      .filter(Boolean))];
    const filteredModels = await filterModelsForMode(models, state.uiMode);
    if (refreshVersion !== modelListRefreshVersion) return;
    modelSelect.innerHTML = '';
    const modelIds = [];
    const seenModelIds = new Set();
    for (const model of filteredModels) {
      const modelId = getRegisteredModelId(model);
      if (!modelId || seenModelIds.has(modelId)) continue;
      const entryModelType = normalizeModelType(model?.modelType);
      if (entryModelType) {
        state.modelTypeCache[modelId] = entryModelType;
      }
      seenModelIds.add(modelId);
      modelIds.push(modelId);
    }
    if (!modelIds.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = `No ${getModeModelLabel(state.uiMode)} models`;
      modelSelect.appendChild(opt);
    } else {
      for (const modelId of modelIds) {
        const opt = document.createElement('option');
        opt.value = modelId;
        opt.textContent = formatModelIdLabel(modelId);
        opt.title = modelId;
        modelSelect.appendChild(opt);
      }
    }
    updateSidebarLayout(models);
    state.modelAvailability = await computeModelAvailability(models);
    await updateStorageInfo();
    await syncModelForMode(state.uiMode);
    renderTranslateCompareSelectors();
    syncTranslateCompareUI();
    if (state.uiMode === 'energy') {
      await preloadEnergyPipelineIfNeeded();
    }
    if (state.uiMode === 'models') {
      await refreshStorageInspector({
        onSelectModel: selectDiagnosticsModel,
        onTryModel: handleStorageTryModel,
        onUnloadActiveModel: unloadActivePipeline,
        onStorageInventoryRefreshed: renderQuickModelPanels,
        onModelsUpdated: refreshModelList,
      });
      renderQuickModelPanels();
    }
    updateDiagnosticsGuidance();
    updateModelEmptyStates();
  } finally {
    if (refreshVersion === modelListRefreshVersion) {
      state.modelAvailabilityLoading = false;
      if (!state.appInitializing) {
        updateStatusIndicator();
      }
    }
  }
}

async function refreshGpuInfo() {
  const deviceEl = $('gpu-device');
  const ramRow = $('gpu-ram-row');
  const ramEl = $('gpu-ram');
  const vramEl = $('gpu-vram');
  const featuresEl = $('gpu-features');
  const vramLabel = $('gpu-vram-label');
  const unifiedNote = $('gpu-unified-note');

  if (!isWebGPUAvailable()) {
    setText(deviceEl, 'WebGPU unavailable');
    setText(vramEl, '--');
    setText(featuresEl, 'none');
    setHidden(ramRow, true);
    setHidden(unifiedNote, true);
    return;
  }

  try {
    await initDevice();
    const caps = getKernelCapabilities();
    const adapter = caps.adapterInfo || {};
    const deviceLabel = [adapter.vendor, adapter.architecture || adapter.device, adapter.description]
      .filter(Boolean)
      .join(' ');
    setText(deviceEl, deviceLabel || 'Unknown GPU');

    if (Number.isFinite(navigator.deviceMemory)) {
      state.systemMemoryBytes = navigator.deviceMemory * 1024 * 1024 * 1024;
      setText(ramEl, `${navigator.deviceMemory} GB`);
      setHidden(ramRow, false);
    } else {
      setHidden(ramRow, true);
    }

    state.gpuMaxBytes = caps.maxBufferSize || 0;
    if (vramLabel) vramLabel.textContent = 'Buffer Limit';
    setText(vramEl, caps.maxBufferSize ? formatBytes(caps.maxBufferSize) : '--');

    const features = [
      caps.hasF16 && 'f16',
      caps.hasSubgroups && 'subgroups',
      caps.hasSubgroupsF16 && 'subgroups-f16',
      caps.hasTimestampQuery && 'timestamp',
    ].filter(Boolean);
    setText(featuresEl, features.length ? features.join(', ') : 'basic');

    let preferUnified = false;
    try {
      const platformConfig = getPlatformConfig();
      preferUnified = Boolean(platformConfig?.platform?.memoryHints?.preferUnifiedMemory);
    } catch {
      preferUnified = false;
    }
    setHidden(unifiedNote, !preferUnified);
  } catch (error) {
    setText(deviceEl, `GPU init failed`);
    setText(vramEl, '--');
    setText(featuresEl, 'none');
    setHidden(ramRow, true);
    setHidden(unifiedNote, true);
    log.warn('DopplerDemo', `GPU init failed: ${error.message}`);
  }
}

function getSelectedModelId() {
  if (state.activeModelId) return state.activeModelId;
  const modelSelect = $('diagnostics-model');
  const selected = modelSelect?.value || '';
  if (selected) {
    state.activeModelId = selected;
    return selected;
  }
  if (modelSelect?.options?.length) {
    const fallback = modelSelect.options[0].value;
    state.activeModelId = fallback || null;
    return fallback || null;
  }
  return null;
}

function pickRandomStarter(pool) {
  if (!Array.isArray(pool) || pool.length === 0) return '';
  const index = Math.floor(Math.random() * pool.length);
  return String(pool[index] || '').trim();
}

function isStarterExampleInput(inputEl) {
  return inputEl?.dataset?.starterExample === '1';
}

function setStarterExampleInput(inputEl, isExample) {
  if (!inputEl) return;
  inputEl.dataset.starterExample = isExample ? '1' : '0';
}

function pickRandomStarterDifferent(pool, currentValue) {
  if (!Array.isArray(pool) || pool.length === 0) return '';
  const current = String(currentValue || '').trim();
  if (pool.length === 1) return String(pool[0] || '').trim();
  for (let attempt = 0; attempt < pool.length * 2; attempt += 1) {
    const next = pickRandomStarter(pool);
    if (next && next !== current) {
      return next;
    }
  }
  return pickRandomStarter(pool);
}

function pickRandomSubset(pool, count) {
  if (!Array.isArray(pool) || pool.length === 0) return [];
  const targetCount = Math.max(1, Math.min(Number(count) || 1, pool.length));
  const copy = pool.slice();
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, targetCount);
}

function refreshEmbeddingDemoDocuments(options = {}) {
  const { force = false } = options;
  const current = Array.isArray(state.embeddingDemoDocuments) ? state.embeddingDemoDocuments : [];
  if (!force && current.length === EMBEDDING_DEMO_DOCUMENT_COUNT) {
    return current;
  }
  state.embeddingDemoDocuments = pickRandomSubset(
    EMBEDDING_DEMO_DOCUMENT_CATALOG,
    EMBEDDING_DEMO_DOCUMENT_COUNT
  );
  renderEmbeddingDocumentSet();
  return state.embeddingDemoDocuments;
}

function renderEmbeddingDocumentSet() {
  const wrap = $('run-embedding-docs');
  const list = $('run-embedding-docs-list');
  if (!wrap || !list) return;
  if (state.uiMode !== 'embedding') {
    setHidden(wrap, true);
    return;
  }
  setHidden(wrap, false);
  const docs = Array.isArray(state.embeddingDemoDocuments) ? state.embeddingDemoDocuments : [];
  if (docs.length === 0) {
    list.innerHTML = '<div class="type-caption">No documents configured.</div>';
    return;
  }
  const rows = docs
    .map((doc, index) => {
      const text = String(doc?.text || '').trim();
      const snippet = text.length > 140 ? `${text.slice(0, 140)}...` : text;
      return `<div class="embedding-doc-item"><div class="type-caption"><strong>${index + 1}. ${doc.title}</strong></div><div class="type-caption">${snippet}</div></div>`;
    })
    .join('');
  list.innerHTML = rows;
}

function applyStarterPrompt(inputEl, pool, options = {}) {
  if (!inputEl) return;
  const { force = false } = options;
  const current = String(inputEl.value || '').trim();
  if (!force && current.length > 0) return;
  const next = force ? pickRandomStarterDifferent(pool, current) : pickRandomStarter(pool);
  if (!next) return;
  inputEl.value = next;
  setStarterExampleInput(inputEl, true);
}

function prefillDemoTextInputs() {
  applyStarterPrompt($('run-prompt'), RUN_STARTER_PROMPTS);
  applyStarterPrompt($('diffusion-prompt'), DIFFUSION_STARTER_PROMPTS);
  applyStarterPrompt($('diffusion-negative'), DIFFUSION_NEGATIVE_STARTER_PROMPTS);
}

function bindStarterPromptInput(inputEl) {
  if (!inputEl) return;
  inputEl.addEventListener('focus', () => {
    if (isStarterExampleInput(inputEl)) {
      inputEl.select();
    }
  });
  inputEl.addEventListener('input', () => {
    setStarterExampleInput(inputEl, false);
  });
}

function syncRunControls() {
  const runPrompt = $('run-prompt');
  const runGenerate = $('run-generate-btn');
  const runStop = $('run-stop-btn');
  const runClear = $('run-clear-btn');
  const runResetKvToggle = $('run-reset-kv-toggle');
  const translateSourceSelect = $('translate-source-language');
  const translateTargetSelect = $('translate-target-language');
  const translateSwapBtn = $('translate-swap-btn');
  const temperatureInput = $('temperature-input');
  const topPInput = $('top-p-input');
  const topKInput = $('top-k-input');
  const maxTokensInput = $('max-tokens-input');
  const availability = getModelAvailability();
  const modeKey = state.uiMode === 'embedding'
    ? 'embedding'
    : (state.uiMode === 'translate' ? 'translate' : 'run');
  const hasCompatibleModel = Number.isFinite(availability[modeKey]) && availability[modeKey] > 0;
  const disabled = state.runGenerating || state.runLoading || state.compareGenerating || state.compareLoading;
  if (runPrompt) runPrompt.disabled = disabled;
  if (runGenerate) runGenerate.disabled = disabled || !hasCompatibleModel;
  if (runClear) runClear.disabled = disabled;
  if (runResetKvToggle) runResetKvToggle.disabled = disabled;
  if (translateSourceSelect) translateSourceSelect.disabled = disabled;
  if (translateTargetSelect) translateTargetSelect.disabled = disabled;
  if (translateSwapBtn) translateSwapBtn.disabled = disabled;
  if (temperatureInput) temperatureInput.disabled = disabled;
  if (topPInput) topPInput.disabled = disabled;
  if (topKInput) topKInput.disabled = disabled;
  if (maxTokensInput) maxTokensInput.disabled = disabled;
  if (runStop) setHidden(runStop, !state.runGenerating);
}

function setRunGenerating(isGenerating) {
  state.runGenerating = Boolean(isGenerating);
  if (!state.runGenerating) {
    state.runPrefilling = false;
  }
  syncRunControls();
  updateStatusIndicator();
}

function setRunLoading(isLoading) {
  state.runLoading = Boolean(isLoading);
  syncRunControls();
  updateStatusIndicator();
}

function setRunAutoLabel(inputId, labelId, value, options) {
  const input = $(inputId);
  const label = $(labelId);
  if (!label) return;
  const hasOverride = input?.value != null && input.value !== '';
  const prefix = hasOverride ? 'default' : 'auto';
  label.textContent = `${prefix}: ${formatAutoValue(value, options)}`;
}

function updateRunAutoLabels() {
  const runtime = getRuntimeConfig();
  const sampling = runtime?.inference?.sampling ?? {};
  const batching = runtime?.inference?.batching ?? {};
  const useTranslateDefaults = state.uiMode === 'translate';
  const defaultTemperature = useTranslateDefaults ? DEFAULT_TRANSLATE_TEMPERATURE : sampling.temperature;
  const defaultTopP = useTranslateDefaults ? DEFAULT_TRANSLATE_TOP_P : sampling.topP;
  const defaultTopK = useTranslateDefaults ? DEFAULT_TRANSLATE_TOP_K : sampling.topK;
  const defaultMaxTokens = useTranslateDefaults
    ? (state.compareEnabled ? TRANSLATE_COMPARE_DEFAULT_MAX_TOKENS : DEFAULT_TRANSLATE_MAX_TOKENS)
    : batching.maxTokens;
  setRunAutoLabel('temperature-input', 'temperature-auto', defaultTemperature);
  setRunAutoLabel('top-p-input', 'top-p-auto', defaultTopP);
  setRunAutoLabel('top-k-input', 'top-k-auto', defaultTopK, { integer: true });
  setRunAutoLabel('max-tokens-input', 'max-tokens-auto', defaultMaxTokens, { integer: true });
}

function formatCharCounter(value, maxLength) {
  const length = String(value || '').length;
  if (Number.isFinite(maxLength) && maxLength > 0) {
    return `${length}/${maxLength}`;
  }
  return String(length);
}

function updateDiffusionCharCounters() {
  const promptEl = $('diffusion-prompt');
  const negativeEl = $('diffusion-negative');
  const promptCountEl = $('diffusion-prompt-count');
  const negativeCountEl = $('diffusion-negative-count');

  if (promptCountEl) {
    const maxLength = promptEl?.maxLength ?? null;
    promptCountEl.textContent = formatCharCounter(promptEl?.value, maxLength);
  }
  if (negativeCountEl) {
    const maxLength = negativeEl?.maxLength ?? null;
    negativeCountEl.textContent = formatCharCounter(negativeEl?.value, maxLength);
  }
}

function buildRunGenerateOptions() {
  if (state.uiMode === 'embedding') {
    return {};
  }
  const isTranslateMode = state.uiMode === 'translate';
  const temperature = readOptionalNumber($('temperature-input'));
  const topP = readOptionalNumber($('top-p-input'));
  const topK = readOptionalNumber($('top-k-input'), { integer: true });
  const maxTokens = readOptionalNumber($('max-tokens-input'), { integer: true });
  const options = {};
  if (temperature != null) {
    options.temperature = Math.max(0, temperature);
  }
  if (topP != null) {
    options.topP = Math.max(0, Math.min(1, topP));
  }
  if (topK != null) {
    options.topK = Math.max(0, topK);
  }
  if (maxTokens != null && maxTokens > 0) {
    options.maxTokens = Math.max(1, maxTokens);
  }
  if (isTranslateMode) {
    if (temperature == null) {
      options.temperature = DEFAULT_TRANSLATE_TEMPERATURE;
    }
    if (topP == null) {
      options.topP = DEFAULT_TRANSLATE_TOP_P;
    }
    if (topK == null) {
      options.topK = DEFAULT_TRANSLATE_TOP_K;
    }
    if (maxTokens == null) {
      options.maxTokens = DEFAULT_TRANSLATE_MAX_TOKENS;
    }
  }
  return options;
}

async function loadPipelineFromStorage(modelId) {
  await openModelStore(modelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) {
    throw new Error('Manifest not found in storage');
  }
  const manifest = parseManifest(manifestText);
  await initDevice();
  const device = getDevice();
  return createPipeline(manifest, {
    gpu: { device },
    runtimeConfig: getRuntimeConfig(),
    onProgress: (progress) => updateProgressFromLoader(progress),
  });
}

// Shared pipeline loader — handles overlay, manifest parse, GPU upload, memory stats.
// Callers set their own loading flag before calling and clear it in their own finally.
async function ensurePipeline(modelId, overlayTitle, modeKey) {
  if (!modelId) throw new Error('Select a model before generating');
  if (state.activePipeline && state.activeModelId === modelId) return state.activePipeline;
  if (state.activePipeline) await unloadActivePipeline();
  showProgressOverlay(overlayTitle, modelId);
  try {
    const pipeline = await loadPipelineFromStorage(modelId);
    state.activePipeline = pipeline;
    state.activeModelId = modelId;
    state.activePipelineModelId = modelId;
    if (pipeline?.manifest?.modelType) {
      state.modelTypeCache[modelId] = normalizeModelType(pipeline.manifest.modelType);
    }
    state.modeModelId[modeKey] = modelId;
    state.lastMemoryStats = pipeline.getMemoryStats?.() ?? null;
    updateMemoryControls();
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
    return pipeline;
  } finally {
    hideProgressOverlay();
  }
}

async function ensureRunPipeline() {
  const modelId = getSelectedModelId();
  const modeKey = state.uiMode === 'embedding'
    ? 'embedding'
    : (state.uiMode === 'translate' ? 'translate' : 'run');
  setRunLoading(true);
  try {
    return await ensurePipeline(modelId, 'Loading Model', modeKey);
  } finally {
    setRunLoading(false);
  }
}

async function ensureDiffusionPipeline() {
  const modelId = getSelectedModelId();
  state.diffusionLoading = true;
  updateStatusIndicator();
  try {
    return await ensurePipeline(modelId, 'Loading Model', 'diffusion');
  } finally {
    state.diffusionLoading = false;
    updateStatusIndicator();
  }
}

function drawDiffusionCanvas(result) {
  const canvas = $('diffusion-canvas');
  if (!canvas || !result) return;
  canvas.width = result.width;
  canvas.height = result.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const imageData = new ImageData(result.pixels, result.width, result.height);
  ctx.putImageData(imageData, 0, 0);
}

async function handleDiffusionRun() {
  if (state.diffusionGenerating || state.diffusionLoading) return;
  const promptEl = $('diffusion-prompt');
  const negativeEl = $('diffusion-negative');
  const stepsEl = $('diffusion-steps');
  const guidanceEl = $('diffusion-guidance');
  const seedEl = $('diffusion-seed');
  const widthEl = $('diffusion-width');
  const heightEl = $('diffusion-height');

  const request = {
    prompt: promptEl?.value?.trim() || '',
    negativePrompt: negativeEl?.value?.trim() || '',
    steps: stepsEl?.value ? Number(stepsEl.value) : undefined,
    guidanceScale: guidanceEl?.value ? Number(guidanceEl.value) : undefined,
    seed: seedEl?.value ? Number(seedEl.value) : undefined,
    width: widthEl?.value ? Number(widthEl.value) : undefined,
    height: heightEl?.value ? Number(heightEl.value) : undefined,
  };
  state.lastDiffusionRequest = { ...request };

  updateDiffusionStatus('Preparing...');
  state.diffusionGenerating = true;
  updateStatusIndicator();
  try {
    const pipeline = await ensureDiffusionPipeline();
    if (!pipeline.generate) {
      throw new Error('Selected model does not support diffusion generation.');
    }
    if (!pipeline.manifest || pipeline.manifest.modelType !== 'diffusion') {
      throw new Error('Selected model is not a diffusion model.');
    }
    updateDiffusionStatus('Generating...');
    const result = await pipeline.generate(request);
    if (result) {
      state.lastDiffusionRequest = {
        ...state.lastDiffusionRequest,
        width: result.width,
        height: result.height,
      };
    }
    if (!Number.isFinite(result?.width) || result.width <= 0 || !Number.isFinite(result?.height) || result.height <= 0) {
      throw new Error('Diffusion output dimensions are invalid.');
    }
    drawDiffusionCanvas(result);
    state.lastInferenceStats = pipeline.getStats?.() ?? null;
    state.lastMemoryStats = pipeline.getMemoryStats?.() ?? state.lastMemoryStats;
    if (state.lastInferenceStats) {
      state.runCounter += 1;
      recordRunLog(state.lastInferenceStats, `#${state.runCounter}`);
    }
    updateDiffusionStatus('Complete');
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
  } catch (error) {
    log.error('DopplerDemo', `Diffusion run failed: ${error.message}`);
    updateDiffusionStatus(`Error: ${error.message}`);
  } finally {
    state.diffusionGenerating = false;
    updateStatusIndicator();
  }
}

function handleDiffusionClear() {
  const canvas = $('diffusion-canvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }
  updateDiffusionStatus('Idle');
}

async function preloadEnergyPipelineIfNeeded() {
  if (state.uiMode !== 'energy') return;
  if (state.energyLoading || state.energyGenerating) return;

  const modelId = getSelectedModelId();
  if (!modelId) return;

  const selectedModelType = normalizeModelType(await getModelTypeForId(modelId));
  if (selectedModelType !== 'energy') return;

  const activeModelType = normalizeModelType(state.activePipeline?.manifest?.modelType);
  if (
    state.activePipeline &&
    state.activeModelId === modelId &&
    activeModelType === 'energy'
  ) {
    return;
  }

  updateEnergyStatus('Loading energy model...');
  try {
    await ensureEnergyPipeline();
    if (!state.energyGenerating) updateEnergyStatus('Ready');
  } catch (error) {
    log.warn('DopplerDemo', `Energy preload skipped: ${error.message}`);
    if (!state.energyGenerating) updateEnergyStatus('Idle');
  }
}

async function ensureEnergyPipeline() {
  const modelId = getSelectedModelId();
  state.energyLoading = true;
  updateStatusIndicator();
  try {
    return await ensurePipeline(modelId, 'Loading Model', 'energy');
  } finally {
    state.energyLoading = false;
    updateStatusIndicator();
  }
}

async function runStandaloneQuintelPipeline(request) {
  const { EnergyPipeline } = await import('../src/energy/index.js');
  const pipeline = new EnergyPipeline();
  await pipeline.initialize({
    runtimeConfig: getRuntimeConfig(),
  });
  pipeline.manifest = {
    modelId: 'quintel-standalone',
    modelType: 'energy',
    energy: {},
  };
  try {
    return await pipeline.generate(request);
  } finally {
    await pipeline.unload();
  }
}

async function handleEnergyRun() {
  if (state.energyGenerating || state.energyLoading) return;
  const demo = getEnergyDemoById(state.energyDemoId) || getEnergyDemoById(DEFAULT_ENERGY_DEMO_ID);
  const problem = 'quintel';
  const size = readOptionalNumber($('energy-quintel-size'), { integer: true });
  const displayThreshold = readOptionalNumber($('energy-quintel-threshold'));
  const countTarget = readOptionalNumber($('energy-quintel-count-target'), { integer: true });
  const mirrorX = $('energy-rule-mirror-x')?.checked ?? false;
  const mirrorY = $('energy-rule-mirror-y')?.checked ?? false;
  const diagonal = $('energy-rule-diagonal')?.checked ?? false;
  const countRule = $('energy-rule-count')?.checked ?? false;
  const symmetryWeight = readOptionalNumber($('energy-weight-symmetry'));
  const countWeight = readOptionalNumber($('energy-weight-count'));
  const binarizeWeight = readOptionalNumber($('energy-weight-binarize'));
  const initMode = $('energy-init-mode')?.value || undefined;
  const initSeed = readOptionalNumber($('energy-init-seed'), { integer: true });
  const initScale = readOptionalNumber($('energy-init-scale'));
  const steps = readOptionalNumber($('energy-steps'), { integer: true });
  const stepSize = readOptionalNumber($('energy-step-size'));
  const gradientScale = readOptionalNumber($('energy-gradient-scale'));
  const convergenceThreshold = readOptionalNumber($('energy-convergence'));

  const request = {
    problem,
    initMode,
    seed: initSeed,
    initScale,
    steps,
    stepSize,
    gradientScale,
    convergenceThreshold,
  };
  const quintelRules = {
    mirrorX,
    mirrorY,
    diagonal,
    count: countRule,
    center: false,
  };
  const quintel = {
    rules: quintelRules,
  };
  if (size != null) quintel.size = size;
  if (Number.isFinite(countTarget)) quintel.countTarget = countTarget;
  const weights = {};
  if (Number.isFinite(symmetryWeight)) weights.symmetry = symmetryWeight;
  if (Number.isFinite(countWeight)) weights.count = countWeight;
  if (Number.isFinite(binarizeWeight)) weights.binarize = binarizeWeight;
  if (Object.keys(weights).length) quintel.weights = weights;
  request.quintel = quintel;

  state.lastEnergyResult = null;
  state.lastEnergyRequest = {
    size,
    displayThreshold,
  };

  updateEnergyStatus('Preparing...');
  state.energyGenerating = true;
  updateStatusIndicator();
  try {
    let result = null;
    let pipelineForStats = null;
    const selectedModelId = getSelectedModelId();
    const selectedModelType = normalizeModelType(await getModelTypeForId(selectedModelId));
    const useStandaloneQuintel = selectedModelType !== 'energy';

    if (useStandaloneQuintel) {
      updateEnergyStatus('Running Quintel...');
      result = await runStandaloneQuintelPipeline(request);
    } else {
      pipelineForStats = await ensureEnergyPipeline();
      if (!pipelineForStats.generate) {
        throw new Error('Selected model does not support energy generation.');
      }
      if (!pipelineForStats.manifest || pipelineForStats.manifest.modelType !== 'energy') {
        throw new Error('Selected model is not an energy model.');
      }
      updateEnergyStatus('Running...');
      result = await pipelineForStats.generate(request);
    }
    state.lastEnergyResult = result;
    if (result?.shape) {
      state.lastEnergyRequest = {
        shape: result.shape,
        size: result.shape[0],
        displayThreshold,
      };
    }
    drawEnergyChart(result?.energyHistory || []);
    updateEnergyStats(result);
    renderEnergyBoard(result?.state, result?.shape ?? size, displayThreshold);
    state.lastInferenceStats = pipelineForStats?.getStats?.() ?? null;
    state.lastMemoryStats = pipelineForStats?.getMemoryStats?.() ?? state.lastMemoryStats;
    if (state.lastInferenceStats) {
      state.runCounter += 1;
      recordRunLog(state.lastInferenceStats, `#${state.runCounter}`, 'energy');
    }
    updateEnergyStatus('Complete');
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
  } catch (error) {
    log.error('DopplerDemo', `Energy run failed: ${error.message}`);
    updateEnergyStatus(`Error: ${error.message}`);
  } finally {
    state.energyGenerating = false;
    updateStatusIndicator();
  }
}

function handleEnergyClear() {
  clearEnergyChart();
  clearEnergyBoard();
  updateEnergyStats(null);
  updateEnergyStatus('Idle');
  state.lastEnergyResult = null;
}

async function handleRunGenerate() {
  if (state.runGenerating || state.runLoading) return;
  if (isTranslateCompareEnabled()) {
    await handleTranslateCompareRun();
    return;
  }
  const promptEl = $('run-prompt');
  const outputEl = $('run-output');
  const prompt = promptEl?.value?.trim() || '';
  const isEmbeddingMode = state.uiMode === 'embedding';
  const isTranslateMode = state.uiMode === 'translate';
  const runResetKvToggle = $('run-reset-kv-toggle');
  const resetContextEachRun = !isEmbeddingMode && Boolean(runResetKvToggle?.checked);
  if (!prompt) {
    updateRunStatus(
      isEmbeddingMode
        ? 'Enter text to embed.'
        : (isTranslateMode ? 'Enter text to translate.' : 'Enter a prompt to generate.')
    );
    return;
  }

  const translateSelection = isTranslateMode ? getTranslateLanguageSelection() : null;
  const translateRequest = isTranslateMode
    ? createTranslateTextRequest(
      prompt,
      translateSelection.sourceCode,
      translateSelection.targetCode
    )
    : null;
  let generationInput = isTranslateMode ? translateRequest : prompt;

  updateRunStatus('Preparing...');
  let pipeline;
  let modelType = null;
  try {
    pipeline = await ensureRunPipeline();
    modelType = normalizeModelType(pipeline?.manifest?.modelType);
    if (isEmbeddingMode && modelType !== 'embedding') {
      throw new Error('Selected model is not an embedding model.');
    }
    if (!isEmbeddingMode && (modelType === 'diffusion' || modelType === 'energy' || modelType === 'embedding')) {
      throw new Error('Selected model is not a text model.');
    }
    if (isTranslateMode && translateSelection) {
      const chatTemplateType = pipeline.manifest?.inference?.chatTemplate?.type;
      if (chatTemplateType !== 'translategemma') {
        // General chat model: build a plain instruction prompt instead of the structured
        // translate request, which only translategemma knows how to interpret.
        const { sourceCode, targetCode } = translateSelection;
        generationInput = `Translate the following from ${sourceCode} to ${targetCode}. Output only the translation, no explanation.\n\n${prompt}`;
      }
    }
    if (resetContextEachRun) {
      pipeline.reset?.();
    }
  } catch (error) {
    updateRunStatus(`Error: ${error.message}`);
    return;
  }

  const controller = new AbortController();
  state.runAbortController = controller;
  state.runPrefilling = !isEmbeddingMode;
  setRunGenerating(true);
  updateRunStatus(isEmbeddingMode ? 'Embedding...' : (isTranslateMode ? 'Translating...' : 'Generating...'));
  if (outputEl) outputEl.textContent = '';

  const options = buildRunGenerateOptions();
  const isEmbeddingModel = modelType === 'embedding';
  let output = '';
  let tokenCount = 0;
  const start = performance.now();
  let firstTokenAt = null;

  try {
    if (isEmbeddingModel) {
      const embedStart = performance.now();
      pipeline.reset?.();
      const result = await pipeline.embed(prompt, options);
      const queryEmbeddingValues = result?.embedding ?? new Float32Array(0);
      const querySummary = summarizeEmbeddingVector(queryEmbeddingValues);
      if (!Number.isFinite(querySummary.dimension) || querySummary.dimension <= 0) {
        throw new Error('No embedding returned.');
      }
      if (querySummary.nonFiniteCount > 0) {
        throw new Error(`Embedding contains non-finite values (${querySummary.nonFiniteCount}/${querySummary.dimension}).`);
      }
      const embeddingDocuments = refreshEmbeddingDemoDocuments({ force: true });
      updateRunStatus('Embedding reference documents...');
      const scoredDocuments = [];
      for (const doc of embeddingDocuments) {
        pipeline.reset?.();
        const docResult = await pipeline.embed(doc.text, options);
        const docEmbeddingValues = docResult?.embedding ?? new Float32Array(0);
        const docSummary = summarizeEmbeddingVector(docEmbeddingValues);
        const score = cosineSimilarity(queryEmbeddingValues, docEmbeddingValues);
        scoredDocuments.push({
          id: doc.id,
          title: doc.title,
          text: doc.text,
          tokens: Number.isFinite(docResult?.tokens?.length) ? docResult.tokens.length : 0,
          dimension: docSummary.dimension,
          nonFinite: docSummary.nonFiniteCount,
          score: Number.isFinite(score) ? Number(score.toFixed(6)) : null,
        });
      }

      const ranked = scoredDocuments
        .slice()
        .sort((a, b) => (b.score ?? Number.NEGATIVE_INFINITY) - (a.score ?? Number.NEGATIVE_INFINITY))
        .map((entry, index) => ({ rank: index + 1, ...entry }));
      const embeddingMs = Math.max(1, performance.now() - embedStart);

      output = JSON.stringify(
        {
          mode: 'embedding',
          query: prompt,
          dimension: querySummary.dimension,
          tokens: result?.tokens?.length ?? 0,
          embedding_preview: querySummary.preview,
          retrieval: {
            documents: scoredDocuments,
            ranked,
            top_match: ranked[0]
              ? { id: ranked[0].id, title: ranked[0].title, score: ranked[0].score }
              : null,
          },
        },
        null,
        2
      );
      state.lastMetrics = {
        ...(state.lastMetrics || {}),
        embeddingDim: querySummary.dimension,
        embeddingMs: Number(embeddingMs.toFixed(2)),
      };
      if (outputEl) outputEl.textContent = output;
      updateRunStatus('Complete');
    } else {
      for await (const token of pipeline.generate(generationInput, {
        ...options,
        signal: controller.signal,
        ...(isTranslateMode ? { useChatTemplate: true } : {}),
      })) {
        if (controller.signal.aborted) break;
        output += token;
        tokenCount += 1;
        const now = performance.now();
        if (!firstTokenAt) {
          firstTokenAt = now;
          if (state.runPrefilling) {
            state.runPrefilling = false;
            updateStatusIndicator();
          }
        }
        if (firstTokenAt) {
          const elapsedDecode = Math.max(1, now - firstTokenAt);
          const liveTokensPerSec = tokenCount / (elapsedDecode / 1000);
          state.lastMetrics = {
            ...(state.lastMetrics || {}),
            liveTokensPerSec,
          };
        }
        if (outputEl) outputEl.textContent = output;
      }
      updateRunStatus(controller.signal.aborted ? 'Stopped' : 'Complete');
    }
  } catch (error) {
    if (controller.signal.aborted) {
      updateRunStatus('Stopped');
    } else {
      updateRunStatus(`Error: ${error.message}`);
    }
  } finally {
    const elapsed = Math.max(1, performance.now() - start);
    const tokensPerSec = tokenCount > 0 ? Number(((tokenCount / elapsed) * 1000).toFixed(2)) : null;
    state.lastMetrics = {
      ...(state.lastMetrics || {}),
      tokensPerSec,
      liveTokensPerSec: null,
    };
    if (translateSelection) {
      state.lastMetrics.translateSource = translateSelection.sourceCode;
      state.lastMetrics.translateTarget = translateSelection.targetCode;
      state.lastMetrics.translateRequest = translateRequest;
    }
    state.lastMemoryStats = pipeline?.getMemoryStats?.() ?? state.lastMemoryStats;
    state.lastInferenceStats = pipeline?.getStats?.() ?? state.lastInferenceStats;
    if (state.lastInferenceStats) {
      state.runCounter += 1;
      recordRunLog(state.lastInferenceStats, `#${state.runCounter}`);
    }
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
    setRunGenerating(false);
    state.runAbortController = null;
  }
}

function stopRunGeneration() {
  if (state.runAbortController) {
    state.runAbortController.abort();
  }
}

function handleRunClear() {
  const promptEl = $('run-prompt');
  const outputEl = $('run-output');
  if (promptEl) {
    promptEl.value = '';
    setStarterExampleInput(promptEl, false);
  }
  if (outputEl) outputEl.textContent = '';
  for (const laneId of getCompareLaneIds()) {
    clearCompareLaneResult(laneId);
    renderTranslateCompareLane(laneId);
  }
  updateRunStatus('Idle');
  syncDeepLinkFromUI();
}

function handleInferencePulseReset() {
  state.lastMetrics = null;
  state.lastInferenceStats = null;
  state.lastMemoryStats = null;
  state.lastDiffusionRequest = null;
  state.lastEnergyRequest = null;
  state.runLog = [];
  state.runCounter = 0;
  for (const laneId of getCompareLaneIds()) {
    clearCompareLaneResult(laneId);
  }

  const snapshot = captureMemorySnapshot();
  updatePerformancePanel(snapshot);
  updateMemoryPanel(snapshot);
  renderRunLog();
  syncTranslateCompareUI();
}

function summarizeEmbeddingVector(values) {
  const dimension = Number.isFinite(values?.length) ? values.length : 0;
  let nonFiniteCount = 0;
  for (let i = 0; i < dimension; i++) {
    if (!Number.isFinite(values[i])) nonFiniteCount++;
  }
  return {
    dimension,
    nonFiniteCount,
    preview: Array.from(values.slice(0, Math.min(16, dimension))).map((v) => Number(v.toFixed(6))),
  };
}

function cosineSimilarity(a, b) {
  if (!ArrayBuffer.isView(a) || !ArrayBuffer.isView(b)) return null;
  if (a.length !== b.length || a.length <= 0) return null;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const av = Number(a[i]);
    const bv = Number(b[i]);
    if (!Number.isFinite(av) || !Number.isFinite(bv)) return null;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA <= 0 || normB <= 0) return null;
  return dot / Math.sqrt(normA * normB);
}

async function unloadActivePipeline() {
  if (!state.activePipeline) return;
  try {
    await state.activePipeline.unload?.();
  } catch (error) {
    log.warn('DopplerDemo', `Unload failed: ${error.message}`);
  }
  state.activePipeline = null;
  state.activePipelineModelId = null;
  state.lastMemoryStats = null;
  state.lastInferenceStats = null;
  updateMemoryControls();
  const snapshot = captureMemorySnapshot();
  updateMemoryPanel(snapshot);
  updatePerformancePanel(snapshot);
}

async function clearAllMemory() {
  await unloadAllCompareLaneRuntimes();
  await unloadActivePipeline();
  destroyBufferPool();
  const snapshot = captureMemorySnapshot();
  updateMemoryPanel(snapshot);
  updatePerformancePanel(snapshot);
}

function startTelemetryLoop() {
  if (state.uiIntervalId) return;

  let telemetryInFlight = false;
  const tick = async () => {
    if (telemetryInFlight) return;
    telemetryInFlight = true;
    try {
      const now = Date.now();
      if (now - state.lastStorageRefresh > 15000) {
        state.lastStorageRefresh = now;
        await updateStorageInfo();
      }
      const snapshot = captureMemorySnapshot();
      updateMemoryPanel(snapshot);
      updatePerformancePanel(snapshot);
    } catch (error) {
      log.warn('DopplerDemo', `Telemetry update failed: ${error.message}`);
    } finally {
      telemetryInFlight = false;
    }
  };

  state.uiIntervalId = setInterval(() => {
    void tick();
  }, 1000);
  void tick();
}

function populateModelPresets() {
  const presetSelect = $('convert-model-preset');
  if (!presetSelect) return;
  presetSelect.innerHTML = '';
  const autoOpt = document.createElement('option');
  autoOpt.value = '';
  autoOpt.textContent = 'auto';
  presetSelect.appendChild(autoOpt);
  for (const presetId of listPresets()) {
    const opt = document.createElement('option');
    opt.value = presetId;
    opt.textContent = presetId;
    presetSelect.appendChild(opt);
  }
}

function populateRuntimePresetSelect(select, entries, fallbackValue) {
  if (!select) return;
  const previous = select.value;
  select.innerHTML = '';
  for (const entry of entries) {
    const opt = document.createElement('option');
    opt.value = entry.id;
    opt.textContent = entry.label;
    select.appendChild(opt);
  }
  const target = previous || fallbackValue;
  if (target !== undefined && entries.some((entry) => entry.id === target)) {
    select.value = target;
    return;
  }
  if (entries.length > 0) {
    select.value = entries[0].id;
  }
}

function populateRuntimePresetSelects() {
  const baseSelect = $('runtime-preset');
  const overrideSelect = $('runtime-config-preset');
  const baseEntries = RUNTIME_PRESET_REGISTRY.filter((entry) => entry.base);
  const overrideEntries = RUNTIME_PRESET_REGISTRY.filter((entry) => entry.override);
  populateRuntimePresetSelect(baseSelect, baseEntries, DEFAULT_RUNTIME_PRESET);
  populateRuntimePresetSelect(overrideSelect, overrideEntries, '');
}

function buildConverterConfig() {
  const presetSelect = $('convert-model-preset');
  const presetId = presetSelect?.value?.trim() || null;
  const weightSelect = $('convert-weight-dtype');
  const weightOverride = weightSelect?.value?.trim().toLowerCase() || null;

  const config = createConverterConfig();
  if (presetId) {
    config.presets.model = presetId;
  }
  if (weightOverride) {
    config.quantization.weights = weightOverride;
  }
  return config;
}

async function runConversion(files, converterConfig, label, modelIdOverride) {
  if (!isConversionSupported()) {
    throw new Error('Browser conversion requires OPFS or IndexedDB.');
  }
  if (modelIdOverride != null) {
    assertValidModelId(modelIdOverride, 'Conversion modelId');
  }
  updateConvertStatus(`Preparing conversion${label ? ` (${label})` : ''}...`, 0);
  state.convertActive = true;
  updateStatusIndicator();
  try {
    const resultModelId = await convertModel(files, {
      modelId: modelIdOverride || undefined,
      converterConfig,
      onProgress: (update) => {
        if (!update) return;
        const percent = Number.isFinite(update.percent) ? update.percent : null;
        const message = update.message || 'Converting...';
        updateConvertStatus(label ? `${message} (${label})` : message, percent);
      },
    });
    updateConvertStatus(`Conversion complete: ${resultModelId}`, 100);
    await refreshModelList();
  } finally {
    state.convertActive = false;
    updateStatusIndicator();
  }
}

function restoreParsedManifest(previousManifest) {
  if (previousManifest) {
    setManifest(previousManifest);
    return;
  }
  clearManifest();
}

async function detectRdrrImport(files) {
  const manifestFile = findPickedFileByBaseName(files, 'manifest.json');
  if (!manifestFile) {
    return { kind: 'none' };
  }

  const manifestText = await manifestFile.text();
  const previousManifest = getManifest();
  let manifest;
  try {
    manifest = parseManifest(manifestText);
  } catch (error) {
    return {
      kind: 'invalid',
      reason: `Found manifest.json but it is not a valid RDRR manifest: ${error.message}`,
    };
  } finally {
    restoreParsedManifest(previousManifest);
  }

  const shardFiles = new Map();
  const missing = [];
  for (const shard of manifest.shards || []) {
    const shardFile = findPickedFileByPath(files, shard.filename);
    if (!shardFile) {
      missing.push(shard.filename || `shard_${shard.index}`);
      continue;
    }
    shardFiles.set(shard.index, shardFile);
  }

  if (missing.length > 0) {
    const preview = missing.slice(0, 3).join(', ');
    const suffix = missing.length > 3 ? ` (+${missing.length - 3} more)` : '';
    return {
      kind: 'invalid',
      reason: `Found RDRR manifest, but shard files are missing: ${preview}${suffix}`,
    };
  }

  let tensorsFile = null;
  if (manifest.tensorsFile) {
    tensorsFile = findPickedFileByPath(files, manifest.tensorsFile);
    if (!tensorsFile) {
      return {
        kind: 'invalid',
        reason: `Found RDRR manifest, but missing tensor map file: ${manifest.tensorsFile}`,
      };
    }
  }

  return {
    kind: 'rdrr',
    manifest,
    manifestText,
    manifestFile,
    shardFiles,
    tensorsFile,
  };
}

async function importRdrrFromFiles(files, detection, label) {
  if (!detection || detection.kind !== 'rdrr') {
    throw new Error('RDRR import requires a valid manifest and shard set.');
  }

  const previousManifest = getManifest();
  state.convertActive = true;
  updateStatusIndicator();
  try {
    const manifest = parseManifest(detection.manifestText);
    const modelId = assertValidModelId(manifest.modelId, 'RDRR manifest modelId');

    await openModelStore(modelId);

    const shards = Array.isArray(manifest.shards) ? manifest.shards : [];
    const totalSteps = shards.length + (manifest.tensorsFile ? 1 : 0) + 2;
    let completed = 0;
    const step = (message) => {
      completed += 1;
      const percent = totalSteps > 0 ? (completed / totalSteps) * 100 : 100;
      updateConvertStatus(label ? `${message} (${label})` : message, percent);
    };

    await saveManifest(JSON.stringify(manifest, null, 2));
    step(`Saved manifest for ${modelId}`);

    if (manifest.tensorsFile) {
      const tensorsFile = detection.tensorsFile || findPickedFileByPath(files, manifest.tensorsFile);
      if (!tensorsFile) {
        throw new Error(`Missing ${manifest.tensorsFile} for RDRR import.`);
      }
      const tensorsText = await tensorsFile.text();
      await saveTensorsToStore(tensorsText);
      step(`Saved ${manifest.tensorsFile}`);
    }

    const tokenizerFilePath = manifest.tokenizer?.file || null;
    let tokenizerJsonFile = tokenizerFilePath ? findPickedFileByPath(files, tokenizerFilePath) : null;
    let tokenizerModelFile = null;
    if (tokenizerJsonFile && getPathBaseName(getPickedFilePath(tokenizerJsonFile)) === 'tokenizer.model') {
      tokenizerModelFile = tokenizerJsonFile;
      tokenizerJsonFile = null;
    }
    if (!tokenizerJsonFile) {
      tokenizerJsonFile = findPickedFileByBaseName(files, 'tokenizer.json');
    }
    if (!tokenizerModelFile) {
      tokenizerModelFile = findPickedFileByBaseName(files, 'tokenizer.model');
    }

    if (tokenizerJsonFile) {
      await saveTokenizer(await tokenizerJsonFile.text());
    }
    if (tokenizerModelFile) {
      await saveTokenizerModel(await tokenizerModelFile.arrayBuffer());
    }

    for (const filename of AUX_IMPORT_FILENAMES) {
      const auxFile = findPickedFileByBaseName(files, filename);
      if (!auxFile) continue;
      await saveAuxFile(filename, await auxFile.arrayBuffer());
    }

    for (let i = 0; i < shards.length; i++) {
      const shard = shards[i];
      const shardFile = detection.shardFiles.get(shard.index) || findPickedFileByPath(files, shard.filename);
      if (!shardFile) {
        throw new Error(`Missing shard file: ${shard.filename}`);
      }
      const data = new Uint8Array(await shardFile.arrayBuffer());
      if (Number.isFinite(shard.size) && data.byteLength !== shard.size) {
        throw new Error(
          `Shard size mismatch for ${shard.filename}: expected ${shard.size} bytes, got ${data.byteLength}`
        );
      }
      await writeShard(shard.index, data, { verify: true });
      step(`Imported shard ${i + 1}/${shards.length}`);
    }

    await registerDownloadedModel(modelId);
    delete state.modelTypeCache[modelId];
    updateConvertStatus(`RDRR import complete: ${modelId}`, 100);
    await refreshModelList();
  } finally {
    restoreParsedManifest(previousManifest);
    state.convertActive = false;
    updateStatusIndicator();
  }
}

async function regenerateManifest(modelId) {
  if (!modelId) {
    throw new Error('Select a model before regenerating the manifest.');
  }

  await openModelStore(modelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) {
    throw new Error('Manifest not found in storage.');
  }

  const manifest = parseManifest(manifestText);
  let tensorMap = manifest.tensors ?? null;
  if (!tensorMap && manifest.tensorsFile) {
    const tensorsText = await loadTensorsFromStore();
    if (!tensorsText) {
      throw new Error('tensors.json not found in storage.');
    }
    tensorMap = JSON.parse(tensorsText);
  }
  if (!tensorMap) {
    throw new Error('Manifest is missing tensor locations.');
  }

  const tensorNames = Object.keys(tensorMap);
  for (const name of tensorNames) {
    const entry = tensorMap[name];
    if (entry) {
      entry.role = classifyTensorRole(name);
    }
  }

  let inference = manifest.inference;
  if (manifest.modelType === 'diffusion') {
    if (!inference) {
      inference = { ...DEFAULT_MANIFEST_INFERENCE, presetId: 'diffusion' };
    }
  } else {
    const rawConfig = manifest.config ?? {};
    const architectureHint = rawConfig.architectures?.[0] ?? rawConfig.model_type ?? '';
    const presetId = manifest.inference?.presetId || detectPreset(rawConfig, architectureHint);
    if (presetId === 'transformer') {
      const modelType = rawConfig.model_type ?? 'unknown';
      throw new Error(
        `Unknown model family: architecture="${architectureHint || 'unknown'}", model_type="${modelType}"`
      );
    }
    const preset = resolvePreset(presetId);
    const modelConfig = rawConfig?.text_config ?? rawConfig ?? {};
    const hiddenSize = modelConfig.hidden_size ?? modelConfig.n_embd ?? modelConfig.d_model ?? modelConfig.model_dim ?? null;
    const numHeads = modelConfig.num_attention_heads ?? modelConfig.n_head ?? modelConfig.num_heads ?? null;
    const derivedHeadDim = (Number.isFinite(hiddenSize) && Number.isFinite(numHeads) && numHeads > 0)
      ? hiddenSize / numHeads
      : null;
    const configHeadDim = Number.isFinite(rawConfig.head_dim) ? rawConfig.head_dim : null;
    const manifestHeadDim = (
      manifest.architecture
      && typeof manifest.architecture === 'object'
      && Number.isFinite(manifest.architecture.headDim)
    )
      ? manifest.architecture.headDim
      : null;
    const headDim = configHeadDim
      ?? manifestHeadDim
      ?? (Number.isFinite(derivedHeadDim) && Math.floor(derivedHeadDim) === derivedHeadDim ? derivedHeadDim : null);
    if (!headDim) {
      throw new Error('Missing headDim in manifest config (head_dim or hidden_size/num_attention_heads).');
    }
    inference = buildManifestInference(
      preset,
      rawConfig,
      headDim,
      manifest.quantizationInfo ?? null,
      tensorNames
    );
  }

  const embeddingOutput = inferEmbeddingOutputConfig(tensorMap);
  if (embeddingOutput && inference?.output) {
    inference = {
      ...inference,
      output: {
        ...inference.output,
        ...embeddingOutput,
      },
    };
  }

  const updatedManifest = {
    ...manifest,
    inference,
    tensors: tensorMap,
    tensorCount: tensorNames.length,
    metadata: {
      ...(manifest.metadata || {}),
      manifestRegeneratedAt: new Date().toISOString(),
    },
  };

  await saveManifest(JSON.stringify(updatedManifest, null, 2));
  if (manifest.tensorsFile) {
    await saveTensorsToStore(JSON.stringify(tensorMap, null, 2));
  }

  return updatedManifest;
}

async function handleRegenerateManifest() {
  if (state.convertActive) return;
  const modelId = getSelectedModelId();
  updateConvertStatus(`Regenerating manifest${modelId ? ` (${modelId})` : ''}...`, 0);
  state.convertActive = true;
  updateStatusIndicator();
  try {
    await regenerateManifest(modelId);
    if (modelId) {
      delete state.modelTypeCache[modelId];
    }
    updateConvertStatus(`Manifest regenerated: ${modelId}`, 100);
    await refreshModelList();
  } catch (error) {
    log.error('DopplerDemo', `Manifest regenerate failed: ${error.message}`);
    updateConvertStatus(`Manifest error: ${error.message}`, 0);
  } finally {
    state.convertActive = false;
    updateStatusIndicator();
  }
}

async function handleConvertFiles() {
  if (state.convertActive) return;
  updateConvertStatus('Select a model folder or files...', 0);
  let files = null;
  let pickedLabel = null;
  try {
    const pickedDirectory = await pickModelDirectory();
    files = pickedDirectory?.files || null;
    pickedLabel = pickedDirectory?.directoryName || null;
  } catch (error) {
    files = null;
  }

  if (!files || files.length === 0) {
    const pickedFiles = await pickModelFiles({ multiple: true });
    files = pickedFiles?.files || null;
  }

  if (!files || files.length === 0) {
    updateConvertStatus('No model files found in the selected folder.', 0);
    return;
  }

  const hasWeights = files.some((file) => {
    const name = file.name.toLowerCase();
    return name.endsWith('.safetensors') || name.endsWith('.gguf');
  });

  const rdrrDetection = await detectRdrrImport(files);
  if (rdrrDetection.kind === 'rdrr') {
    updateConvertStatus(
      `Detected pre-converted RDRR package${pickedLabel ? ` in ${pickedLabel}` : ''}. Importing...`,
      0
    );
    await importRdrrFromFiles(files, rdrrDetection, pickedLabel);
    return;
  }
  if (rdrrDetection.kind === 'invalid' && !hasWeights) {
    updateConvertStatus(rdrrDetection.reason, 0);
    return;
  }
  if (rdrrDetection.kind === 'invalid' && hasWeights) {
    log.warn('DopplerDemo', rdrrDetection.reason);
  }

  if (!hasWeights) {
    updateConvertStatus('Missing .safetensors or .gguf in the selected folder.', 0);
    return;
  }

  const modelIdOverride = await deriveModelIdFromFiles(files, pickedLabel);
  if (!modelIdOverride) {
    updateConvertStatus(
      'Missing valid modelId. Use 2-128 chars: letters/numbers plus dot, underscore, hyphen.',
      0
    );
    return;
  }

  updateConvertStatus(
    `Found ${files.length} files${pickedLabel ? ` in ${pickedLabel}` : ''}. Starting conversion...`,
    0
  );
  const converterConfig = buildConverterConfig();
  await runConversion(files, converterConfig, pickedLabel, modelIdOverride);
}

async function handleConvertUrls() {
  const urlInput = $('convert-url-input');
  if (!urlInput) return;
  const urls = urlInput.value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  if (!urls.length) return;

  const directRdrrBaseUrl = urls.length === 1 ? resolveDirectRdrrBaseUrlFromInput(urls[0]) : '';
  if (directRdrrBaseUrl) {
    updateConvertStatus('Detected direct RDRR manifest URL. Importing prebuilt package...', 0);
    await importRdrrFromBaseUrl(directRdrrBaseUrl);
    updateConvertStatus('Imported prebuilt RDRR package from manifest URL.', 100);
    return;
  }

  if (!state.quickModelCatalogLoading && getQuickCatalogEntries().length === 0) {
    await loadQuickModelCatalog();
  }
  const registryEntry = findQuickCatalogEntryForRegistryInput(urls);
  if (registryEntry) {
    updateConvertStatus(
      `Found ${registryEntry.modelId} in Doppler registry. Importing prebuilt RDRR instead of converting...`,
      0
    );
    await importQuickModelEntry(registryEntry);
    updateConvertStatus(`Imported prebuilt RDRR package: ${registryEntry.modelId}`, 100);
    return;
  }

  updateConvertStatus('No prebuilt RDRR match found in registry. Starting conversion...', 0);
  const converterConfig = buildConverterConfig();
  const sources = await createRemoteModelSources(urls, { converterConfig });
  await runConversion(sources, converterConfig);
}

async function handleDiagnosticsRun(mode) {
  const profileSelect = $('diagnostics-profile');
  const modelSelect = $('diagnostics-model');
  const presetSelect = $('runtime-preset');
  const selections = state.diagnosticsSelections[state.uiMode] || {};
  const selectedProfileId = profileSelect?.value || selections.profile || '';
  const selectedProfile = decodeDiagnosticsProfileId(selectedProfileId);
  const suite = selectedProfile?.suite || selections.suite || getDiagnosticsDefaultSuite(state.uiMode);
  const modelId = modelSelect?.value || null;
  const runtimePreset = selectedProfile?.preset || selections.preset || presetSelect?.value || DEFAULT_RUNTIME_PRESET;
  if (selectedProfile) {
    storeDiagnosticsSelection(state.uiMode, {
      profile: selectedProfileId,
      suite: selectedProfile.suite,
      preset: selectedProfile.preset,
    });
  }
  if (presetSelect && presetSelect.value !== runtimePreset) {
    presetSelect.value = runtimePreset;
  }
  if (profileSelect && selectedProfileId && profileSelect.value !== selectedProfileId) {
    profileSelect.value = selectedProfileId;
  }
  const captureOutput = runtimePreset === 'modes/debug';
  const previousRuntime = cloneRuntimeConfig(getRuntimeConfig());
  let runtimeConfig = state.diagnosticsRuntimeConfig;

  updateDiagnosticsStatus(`${mode === 'verify' ? 'Verifying' : 'Running'} ${suite}...`);
  updateDiagnosticsReport('');
  clearDiagnosticsOutput();
  try {
    if (!runtimeConfig || state.diagnosticsRuntimePresetId !== runtimePreset) {
      runtimeConfig = await refreshDiagnosticsRuntimeConfig(runtimePreset);
    }
    if (mode === 'verify') {
      const result = await controller.verifySuite(
        modelId ? { sources: { browser: { id: modelId } } } : null,
        {
          suite,
          runtimePreset,
          modelId,
          runtimeConfig,
        }
      );
      state.lastReport = result.report;
      state.lastReportInfo = result.reportInfo ?? null;
      state.lastMetrics = result.metrics ?? null;
      state.lastDiagnosticsSuite = result.suite ?? suite;
      updateDiagnosticsStatus('Verified');
      updateDiagnosticsReport(result.report?.timestamp || new Date().toISOString());
      renderDiagnosticsOutput({ suite, modelId, report: result.report }, suite, false);
      return;
    }

    if (state.activePipeline) {
      await unloadActivePipeline();
    }

    const options = {
      suite,
      runtimePreset,
      modelId,
      runtimeConfig,
      captureOutput,
    };
    const result = await controller.runSuite(
      modelId ? { sources: { browser: { id: modelId } } } : null,
      { ...options, keepPipeline: true }
    );
    state.lastReport = result.report;
    state.lastReportInfo = result.reportInfo;
    state.lastMetrics = result.metrics ?? null;
    state.lastDiagnosticsSuite = result.suite;
    if (result.memoryStats) {
      state.lastMemoryStats = result.memoryStats;
    }
    if (result.pipeline !== undefined) {
      state.activePipeline = result.pipeline;
    }
    state.activeModelId = modelId || null;
    state.lastInferenceStats = result.pipeline?.getStats?.() ?? state.lastInferenceStats;
    if (state.lastInferenceStats) {
      state.runCounter += 1;
      recordRunLog(state.lastInferenceStats, `#${state.runCounter}`);
    }
    if (result.suite === 'diffusion' && result.metrics) {
      state.lastDiffusionRequest = {
        width: result.metrics.width,
        height: result.metrics.height,
        steps: result.metrics.steps,
      };
    }
    if (result.suite === 'energy' && result.metrics) {
      const shape = Array.isArray(result.metrics.shape) ? result.metrics.shape : null;
      if (shape) {
        state.lastEnergyRequest = {
          shape,
          height: shape[0],
          width: shape[1],
          channels: shape[2],
        };
      }
      if (Array.isArray(result.metrics.energyHistory)) {
        drawEnergyChart(result.metrics.energyHistory);
      }
      updateEnergyStats({
        steps: result.metrics.steps,
        energy: result.metrics.energy,
        dtype: result.metrics.dtype,
        shape,
        stateStats: result.metrics.stateStats,
      });
    }
    updateDiagnosticsStatus(`Complete (${result.suite})`);
    if (result.reportInfo?.path) {
      updateDiagnosticsReport(result.reportInfo.path);
    } else if (result.report?.timestamp) {
      updateDiagnosticsReport(result.report.timestamp);
    }
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
    updateMemoryControls();
    renderDiagnosticsOutput(result, suite, captureOutput);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    updateDiagnosticsStatus(message, true);
    const timestamp = new Date().toISOString();
    const report = {
      suite,
      modelId,
      runtimePreset,
      timestamp,
      results: [{ name: mode === 'verify' ? 'verify-config' : 'run', passed: false, error: message }],
      metrics: { error: true, mode },
      output: { error: message },
    };
    state.lastReport = report;
    state.lastReportInfo = null;
    state.lastMetrics = report.metrics;
    state.lastDiagnosticsSuite = suite;
    updateDiagnosticsReport(timestamp);
    renderDiagnosticsOutput({ suite, modelId, report }, suite, captureOutput);
  } finally {
    setRuntimeConfig(previousRuntime);
    updateRunAutoLabels();
  }
}

function exportDiagnosticsReport() {
  if (!state.lastReport) {
    updateDiagnosticsStatus('No report available to export', true);
    return;
  }
  const timestamp = state.lastReport.timestamp || new Date().toISOString();
  const safeTimestamp = timestamp.replace(/[:]/g, '-');
  const filename = `doppler-report-${safeTimestamp}.json`;
  const blob = new Blob([JSON.stringify(state.lastReport, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function serializeTypedArray(value) {
  if (!value) return null;
  if (ArrayBuffer.isView(value)) return Array.from(value);
  return value;
}

function serializeSchedule(schedule) {
  if (!schedule) return null;
  return {
    slotAssignments: serializeTypedArray(schedule.slotAssignments),
    slotEngines: Array.isArray(schedule.slotEngines) ? schedule.slotEngines.slice() : schedule.slotEngines,
    slotIndices: Array.isArray(schedule.slotIndices) ? schedule.slotIndices.slice() : schedule.slotIndices,
  };
}

function serializeOps(ops) {
  if (!Array.isArray(ops)) return null;
  return ops.map((op) => ({
    id: op?.id ?? null,
    engine: op?.engine ?? null,
    slot: Array.isArray(op?.slot) ? op.slot.slice() : op?.slot ?? null,
    offloadable: !!op?.offloadable,
    meta: op?.meta ?? null,
  }));
}

function exportEnergyRun() {
  if (!state.lastEnergyResult) {
    updateEnergyStatus('No energy run available to export.');
    return;
  }
  const payload = {
    timestamp: new Date().toISOString(),
    problem: 'quintel',
    result: {
      backend: state.lastEnergyResult.backend ?? null,
      dtype: state.lastEnergyResult.dtype ?? null,
      shape: state.lastEnergyResult.shape ?? null,
      steps: state.lastEnergyResult.steps ?? null,
      energy: state.lastEnergyResult.energy ?? null,
      metrics: state.lastEnergyResult.metrics ?? null,
      baseline: state.lastEnergyResult.baseline ?? null,
      candidates: state.lastEnergyResult.candidates ?? null,
      energyHistory: state.lastEnergyResult.energyHistory ?? null,
      schedule: serializeSchedule(state.lastEnergyResult.schedule),
    },
  };
  const filename = `doppler-energy-export-${payload.timestamp.replace(/[:]/g, '-')}.json`;
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function bindUI() {
  const errorModal = $('error-modal');
  const errorClose = $('error-close');
  const advancedNav = $('advanced-nav');
  const convertBtn = $('convert-btn');
  const convertUrlBtn = $('convert-url-btn');
  const regenManifestBtn = $('regen-manifest-btn');
  const downloadStart = $('download-start-btn');
  const downloadPause = $('download-pause-btn');
  const downloadResume = $('download-resume-btn');
  const downloadCancel = $('download-cancel-btn');
  const downloadRefresh = $('download-refresh-btn');
  const runtimePreset = $('runtime-preset');
  const runtimeFile = $('runtime-config-file');
  const runtimeClear = $('runtime-config-clear');
  const runtimeConfigPreset = $('runtime-config-preset');
  const diagnosticsModelSelect = $('diagnostics-model');
  const diagnosticsProfile = $('diagnostics-profile');
  const diagnosticsRun = $('diagnostics-run-btn');
  const diagnosticsVerify = $('diagnostics-verify-btn');
  const diagnosticsExport = $('diagnostics-export-btn');
  const distillTeacherFile = $('distill-teacher-file');
  const distillTeacherJson = $('distill-teacher-json');
  const distillWorkloadSelect = $('distill-workload-select');
  const distillReplayBtn = $('distill-replay-btn');
  const distillExportBtn = $('distill-export-btn');
  const unloadModelBtn = $('unload-model-btn');
  const clearMemoryBtn = $('clear-memory-btn');
  const modelsQuickModelsList = $('models-list');
  const runPrompt = $('run-prompt');
  const runPromptShuffle = $('run-prompt-shuffle');
  const runGenerate = $('run-generate-btn');
  const runStop = $('run-stop-btn');
  const runClear = $('run-clear-btn');
  const translateSourceLanguage = $('translate-source-language');
  const translateTargetLanguage = $('translate-target-language');
  const translateSwapBtn = $('translate-swap-btn');
  const translateComparePreset = $('translate-compare-preset');
  const translateCompareRun = $('translate-compare-run-btn');
  const translateCompareExport = $('translate-compare-export-btn');
  const translateCompareExportLatest = $('translate-compare-export-latest-btn');
  const translateCompareShare = $('translate-compare-share-btn');
  const translateSmokeGrid = $('translate-smoke-grid');
  const translateHistoryList = $('translate-history-list');
  const translateHistoryClear = $('translate-history-clear-btn');
  const translateViewSingleBtn = $('translate-view-single-btn');
  const translateViewCompareBtn = $('translate-view-compare-btn');
  const translateLayoutSingleBtn = $('translate-layout-single-btn');
  const translateLayoutCompareBtn = $('translate-layout-compare-btn');
  const translateLeftEngine = $('translate-left-engine');
  const translateRightEngine = $('translate-right-engine');
  const translateLeftModel = $('translate-left-model');
  const translateRightModel = $('translate-right-model');
  const pulseReset = $('pulse-reset-btn');
  const temperatureInput = $('temperature-input');
  const topPInput = $('top-p-input');
  const topKInput = $('top-k-input');
  const maxTokensInput = $('max-tokens-input');
  const diffusionPrompt = $('diffusion-prompt');
  const diffusionPromptShuffle = $('diffusion-prompt-shuffle');
  const diffusionNegative = $('diffusion-negative');
  const diffusionSteps = $('diffusion-steps');
  const diffusionGuidance = $('diffusion-guidance');
  const diffusionSeed = $('diffusion-seed');
  const diffusionWidth = $('diffusion-width');
  const diffusionHeight = $('diffusion-height');
  const diffusionRun = $('diffusion-run-btn');
  const diffusionClear = $('diffusion-clear-btn');
  const energyDemoSelect = $('energy-demo-select');
  const energyRun = $('energy-run-btn');
  const energyExport = $('energy-export-btn');
  const energyClear = $('energy-clear-btn');
  const closeAdvancedNav = () => {
    if (advancedNav instanceof HTMLDetailsElement) {
      advancedNav.open = false;
    }
  };

  if (advancedNav instanceof HTMLDetailsElement) {
    advancedNav.open = false;
    document.addEventListener('click', (event) => {
      if (!advancedNav.open) return;
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (!advancedNav.contains(target)) {
        advancedNav.open = false;
      }
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        advancedNav.open = false;
      }
    });
  }

  errorClose?.addEventListener('click', () => hideErrorModal());
  errorModal?.addEventListener('click', (event) => {
    if (event.target === errorModal) {
      hideErrorModal();
    }
  });

  const onQuickModelAction = (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest('button[data-quick-action][data-quick-model-id]');
    if (!(button instanceof HTMLButtonElement)) return;
    const action = button.dataset.quickAction || '';
    const modelId = button.dataset.quickModelId || '';
    if (!action || !modelId) return;
    runQuickModelAction(action, modelId).catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      updateConvertStatus(`Quick model action failed: ${message}`, 0);
      updateDiagnosticsStatus(`Quick model action failed: ${message}`, true);
    });
  };

  modelsQuickModelsList?.addEventListener('click', onQuickModelAction);
  bindStarterPromptInput(runPrompt);
  bindStarterPromptInput(diffusionPrompt);
  bindStarterPromptInput(diffusionNegative);
  populateTranslateLanguageControls();

  const syncTranslateDirection = () => {
    const sourceCode = normalizeTranslateLanguageCode(translateSourceLanguage?.value, DEFAULT_TRANSLATE_SOURCE);
    if (translateSourceLanguage instanceof HTMLSelectElement) {
      translateSourceLanguage.value = sourceCode;
    }
    let targetCode = normalizeTranslateLanguageCode(translateTargetLanguage?.value, DEFAULT_TRANSLATE_TARGET);
    if (targetCode === sourceCode) {
      targetCode = sourceCode === DEFAULT_TRANSLATE_TARGET
        ? DEFAULT_TRANSLATE_SOURCE
        : DEFAULT_TRANSLATE_TARGET;
    }
    if (translateTargetLanguage instanceof HTMLSelectElement) {
      translateTargetLanguage.value = targetCode;
    }
    state.activeCompareSmokeSampleId = null;
    renderTranslateCompareSmokePanel();
    syncDeepLinkFromUI();
  };
  translateSourceLanguage?.addEventListener('change', syncTranslateDirection);
  translateTargetLanguage?.addEventListener('change', syncTranslateDirection);
  translateSwapBtn?.addEventListener('click', () => {
    swapTranslateLanguages();
    syncTranslateDirection();
  });
  translateViewSingleBtn?.addEventListener('click', () => setTranslateCompareEnabled(false));
  translateViewCompareBtn?.addEventListener('click', () => setTranslateCompareEnabled(true));
  translateLayoutSingleBtn?.addEventListener('click', () => setTranslateCompareEnabled(false));
  translateLayoutCompareBtn?.addEventListener('click', () => setTranslateCompareEnabled(true));
  translateComparePreset?.addEventListener('change', () => {
    applyTranslateComparePreset(translateComparePreset.value || 'proof').catch((error) => {
      updateRunStatus(`Compare preset error: ${error.message}`);
    });
    syncDeepLinkFromUI();
  });
  translateCompareRun?.addEventListener('click', () => {
    setTranslateCompareEnabled(true);
    handleTranslateCompareRun().catch((error) => {
      updateRunStatus(`Compare error: ${error.message}`);
    });
  });
  translateCompareExport?.addEventListener('click', () => {
    exportTranslateCompareArtifactPayload(getLatestTranslateCompareArtifact());
  });
  translateCompareExportLatest?.addEventListener('click', () => {
    exportTranslateCompareArtifactPayload(getLatestTranslateCompareArtifact());
  });
  translateCompareShare?.addEventListener('click', () => {
    setTranslateCompareEnabled(true);
    copyTranslateCompareShareLink().catch((error) => {
      updateRunStatus(`Share error: ${error.message}`);
    });
  });
  translateHistoryClear?.addEventListener('click', () => {
    if (typeof globalThis.confirm === 'function' && !globalThis.confirm('Clear saved compare history?')) {
      return;
    }
    state.compareHistory = [];
    state.lastCompareArtifact = null;
    persistTranslateCompareHistory();
    syncTranslateCompareUI();
  });
  document.querySelectorAll('[data-compare-history-filter]').forEach((button) => {
    button.addEventListener('click', () => {
      state.compareHistoryFilter = normalizeTranslateCompareHistoryFilter(button.dataset.compareHistoryFilter);
      syncTranslateCompareUI();
    });
  });
  translateSmokeGrid?.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest('button[data-compare-smoke-sample]');
    if (!(button instanceof HTMLButtonElement)) return;
    applyTranslateCompareSmokeSample(button.dataset.compareSmokeSample || '');
  });
  translateHistoryList?.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest('button[data-compare-history-export]');
    if (!(button instanceof HTMLButtonElement)) return;
    exportTranslateCompareHistoryArtifactById(button.dataset.compareHistoryExport || '');
  });

  const bindCompareLaneControls = (laneId, engineSelect, modelSelect) => {
    engineSelect?.addEventListener('change', () => {
      const lane = getCompareLane(laneId);
      if (!lane) return;
      lane.engine = resolveText(engineSelect.value, 'doppler');
      clearCompareLaneResult(laneId);
      populateCompareLaneModelSelect(laneId);
      renderTranslateCompareLane(laneId);
      syncDeepLinkFromUI();
    });
    modelSelect?.addEventListener('change', () => {
      const lane = getCompareLane(laneId);
      if (!lane) return;
      lane.modelId = resolveText(modelSelect.value, '');
      lane.tjsModelId = lane.engine === 'transformersjs'
        ? resolveTransformersModelIdForLane(lane)
        : '';
      clearCompareLaneResult(laneId);
      renderTranslateCompareLane(laneId);
      syncDeepLinkFromUI();
    });
  };

  bindCompareLaneControls('left', translateLeftEngine, translateLeftModel);
  bindCompareLaneControls('right', translateRightEngine, translateRightModel);
  document.querySelectorAll('.task-tab').forEach((button) => {
    button.addEventListener('click', () => {
      if (button.hidden) return;
      closeAdvancedNav();
      const task = button.dataset.task || 'run';
      setUiTask(task).catch((error) => {
        log.warn('DopplerDemo', `Task switch failed: ${error.message}`);
      });
    });
  });

  document.querySelectorAll('.mode-subtab').forEach((button) => {
    button.addEventListener('click', () => {
      if (button.hidden) return;
      closeAdvancedNav();
      const mode = button.dataset.mode || 'run';
      setUiMode(mode).catch((error) => {
        log.warn('DopplerDemo', `Mode switch failed: ${error.message}`);
      });
    });
  });

  [
    'run',
    'diffusion',
    'energy',
    'diagnostics',
  ].forEach((scope) => {
    const button = $(`${scope}-empty-notice-btn`);
    button?.addEventListener('click', () => {
      handleEmptyNoticeAction(scope).catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        updateConvertStatus(`Quick model action failed: ${message}`, 0);
        updateDiagnosticsStatus(`Quick model action failed: ${message}`, true);
      });
    });
  });

  convertBtn?.addEventListener('click', () => {
    resetConvertStatus();
    handleConvertFiles().catch((error) => {
      updateConvertStatus(`Conversion error: ${error.message}`);
    });
  });

  convertUrlBtn?.addEventListener('click', () => {
    resetConvertStatus();
    handleConvertUrls().catch((error) => {
      updateConvertStatus(`Conversion error: ${error.message}`);
    });
  });

  regenManifestBtn?.addEventListener('click', () => {
    resetConvertStatus();
    handleRegenerateManifest().catch((error) => {
      updateConvertStatus(`Manifest error: ${error.message}`);
    });
  });

  downloadStart?.addEventListener('click', () => {
    startDownload();
  });

  downloadPause?.addEventListener('click', () => {
    pauseActiveDownload();
  });

  downloadResume?.addEventListener('click', () => {
    resumeActiveDownload();
  });

  downloadCancel?.addEventListener('click', () => {
    cancelActiveDownload();
  });

  downloadRefresh?.addEventListener('click', () => {
    refreshDownloads();
  });

  runtimePreset?.addEventListener('change', () => {
    const mode = state.uiMode;
    storeDiagnosticsSelection(mode, { preset: runtimePreset.value || DEFAULT_RUNTIME_PRESET, profile: '' });
    if (runtimePreset.value !== 'modes/debug') {
      clearDiagnosticsOutput();
    }
    applySelectedRuntimePreset();
  });

  diagnosticsModelSelect?.addEventListener('change', () => {
    selectDiagnosticsModel(diagnosticsModelSelect.value || null);
  });

  diagnosticsProfile?.addEventListener('change', () => {
    const selectedProfileId = diagnosticsProfile.value || '';
    const selectedProfile = decodeDiagnosticsProfileId(selectedProfileId);
    if (selectedProfile) {
      storeDiagnosticsSelection(state.uiMode, {
        profile: selectedProfileId,
        suite: selectedProfile.suite,
        preset: selectedProfile.preset,
      });
    }
    updateDiagnosticsGuidance();
  });

  runtimeFile?.addEventListener('change', () => {
    const file = runtimeFile.files?.[0] || null;
    handleRuntimeConfigFile(file);
  });

  runtimeConfigPreset?.addEventListener('change', () => {
    const presetId = runtimeConfigPreset.value || '';
    if (runtimeFile) {
      runtimeFile.value = '';
    }
    applyRuntimeConfigPreset(presetId);
  });

  runtimeClear?.addEventListener('click', () => {
    state.runtimeOverride = null;
    state.runtimeOverrideBase = null;
    state.runtimeOverrideLabel = null;
    if (runtimeFile) {
      runtimeFile.value = '';
    }
    if (runtimeConfigPreset) {
      runtimeConfigPreset.value = '';
    }
    applySelectedRuntimePreset();
  });

  diagnosticsRun?.addEventListener('click', () => handleDiagnosticsRun('run'));
  diagnosticsVerify?.addEventListener('click', () => handleDiagnosticsRun('verify'));
  diagnosticsExport?.addEventListener('click', exportDiagnosticsReport);
  distillTeacherFile?.addEventListener('change', async () => {
    try {
      const file = distillTeacherFile.files?.[0] || null;
      if (!file || !(distillTeacherJson instanceof HTMLTextAreaElement)) return;
      const text = await readFileAsText(file);
      distillTeacherJson.value = text;
      setDistillStatus(`Loaded teacher report file: ${file.name}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setDistillStatus(`Failed to read teacher report file: ${message}`, true);
    }
  });
  distillWorkloadSelect?.addEventListener('change', () => {
    if (!distillWorkloadSelect.value) {
      setDistillStatus('Workload traceability disabled (None selected).');
      return;
    }
    const workload = findDistillWorkloadById(distillWorkloadSelect.value);
    if (!workload) {
      setDistillStatus('Selected workload pack is unavailable.', true);
      return;
    }
    const sha = workload.sha256 ? ` sha256:${workload.sha256.slice(0, 12)}...` : '';
    setDistillStatus(`Selected workload pack: ${workload.id}.${sha}`);
  });
  distillReplayBtn?.addEventListener('click', () => {
    handleDistillReplay().catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      setDistillStatus(`Distill replay failed: ${message}`, true);
    });
  });
  distillExportBtn?.addEventListener('click', () => {
    exportDistillReplay();
  });

  unloadModelBtn?.addEventListener('click', () => {
    unloadActivePipeline().catch((error) => {
      log.warn('DopplerDemo', `Unload failed: ${error.message}`);
    });
  });

  clearMemoryBtn?.addEventListener('click', () => {
    clearAllMemory().catch((error) => {
      log.warn('DopplerDemo', `Clear memory failed: ${error.message}`);
    });
  });

  runGenerate?.addEventListener('click', () => {
    handleRunGenerate().catch((error) => {
      log.warn('DopplerDemo', `Run generate failed: ${error.message}`);
    });
  });

  runPrompt?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      handleRunGenerate().catch((error) => {
        log.warn('DopplerDemo', `Run generate failed: ${error.message}`);
      });
    }
  });
  runPrompt?.addEventListener('input', () => {
    if (state.uiMode === 'translate') {
      state.activeCompareSmokeSampleId = null;
      renderTranslateCompareSmokePanel();
      syncDeepLinkFromUI();
    }
  });

  runPromptShuffle?.addEventListener('click', () => {
    applyStarterPrompt(runPrompt, getRunStarterPromptPool(), { force: true });
    if (state.uiMode === 'embedding') {
      refreshEmbeddingDemoDocuments({ force: true });
    }
    syncDeepLinkFromUI();
    runPrompt?.focus();
    runPrompt?.select();
  });

  runStop?.addEventListener('click', () => {
    stopRunGeneration();
  });

  runClear?.addEventListener('click', () => {
    handleRunClear();
  });
  pulseReset?.addEventListener('click', () => {
    handleInferencePulseReset();
  });

  temperatureInput?.addEventListener('input', updateRunAutoLabels);
  topPInput?.addEventListener('input', updateRunAutoLabels);
  topKInput?.addEventListener('input', updateRunAutoLabels);
  maxTokensInput?.addEventListener('input', updateRunAutoLabels);
  diffusionPrompt?.addEventListener('input', updateDiffusionCharCounters);
  diffusionNegative?.addEventListener('input', updateDiffusionCharCounters);

  diffusionPromptShuffle?.addEventListener('click', () => {
    applyStarterPrompt(diffusionPrompt, DIFFUSION_STARTER_PROMPTS, { force: true });
    applyStarterPrompt(diffusionNegative, DIFFUSION_NEGATIVE_STARTER_PROMPTS, { force: true });
    updateDiffusionCharCounters();
    diffusionPrompt?.focus();
    diffusionPrompt?.select();
  });

  diffusionClear?.addEventListener('click', () => {
    if (diffusionPrompt) {
      diffusionPrompt.value = '';
      setStarterExampleInput(diffusionPrompt, false);
    }
    if (diffusionNegative) {
      diffusionNegative.value = '';
      setStarterExampleInput(diffusionNegative, false);
    }
    if (diffusionSteps) diffusionSteps.value = '20';
    if (diffusionGuidance) diffusionGuidance.value = '7.5';
    if (diffusionSeed) diffusionSeed.value = '';
    if (diffusionWidth) diffusionWidth.value = '256';
    if (diffusionHeight) diffusionHeight.value = '256';
    updateDiffusionCharCounters();
    handleDiffusionClear();
  });

  diffusionRun?.addEventListener('click', () => {
    handleDiffusionRun().catch((error) => {
      log.error('DopplerDemo', `Diffusion run failed: ${error.message}`);
      updateDiffusionStatus(`Error: ${error.message}`);
    });
  });

  energyDemoSelect?.addEventListener('change', () => {
    const demoId = energyDemoSelect.value || DEFAULT_ENERGY_DEMO_ID;
    const demo = getEnergyDemoById(demoId);
    if (!demo) return;
    state.energyDemoId = demo.id;
    setText($('energy-demo-description'), demo.description || '');
    setEnergyMetricLabels(demo.problem || 'quintel');
    toggleEnergyProblemControls(demo.problem || 'quintel');
    applyEnergyDemoDefaults(demo);
  });

  energyClear?.addEventListener('click', () => {
    const demoId = state.energyDemoId || DEFAULT_ENERGY_DEMO_ID;
    const demo = getEnergyDemoById(demoId) || getEnergyDemoById(DEFAULT_ENERGY_DEMO_ID);
    if (demo) {
      applyEnergyDemoDefaults(demo);
    }
    handleEnergyClear();
  });

  energyRun?.addEventListener('click', () => {
    handleEnergyRun().catch((error) => {
      log.error('DopplerDemo', `Energy run failed: ${error.message}`);
      updateEnergyStatus(`Error: ${error.message}`);
    });
  });

  energyExport?.addEventListener('click', () => {
    exportEnergyRun();
  });

  populateDistillWorkloadSelect();
  setDistillOutput(state.distillLastReplay);
  if (!state.distillWorkloadsLoading && !state.distillWorkloadsError) {
    setDistillStatus('Ready.');
  }
  updateRunAutoLabels();
  updateDiffusionCharCounters();
}

function setAppBootstrapMessage(message) {
  const overlayText = $('app-bootstrap-message');
  if (!overlayText) return;
  setText(overlayText, typeof message === 'string' && message.trim() ? message.trim() : 'Loading...');
}

function setAppBootstrapVisible(visible, message = null) {
  const overlay = $('app-bootstrap-overlay');
  const body = document.body;
  if (!overlay) return;
  if (typeof message === 'string' && message.trim()) {
    setAppBootstrapMessage(message);
  }
  if (body) {
    if (visible) {
      body.classList.add('app-booting');
    } else {
      body.classList.remove('app-booting');
    }
  }
  setHidden(overlay, !visible);
}

function syncMobileAdvanced() {
  const isDesktop = window.matchMedia('(min-width: 778px)').matches;
  document.querySelectorAll('details.mobile-advanced').forEach((el) => {
    if (isDesktop) {
      el.setAttribute('open', '');
    } else {
      el.removeAttribute('open');
    }
  });
}

async function init() {
  state.appInitializing = true;
  ensureTranslateCompareRuntimeState();
  hydrateTranslateCompareHistory();
  setStatusIndicator('Initializing...', 'info');
  setAppBootstrapMessage('Loading...');
  console.log('[Bootstrap] Loading... evaluating demo module graph and preparing bootstrap overlay.');
  setAppBootstrapVisible(true);
  try {
    ensurePrimaryModeControlStack();
    syncMobileAdvanced();
    window.matchMedia('(min-width: 778px)').addEventListener('change', syncMobileAdvanced);
    const deepLinkState = readDeepLinkStateFromLocation();
    state.surface = normalizeSurface(deepLinkState.surface, state.surface || 'demo');
    syncSurfaceUI(state.surface);
    if (deepLinkState.mode) {
      state.uiMode = resolveModeForSurface(deepLinkState.mode, state.surface);
    }
    if (deepLinkState.task) {
      const deepLinkTask = deepLinkState.mode
        ? getTaskForMode(deepLinkState.mode, deepLinkState.task)
        : deepLinkState.task;
      state.uiTask = resolveTaskForSurface(deepLinkTask, state.surface, deepLinkState.mode || state.uiMode);
    }
    if (!deepLinkState.mode && deepLinkState.task) {
      state.uiMode = resolveModeForTask(state.uiTask, state.surface, state.uiMode);
    }
    state.uiTask = resolveTaskForSurface(getTaskForMode(state.uiMode, state.uiTask || 'run'), state.surface, state.uiMode);
    applyDeepLinkStateToUI(deepLinkState);
    prefillDemoTextInputs();
    updateDiffusionCharCounters();
    configureDownloadCallbacks({
      onModelRegistered: registerDownloadedModel,
      onModelsUpdated: refreshModelList,
      onProgress: handleDownloadProgressEvent,
      onStateChange: handleDownloadStateChangeEvent,
    });
    populateModelPresets();
    populateRuntimePresetSelects();
    populateEnergyDemoSelect();
    setStatusIndicator('Initializing...', 'info');
    console.log('[Bootstrap] Initializing... running startup tasks: quick model catalog fetch, WebGPU capability init, and download-state refresh.');

    const startupTasks = Promise.all([
      loadTranslateCompareProfiles(),
      loadTranslateCompareEvidence(),
      loadQuickModelCatalog(),
      loadDistillWorkloadRegistry(),
      refreshGpuInfo(),
      refreshDownloads(),
    ]);
    await startupTasks;
    state.compareDeviceLabel = await resolveWebGpuDeviceLabel('WebGPU');

    await setUiMode(state.uiMode, { task: state.uiTask });
    bindUI();
    applyDeepLinkStateToUI(deepLinkState);
    await applyTranslateComparePreset(state.comparePresetId || 'proof', { preserveExisting: true });
    syncTranslateCompareUI();
    updateMemoryControls();
    startTelemetryLoop();
    setRunLoading(false);
    setRunGenerating(false);
    updateRunStatus('Idle');
    updateDiffusionStatus('Idle');
    updateEnergyStatus('Idle');
  } finally {
    state.appInitializing = false;
    setAppBootstrapVisible(false);
    updateStatusIndicator();
  }
}

init().catch((error) => {
  log.error('DopplerDemo', `Demo init failed: ${error.message}`);
});
