export const ENERGY_DEMOS = [
  {
    id: 'quintel-cross',
    problem: 'quintel',
    label: 'Quintel: Cross (mirror + count)',
    description: 'Mirror X/Y with a count target. Produces a symmetric cross pattern.',
    defaults: {
      quintel: {
        size: 5,
        threshold: 0.2,
        countTarget: 12,
        mirrorX: true,
        mirrorY: true,
        diagonal: false,
        countRule: true,
        symmetryWeight: 1.0,
        countWeight: 0.2,
        binarizeWeight: 0.01,
        initMode: 'uniform',
        initSeed: 1337,
        initScale: 0.6,
      },
      loop: {
        steps: 20000,
        stepSize: 0.02,
        gradientScale: 0.5,
        convergence: 0.00001,
      },
    },
  },
];

export const DEFAULT_ENERGY_DEMO_ID = ENERGY_DEMOS[0]?.id || 'quintel-cross';

export const ENERGY_METRIC_LABELS = {
  quintel: {
    symmetry: 'Symmetry',
    count: 'Count',
    binarize: 'Binarize',
  },
};

export const RUNTIME_PRESET_REGISTRY = [
  { id: '', label: 'none', base: false, override: true },
  { id: 'profiles/default', label: 'profiles/default', base: true, override: false },
  { id: 'profiles/verbose-trace', label: 'profiles/verbose-trace', base: true, override: false },
  { id: 'profiles/throughput', label: 'profiles/throughput', base: true, override: false },
  { id: 'profiles/vector-stability', label: 'profiles/vector-stability', base: true, override: false },
  { id: 'profiles/vector-throughput', label: 'profiles/vector-throughput', base: true, override: false },
  { id: 'profiles/production', label: 'profiles/production', base: false, override: true },
  { id: 'profiles/low-memory', label: 'profiles/low-memory', base: false, override: true },
  { id: 'profiles/simulation', label: 'profiles/simulation', base: false, override: true },
  { id: 'profiles/trace-layers', label: 'profiles/trace-layers', base: false, override: true },
  { id: 'kernels/safe-q4k', label: 'kernels/safe-q4k', base: false, override: true },
  { id: 'kernels/fused-q4k', label: 'kernels/fused-q4k', base: false, override: true },
  { id: 'kernels/dequant-f16-q4k', label: 'kernels/dequant-f16-q4k', base: false, override: true },
  { id: 'kernels/dequant-f32-q4k', label: 'kernels/dequant-f32-q4k', base: false, override: true },
  { id: 'compute/f16-activations', label: 'compute/f16-activations', base: false, override: true },
  { id: 'compute/f16-batched', label: 'compute/f16-batched', base: false, override: true },
  { id: 'platform/metal-apple-q4k', label: 'platform/metal-apple-q4k', base: false, override: true },
  { id: 'model/gemma3-layer-probe', label: 'model/gemma3-layer-probe', base: false, override: true },
  { id: 'model/gemma2-pipeline', label: 'model/gemma2-pipeline', base: false, override: true },
  { id: 'model/gemma2-pipeline-debug', label: 'model/gemma2-pipeline-debug', base: false, override: true },
  { id: 'experiments/verify/gemma3-verify', label: 'experiments/verify/gemma3-verify', base: false, override: true },
  { id: 'experiments/debug/gemma3-debug-q4k', label: 'experiments/debug/gemma3-debug-q4k', base: false, override: true },
];

export const DIAGNOSTICS_SUITE_INFO = {
  inference: {
    description: 'Runs a short generation with the Active model.',
    requiresModel: true,
    requiresBenchIntent: false,
  },
  bench: {
    description: 'Benchmarks tokens/sec for the Active model.',
    requiresModel: true,
    requiresBenchIntent: true,
  },
  debug: {
    description: 'Runs inference with debug tracing enabled by runtime config.',
    requiresModel: true,
    requiresBenchIntent: false,
  },
  training: {
    description: 'Runs training/distillation validation checks on the Active model.',
    requiresModel: true,
    requiresBenchIntent: false,
  },
  diffusion: {
    description: 'Benchmarks diffusion generation using the Active model.',
    requiresModel: true,
    requiresBenchIntent: true,
  },
  energy: {
    description: 'Runs an energy loop with the Active model and reports convergence stats.',
    requiresModel: true,
    requiresBenchIntent: false,
  },
  kernels: {
    description: 'Validates GPU kernels only (no model required).',
    requiresModel: false,
    requiresBenchIntent: false,
  },
};

export const DIAGNOSTICS_SUITE_ORDER = [
  'inference',
  'training',
  'diffusion',
  'energy',
  'kernels',
];

export const BENCH_INTENTS = new Set(['investigate', 'calibrate']);
export const DEFAULT_RUNTIME_PRESET = 'profiles/verbose-trace';
export const DIAGNOSTICS_DEFAULTS = {
  run: { suite: 'inference', preset: 'profiles/default' },
  translate: { suite: 'inference', preset: 'profiles/default' },
  embedding: { suite: 'embedding', preset: 'profiles/vector-stability' },
  diffusion: { suite: 'diffusion', preset: 'profiles/default' },
  energy: { suite: 'energy', preset: 'profiles/default' },
  diagnostics: { suite: 'inference', preset: 'profiles/default' },
};
