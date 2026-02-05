import { VLIW_SPEC_PRESETS } from './energy/specs.js';

export const VLIW_DATASETS = {
  'vliw-simd-frozen': {
    id: 'vliw-simd-frozen',
    label: 'VLIW SIMD schedule (frozen workload)',
    spec: VLIW_SPEC_PRESETS.frozen_ub_energy_1385,
    mode: 'parity',
    capsMode: 'slot_limits',
    source: 'generator/ub_energy_bundle_1385.py',
  },
  'vliw-simd-real': {
    id: 'vliw-simd-real',
    label: 'VLIW SIMD schedule (real spec)',
    path: 'data/vliw-simd-real.json',
  },
  'vliw-simd': {
    id: 'vliw-simd',
    label: 'VLIW SIMD schedule (full kernel)',
    path: 'data/vliw-simd.json',
  },
};

export const ENERGY_DEMOS = [
  {
    id: 'quintel-cross',
    problem: 'quintel',
    label: 'Quintel: Cross (mirror + count)',
    description: 'Mirror X/Y with a count target. Produces a symmetric cross pattern.',
    defaults: {
      size: 5,
      displayThreshold: 0.2,
      countTarget: 12,
      rules: {
        mirrorX: true,
        mirrorY: true,
        diagonal: false,
        count: true,
      },
      weights: {
        symmetry: 1.0,
        count: 0.2,
        binarize: 0.01,
      },
      init: {
        mode: 'uniform',
        seed: 1337,
        scale: 0.6,
      },
      loop: {
        steps: 20000,
        stepSize: 0.02,
        gradientScale: 0.5,
        convergence: 0.00001,
      },
    },
  },
  {
    id: 'vliw-simd',
    problem: 'vliw',
    label: 'VLIW SIMD: Spec → DAG → Schedule',
    description: 'Layered energy: spec constraints → DAG build → schedule cycles under slot caps.',
    defaults: {
      displayThreshold: 0.5,
      vliw: {
        dataset: 'vliw-simd-frozen',
        bundleLimit: 0,
        mode: 'parity',
        scoreMode: 'bundle',
        restarts: 2,
        temperatureStart: 3.0,
        temperatureDecay: 0.99,
        mutationCount: 8,
        policy: 'weights',
        schedulerPolicies: ['mix'],
        specSearch: {
          enabled: true,
          restarts: 2,
          steps: 32,
          temperatureStart: 2.5,
          temperatureDecay: 0.95,
          mutationCount: 2,
          seed: 1337,
          penaltyGate: 2,
          cycleLambda: 1.0,
          lbPenalty: 100,
          targetCycles: 0,
          scoreMode: 'bundle',
          innerSteps: 32,
          constraints: {
            mode: 'parity',
            fallbackCycles: 10000,
          },
        },
      },
      init: {
        mode: 'baseline',
        seed: 1337,
        scale: 0.35,
      },
      loop: {
        steps: 32,
        stepSize: 0.15,
        gradientScale: 1.0,
        convergence: 0,
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
  vliw: {
    symmetry: 'Cycles',
    count: 'Utilization',
    binarize: 'Violations',
  },
};

export const RUNTIME_PRESET_REGISTRY = [
  { id: '', label: 'none', base: false, override: true },
  { id: 'modes/debug', label: 'modes/debug', base: true, override: false },
  { id: 'modes/bench', label: 'modes/bench', base: true, override: false },
  { id: 'modes/production', label: 'modes/production', base: false, override: true },
  { id: 'modes/low-memory', label: 'modes/low-memory', base: false, override: true },
  { id: 'modes/simulation', label: 'modes/simulation', base: false, override: true },
  { id: 'modes/trace-layers', label: 'modes/trace-layers', base: false, override: true },
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
  { id: 'experiments/gemma3-verify', label: 'experiments/gemma3-verify', base: false, override: true },
  { id: 'experiments/gemma3-debug-q4k', label: 'experiments/gemma3-debug-q4k', base: false, override: true },
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
  'debug',
  'bench',
  'diffusion',
  'energy',
  'kernels',
];

export const BENCH_INTENTS = new Set(['investigate', 'calibrate']);
export const DEFAULT_RUNTIME_PRESET = 'modes/debug';
export const DIAGNOSTICS_DEFAULTS = {
  run: { suite: 'inference' },
  diffusion: { suite: 'diffusion' },
  energy: { suite: 'energy' },
  diagnostics: { suite: 'inference' },
  kernels: { suite: 'kernels' },
};
