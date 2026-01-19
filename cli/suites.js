


export const KERNEL_TESTS = [
  'matmul',
  'matmul-q4k',
  'matmul-q4k-large',
  'attention',
  'rmsnorm',
  'softmax',
  'rope',
  'silu',
  'swiglu',
  'gather',
  'scatter-add',
  'moe-gather',
  'residual',
  'scale',
  'topk',
  'dequant',
  'dequant-q4k',
  'dequant-q4k-f16',
  'matmul-f16w',
  'dequant-q6k',
  'sample',
];

export const TRAINING_TESTS = [
  'loss-forward',
  'softmax-backward',
  'cross-entropy-backward',
  'rmsnorm-backward',
  'matmul-backward',
  'parity-fixture',
  'training-leak-perf',
];

export const KERNEL_BENCHMARKS = [
  'matmul',
  'attention',
  'softmax',
  'rmsnorm',
  'silu',
  'rope',
  'moe',
];

export const QUICK_TESTS = ['matmul', 'rmsnorm', 'softmax', 'gather'];
