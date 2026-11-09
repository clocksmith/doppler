import { runPhysicalBrowserRegression } from '../helpers/physical-browser-regression.js';

await runPhysicalBrowserRegression({
  modulePath: '/tests/kernels/qwen-gqa-softmax-race.js',
  exportName: 'checkQwenGqaSoftmaxRace',
  sourcePaths: ['src/gpu/kernels/backward/qwen_gqa_softmax_recompute.wgsl',
    'src/gpu/kernels/backward/qwen_gqa_attention_backward.js', 'tests/kernels/qwen-gqa-softmax-race.js',
    'src/experimental/training/attention-backward.js'],
  defaultOutput: '.cache/qwen-softmax-race-20261109/browser.json',
  scope: 'Qwen GQA softmax and backward operator checks; no training-run qualification',
});
