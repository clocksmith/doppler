import { runPhysicalBrowserRegression } from '../helpers/physical-browser-regression.js';

await runPhysicalBrowserRegression({
  modulePath: '/tests/kernels/rmsnorm-f16-small-race.js',
  exportName: 'checkSmallF16RmsNorm',
  sourcePaths: ['src/gpu/kernels/rmsnorm_f16.wgsl', 'tests/kernels/rmsnorm-f16-small-race.js'],
  defaultOutput: '.cache/gemma-consistency-20261109/browser.json',
  scope: 'Small F16 RMSNorm operator repeatability; not full-model qualification',
});
