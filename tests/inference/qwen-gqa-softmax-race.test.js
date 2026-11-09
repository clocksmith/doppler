import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice } from '../../src/gpu/device.js';
import { checkQwenGqaSoftmaxRace } from '../kernels/qwen-gqa-softmax-race.js';

const gpu = await probeNodeGPU({ installFileFetchShim: true });
if (!gpu.ready) {
  console.log(`qwen-gqa-softmax-race: skipped (${gpu.reason})`);
} else {
  try { console.log(JSON.stringify(await checkQwenGqaSoftmaxRace())); }
  finally { destroyDevice(); }
}
