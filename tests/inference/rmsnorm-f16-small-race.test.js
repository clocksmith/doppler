import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice } from '../../src/gpu/device.js';
import { checkSmallF16RmsNorm } from '../kernels/rmsnorm-f16-small-race.js';

const gpu = await probeNodeGPU({ installFileFetchShim: true });
if (!gpu.ready) {
  console.log(`rmsnorm-f16-small-race: skipped (${gpu.reason})`);
} else {
  try {
    console.log(JSON.stringify(await checkSmallF16RmsNorm()));
  } finally { destroyDevice(); }
}
