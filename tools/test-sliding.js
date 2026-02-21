import { getDevice } from '../src/gpu/device.js';
import { resolveAttentionPlanForTest } from '../src/gpu/kernels/attention.js';

async function testSlidingWindow() {
    const device = await getDevice();
    console.log("Got Device");

    const seqLen = 1;
    const kvLen = 10000;
    const headDim = 256;
    const numHeads = 4;
    const kvDtype = 'f16';
    const qDtype = 'f32';
    const sharedLimit = 32768;
    const caps = device.limits;
    const layerIdx = 0;

    const plan = resolveAttentionPlanForTest(seqLen, kvLen, headDim, numHeads, kvDtype, qDtype, sharedLimit, caps, layerIdx, false);
    console.log("Kernel Plan Selected:", plan);

    process.exit(0);
}

testSlidingWindow().catch(console.error);
