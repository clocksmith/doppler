


import { test, expect } from './setup.js';

test.describe('RMSNormOffset', () => {
    test('should apply (1+w) scaling when offset is enabled', async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async () => {
            const batchSize = 1;
            const hiddenSize = 4;

            // Input is all ones. RMS(ones) = 1.0.
            const input = new Float32Array([1, 1, 1, 1]);

            // Weight is all zeros.
            const weight = new Float32Array([0, 0, 0, 0]);

            const gpu = await window.testHarness.getGPU();

            // Base case: offset=false. Result should be x * w = 1.0 * 0.0 = 0.0.
            const actualBase = await window.testHarness.runRMSNorm(
                gpu.device, input, weight, batchSize, hiddenSize, 1e-6,
                { rmsNormWeightOffset: false }
            );

            // Offset case: offset=true. Result should be x * (1+w) = 1.0 * (1+0) = 1.0.
            const actualOffset = await window.testHarness.runRMSNorm(
                gpu.device, input, weight, batchSize, hiddenSize, 1e-6,
                { rmsNormWeightOffset: true }
            );

            return { actualBase, actualOffset };
        });

        // Verify base case (0.0)
        expect(result.actualBase[0]).toBeCloseTo(0.0, 4);

        // Verify offset case (1.0)
        expect(result.actualOffset[0]).toBeCloseTo(1.0, 4);
    });
});
