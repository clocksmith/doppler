/**
 * BF16/F16/F32 Cast Kernel Correctness Tests
 *
 * Tests type conversion kernels that are critical for weight loading.
 */

import { test, expect } from './setup.js';

interface CastResult {
  maxError: number;
  hasNaN?: boolean;
  sampleInput?: number[];
  sampleOutput?: number[];
  allZero?: boolean;
}

// Reference: Convert F32 to BF16 (truncate mantissa)
function f32ToBf16(f32: number): number {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, f32, true);
  const bits = view.getUint32(0, true);
  // BF16 is upper 16 bits of F32
  return (bits >> 16) & 0xFFFF;
}

// Reference: Convert BF16 to F32
function bf16ToF32(bf16: number): number {
  const view = new DataView(new ArrayBuffer(4));
  // BF16 is upper 16 bits, lower 16 bits are zero
  view.setUint32(0, bf16 << 16, true);
  return view.getFloat32(0, true);
}

// Reference: Convert F32 to F16
function f32ToF16(f32: number): number {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, f32, true);
  const bits = view.getUint32(0, true);

  const sign = (bits >> 31) & 0x1;
  const exp = (bits >> 23) & 0xFF;
  const mant = bits & 0x7FFFFF;

  let h_exp: number, h_mant: number;

  if (exp === 0) {
    // Zero or denormal
    h_exp = 0;
    h_mant = 0;
  } else if (exp === 0xFF) {
    // Inf or NaN
    h_exp = 0x1F;
    h_mant = mant ? 0x200 : 0; // NaN or Inf
  } else {
    const newExp = exp - 127 + 15;
    if (newExp >= 0x1F) {
      // Overflow to Inf
      h_exp = 0x1F;
      h_mant = 0;
    } else if (newExp <= 0) {
      // Underflow to zero
      h_exp = 0;
      h_mant = 0;
    } else {
      h_exp = newExp;
      h_mant = mant >> 13;
    }
  }

  return (sign << 15) | (h_exp << 10) | h_mant;
}

// Reference: Convert F16 to F32
function f16ToF32(f16: number): number {
  const sign = (f16 >> 15) & 0x1;
  const exp = (f16 >> 10) & 0x1F;
  const mant = f16 & 0x3FF;

  let f32Bits: number;
  if (exp === 0) {
    if (mant === 0) {
      f32Bits = sign << 31;
    } else {
      // Denormal
      let e = -1;
      let m = mant;
      while ((m & 0x400) === 0) {
        m <<= 1;
        e--;
      }
      f32Bits = (sign << 31) | ((e + 127) << 23) | ((m & 0x3FF) << 13);
    }
  } else if (exp === 0x1F) {
    f32Bits = (sign << 31) | 0x7F800000 | (mant << 13);
  } else {
    f32Bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }

  const view = new DataView(new ArrayBuffer(4));
  view.setUint32(0, f32Bits, true);
  return view.getFloat32(0, true);
}

test.describe('Type Casting Kernels', () => {
  test.describe('BF16 to F32 Conversion', () => {
    test('should convert BF16 to F32 correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<CastResult> => {
        const numElements = 256;
        const testValues = new Float32Array(numElements);

        for (let i = 0; i < numElements; i++) {
          testValues[i] = (Math.random() - 0.5) * 4.0;
        }

        // Convert F32 to BF16
        const bf16Data = new Uint16Array(numElements);
        for (let i = 0; i < numElements; i++) {
          const view = new DataView(new ArrayBuffer(4));
          view.setFloat32(0, testValues[i], true);
          const bits = view.getUint32(0, true);
          bf16Data[i] = (bits >> 16) & 0xFFFF;
        }

        // Expected F32 from BF16 (reference)
        const expectedF32 = new Float32Array(numElements);
        for (let i = 0; i < numElements; i++) {
          const view = new DataView(new ArrayBuffer(4));
          view.setUint32(0, bf16Data[i] << 16, true);
          expectedF32[i] = view.getFloat32(0, true);
        }

        const gpu = await window.testHarness.getGPU();
        const device = gpu.device;

        const inputBuffer = device.createBuffer({
          size: bf16Data.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(inputBuffer, 0, bf16Data);

        const { runBF16ToF32 } = await import('../../gpu/kernels/cast.js');
        const outputBuffer = await runBF16ToF32(inputBuffer, numElements);

        const staging = device.createBuffer({
          size: numElements * 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(outputBuffer, 0, staging, 0, numElements * 4);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const actualF32 = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();

        let maxError = 0;
        let hasNaN = false;
        for (let i = 0; i < numElements; i++) {
          if (isNaN(actualF32[i])) hasNaN = true;
          const error = Math.abs(expectedF32[i] - actualF32[i]);
          maxError = Math.max(maxError, error);
        }

        inputBuffer.destroy();
        outputBuffer.destroy();

        return { maxError, hasNaN, allZero: actualF32.every(v => v === 0) };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.allZero).toBe(false);
      expect(result.maxError).toBeLessThan(1e-6);
    });
  });

  test.describe('F32 to F16 Conversion', () => {
    test('should convert F32 to F16 correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<CastResult> => {
        const numElements = 256;
        const testValues = new Float32Array(numElements);

        for (let i = 0; i < numElements; i++) {
          testValues[i] = (Math.random() - 0.5) * 2.0;
        }

        const gpu = await window.testHarness.getGPU();
        const device = gpu.device;

        const inputBuffer = device.createBuffer({
          size: testValues.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(inputBuffer, 0, testValues);

        const { castF32ToF16 } = await import('../../gpu/kernels/cast.js');
        const outputBuffer = await castF32ToF16(inputBuffer, numElements);

        const staging = device.createBuffer({
          size: numElements * 2,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(outputBuffer, 0, staging, 0, numElements * 2);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const actualF16 = new Uint16Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();

        // Convert F16 back to F32 for comparison
        let maxError = 0;
        let hasNaN = false;
        for (let i = 0; i < numElements; i++) {
          const f16 = actualF16[i];
          const sign = (f16 >> 15) & 0x1;
          const exp = (f16 >> 10) & 0x1F;
          const mant = f16 & 0x3FF;

          let val: number;
          if (exp === 0) {
            val = mant === 0 ? 0 : Math.pow(2, -14) * (mant / 1024);
          } else if (exp === 31) {
            val = mant === 0 ? Infinity : NaN;
          } else {
            val = Math.pow(2, exp - 15) * (1 + mant / 1024);
          }
          const actualF32 = sign ? -val : val;

          if (isNaN(actualF32)) hasNaN = true;
          const error = Math.abs(testValues[i] - actualF32);
          maxError = Math.max(maxError, error);
        }

        inputBuffer.destroy();
        outputBuffer.destroy();

        return { maxError, hasNaN, allZero: actualF16.every(v => v === 0) };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.allZero).toBe(false);
      expect(result.maxError).toBeLessThan(0.01); // F16 precision loss
    });
  });

  test.describe('BF16 to F16 Conversion', () => {
    test('should convert BF16 to F16 correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<CastResult> => {
        // Create test data: F32 values that we'll convert to BF16, then to F16
        const numElements = 256;
        const testValues = new Float32Array(numElements);

        // Use a range of values typical for embeddings
        for (let i = 0; i < numElements; i++) {
          testValues[i] = (Math.random() - 0.5) * 2.0; // Range [-1, 1]
        }

        // Convert F32 to BF16 (simulating what SafeTensors stores)
        const bf16Data = new Uint16Array(numElements);
        for (let i = 0; i < numElements; i++) {
          const view = new DataView(new ArrayBuffer(4));
          view.setFloat32(0, testValues[i], true);
          const bits = view.getUint32(0, true);
          bf16Data[i] = (bits >> 16) & 0xFFFF;
        }

        // Expected F16 output (via F32 intermediate)
        const expectedF16 = new Uint16Array(numElements);
        for (let i = 0; i < numElements; i++) {
          // BF16 -> F32 -> F16
          const view = new DataView(new ArrayBuffer(4));
          view.setUint32(0, bf16Data[i] << 16, true);
          const f32Val = view.getFloat32(0, true);

          // F32 -> F16
          const f32Bits = view.getUint32(0, true);
          const sign = (f32Bits >> 31) & 0x1;
          const exp = (f32Bits >> 23) & 0xFF;
          const mant = f32Bits & 0x7FFFFF;

          let h_exp: number, h_mant: number;
          if (exp === 0) {
            h_exp = 0;
            h_mant = 0;
          } else if (exp === 0xFF) {
            h_exp = 0x1F;
            h_mant = mant ? 0x200 : 0;
          } else {
            const newExp = exp - 127 + 15;
            if (newExp >= 0x1F) {
              h_exp = 0x1F;
              h_mant = 0;
            } else if (newExp <= 0) {
              h_exp = 0;
              h_mant = 0;
            } else {
              h_exp = newExp;
              h_mant = mant >> 13;
            }
          }
          expectedF16[i] = (sign << 15) | (h_exp << 10) | h_mant;
        }

        // Run GPU kernel
        const gpu = await window.testHarness.getGPU();
        const device = gpu.device;

        // Create input buffer with BF16 data
        const inputBuffer = device.createBuffer({
          size: bf16Data.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(inputBuffer, 0, bf16Data);

        // Import and run the kernel
        const { runBF16ToF16 } = await import('../../gpu/kernels/cast.js');
        const outputBuffer = await runBF16ToF16(inputBuffer, numElements, 'test_bf16_to_f16');

        // Read back result
        const staging = device.createBuffer({
          size: numElements * 2,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(outputBuffer, 0, staging, 0, numElements * 2);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const actualF16 = new Uint16Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();

        // Compare
        let maxError = 0;
        let hasNaN = false;
        const sampleInput: number[] = [];
        const sampleOutput: number[] = [];

        for (let i = 0; i < numElements; i++) {
          // Convert both to F32 for comparison
          const expectedF32 = f16ToF32Impl(expectedF16[i]);
          const actualF32 = f16ToF32Impl(actualF16[i]);

          if (isNaN(actualF32)) hasNaN = true;

          const error = Math.abs(expectedF32 - actualF32);
          maxError = Math.max(maxError, error);

          if (i < 5) {
            sampleInput.push(testValues[i]);
            sampleOutput.push(actualF32);
          }
        }

        // Helper function inline
        function f16ToF32Impl(f16: number): number {
          const sign = (f16 >> 15) & 0x1;
          const exp = (f16 >> 10) & 0x1F;
          const mant = f16 & 0x3FF;

          let f32Bits: number;
          if (exp === 0) {
            if (mant === 0) {
              f32Bits = sign << 31;
            } else {
              let e = -1;
              let m = mant;
              while ((m & 0x400) === 0) {
                m <<= 1;
                e--;
              }
              f32Bits = (sign << 31) | ((e + 127) << 23) | ((m & 0x3FF) << 13);
            }
          } else if (exp === 0x1F) {
            f32Bits = (sign << 31) | 0x7F800000 | (mant << 13);
          } else {
            f32Bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
          }

          const view = new DataView(new ArrayBuffer(4));
          view.setUint32(0, f32Bits, true);
          return view.getFloat32(0, true);
        }

        inputBuffer.destroy();
        outputBuffer.destroy();

        return {
          maxError,
          hasNaN,
          sampleInput,
          sampleOutput,
          allZero: actualF16.every(v => v === 0),
        };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.allZero).toBe(false);
      expect(result.maxError).toBeLessThan(1e-3); // F16 precision
    });

    test('should preserve sign and magnitude for typical embedding values', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<CastResult> => {
        // Test with values typical for embeddings: small magnitudes around 0.01-0.1
        const numElements = 64;
        const testValues = [
          0.0674, -0.0479, 0.0150, -0.0058, 0.0089, // Values seen in debug
          0.5, -0.5, 0.1, -0.1, 0.01, -0.01,
          1.0, -1.0, 2.0, -2.0,
        ];

        // Pad to numElements
        while (testValues.length < numElements) {
          testValues.push((Math.random() - 0.5) * 0.2);
        }

        // Convert to BF16
        const bf16Data = new Uint16Array(numElements);
        for (let i = 0; i < numElements; i++) {
          const view = new DataView(new ArrayBuffer(4));
          view.setFloat32(0, testValues[i], true);
          const bits = view.getUint32(0, true);
          bf16Data[i] = (bits >> 16) & 0xFFFF;
        }

        // Run GPU kernel
        const gpu = await window.testHarness.getGPU();
        const device = gpu.device;

        const inputBuffer = device.createBuffer({
          size: bf16Data.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(inputBuffer, 0, bf16Data);

        const { runBF16ToF16 } = await import('../../gpu/kernels/cast.js');
        const outputBuffer = await runBF16ToF16(inputBuffer, numElements, 'test_bf16_to_f16');

        // Read back
        const staging = device.createBuffer({
          size: numElements * 2,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(outputBuffer, 0, staging, 0, numElements * 2);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const actualF16 = new Uint16Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();

        // Convert F16 back to F32 for comparison
        const actualF32: number[] = [];
        for (let i = 0; i < numElements; i++) {
          const f16 = actualF16[i];
          const sign = (f16 >> 15) & 0x1;
          const exp = (f16 >> 10) & 0x1F;
          const mant = f16 & 0x3FF;

          let val: number;
          if (exp === 0) {
            val = mant === 0 ? 0 : Math.pow(2, -14) * (mant / 1024);
          } else if (exp === 31) {
            val = mant === 0 ? Infinity : NaN;
          } else {
            val = Math.pow(2, exp - 15) * (1 + mant / 1024);
          }
          actualF32.push(sign ? -val : val);
        }

        // Compare first 15 values
        let maxError = 0;
        for (let i = 0; i < 15; i++) {
          const error = Math.abs(testValues[i] - actualF32[i]);
          maxError = Math.max(maxError, error);
        }

        inputBuffer.destroy();
        outputBuffer.destroy();

        return {
          maxError,
          sampleInput: testValues.slice(0, 10),
          sampleOutput: actualF32.slice(0, 10),
        };
      });

      // Check that output values are close to input
      expect(result.maxError).toBeLessThan(0.01); // Allow for BF16->F16 precision loss
      console.log('Sample input:', result.sampleInput);
      console.log('Sample output:', result.sampleOutput);
    });
  });
});
