import * as backwardKernels from '../gpu/kernels/backward/index.js';
import { runResidualAdd } from '../gpu/kernels/residual.js';
import { runScale } from '../gpu/kernels/index.js';
import { getDevice } from '../gpu/device.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../memory/buffer-pool.js';
import { createTensor } from '../gpu/tensor.js';
import { attentionBackwardCpu } from './attention-backward.js';
import { f16ToF32Array, f32ToF16Array } from '../inference/kv-cache/types.js';

export const OpType = {
  EMBED: 'embed',
  MATMUL: 'matmul',
  RMSNORM: 'rmsnorm',
  RESIDUAL_ADD: 'residual_add',
  ROW_SLICE: 'row_slice',
  LAYERNORM: 'layernorm',
  ATTENTION: 'attention',
  SOFTMAX: 'softmax',
  ROPE: 'rope',
  SILU: 'silu',
  SILU_ROWSPLIT: 'silu_rowsplit',
  GELU: 'gelu',
  SCALE: 'scale',
  CROSS_ENTROPY: 'cross_entropy',
  BIAS_ADD: 'bias_add',
  UPSAMPLE2D: 'upsample2d',
  PIXEL_SHUFFLE: 'pixel_shuffle',
  GROUPNORM: 'groupnorm',
  CONV2D: 'conv2d',
};

const MAX_RESIDUAL_ELEMENTS_PER_DISPATCH = 65535 * 256;

export class AutogradTape {
  constructor(registry) {
    this.registry = registry;
    this.records = [];
  }

  watch(tensor) {
    return tensor;
  }

  async record(op, fn, inputs, options = {}) {
    const output = await fn(...inputs);
    this.records.push({ op, inputs, output, options });
    return output;
  }

  async backward(gradOutput) {
    const grads = new Map();
    const seeds = this.normalizeBackwardSeeds(gradOutput);
    for (const seed of seeds) {
      await this.accumulateGrad(grads, seed.tensor, seed.grad);
    }

    for (let i = this.records.length - 1; i >= 0; i -= 1) {
      const record = this.records[i];
      const entry = this.registry.ops[record.op];
      if (!entry) {
        continue;
      }

      const gradOut = grads.get(record.output);
      if (!gradOut) {
        continue;
      }

      const gradsOut = await this.runBackward(entry.backward, record, gradOut);
      for (const { input, grad } of gradsOut) {
        if (input && grad) {
          await this.accumulateGrad(grads, input, grad);
        }
      }
    }

    return grads;
  }

  isTensorLike(value) {
    return !!value
      && typeof value === 'object'
      && Array.isArray(value.shape)
      && value.buffer != null;
  }

  normalizeBackwardSeeds(gradOutput) {
    const seeds = [];
    const pushSeed = (tensor, grad) => {
      if (!this.isTensorLike(tensor) || !this.isTensorLike(grad)) {
        return;
      }
      seeds.push({ tensor, grad });
    };
    if (gradOutput instanceof Map) {
      for (const [tensor, grad] of gradOutput.entries()) {
        pushSeed(tensor, grad);
      }
      return seeds;
    }
    if (Array.isArray(gradOutput)) {
      for (const seed of gradOutput) {
        if (!seed || typeof seed !== 'object') continue;
        pushSeed(seed.tensor || seed.output || null, seed.grad || null);
      }
      return seeds;
    }
    if (gradOutput && typeof gradOutput === 'object' && Array.isArray(gradOutput.seeds)) {
      for (const seed of gradOutput.seeds) {
        if (!seed || typeof seed !== 'object') continue;
        pushSeed(seed.tensor || seed.output || null, seed.grad || null);
      }
      return seeds;
    }
    const last = this.records[this.records.length - 1];
    if (last && this.isTensorLike(gradOutput)) {
      pushSeed(last.output, gradOutput);
    }
    return seeds;
  }

  async runBackward(backwardName, record, gradOut) {
    const entry = this.registry.ops[record.op];

    if (backwardName === 'residual_add_backward') {
      const gradA = gradOut;
      const gradB = await runScale(gradOut, 1, { inplace: false });
      return [
        { input: record.inputs[0], grad: gradA },
        { input: record.inputs[1], grad: gradB },
      ];
    }

    if (backwardName === 'row_slice_backward') {
      const rows = Math.max(1, Math.floor(Number(record.options?.rows) || 0));
      const cols = Math.max(1, Math.floor(Number(record.options?.cols) || 0));
      const rowIndex = Math.max(0, Math.min(rows - 1, Math.floor(Number(record.options?.rowIndex) || 0)));
      const gradInput = await this.expandRowSliceGrad(gradOut, rows, cols, rowIndex);
      return [{ input: record.inputs[0], grad: gradInput }];
    }

    if (backwardName === 'silu_rowsplit_backward') {
      const numTokens = Math.max(1, Math.floor(Number(record.options?.numTokens) || 0));
      const dim = Math.max(1, Math.floor(Number(record.options?.dim) || 0));
      const activation = String(record.options?.activation || 'silu').toLowerCase();
      const swigluLimit = Number(record.options?.swigluLimit);
      const gradInput = await this.runSiluRowsplitBackwardCpu(
        record.inputs[0],
        gradOut,
        { numTokens, dim, activation, swigluLimit: Number.isFinite(swigluLimit) ? swigluLimit : 0 }
      );
      return [{ input: record.inputs[0], grad: gradInput }];
    }

    if (backwardName === 'embed_backward') {
      const [indices, embeddings] = record.inputs;
      const gradWeight = await backwardKernels.runEmbedBackward(indices, gradOut, {
        numTokens: Math.max(1, Math.floor(Number(record.options?.numTokens) || 0)),
        hiddenSize: Math.max(1, Math.floor(Number(record.options?.hiddenSize) || 0)),
        vocabSize: Math.max(1, Math.floor(Number(record.options?.vocabSize) || 0)),
        transpose: record.options?.transpose === true,
        indexOffset: Math.max(0, Math.floor(Number(record.options?.indexOffset) || 0)),
      });
      return [{ input: embeddings, grad: gradWeight }];
    }

    // Special case for attention which has CPU fallback and complex internal logic
    if (backwardName === 'attention_backward') {
      const [q, k, v, softmax] = record.inputs;
      const { seqLen, numHeads, headDim, scale } = record.options;
      const recomputeForward = record.options.recomputeForward === true || !softmax;
      const { gradQ, gradK, gradV } = recomputeForward
        ? await attentionBackwardCpu(
          q, k, v, null, gradOut,
          { seqLen, numHeads, headDim, scale, causal: record.options.causal }
        )
        : await backwardKernels.runAttentionBackward(
          q, k, v, softmax, gradOut,
          { seqLen, numHeads, headDim, scale, causal: record.options.causal }
        ).catch(() => attentionBackwardCpu(
          q, k, v, softmax, gradOut,
          { seqLen, numHeads, headDim, scale, causal: record.options.causal }
        ));
      return [
        { input: q, grad: gradQ },
        { input: k, grad: gradK },
        { input: v, grad: gradV },
      ];
    }

    const kernelFnName = `run${backwardName.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join('')}`;
    const kernelFn = backwardKernels[kernelFnName];

    if (!kernelFn) {
      throw new Error(`Backward kernel function "${kernelFnName}" not found for "${backwardName}"`);
    }

    // Prepare options from registry metadata
    const options = { ...record.options };
    if (entry.params) {
      for (const param of entry.params) {
        if (options[param] === undefined && record.options[param] !== undefined) {
          options[param] = record.options[param];
        }
      }
    }

    // Standard kernels: (input, weight, gradOut, options) or (input, gradOut, options)
    // Map inputs based on registry "grads" metadata
    const result = await kernelFn(...record.inputs, gradOut, options);

    // Map result back to inputs
    const outputs = [];
    if (result && typeof result === 'object' && !result.buffer) {
      // Multiple gradients returned as object (e.g. { gradInput, gradWeight })
      for (const [key, grad] of Object.entries(result)) {
        // Map 'gradInput' to entry.grads[0], 'gradWeight' to entry.grads[1], etc.
        // This is a bit heuristic, but follows our naming convention.
        if (key === 'gradInput') outputs.push({ input: record.inputs[0], grad });
        else if (key === 'gradWeight') outputs.push({ input: record.inputs[1], grad });
        else if (key === 'gradBias') outputs.push({ input: record.inputs[2], grad });
        else if (key === 'gradGamma') outputs.push({ input: record.inputs[1], grad });
      }
    } else {
      // Single gradient returned as Tensor
      outputs.push({ input: record.inputs[0], grad: result });
    }

    return outputs;
  }

  async readTensorAsF32(tensor) {
    const raw = await readBuffer(tensor.buffer);
    if (tensor.dtype === 'f16') {
      return f16ToF32Array(new Uint16Array(raw));
    }
    return new Float32Array(raw);
  }

  async expandRowSliceGrad(gradOut, rows, cols, rowIndex) {
    const gradRow = await this.readTensorAsF32(gradOut);
    const expanded = new Float32Array(rows * cols);
    const rowOffset = rowIndex * cols;
    const copyCount = Math.min(cols, gradRow.length);
    expanded.set(gradRow.subarray(0, copyCount), rowOffset);
    const dtype = gradOut.dtype === 'f16' ? 'f16' : 'f32';
    const payload = dtype === 'f16' ? f32ToF16Array(expanded) : expanded;
    const outBuffer = acquireBuffer(payload.byteLength, undefined, 'row_slice_backward_output');
    uploadData(outBuffer, payload);
    return createTensor(outBuffer, dtype, [rows, cols], 'row_slice_backward_output');
  }

  resolveSiluRowsplitGate(gateValue, activation) {
    if (activation === 'gelu') {
      const sqrtTwoOverPi = 0.7978845608;
      const coeff = 0.044715;
      const gateCubed = gateValue * gateValue * gateValue;
      const inner = Math.max(-15, Math.min(15, sqrtTwoOverPi * (gateValue + (coeff * gateCubed))));
      const tanhInner = Math.tanh(inner);
      const activated = 0.5 * gateValue * (1 + tanhInner);
      const sechSq = 1 - (tanhInner * tanhInner);
      const innerDeriv = sqrtTwoOverPi * (1 + (3 * coeff * gateValue * gateValue));
      const derivative = 0.5 * (1 + tanhInner) + (0.5 * gateValue * sechSq * innerDeriv);
      return { activated, derivative };
    }
    const clamped = Math.max(-15, Math.min(15, gateValue));
    const sigmoid = 1 / (1 + Math.exp(-clamped));
    const activated = gateValue * sigmoid;
    const derivative = sigmoid * (1 + (gateValue * (1 - sigmoid)));
    return { activated, derivative };
  }

  async runSiluRowsplitBackwardCpu(input, gradOut, options) {
    const numTokens = Math.max(1, Math.floor(Number(options?.numTokens) || 0));
    const dim = Math.max(1, Math.floor(Number(options?.dim) || 0));
    const activation = options?.activation === 'gelu' ? 'gelu' : 'silu';
    const swigluLimit = Number.isFinite(options?.swigluLimit) ? options.swigluLimit : 0;
    const inputData = await this.readTensorAsF32(input);
    const gradData = await this.readTensorAsF32(gradOut);
    const totalInput = numTokens * dim * 2;
    const output = new Float32Array(totalInput);

    for (let row = 0; row < numTokens; row += 1) {
      const rowBase = row * dim * 2;
      const gradBase = row * dim;
      for (let col = 0; col < dim; col += 1) {
        const gateIndex = rowBase + col;
        const upIndex = rowBase + dim + col;
        if (gateIndex >= inputData.length || upIndex >= inputData.length) continue;
        const gradIndex = gradBase + col;
        if (gradIndex >= gradData.length) continue;

        const gateValue = inputData[gateIndex];
        const upValue = inputData[upIndex];
        const gradValue = gradData[gradIndex];
        const gate = this.resolveSiluRowsplitGate(gateValue, activation);
        const product = gate.activated * upValue;
        if (swigluLimit > 0 && Math.abs(product) > swigluLimit) {
          continue;
        }
        output[gateIndex] = gradValue * gate.derivative * upValue;
        output[upIndex] = gradValue * gate.activated;
      }
    }

    const dtype = gradOut.dtype === 'f16' ? 'f16' : 'f32';
    const payload = dtype === 'f16' ? f32ToF16Array(output) : output;
    const outBuffer = acquireBuffer(payload.byteLength, undefined, 'silu_rowsplit_backward_output');
    uploadData(outBuffer, payload);
    return createTensor(outBuffer, dtype, [numTokens, dim * 2], 'silu_rowsplit_backward_output');
  }

  async accumulateLargeGradF32(existing, grad, size, shape) {
    const device = getDevice();
    if (!device) {
      throw new Error('Autograd requires active GPU device for large gradient accumulation.');
    }
    const bytesPerElement = 4;
    const outputBuffer = acquireBuffer(size * bytesPerElement, undefined, 'grad_accum_large_output');
    try {
      for (let offset = 0; offset < size; offset += MAX_RESIDUAL_ELEMENTS_PER_DISPATCH) {
        const chunkElements = Math.min(MAX_RESIDUAL_ELEMENTS_PER_DISPATCH, size - offset);
        const chunkBytes = chunkElements * bytesPerElement;
        const chunkOffsetBytes = offset * bytesPerElement;

        let aChunkBuffer = null;
        let bChunkBuffer = null;
        let summedChunkBuffer = null;
        try {
          aChunkBuffer = acquireBuffer(chunkBytes, undefined, 'grad_accum_large_a_chunk');
          bChunkBuffer = acquireBuffer(chunkBytes, undefined, 'grad_accum_large_b_chunk');
          const copyIn = device.createCommandEncoder();
          copyIn.copyBufferToBuffer(existing.buffer, chunkOffsetBytes, aChunkBuffer, 0, chunkBytes);
          copyIn.copyBufferToBuffer(grad.buffer, chunkOffsetBytes, bChunkBuffer, 0, chunkBytes);
          device.queue.submit([copyIn.finish()]);

          const aChunk = createTensor(aChunkBuffer, 'f32', [chunkElements], 'grad_accum_large_a_tensor');
          const bChunk = createTensor(bChunkBuffer, 'f32', [chunkElements], 'grad_accum_large_b_tensor');
          const summedChunk = await runResidualAdd(aChunk, bChunk, chunkElements);
          summedChunkBuffer = summedChunk?.buffer ?? null;

          const copyOut = device.createCommandEncoder();
          copyOut.copyBufferToBuffer(summedChunk.buffer, 0, outputBuffer, chunkOffsetBytes, chunkBytes);
          device.queue.submit([copyOut.finish()]);
        } finally {
          if (aChunkBuffer) {
            releaseBuffer(aChunkBuffer);
          }
          if (bChunkBuffer) {
            releaseBuffer(bChunkBuffer);
          }
          if (summedChunkBuffer && summedChunkBuffer !== outputBuffer) {
            releaseBuffer(summedChunkBuffer);
          }
        }
      }

      return createTensor(outputBuffer, 'f32', [...shape], 'grad_accum_large_output');
    } catch (error) {
      releaseBuffer(outputBuffer);
      throw error;
    }
  }


  async accumulateGrad(grads, input, grad) {
    const existing = grads.get(input);
    if (!existing) {
      grads.set(input, grad);
      return;
    }
    const size = grad.shape.reduce((acc, value) => acc * value, 1);
    const useChunkedResidual = (
      size > MAX_RESIDUAL_ELEMENTS_PER_DISPATCH
      && existing.dtype === 'f32'
      && grad.dtype === 'f32'
      && Array.isArray(grad.shape)
    );
    const summed = useChunkedResidual
      ? await this.accumulateLargeGradF32(existing, grad, size, grad.shape)
      : await runResidualAdd(existing, grad, size);
    grads.set(input, summed);
    if (existing.buffer !== summed.buffer) {
      releaseBuffer(existing.buffer);
    }
    if (grad.buffer !== summed.buffer && grad.buffer !== existing.buffer) {
      releaseBuffer(grad.buffer);
    }
  }

  reset() {
    this.records = [];
  }
}
