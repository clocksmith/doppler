import * as backwardKernels from '../gpu/kernels/backward/index.js';
import { runResidualAdd } from '../gpu/kernels/residual.js';
import { releaseBuffer } from '../memory/buffer-pool.js';
import { attentionBackwardCpu } from './attention-backward.js';

export const OpType = {
  EMBED: 'embed',
  MATMUL: 'matmul',
  RMSNORM: 'rmsnorm',
  LAYERNORM: 'layernorm',
  ATTENTION: 'attention',
  SOFTMAX: 'softmax',
  ROPE: 'rope',
  SILU: 'silu',
  GELU: 'gelu',
  SCALE: 'scale',
  CROSS_ENTROPY: 'cross_entropy',
  BIAS_ADD: 'bias_add',
  UPSAMPLE2D: 'upsample2d',
  PIXEL_SHUFFLE: 'pixel_shuffle',
  GROUPNORM: 'groupnorm',
  CONV2D: 'conv2d',
};

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
    const last = this.records[this.records.length - 1];
    if (last) {
      grads.set(last.output, gradOutput);
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

  async runBackward(backwardName, record, gradOut) {
    const entry = this.registry.ops[record.op];
    
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

  async accumulateGrad(grads, input, grad) {
    const existing = grads.get(input);
    if (!existing) {
      grads.set(input, grad);
      return;
    }
    const size = grad.shape.reduce((acc, value) => acc * value, 1);
    const summed = await runResidualAdd(existing, grad, size);
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

  async accumulateGrad(grads, input, grad) {
    const existing = grads.get(input);
    if (!existing) {
      grads.set(input, grad);
      return;
    }
    const size = grad.shape.reduce((acc, value) => acc * value, 1);
    const summed = await runResidualAdd(existing, grad, size);
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
