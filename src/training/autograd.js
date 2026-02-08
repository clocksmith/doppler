import {
  runEmbedBackward,
  runMatmulBackward,
  runSoftmaxBackward,
  runRmsNormBackward,
  runAttentionBackward,
  runRoPEBackward,
  runSiluBackward,
  runGeluBackward,
  runScaleBackward,
  runCrossEntropyBackward,
  runLayerNormBackward,
  runBiasAddBackward,
  runUpsample2DBackward,
  runPixelShuffleBackward,
  runGroupNormBackward,
  runConv2DBackward,
} from '../gpu/kernels/backward/index.js';
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
        await this.accumulateGrad(grads, input, grad);
      }
    }

    return grads;
  }

  async runBackward(backwardName, record, gradOut) {
    switch (backwardName) {
      case 'embed_backward': {
        const [indices, weight] = record.inputs;
        if (!indices || !weight) {
          throw new Error('embed backward requires [indices, weight] inputs');
        }
        const { numTokens, hiddenSize, vocabSize, transpose, indexOffset } = record.options;
        const gradWeight = await runEmbedBackward(indices, gradOut, {
          numTokens,
          hiddenSize,
          vocabSize,
          transpose,
          indexOffset,
        });
        return [{ input: weight, grad: gradWeight }];
      }
      case 'matmul_backward': {
        const [input, weight] = record.inputs;
        const { M, N, K, transposeB, computeGradInput, computeGradWeight } = record.options;
        const { gradInput, gradWeight } = await runMatmulBackward(input, weight, gradOut, {
          M,
          N,
          K,
          transposeB,
          computeGradInput,
          computeGradWeight,
        });
        const outputs = [];
        if (gradInput) {
          outputs.push({ input, grad: gradInput });
        }
        if (gradWeight) {
          outputs.push({ input: weight, grad: gradWeight });
        }
        return outputs;
      }
      case 'softmax_backward': {
        const input = record.output;
        const { rows, cols } = record.options;
        const gradInput = await runSoftmaxBackward(input, gradOut, { rows, cols });
        return [{ input, grad: gradInput }];
      }
      case 'rmsnorm_backward': {
        const [input, weight] = record.inputs;
        const { numTokens, hiddenSize, eps } = record.options;
        const gradInput = await runRmsNormBackward(input, weight, gradOut, { numTokens, hiddenSize, eps });
        return [{ input, grad: gradInput }];
      }
      case 'layernorm_backward': {
        const [input, weight, bias] = record.inputs;
        const { numTokens, hiddenSize, eps } = record.options;
        const { gradInput, gradWeight, gradBias } = await runLayerNormBackward(input, weight, gradOutput, { numTokens, hiddenSize, eps });
        return [
          { input, grad: gradInput },
          { input: weight, grad: gradWeight },
          { input: bias, grad: gradBias },
        ];
      }
      case 'attention_backward': {
        const [q, k, v, softmax] = record.inputs;
        const { seqLen, numHeads, headDim, scale } = record.options;
        const recomputeForward = record.options.recomputeForward === true || !softmax;
        const { gradQ, gradK, gradV } = recomputeForward
          ? await attentionBackwardCpu(
            q,
            k,
            v,
            null,
            gradOut,
            { seqLen, numHeads, headDim, scale, causal: record.options.causal }
          )
          : await runAttentionBackward(
            q,
            k,
            v,
            softmax,
            gradOut,
            { seqLen, numHeads, headDim, scale, causal: record.options.causal }
          ).catch(() => attentionBackwardCpu(
            q,
            k,
            v,
            softmax,
            gradOut,
            { seqLen, numHeads, headDim, scale, causal: record.options.causal }
          ));
        return [
          { input: q, grad: gradQ },
          { input: k, grad: gradK },
          { input: v, grad: gradV },
        ];
      }
      case 'rope_backward': {
        const [input, freqsCos, freqsSin] = record.inputs;
        const { seqLen, numHeads, headDim, startPos } = record.options;
        const gradInput = await runRoPEBackward(gradOut, freqsCos, freqsSin, {
          seqLen, numHeads, headDim, startPos,
        });
        return [{ input, grad: gradInput }];
      }
      case 'silu_backward': {
        const input = record.inputs[0];
        const gradInput = await runSiluBackward(input, gradOut, record.options);
        return [{ input, grad: gradInput }];
      }
      case 'gelu_backward': {
        const input = record.inputs[0];
        const gradInput = await runGeluBackward(input, gradOut, record.options);
        return [{ input, grad: gradInput }];
      }
      case 'scale_backward': {
        const input = record.inputs[0];
        const { scale } = record.options;
        const gradInput = await runScaleBackward(input, gradOut, { scale });
        return [{ input, grad: gradInput }];
      }
      case 'cross_entropy_backward': {
        const [softmax, targets] = record.inputs;
        const { numTokens, vocabSize } = record.options;
        const gradInput = await runCrossEntropyBackward(softmax, targets, gradOut, { numTokens, vocabSize });
        return [{ input: softmax, grad: gradInput }];
      }
      case 'bias_add_backward': {
        const [input, bias] = record.inputs;
        const { numTokens, dim } = record.options;
        const gradBias = await runBiasAddBackward(gradOut, { numTokens, dim });
        return [
          { input, grad: gradOut }, // gradInput is just gradOut
          { input: bias, grad: gradBias },
        ];
      }
      case 'upsample2d_backward': {
        const [input] = record.inputs;
        const gradInput = await runUpsample2DBackward(gradOut, record.options);
        return [{ input, grad: gradInput }];
      }
      case 'pixel_shuffle_backward': {
        const [input] = record.inputs;
        const gradInput = await runPixelShuffleBackward(gradOut, record.options);
        return [{ input, grad: gradInput }];
      }
      case 'groupnorm_backward': {
        const [input, weight, bias] = record.inputs;
        const gradInput = await runGroupNormBackward(input, weight, gradOut, record.options);
        return [{ input, grad: gradInput }];
      }
      case 'conv2d_backward': {
        const [input, weight, bias] = record.inputs;
        const { gradInput, gradWeight } = await runConv2DBackward(input, weight, gradOut, record.options);
        return [
          { input, grad: gradInput },
          { input: weight, grad: gradWeight },
        ];
      }
      default:
        throw new Error(`Backward kernel "${backwardName}" not implemented`);
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
