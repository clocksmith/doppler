import { runProbes, tensorProbeOptions } from "../observation/probes.js";
import { readBuffer } from '../../../../memory/buffer-pool.js';
import { trace, isTraceEnabled } from '../../../../debug/index.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { decodeReadback } from '../debug-utils/readback-decoding.js';
import { getLogitsHealth } from '../debug-utils/health-metrics.js';
import { shouldDebugLayer } from './types.js';

export function shouldTraceRecordedHealth(layerIdx, debugFlags) {
  const debugLayers = debugFlags?.debugLayers;
  return isTraceEnabled('logits')
    && Array.isArray(debugLayers)
    && shouldDebugLayer(layerIdx, debugLayers);
}

export function enqueueRecordedTensorHealth(recorder, label, tensor, dtype, elementCount) {
  if (!recorder || !tensor?.buffer || !Number.isFinite(elementCount) || elementCount <= 0) return;
  const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype });
  recorder.enqueueCompletionTask(async () => {
    const data = await readBuffer(tensor.buffer, elementCount * bytesPerElement);
    trace.logits(label, getLogitsHealth(decodeReadback(data, dtype)));
  });
}

export async function observeRecordedAttentionProjections({ qTensor, kTensor, vTensor, config, state, debugFlags, recorder }) {
  const { layerIdx, numTokens, numHeads, numKVHeads, headDim } = config;
  if (qTensor) {
    await runProbes('q_proj', qTensor.buffer, {
      layerIdx, numTokens, hiddenSize: numHeads * headDim,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, ...tensorProbeOptions(qTensor),
    });
  }
  if (kTensor) {
    await runProbes('k_proj', kTensor.buffer, {
      layerIdx, numTokens, hiddenSize: numKVHeads * headDim,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, ...tensorProbeOptions(kTensor),
    });
  }
  if (vTensor) {
    await runProbes('v_proj', vTensor.buffer, {
      layerIdx, numTokens, hiddenSize: numKVHeads * headDim,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, ...tensorProbeOptions(vTensor),
    });
  }
  if (qTensor && shouldTraceRecordedHealth(layerIdx, debugFlags)) {
    enqueueRecordedTensorHealth(
      recorder,
      `L${layerIdx}.q_proj_HEALTH`,
      qTensor,
      qTensor.dtype,
      numTokens * numHeads * headDim
    );
    enqueueRecordedTensorHealth(
      recorder,
      `L${layerIdx}.k_proj_HEALTH`,
      kTensor,
      kTensor?.dtype ?? null,
      numTokens * numKVHeads * headDim
    );
    enqueueRecordedTensorHealth(
      recorder,
      `L${layerIdx}.v_proj_HEALTH`,
      vTensor,
      vTensor?.dtype ?? null,
      numTokens * numKVHeads * headDim
    );
  }
}

export async function observeRecordedAttentionQKNorm({ qTensor, kTensor, config, state, debugFlags, recorder }) {
  const { layerIdx, numTokens, numHeads, numKVHeads, headDim } = config;

    await runProbes('q_norm', qTensor.buffer, {
      layerIdx, numTokens, hiddenSize: numHeads * headDim,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, ...tensorProbeOptions(qTensor),
    });
    if (kTensor) {
      await runProbes('k_norm', kTensor.buffer, {
        layerIdx, numTokens, hiddenSize: numKVHeads * headDim,
        probes: state.debugProbes, recorder,
        operatorDiagnostics: state.operatorDiagnostics, ...tensorProbeOptions(kTensor),
      });
    }
    if (shouldTraceRecordedHealth(layerIdx, debugFlags)) {
      enqueueRecordedTensorHealth(
        recorder,
        `L${layerIdx}.q_norm_HEALTH`,
        qTensor,
        qTensor.dtype,
        numTokens * numHeads * headDim
      );
      enqueueRecordedTensorHealth(
        recorder,
        `L${layerIdx}.k_norm_HEALTH`,
        kTensor,
        kTensor?.dtype ?? null,
        numTokens * numKVHeads * headDim
      );
    }

}
