import { isTraceEnabled, trace } from '../../../../debug/index.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { readBuffer } from '../../../../memory/buffer-pool.js';
import { getLogitsHealth, decodeReadback, shouldDebugLayerOutput } from '../debug-utils.js';

/** @type {import('./activation-observation.js').captureTrainingActivation} */
export async function captureTrainingActivation(context, layerIdx, tensor, numTokens, hiddenSize) {
  const capture = context.trainingCapture;
  if (!capture || capture.layerIdx !== layerIdx || capture.stage !== 'ffn_act') return;
  await capture.capture({
    layerIdx,
    stage: 'ffn_act',
    tensor,
    numTokens,
    hiddenSize,
    recorder: context.recorder ?? null,
  });
}
/** @type {import('./activation-observation.js').enqueueRecordedDenseHealth} */
export function enqueueRecordedDenseHealth(context, layerIdx, label, tensor, elementCount) {
  const recorder = context.recorder ?? null;
  if (!recorder || !isTraceEnabled('logits') || !shouldDebugLayerOutput(layerIdx, context.debugLayers)) {
    return;
  }
  if (!tensor?.buffer || !Number.isFinite(elementCount) || elementCount <= 0) {
    return;
  }
  const dtype = tensor.dtype;
  const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype });
  recorder.enqueueCompletionTask(async () => {
    const data = await readBuffer(tensor.buffer, elementCount * bytesPerElement);
    trace.logits(`L${layerIdx}.${label}_HEALTH`, getLogitsHealth(decodeReadback(data, dtype)));
  });
}
