import { readBuffer } from '../../../../memory/buffer-pool.js';
import { trace, isTraceEnabled } from '../../../../debug/index.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { decodeReadback, getLogitsHealth } from '../debug-utils/index.js';
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
