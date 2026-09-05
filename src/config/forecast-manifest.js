import { computeCanonicalSha256 } from '../formats/canonical-hash.js';

export const FORECAST_MANIFEST_SCHEMA = 'doppler.forecast-manifest/v1';

export function validateForecastManifest(manifest, pack, targetPlan) {
  if (manifest?.schema !== FORECAST_MANIFEST_SCHEMA || manifest.modelId !== pack.modelId) {
    throw new Error('Forecast program requires its signed numeric forecasting manifest.');
  }
  const contract = manifest.forecast;
  if (!contract || !Number.isSafeInteger(contract.contextLength) || contract.contextLength < 1
    || contract.contextLength > 2048 || !Number.isSafeInteger(contract.predictionLength)
    || contract.predictionLength < 1 || contract.predictionLength > 64
    || contract.inputDtype !== 'f32' || contract.outputDtype !== 'f32'
    || contract.missingInput !== 'left-pad-masked-zero' || contract.outputLayout !== 'time-quantile'
    || !Array.isArray(contract.quantiles) || contract.quantiles.length < 1
    || contract.quantiles.some((q, i) => !Number.isFinite(q) || q <= 0 || q >= 1
      || (i > 0 && q <= contract.quantiles[i - 1]))) {
    throw new Error('Invalid forecast dimensions, dtype, missingness or quantile contract.');
  }
  if (targetPlan.dtypes.activation !== contract.inputDtype || targetPlan.dtypes.weight !== 'f32'
    || targetPlan.dtypes.kv !== 'none' || targetPlan.memoryLayout.kvCacheLayout !== 'none'
    || !Array.isArray(targetPlan.phases.forecast) || Object.keys(targetPlan.phases).length !== 1
    || manifest.executionGraphHash !== targetPlan.executionGraphHash
    || computeCanonicalSha256(manifest.execution) !== targetPlan.executionGraphHash
    || computeCanonicalSha256(manifest.execution) !== computeCanonicalSha256(pack.program.execution)
    || computeCanonicalSha256(manifest.execution.steps) !== computeCanonicalSha256(targetPlan.phases.forecast)) {
    throw new Error('Forecast manifest does not match the signed execution graph and dtype lane.');
  }
  const slots = new Map(targetPlan.memoryLayout.bufferSlots.map(slot => [slot.slotId, slot]));
  for (const [field, bytes, usage] of [
    ['inputSlot', contract.contextLength * 4, 'storage'], ['maskSlot', contract.contextLength * 4, 'storage'],
    ['requestSlot', 16, 'uniform'], ['outputSlot', contract.predictionLength * contract.quantiles.length * 4, 'storage'],
  ]) {
    const slot = slots.get(contract[field]);
    if (!slot || slot.owner !== 'runtime' || slot.size.op !== 'constant' || slot.size.bytes !== bytes
      || !slot.usage?.includes(usage) || !slot.usage.includes(field === 'outputSlot' ? 'copy-src' : 'copy-dst')) {
      throw new Error(`Forecast ${field} has an invalid memory binding.`);
    }
  }
  if (!Array.isArray(manifest.uploads) || manifest.uploads.length === 0) throw new Error('Forecast uploads are required.');
  const uploaded = new Set();
  for (const upload of manifest.uploads) {
    const slot = slots.get(upload.slotId);
    const artifact = pack.artifacts.find(entry => entry.artifactId === upload.artifactId);
    if (!slot || slot.owner !== 'runtime' || slot.size.op !== 'constant' || !slot.usage?.includes('copy-dst')
      || !artifact || !Number.isSafeInteger(upload.offsetBytes) || upload.offsetBytes < 0
      || !Number.isSafeInteger(upload.sizeBytes) || upload.sizeBytes !== slot.size.bytes
      || upload.offsetBytes + upload.sizeBytes > artifact.sizeBytes || upload.offsetBytes % 4 !== 0
      || uploaded.has(upload.slotId)
      || [contract.inputSlot, contract.maskSlot, contract.requestSlot, contract.outputSlot].includes(upload.slotId)) {
      throw new Error('Invalid or conflicting forecast artifact upload.');
    }
    uploaded.add(upload.slotId);
  }
  return contract;
}
