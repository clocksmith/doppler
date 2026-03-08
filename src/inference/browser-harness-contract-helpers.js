import { getInferenceLayerPatternContractArtifact } from '../rules/rule-registry.js';
import { isPlainObject } from '../utils/plain-object.js';
import { validateBrowserSuiteMetrics } from '../config/schema/browser-suite-metrics.schema.js';
import { buildExecutionContractArtifact } from '../config/execution-contract-check.js';
import { buildManifestRequiredInferenceFieldsArtifact } from '../config/required-inference-fields-contract-check.js';

export function buildSuiteContractMetrics(suite, baseMetrics, manifest) {
  const executionContractArtifact = buildExecutionContractArtifact(manifest);
  const executionV0GraphContractArtifact = executionContractArtifact?.executionV0?.graph ?? null;
  const layerPatternContractArtifact = getInferenceLayerPatternContractArtifact();
  const requiredInferenceFieldsArtifact = manifest?.modelType === 'transformer'
    && isPlainObject(manifest?.inference?.attention)
    ? buildManifestRequiredInferenceFieldsArtifact(
      manifest?.inference ?? null,
      `${manifest?.modelId ?? 'unknown'}.inference`
    )
    : null;
  return validateBrowserSuiteMetrics({
    ...baseMetrics,
    schemaVersion: 1,
    source: 'doppler',
    suite,
    ...(executionContractArtifact ? { executionContractArtifact } : {}),
    executionV0GraphContractArtifact,
    layerPatternContractArtifact,
    requiredInferenceFieldsArtifact,
  });
}
