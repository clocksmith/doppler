import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

import {
  buildProductPortfolioCoherenceReport,
  validateProductPortfolioCoherence,
} from '../../tools/check-product-portfolio-coherence.js';

const PATHS = {
  contract: 'tools/policies/product-portfolio-coherence.json',
  catalog: 'models/catalog.json',
  productIntegrations: 'tools/policies/product-integration-qualification.json',
  providerConformance: 'tools/policies/provider-conformance.json',
  runtimeOwnership: 'benchmarks/vendors/runtime-ownership-decisions.json',
  bunQualification: 'tools/policies/bun-product-qualification.json',
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

const sources = Object.fromEntries(await Promise.all(Object.entries(PATHS).map(
  async ([name, filePath]) => [name, JSON.parse(await fs.readFile(filePath, 'utf8'))]
)));

{
  const report = await buildProductPortfolioCoherenceReport();
  assert.equal(report.ok, true, report.errors.join('\n'));
  assert.equal(report.workloads.length, 3);
  assert.deepEqual(report.workloads.map((entry) => entry.workload), [
    'generation',
    'embedding',
    'reranking',
  ]);
  assert.equal(
    report.workloads.find((entry) => entry.workload === 'generation')?.resolvedArtifactVariantId,
    'sha256:b0e11d284d95b1e5f3f6819f978bc4b9a353a73f18054d137aaacf6c4de7cd56'
  );
  assert.equal(
    report.workloads.find((entry) => entry.workload === 'embedding')?.resolvedArtifactVariantId,
    'sha256:18a4175e7ff511ec88b6c7a45406c31f71e747dd36e509ab8cc25a2263f85d7c'
  );
  assert.equal(
    report.workloads.find((entry) => entry.workload === 'reranking')?.resolvedArtifactVariantId,
    'sha256:c9da235a2ad1d59bff76230a08806c0d9f7dfc468864c22ea3dbfbb20185ba82'
  );
  assert.deepEqual(report.requiredGates, [
    'product-integration',
    'provider-conformance',
    'runtime-ownership',
    'bun-product',
  ]);
}

{
  const driftedModel = clone(sources);
  driftedModel.runtimeOwnership.decisions[0].logicalModelId = 'different-generation-model';
  const report = validateProductPortfolioCoherence(driftedModel);
  assert.equal(report.ok, false);
  assert.ok(
    report.errors.some((error) => error.includes(
      'runtime ownership.qwen35-generation-runtime-ownership.logicalModelId'
    )),
    report.errors.join('\n')
  );
}

{
  const driftedManifest = clone(sources);
  driftedManifest.bunQualification.qualifications[1].manifestVariantId = 'different-variant';
  const report = validateProductPortfolioCoherence(driftedManifest);
  assert.equal(report.ok, false);
  assert.ok(
    report.errors.some((error) => error.includes(
      'Bun qualification.embeddinggemma-bun-product.manifestVariantId'
    )),
    report.errors.join('\n')
  );
}

{
  const driftedCorrectness = clone(sources);
  driftedCorrectness.providerConformance.suites[2].correctnessClass = 'exact-token';
  const report = validateProductPortfolioCoherence(driftedCorrectness);
  assert.equal(report.ok, false);
  assert.ok(
    report.errors.some((error) => error.includes(
      'provider conformance.qwen3-reranking-browser-node.correctnessClass'
    )),
    report.errors.join('\n')
  );
}

{
  const driftedCatalog = clone(sources);
  const model = driftedCatalog.catalog.models.find((entry) => (
    entry.modelId === 'google-embeddinggemma-300m-q4k-ehf16-af32'
  ));
  model.manifestVariantId = 'different-catalog-variant';
  const report = validateProductPortfolioCoherence(driftedCatalog);
  assert.equal(report.ok, false);
  assert.ok(
    report.errors.some((error) => error.includes(
      'model catalog.google-embeddinggemma-300m-q4k-ehf16-af32.manifestVariantId'
    )),
    report.errors.join('\n')
  );
}

{
  const mixedArtifacts = clone(sources);
  mixedArtifacts.productIntegrations.integrations[0].resolvedArtifactVariantId =
    `sha256:${'a'.repeat(64)}`;
  mixedArtifacts.providerConformance.suites[0].resolvedArtifactVariantId =
    `sha256:${'b'.repeat(64)}`;
  const report = validateProductPortfolioCoherence(mixedArtifacts);
  assert.equal(report.ok, false);
  assert.ok(
    report.errors.some((error) => error.includes(
      'resolved artifact identity differs across gates'
    )),
    report.errors.join('\n')
  );
}

{
  const missingBinding = clone(sources);
  missingBinding.productIntegrations.integrations = missingBinding.productIntegrations.integrations
    .filter((entry) => entry.id !== 'columbo-local-reranking');
  const report = validateProductPortfolioCoherence(missingBinding);
  assert.equal(report.ok, false);
  assert.ok(
    report.errors.some((error) => error.includes(
      'does not identify a product integration row: columbo-local-reranking'
    )),
    report.errors.join('\n')
  );
}

console.log('product-portfolio-coherence.test: ok');
