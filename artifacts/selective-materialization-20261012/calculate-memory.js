// Evidence-only arithmetic. No WebGPU, model execution, or runtime configuration.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';
const directory = dirname(fileURLToPath(import.meta.url));
const root = resolve(directory, '../..');
const inventory = JSON.parse(readFileSync(resolve(directory, 'memory-inventory.json')));
const c = inventory.contract;
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
assert.equal(digest(readFileSync(resolve(directory, 'inactive-precision-proposal.json'))), inventory.proposalSha256);
const architecture = inventory.model.architecture;
assert.equal(architecture.hiddenSize, 1024);
assert.equal(architecture.intermediateSize, 3584);
assert.equal(architecture.numAttentionHeads * architecture.headDim, 2048);
assert.equal(architecture.numKeyValueHeads * architecture.headDim, 512);
assert.equal(2 * architecture.linearNumKeyHeads * architecture.linearKeyHeadDim
  + architecture.linearNumValueHeads * architecture.linearValueHeadDim, 6144);
assert.equal(architecture.linearConvKernelDim, 4);
for (const start of [0, 12]) {
  const layers = inventory.model.layerTypes.layerTypes.slice(start, start + 12);
  assert.equal(layers.filter(type => type === 'linear_attention').length, c.linearLayersPerPartition);
  assert.equal(layers.filter(type => type === 'full_attention').length, c.fullLayersPerPartition);
}
for (const source of inventory.sources) {
  assert.equal(digest(readFileSync(resolve(root, source.path))), source.sha256, source.path);
}
function bucket(bytes) {
  assert(Number.isSafeInteger(bytes) && bytes > 0);
  const aligned = Math.ceil(bytes / c.pool.alignment) * c.pool.alignment;
  if (aligned <= c.pool.minBucket) return c.pool.minBucket;
  if (aligned >= c.pool.largeThreshold) return Math.ceil(aligned / c.pool.largeStep) * c.pool.largeStep;
  return 2 ** Math.ceil(Math.log2(aligned));
}
const sum = values => values.reduce((a, b) => a + b, 0);
const product = shape => shape.reduce((a, b) => a * b, 1);
const rowBytes = (row, tokens) => tokens * row.shape[1] * c.lowComponentBytes;
function layer(type, phase, tokens) {
  const rows = inventory.rows.filter(row => (row.layer === 'all' || row.layer === type)
    && (row.phase === 'both' || row.phase === phase)).map(row => ({
    id: row.id, shape: [tokens, row.shape[1]], logicalBytes: rowBytes(row, tokens),
    bucketedBytes: bucket(rowBytes(row, tokens)),
  }));
  const epsilonBytes = type === 'linear' ? inventory.persistent.epsilon.conservativePooledAllocationBytes : 0;
  return { rows, logicalBytes: sum(rows.map(row => row.logicalBytes)),
    bucketedBytes: sum(rows.map(row => row.bucketedBytes)) + epsilonBytes, epsilonBytes };
}
const remainder = c.promptTokens % c.prefillTokensPerChunk;
const cases = {};
for (const tokens of [c.prefillTokensPerChunk, remainder, 1]) {
  const phase = tokens === 1 ? 'decode' : 'prefill';
  const linear = layer('linear', phase, tokens);
  const full = layer('full', phase, tokens);
  // Carry/output, staging-readback and final normalization/last-position input.
  // All three charged together even though the endpoint phases need not overlap.
  const boundaryReserve = 3 * bucket(tokens * 1024 * 4);
  const layerBytes = phase === 'prefill' ? Math.max(linear.bucketedBytes, full.bucketedBytes)
    : c.linearLayersPerPartition * linear.bucketedBytes + c.fullLayersPerPartition * full.bucketedBytes;
  cases[`${phase}-${tokens}`] = { linear, full, boundaryReserve,
    oneAttemptMappedScratchBound: layerBytes + boundaryReserve,
    twoAttemptMappedScratchBound: c.attempts * (layerBytes + boundaryReserve) };
}
const convLogical = c.linearLayersPerPartition * product(inventory.persistent.convHistoryShape) * 4;
const convBucketed = c.linearLayersPerPartition * bucket(product(inventory.persistent.convHistoryShape) * 4);
const approvedStateExtra = c.linearLayersPerPartition * product(inventory.persistent.stateShape) * 4;
const rope = sum(inventory.persistent.ropeLowShapes.map(shape => bucket(product(shape) * 4)));
const scalarSetupReserve = 2 * bucket(128); // prospective 32-lane frequency residual + scalar params
const newPersistent = c.attempts * convBucketed + rope;
const worstScratch = Math.max(...Object.values(cases).map(x => x.twoAttemptMappedScratchBound));
const sizes = new Set([bucket(product(inventory.persistent.convHistoryShape) * 4), 256]);
for (const tokens of [1, remainder, c.prefillTokensPerChunk]) {
  for (const row of inventory.rows) sizes.add(bucket(rowBytes(row, tokens)));
  sizes.add(bucket(tokens * 1024 * 4));
}
// Separate STORAGE and MAP_READ usages are separate pools. Conservatively allow
// the boundary readback sizes another per-bucket quota. The global64 limit still applies.
const idleSlots = [...sizes].flatMap(size => Array(c.pool.maxPerBucket).fill(size));
for (const tokens of [1, remainder, c.prefillTokensPerChunk]) {
  idleSlots.push(...Array(c.pool.maxPerBucket).fill(bucket(tokens * 1024 * 4)));
}
idleSlots.sort((a, b) => b - a);
const idlePoolBound = sum(idleSlots.slice(0, c.pool.maxPooledBuffers));
const output = {
  schema: 'doppler.inactive-paired-intermediate-memory-calculation/v1',
  qualified: false,
  inventorySha256: digest(readFileSync(resolve(directory, 'memory-inventory.json'))),
  calculatorSha256: digest(readFileSync(fileURLToPath(import.meta.url))),
  budget: c.budget,
  allocationScope: 'Additional GPUBuffer sidecars in the declared calculation design; not process RAM, hardware residency, or a complete new-runtime peak.',
  chunkRows: [256,256,256,256,256,256,52],
  cases,
  persistent: { convolutionLowLogicalPerAttempt: convLogical, convolutionLowBucketedPerAttempt: convBucketed,
    convolutionLowTwoAttempts: c.attempts * convBucketed,
    sharedRotaryLowTables: rope, setupResidualAndParamsReserve: scalarSetupReserve,
    newBroadPersistentTwoAttempts: newPersistent,
    alreadyApprovedRecurrentExtraPerAttempt: approvedStateExtra,
    alreadyApprovedRecurrentExtraTwoAttempts: c.attempts * approvedStateExtra },
  bounds: { twoAttemptActiveBroadSidecars: newPersistent + worstScratch + scalarSetupReserve,
    additionalIdlePoolInventoriedSizes: idlePoolBound,
    twoAttemptBroadPlusInventoriedIdlePool: newPersistent + worstScratch + scalarSetupReserve + idlePoolBound,
    includingPreviouslyApprovedStateLowsComparedWithHistoricalF32: newPersistent + worstScratch + scalarSetupReserve + idlePoolBound + c.attempts * approvedStateExtra,
    qualification: 'Conditional envelope for listed sizes/lifetimes and settled submissions only. No bound on unbounded deferred destruction, unseen workspace, allocator implementation changes, or device-internal allocations.' },
  unchangedKv: { reservedPositionsPerAttempt: c.promptTokens + c.maxOutputTokens,
    bytesPerPartitionPerAttempt: 3 * 2 * (c.promptTokens + c.maxOutputTokens) * 2 * 256 * 2,
    bytesPerPartitionTwoAttempts: 2 * 3 * 2 * (c.promptTokens + c.maxOutputTokens) * 2 * 256 * 2,
    upperPrepared8192PositionsPerAttempt: 3 * 2 * 8192 * 2 * 256 * 2,
    note: 'F16 K/V remains unchanged; each attempt reserves prompt+maximum output, not observed1627 output length.' },
  transport: { extraHostPayloadPer1588Tokens: c.promptTokens * 1024 * 4,
    pairedHostPayloadPer1588Tokens: c.promptTokens * 1024 * 8,
    twoPayloadExtraLogicalBytes: c.attempts * c.promptTokens * 1024 * 4,
    extraDecodePayloadBytes: 1024 * 4,
    note: 'CPU payload bytes, not GPU budget. Structured clones, network copies and host staging multiplicity are not bounded here. GPU chunks/readback sidecars are charged above.' },
  observed: inventory.observed,
  historicalPeakSensitivityOnly: inventory.observed.hosts.map(host => {
    const duplicatedNonweightPeak = host.weightBytes + 2 * (host.peakBytes - host.weightBytes);
    const sumWithProposal = duplicatedNonweightPeak + newPersistent + worstScratch + scalarSetupReserve
      + idlePoolBound + 2 * approvedStateExtra;
    return { ...host, duplicatedNonweightPeak, sumWithProposal, remainingToCap: c.budget - sumWithProposal,
      caveat: 'Arithmetic sensitivity, not measured concurrent peak or proof of a total upper bound. Historical nonweight peak includes reusable buffers; duplication and new idle envelope deliberately overcount some bytes, and can miss new workspaces.' };
  }),
  checks: { sourceHashes: inventory.sources.length, exactIntegerArithmetic: true, noRuntimeOrGpuExecution: true },
};
assert.equal(sum(output.chunkRows), 1588);
assert.equal(approvedStateExtra, 9437184);
assert.equal(convLogical, 884736);
assert.equal(convBucketed, 1179648);
assert.equal(rope, 2097152);
assert.equal(output.unchangedKv.bytesPerPartitionPerAttempt, 34922496);
assert.equal(output.transport.extraHostPayloadPer1588Tokens, 6504448);
assert.equal(layer('linear', 'decode', 1).rows.some(row => row.id === 'linear-convolution-output'), false);
assert.equal(layer('linear', 'decode', 1).rows.some(row => row.id === 'ffn-gate'), false);
assert.equal(inventory.rows.filter(row => row.id.includes('readout')).length, 1);
assert.equal(inventory.rows.filter(row => row.id.includes('rope')).length, 2);
const resultPath = resolve(directory, 'memory-calculation.json');
const text = `${JSON.stringify(output, null, 2)}\n`;
if (process.argv.includes('--check')) assert.equal(readFileSync(resultPath, 'utf8'), text);
else writeFileSync(resultPath, text);
console.log(JSON.stringify({ persistent: output.persistent, bounds: output.bounds,
  cases: Object.fromEntries(Object.entries(cases).map(([key,value]) => [key, value.twoAttemptMappedScratchBound])),
  sensitivity: output.historicalPeakSensitivityOnly.map(x => ({host:x.host,index:x.partitionIndex,total:x.sumWithProposal,margin:x.remainingToCap})) }, null, 2));
