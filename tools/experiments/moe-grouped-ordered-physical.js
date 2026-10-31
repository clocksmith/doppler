import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { readJson, sha256, sourceInventory } from './native-qualification/source-proof.js';
import { orderedPackedFixture, orderedInputFixture, compareF32Bits } from './moe-grouped-ordered-reference.js';
import { orderedGroupedDeclarations } from './moe-grouped-ordered-program.js';

const repository = fileURLToPath(new URL('../../', import.meta.url));
const controlRoot = path.join(repository, 'NATIVE_LONG_EQUIVALENCE/runtime-control');
const controlCommit = '0b9c2cc3bc77a9750f63e7479d0059e695751a5e';
const importControl = name => import(pathToFileURL(path.join(controlRoot, 'src', name)).href);

// Called only by the existing guarded physical runner after its lock/provider
// admission. No GPU startup occurs when importing this preparation module.
export async function runOrderedGroupedPhysical({ device, info, probe, outputDirectory }) {
  assert.ok(outputDirectory, '--ordered requires --output-dir; never overwrite a prior receipt');
  const output = path.resolve(outputDirectory);
  await fs.mkdir(output); // Exclusive: prior failures are immutable inputs to diagnosis.
  const save = async (name, value) => fs.writeFile(path.join(output, name), JSON.stringify(value, null, 2) + '\n');
  const accepted = await readJson(path.join(repository, 'NATIVE_LONG_EQUIVALENCE/preflight.json'));
  assert.equal(accepted.passed, true);
  assert.equal(accepted.sources.control.commit, controlCommit);
  const expectedControl = { files: accepted.sources.control.files, symlinks: accepted.sources.control.symlinks };
  assert.deepEqual(await sourceInventory(controlRoot), expectedControl, 'Frozen 0b9 source inventory drift');
  const candidateInventory = await sourceInventory(repository);
  const fixtureFiles = ['tools/experiments/moe-grouped-physical.js', 'tools/experiments/run-moe-grouped-physical.sh',
    'tools/experiments/moe-grouped-ordered-physical.js', 'tools/experiments/moe-grouped-ordered-reference.js',
    'tools/experiments/moe-grouped-ordered-program.js'];
  const fixtureHashes = {};
  for (const name of fixtureFiles) fixtureHashes[name] = sha256(await fs.readFile(path.join(repository, name)));
  await save('source-identity.json', { baseline: { commit: controlCommit, ...expectedControl },
    candidate: candidateInventory, fixtureHashes,
    origin: 'Retained accepted Git-proven inventory rehashed without invoking Git; live candidate files are content-bound.' });
  const { getKernelCapabilities, getPlatformConfig } = await import('../../src/gpu/device.js');
  const { createTensor } = await import('../../src/gpu/tensor.js');
  const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
  const pool = await import('../../src/memory/buffer-pool.js');
  const { createCommandRecorder } = await import('../../src/gpu/command-recorder.js');
  const { getKernelConfig, createKernelRegistry, enterKernelRegistry } = await import('../../src/gpu/kernels/kernel-configs.js');
  const { groupedPrefillGeometry, executeGroupedSort, executeGroupedProjection } = await import('../../src/gpu/kernels/moe-grouped.js');
  const { executeGroupedPrefill } = await import('../../src/inference/pipelines/text/moe/grouped-prefill.js');
  const { runSiLURowSplit, recordSiLURowSplit } = await import('../../src/gpu/kernels/silu.js');
  const { runRouteScatter, recordRouteScatter } = await import('../../src/gpu/kernels/moe-route.js');
  const baselineDevice = await importControl('gpu/device.js');
  baselineDevice.setDevice(device, { capabilities: getKernelCapabilities(), platformConfig: getPlatformConfig() });
  const baseline = await importControl('gpu/kernels/moe-route.js');
  const baselineActivation = await importControl('gpu/kernels/silu.js');
  const baselinePool = await importControl('memory/buffer-pool.js');
  const baselineCommands = await importControl('gpu/command-recorder.js');
  const baselineConfig = await importControl('gpu/kernels/kernel-configs.js');
  const orderedGu = getKernelConfig('moe_grouped_gu', 'native_f32_ordered');
  const orderedDown = getKernelConfig('moe_grouped_down', 'native_f32_ordered');
  const sort = getKernelConfig('moe_grouped_sort', 'native_f32');
  const declarations = orderedGroupedDeclarations();
  const kernelPath = { prefill: { steps: Object.entries(declarations)
    .map(([op, declaration]) => ({ op, ...declaration })) }, decode: { steps: [] } };
  const referenceConfigs = Object.fromEntries(['q4_0', 'q4_1', 'q8_0'].flatMap(dtype =>
    [`${dtype}_split_f32`, `${dtype}_f32`].map(variant =>
      [variant, baselineConfig.getKernelConfig('moe_route_matmul', variant)])));
  for (const config of Object.values(referenceConfigs)) {
    assert.equal(config.wgslOverrides.THREADS_PER_COL_GEMV, 32, 'Pinned baseline must use 32-lane tree');
    assert.equal(config.wgslOverrides.WORKGROUP_SIZE,
      config.wgslOverrides.COLS_PER_WG * config.wgslOverrides.THREADS_PER_COL_GEMV);
  }
  for (const config of [orderedGu, orderedDown]) {
    assert.equal(config.wgslOverrides.REDUCTION_LANES, 32, 'Ordered candidate must preserve 32-lane FMA/tree');
    assert.equal(config.wgslOverrides.GROUP_SIZE, 8, 'Ordered candidate retains eight-route reuse');
    assert.equal(config.variantMetadata.reductionLanes, 32);
    assert.equal(config.shaderFile, declarations[config.operation].kernel);
    assert.equal(config.entryPoint, declarations[config.operation].entry);
  }
  await save('kernel-contract.json', { kernelPath, candidate: { sort, gu: orderedGu, down: orderedDown },
    baseline: referenceConfigs });
  const report = { status: 'running', adapter: info, provider: probe.providerReceipt,
    scope: 'Exact pinned route-GU/down operator comparison; activation/scatter separately gated; no model qualification or performance claim.',
    comparison: { metric: 'F32 bitwise equality including signed zero', tolerance: null, finiteRequired: true },
    baselineCommit: controlCommit, realModelCaptures: 'Not admitted: requested layer0 directory is absent.',
    receipts: [] };
  await save('receipt.json', report);
  const errors = [];
  const onError = event => errors.push(event.error.message);
  device.addEventListener('uncapturederror', onError);
  async function capture(name, tensor, elements) {
    const raw = new Uint8Array(await pool.readBuffer(tensor.buffer, elements * 4));
    await fs.writeFile(path.join(output, `${name}.f32`), raw, { flag: 'wx' });
    return { values: new Float32Array(raw.buffer, raw.byteOffset, elements),
      artifact: { file: `${name}.f32`, bytes: raw.length, sha256: sha256(raw), elements, dtype: 'f32', byteOrder: 'little-endian' } };
  }
  async function compare(name, candidate, reference, elements) {
    const actual = await capture(`${name}-candidate`, candidate, elements);
    const expected = await capture(`${name}-baseline`, reference, elements);
    return { ...compareF32Bits(actual.values, expected.values), candidate: actual.artifact, baseline: expected.artifact };
  }
  const held = [];
  const keep = (buffer, release = pool.releaseBuffer) => { held.push({ buffer, release }); return buffer; };
  const upload = (data, label) => {
    const buffer = keep(pool.acquireBuffer(data.byteLength, undefined, label));
    pool.uploadData(buffer, data); return buffer;
  };
  const keepTensor = (tensor, release) => { keep(tensor.buffer, release); return tensor; };
  async function releaseSince(mark) {
    await device.queue.onSubmittedWorkDone();
    for (const { buffer, release } of held.splice(mark)) release(buffer);
  }
  try {
    for (const dtype of ['q4_0', 'q4_1', 'q8_0']) {
      for (const [H, I, tokenCounts] of [[2048, 512, [1, 33, 256]], [96, 96, [7, 9]]]) {
        const bankMark = held.length, E = 256, topK = 8;
        const weights = {}, weightReceipts = {};
        for (const [name, shape, salt] of [['gate', [E, I, H], 1], ['up', [E, I, H], 4], ['down', [E, H, I], 7]]) {
          const raw = orderedPackedFixture(dtype, shape, salt);
          weightReceipts[name] = { dtype, shape, salt, bytes: raw.byteLength, sha256: sha256(raw),
            fixture: 'orderedPackedFixture: signed original GGUF scale/min/nibble/signed-byte blocks' };
          weights[name] = createWeightBuffer(upload(raw, `ordered_${name}`), dtype, 'row', shape);
        }
        await save(`${dtype}-h${H}-i${I}-weights.json`, weightReceipts);
        for (const T of tokenCounts) for (const recorded of [false, true]) {
          const mark = held.length, id = `${dtype}-h${H}-i${I}-t${T}-${recorded ? 'recorded' : 'immediate'}`;
          const options = { numTokens: T, numExperts: E, topK, hiddenSize: H, intermediateSize: I,
            kernelPath, phase: 'prefill', layerIdx: 0 };
          const sizes = groupedPrefillGeometry(options, device.limits);
          const ids = Uint32Array.from({ length: T * topK }, (_, route) => {
            const token = Math.floor(route / topK), slot = route % topK;
            return slot === 0 ? 0 : 1 + (token % 3) * 16 + slot;
          });
          const mixing = Float32Array.from({ length: T * topK }, (_, i) => (i % topK + 1) / 36);
          const inputValues = orderedInputFixture(T, H, 3);
          const downValues = orderedInputFixture(T * topK, I, 5);
          const input = createTensor(upload(inputValues, 'ordered_input'), 'f32', [T, H]);
          const downInput = createTensor(upload(downValues, 'ordered_down_input'), 'f32', [T * topK, I]);
          const indices = upload(ids, 'ordered_indices'), routingWeights = upload(mixing, 'ordered_route_weights');
          const groups = keep(pool.acquireBuffer(sizes.groupBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC));
          const indirect = keep(pool.acquireBuffer(24, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC));
          const recorder = recorded ? createCommandRecorder(`ordered-${id}`) : null;
          const referenceRecorder = recorded ? baselineCommands.createCommandRecorder(`baseline-${id}`) : null;
          const route = referenceRecorder ? baseline.recordRouteQ4Matmul.bind(null, referenceRecorder) : baseline.runRouteQ4Matmul;
          const activateBaseline = referenceRecorder ? baselineActivation.recordSiLURowSplit.bind(null, referenceRecorder) : baselineActivation.runSiLURowSplit;
          const activate = recorder ? recordSiLURowSplit.bind(null, recorder) : runSiLURowSplit;
          const scatterBaseline = referenceRecorder ? baseline.recordRouteScatter.bind(null, referenceRecorder) : baseline.runRouteScatter;
          const scatter = recorder ? recordRouteScatter.bind(null, recorder) : runRouteScatter;
          const routeOptions = { numRoutes: T * topK, numExperts: E, topK, alpha: 1 };
          const activationOptions = { numTokens: T * topK, dim: I, activation: 'silu', swigluLimit: null };
          const scatterOptions = { numTokens: T, hiddenSize: H, topK };
          try {
            await executeGroupedSort(recorder, indices, groups, indirect, options);
            const gu = keepTensor(await executeGroupedProjection(recorder, 'moe_grouped_gu', input,
              weights.gate, weights.up, groups, indirect, options));
            const guBaseline = keepTensor(await route(input, indices, weights.gate,
              { ...routeOptions, N: I * 2, K: H, inputMode: 'token', upWeights: weights.up }), baselinePool.releaseBuffer);
            const down = keepTensor(await executeGroupedProjection(recorder, 'moe_grouped_down', downInput,
              weights.down, null, groups, indirect, options));
            const downBaseline = keepTensor(await route(downInput, indices, weights.down,
              { ...routeOptions, N: H, K: I, inputMode: 'route' }), baselinePool.releaseBuffer);
            const activated = keepTensor(await activate(gu, activationOptions));
            const activatedBaseline = keepTensor(await activateBaseline(guBaseline, activationOptions), baselinePool.releaseBuffer);
            const chainedDown = keepTensor(await executeGroupedProjection(recorder, 'moe_grouped_down', activated,
              weights.down, null, groups, indirect, options));
            const chainedDownBaseline = keepTensor(await route(activatedBaseline, indices, weights.down,
              { ...routeOptions, N: H, K: I, inputMode: 'route' }), baselinePool.releaseBuffer);
            const scattered = keepTensor(await scatter(chainedDown, routingWeights, scatterOptions));
            const scatteredBaseline = keepTensor(await scatterBaseline(chainedDownBaseline, routingWeights, scatterOptions), baselinePool.releaseBuffer);
            const production = keepTensor(await executeGroupedPrefill(recorder, { ...options,
              inputTensor: input, indicesBuffer: indices, weightsBuffer: routingWeights,
              weights: { ...weights, numExperts: E }, swigluLimit: null }));
            if (referenceRecorder) await referenceRecorder.submitAndWait();
            if (recorder) await recorder.submitAndWait();
            const schedule = new Uint32Array(await pool.readBuffer(groups, sizes.groupBytes));
            const args = new Uint32Array(await pool.readBuffer(indirect, 24));
            const seen = [];
            let chunk = 0, pair = 0;
            for (let expert = 0; expert < E; expert++) {
              const routes = Array.from(ids).flatMap((value, i) => value === expert ? [i] : []);
              for (let start = 0; start < routes.length; start += 8) {
                const selected = routes.slice(start, start + 8);
                assert.deepEqual(Array.from(schedule.slice(chunk * 4, chunk * 4 + 4)), [pair, selected.length, expert, 0]);
                const actual = Array.from(schedule.slice(sizes.pairOffset + pair, sizes.pairOffset + pair + selected.length));
                assert.deepEqual(actual, selected); seen.push(...actual);
                pair += selected.length; chunk++;
              }
            }
            assert.equal(new Set(seen).size, T * topK, 'Every routed slot appears exactly once');
            assert.deepEqual(Array.from(args), [sizes.guGroups, chunk, 1, sizes.downGroups, chunk, 1]);
            await fs.writeFile(path.join(output, `${id}-input.f32`), inputValues, { flag: 'wx' });
            await fs.writeFile(path.join(output, `${id}-down-input.f32`), downValues, { flag: 'wx' });
            await fs.writeFile(path.join(output, `${id}-route-ids.u32`), ids, { flag: 'wx' });
            await fs.writeFile(path.join(output, `${id}-schedule.u32`), schedule, { flag: 'wx' });
            const gates = {
              gateUp: await compare(`${id}-gate-up`, gu, guBaseline, T * topK * I * 2),
              down: await compare(`${id}-down`, down, downBaseline, T * topK * H),
              activation: await compare(`${id}-activation`, activated, activatedBaseline, T * topK * I),
              chainedDown: await compare(`${id}-chained-down`, chainedDown, chainedDownBaseline, T * topK * H),
              scatter: await compare(`${id}-scatter`, scattered, scatteredBaseline, T * H),
              productionGroupedPrefill: await compare(`${id}-production`, production, scatteredBaseline, T * H),
            };
            const receipt = { id, dtype, H, I, T, E, topK, recorded, sizes, chunks: chunk,
              weights: `${dtype}-h${H}-i${I}-weights.json`, gates,
              passed: Object.values(gates).every(gate => gate.passed) };
            report.receipts.push(receipt);
            await save(`${id}.json`, receipt); await save('receipt.json', report);
            console.log(JSON.stringify({ case: id, passed: receipt.passed,
              mismatches: Object.fromEntries(Object.entries(gates).map(([name, gate]) => [name, gate.mismatches])) }));
          } finally {
            if (referenceRecorder) await referenceRecorder.abort();
            if (recorder) await recorder.abort();
            await releaseSince(mark);
          }
        }
        if (dtype === 'q4_0' && H === 96) {
          const mark = held.length;
          const input = createTensor(upload(new Float32Array(H), 'ordered_failure_input'), 'f32', [1, H]);
          const indices = upload(Uint32Array.from({ length: topK }, (_, i) => i), 'ordered_failure_indices');
          const routingWeights = upload(new Float32Array(topK), 'ordered_failure_weights');
          const before = pool.getBufferPool().getStats().activeBuffers;
          const recorder = createCommandRecorder('ordered-injected-failure');
          const leave = enterKernelRegistry(createKernelRegistry({ validators: {
            moe_grouped_gu: { native_f32_ordered: { id: 'ordered-fixture-injected-failure',
              validate() { throw new Error('injected ordered GU failure'); } } },
          } }));
          try {
            await assert.rejects(executeGroupedPrefill(recorder, { inputTensor: input,
              indicesBuffer: indices, weightsBuffer: routingWeights, weights: { ...weights, numExperts: E },
              numTokens: 1, topK, hiddenSize: H, intermediateSize: I, swigluLimit: null,
              kernelPath, phase: 'prefill', layerIdx: 0 }), /injected ordered GU failure/);
          } finally { leave(); await recorder.abort(); }
          assert.equal(pool.getBufferPool().getStats().activeBuffers, before,
            'Ordered recorded failure must return every owned schedule, indirect and output allocation after abort');
          report.recordedFailureCleanup = { passed: true, before, after: pool.getBufferPool().getStats().activeBuffers };
          await save('receipt.json', report);
          await releaseSince(mark);
        }
        await releaseSince(bankMark);
      }
    }
    assert.deepEqual(errors, [], 'GPU validation errors');
    assert.deepEqual(await sourceInventory(controlRoot), expectedControl, 'Frozen baseline changed during run');
    assert.deepEqual(await sourceInventory(repository), candidateInventory, 'Candidate changed during run');
    report.status = report.receipts.every(receipt => receipt.passed) ? 'pass' : 'fail';
    await save('receipt.json', report);
    assert.equal(report.status, 'pass', 'Ordered component bitwise gates failed; retain all raw receipts');
    return report;
  } catch (error) {
    report.status = 'fail'; report.error = String(error?.stack ?? error); report.gpuErrors = errors;
    await save('receipt.json', report); throw error;
  } finally {
    await releaseSince(0);
    baselinePool.destroyBufferPool(device);
    device.removeEventListener('uncapturederror', onError);
  }
}
