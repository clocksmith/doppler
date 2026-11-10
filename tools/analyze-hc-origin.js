#!/usr/bin/env node
// Offline first-HC arithmetic attribution. Never opens a GPU or qualifies a model.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import {
  compareF32Bits,
  witnessHyperConnectionNorm,
  witnessMatmulRowSubgroupWave64,
  witnessLowRankSilu,
  witnessMixSigmoid,
  witnessHyperConnectionBranch,
  witnessInjectGate,
} from "./layer25-wgsl-witness-core.js";

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const STAGES = [
  "norm",
  "down",
  "lowRank",
  "up",
  "mix",
  "branchInput",
  "logits",
  "gate",
];

function difference(actual, reference) {
  const bits = compareF32Bits(actual, reference);
  let squared = 0,
    referenceSquared = 0;
  for (let i = 0; i < actual.length; i++) {
    squared += (actual[i] - reference[i]) ** 2;
    referenceSquared += reference[i] ** 2;
  }
  return {
    ...bits,
    differenceRms: Math.sqrt(squared / actual.length),
    sourceRms: Math.sqrt(referenceSquared / actual.length),
    relativeRms:
      referenceSquared === 0 ? null : Math.sqrt(squared / referenceSquared),
  };
}

export function analyzeHcOrigin(packet, values) {
  assert.equal(packet.schema, "doppler.hc-origin-witness/v1");
  assert.equal(packet.qualificationAuthority, false);
  assert.equal(packet.arithmetic, "amd-wave64-ascending-dpp");
  const { rows, streamCount, streamWidth, projectionRank, normEps } =
    packet.geometry;
  for (const n of [rows, streamCount, streamWidth, projectionRank])
    assert(Number.isSafeInteger(n) && n > 0);
  assert(streamCount > 1 && streamWidth % 4 === 0 && projectionRank % 4 === 0);
  assert(Number.isFinite(normEps) && normEps > 0);
  assert.equal(packet.positions.length, rows);
  assert.equal(new Set(packet.positions).size, rows);
  assert(
    Number.isSafeInteger(packet.originalRows) && packet.originalRows >= rows,
  );
  assert(
    packet.positions.every(
      (p) => Number.isSafeInteger(p) && p >= 0 && p < packet.originalRows,
    ),
  );
  const width = streamCount * streamWidth;
  const dimensions = {
    input: rows * width,
    normWeight: width,
    downWeight: projectionRank * width,
    upWeight: width * projectionRank,
    injectWeight: streamCount * width,
  };
  for (const [name, length] of Object.entries(dimensions)) {
    assert(
      values[name] instanceof Float32Array && values[name].length === length,
      `Invalid ${name} shape`,
    );
  }
  const columns = {
    norm: width,
    down: projectionRank,
    lowRank: projectionRank,
    up: width,
    mix: width,
    branchInput: streamWidth,
    logits: streamCount,
    gate: streamCount,
  };
  for (const stage of STAGES)
    for (const prefix of ["gpu", "source"]) {
      assert(
        values[`${prefix}.${stage}`] instanceof Float32Array &&
          values[`${prefix}.${stage}`].length === rows * columns[stage],
        `Invalid ${prefix}.${stage} shape`,
      );
    }
  for (const array of Object.values(values))
    assert(array.every(Number.isFinite), "Nonfinite witness payload");
  const normOptions = { streamCount, streamWidth, eps: normEps };
  const norm = witnessHyperConnectionNorm(
    values.input,
    values.normWeight,
    rows,
    normOptions,
  );
  const down = witnessMatmulRowSubgroupWave64(
    norm,
    rows,
    width,
    values.downWeight,
    projectionRank,
  );
  const lowRank = witnessLowRankSilu(down, streamCount);
  const up = witnessMatmulRowSubgroupWave64(
    lowRank,
    rows,
    projectionRank,
    values.upWeight,
    width,
  );
  const mix = witnessMixSigmoid(up);
  const branchInput = witnessHyperConnectionBranch(
    norm,
    mix,
    rows,
    streamCount,
    streamWidth,
  );
  const logits = witnessMatmulRowSubgroupWave64(
    norm,
    rows,
    width,
    values.injectWeight,
    streamCount,
  );
  const gate = witnessInjectGate(logits, streamCount);
  const chain = { norm, down, lowRank, up, mix, branchInput, logits, gate };
  const stages = STAGES.map((stage) => ({
    stage,
    gpu: compareF32Bits(chain[stage], values[`gpu.${stage}`]),
    source: difference(chain[stage], values[`source.${stage}`]),
  }));
  const earliestSampledDifference = packet.positions.map((position, row) => {
    for (const stage of STAGES) {
      const start = row * columns[stage],
        end = start + columns[stage];
      const comparison = difference(
        values[`gpu.${stage}`].subarray(start, end),
        values[`source.${stage}`].subarray(start, end),
      );
      if (!comparison.allExact) return { position, stage, ...comparison };
    }
    return { position, stage: null };
  });
  const wrongOrder = witnessMatmulRowSubgroupWave64(
    norm,
    rows,
    width,
    values.downWeight,
    projectionRank,
    { reductionOrder: "top-down" },
  );
  const wrongNorm = witnessHyperConnectionNorm(
    values.input,
    values.normWeight,
    rows,
    { ...normOptions, weightOffset: false },
  );
  const corrupted = values["gpu.down"].slice();
  new Uint32Array(corrupted.buffer)[0] ^= 1;
  const negatives = {
    reversedReduction: compareF32Bits(wrongOrder, values["gpu.down"]),
    missingNormOffset: compareF32Bits(wrongNorm, values["gpu.norm"]),
    corruptedOutput: compareF32Bits(down, corrupted),
  };
  return {
    schema: "doppler.hc-origin-witness-result/v1",
    scope:
      "Chained backend arithmetic on sampled first-HC rows; no intermediate resets or model acceptance",
    normalModelQualified: false,
    qualificationAuthority: false,
    passed: stages.every((s) => s.gpu.allExact),
    geometry: packet.geometry,
    positions: packet.positions,
    stages,
    earliestSampledDifference,
    negativeControls: negatives,
    newGpuInvocations: 0,
    runtimeChanged: false,
    acceptanceChanged: false,
  };
}

export function readHcOriginPacket(protocolPath, dataDirectory) {
  const bytes = fs.readFileSync(protocolPath),
    packet = JSON.parse(bytes);
  const values = {},
    payloadIdentities = {};
  for (const [name, record] of Object.entries(packet.payloads)) {
    assert.equal(
      path.basename(record.file),
      record.file,
      "Payload paths must be local basenames",
    );
    const raw = fs.readFileSync(path.join(dataDirectory, record.file));
    assert.match(record.sha256, /^[a-f0-9]{64}$/);
    assert.equal(sha(raw), record.sha256, `Payload identity: ${name}`);
    assert.equal(raw.length, record.bytes, `Payload bytes: ${name}`);
    assert.equal(raw.length % 4, 0);
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    values[name] = Float32Array.from({ length: raw.length / 4 }, (_, i) =>
      view.getFloat32(i * 4, true),
    );
    payloadIdentities[name] = { sha256: record.sha256, bytes: raw.length };
  }
  return {
    packet,
    values,
    identities: { protocolSha256: sha(bytes), payloads: payloadIdentities },
  };
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  const [protocolPath, dataDirectory, outputPath, ...extra] =
    process.argv.slice(2);
  assert(
    protocolPath && dataDirectory && outputPath && !extra.length,
    "Usage: node tools/analyze-hc-origin.js PROTOCOL.json DATA_DIRECTORY NEW_OUTPUT.json",
  );
  const { packet, values, identities } = readHcOriginPacket(
    protocolPath,
    dataDirectory,
  );
  const result = {
    ...analyzeHcOrigin(packet, values),
    identities,
    bindings: packet.bindings,
  };
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2) + "\n", {
    flag: "wx",
  });
  console.log(
    JSON.stringify({
      passed: result.passed,
      earliestSampledDifference: result.earliestSampledDifference,
    }),
  );
  if (!result.passed) process.exitCode = 1;
}
