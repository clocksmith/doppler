import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { analyzeHcOrigin, readHcOriginPacket } from "./analyze-hc-origin.js";

function fixture() {
  // Analytic oracle: RMS(±1) with eps=3 gives invRms=1/2;
  // offset weight 1 gives norm=±1. Zero projections give mix=1/2,
  // branch=±1/2 and injection gate=1. No witness generates expected data.
  const rows = 2,
    streamWidth = 64,
    streamCount = 4,
    width = 256,
    rank = 4;
  const signed = (n) =>
    Float32Array.from({ length: rows * n }, (_, i) => (i < n ? 1 : -1));
  const packet = {
    schema: "doppler.hc-origin-witness/v1",
    qualificationAuthority: false,
    arithmetic: "amd-wave64-ascending-dpp",
    positions: [1717, 2271],
    originalRows: 2304,
    geometry: {
      rows,
      streamWidth,
      streamCount,
      projectionRank: rank,
      normEps: 3,
    },
  };
  const values = {
    input: signed(width),
    normWeight: new Float32Array(width).fill(1),
    downWeight: new Float32Array(rank * width),
    upWeight: new Float32Array(width * rank),
    injectWeight: new Float32Array(streamCount * width),
  };
  const expected = {
    norm: signed(width),
    down: new Float32Array(rows * rank),
    lowRank: new Float32Array(rows * rank),
    up: new Float32Array(rows * width),
    mix: new Float32Array(rows * width).fill(0.5),
    branchInput: Float32Array.from(signed(streamWidth), (v) => v / 2),
    logits: new Float32Array(rows * streamCount),
    gate: new Float32Array(rows * streamCount).fill(1),
  };
  for (const [stage, data] of Object.entries(expected)) {
    values[`gpu.${stage}`] = data.slice();
    values[`source.${stage}`] = data.slice();
  }
  return { packet, values };
}

test("analytic HC oracle preserves explicit epsilon, signs and diagnostic scope", () => {
  const { packet, values } = fixture(),
    result = analyzeHcOrigin(packet, values);
  assert.equal(result.passed, true);
  assert(
    result.stages.every((stage) => stage.gpu.allExact && stage.source.allExact),
  );
  assert(result.earliestSampledDifference.every((row) => row.stage === null));
  assert.equal(result.negativeControls.corruptedOutput.allExact, false);
  assert.equal(result.negativeControls.missingNormOffset.allExact, false);
  // Zero dot products cannot distinguish reduction order; report that honestly.
  assert.equal(result.negativeControls.reversedReduction.allExact, true);
  assert.equal(result.qualificationAuthority, false);
  assert.equal(result.normalModelQualified, false);
  assert.equal(result.newGpuInvocations, 0);
});

test("one-ULP corruption fails arithmetic validation even with a valid payload identity", () => {
  const { packet, values } = fixture();
  new Uint32Array(values["gpu.branchInput"].buffer)[0] ^= 1;
  const result = analyzeHcOrigin(packet, values);
  assert.equal(result.passed, false);
  assert.equal(
    result.stages.find((row) => row.stage === "branchInput").gpu.maxUlp,
    1,
  );
});

test("source differences locate their sampled boundary without becoming an acceptance failure", () => {
  const { packet, values } = fixture();
  values["source.down"][4] = 0.25;
  const result = analyzeHcOrigin(packet, values);
  assert.equal(result.passed, true);
  assert.deepEqual(
    result.earliestSampledDifference.map((row) => [row.position, row.stage]),
    [
      [1717, null],
      [2271, "down"],
    ],
  );
  assert.equal(result.normalModelQualified, false);
});

test("unsupported arithmetic, wrong geometry and nonfinite data fail closed", () => {
  for (const mutate of [
    (p) => {
      p.qualificationAuthority = true;
    },
    (p) => {
      p.arithmetic = "unspecified-gpu";
    },
    (p) => {
      p.geometry.rows = 3;
    },
    (p) => {
      p.positions[1] = p.positions[0];
    },
    (p) => {
      p.positions[1] = p.originalRows;
    },
    (p) => {
      p.geometry.normEps = 0;
    },
  ]) {
    const { packet, values } = fixture();
    mutate(packet);
    assert.throws(() => analyzeHcOrigin(packet, values));
  }
  const { packet, values } = fixture();
  values.input[0] = NaN;
  assert.throws(() => analyzeHcOrigin(packet, values), /Nonfinite/);
});

test("packet loading rejects changed bytes and paths outside the payload directory", () => {
  const directory = fs.mkdtempSync(
    path.join(os.tmpdir(), "doppler-hc-origin-"),
  );
  try {
    const bytes = Buffer.alloc(4);
    bytes.writeFloatLE(1.25);
    fs.writeFileSync(path.join(directory, "input.f32"), bytes);
    const packet = {
      payloads: {
        input: {
          file: "input.f32",
          bytes: 4,
          sha256: createHash("sha256").update(bytes).digest("hex"),
        },
      },
    };
    const protocol = path.join(directory, "packet.json");
    const save = () => fs.writeFileSync(protocol, JSON.stringify(packet));
    save();
    assert.equal(readHcOriginPacket(protocol, directory).values.input[0], 1.25);
    bytes.writeFloatLE(2.5);
    fs.writeFileSync(path.join(directory, "input.f32"), bytes);
    assert.throws(
      () => readHcOriginPacket(protocol, directory),
      /Payload identity/,
    );
    packet.payloads.input.file = "../input.f32";
    save();
    assert.throws(
      () => readHcOriginPacket(protocol, directory),
      /local basenames/,
    );
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
