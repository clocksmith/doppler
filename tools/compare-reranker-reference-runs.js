#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { evaluateRerankReference } from '../src/config/rerank-reference.js';
import { assertRerankerRunCoverage, buildRerankerReferenceSchedule } from './reranker-reference-schedule.js';

const digest = bytes => `sha256:${createHash('sha256').update(bytes).digest('hex')}`;

// Recompute quality from raw outputs. Qualifier pass flags cannot substitute for
// the frozen source oracle or establish a comparable performance population.
export function auditRerankerReferencePair(doppler, tjs, references) {
  const qualityFailures = [];
  const scopeFailures = [];
  const timingFailures = [];
  const checks = [];
  function check(failures, label, operation) {
    try { operation(); } catch (error) { failures.push({ label, message: error.message }); }
  }
  for (const [engine, report] of Object.entries({ doppler, tjs })) {
    check(qualityFailures, `${engine}.execution`, () => {
      assert.equal(report.passed, true); assert.deepEqual(report.cleanup, []);
      assert.equal(report.phases.length, report.config.repeatRuns);
      assert(report.phases.length > 0);
      assert.deepEqual(report.config.references.map(input => input.digest), references.map(input => input.digest));
      const schedule = buildRerankerReferenceSchedule(report.sampling);
      assert.deepEqual(report.runSchedule, schedule);
      for (const [repeat, phase] of report.phases.entries()) {
        assert.equal(phase.repeat, repeat);
        assertRerankerRunCoverage(phase.observation.runs, schedule, references.length);
        for (const run of phase.observation.runs) {
          const reference = references[run.referenceIndex].reference;
          const input = engine === 'doppler'
            ? { query: run.receipt.evidence.query, documents: run.receipt.evidence.documents }
            : { query: run.query, documents: run.documents };
          const comparison = evaluateRerankReference(reference, { input,
            scoringConfig: phase.observation.scoringConfig, outputs: run.scores });
          checks.push({ engine, repeat, phase: run.phase, iteration: run.iteration,
            referenceIndex: run.referenceIndex, ...comparison });
          assert(comparison.passed, 'Raw output differs from the unchanged source oracle.');
        }
      }
    });
    check(scopeFailures, `${engine}.physical-browser`, () => {
      for (const field of ['memorySamplerDigest', 'scheduleDigest', 'qualifierDigest']) {
        assert.match(report[field], /^sha256:[0-9a-f]{64}$/);
      }
      assert.equal(report.hardware.isFallbackAdapter, false);
      for (const field of ['vendor', 'architecture', 'device', 'description']) {
        assert.equal(typeof report.hardware[field], 'string'); assert(report.hardware[field].length > 0);
      }
      for (const field of ['product', 'revision', 'userAgent', 'jsVersion']) {
        assert.equal(typeof report.browser[field], 'string'); assert(report.browser[field].length > 0);
      }
    });
    check(timingFailures, `${engine}.timed-population`, () => {
      assert(report.sampling !== null, 'Reference qualification is not a timed population.');
      assert.equal(report.config.repeatRuns, 1, 'Paired samples require a fresh browser per opening.');
      assert.deepEqual(report.config.cachePolicy, {
        browser: 'fresh-profile', model: 'first-open', transport: 'local-http', operatingSystem: 'uncontrolled',
      }, 'Cache scope must be explicit; fresh browser profiles do not flush the operating-system cache.');
      for (const phase of report.phases) {
        assert(Number.isFinite(phase.observation.modelLoadMs) && phase.observation.modelLoadMs > 0);
        for (const run of phase.observation.runs) assert(Number.isFinite(run.durationMs) && run.durationMs > 0);
      }
      assert(Number.isFinite(report.peakRendererRssBytes) && report.peakRendererRssBytes > 0);
      assert(Number.isFinite(report.startup.modelReadyMs) && report.startup.modelReadyMs > 0);
      assert(Number.isFinite(report.startup.firstResultMs) && report.startup.firstResultMs > report.startup.modelReadyMs);
      assert.equal(report.memoryError, undefined);
    });
  }
  check(scopeFailures, 'observed-scope-equality', () => {
    assert.equal(doppler.schema, 'doppler.installed-browser-reranker-qualification-result/v1');
    assert.equal(tjs.schema, 'doppler.transformersjs-reranker-qualification-result/v1');
    assert.deepEqual(doppler.hardware, tjs.hardware);
    assert.deepEqual(doppler.browser, tjs.browser);
    assert.deepEqual(doppler.config.launchArgs, tjs.config.launchArgs);
    assert.equal(doppler.config.sampleIntervalMs, tjs.config.sampleIntervalMs);
    assert.equal(doppler.memorySamplerDigest, tjs.memorySamplerDigest);
    assert.equal(doppler.scheduleDigest, tjs.scheduleDigest);
    assert.deepEqual(doppler.sampling, tjs.sampling);
    assert.equal(doppler.startup.scope, tjs.startup.scope);
  });
  check(scopeFailures, 'transformersjs.strict-provider', () => {
    for (const { observation } of tjs.phases) {
      assert.equal(observation.executionProviderMode, 'webgpu-only');
      for (const field of ['fallbackUsed', 'executionProviderFallbackUsed', 'ortProxyFallbackUsed']) assert.equal(observation[field], false);
      assert.equal(observation.requestedDtype, tjs.config.dtype);
      assert.equal(observation.effectiveDtype, tjs.config.dtype);
    }
  });
  return { schema: 'doppler.reranker-reference-pair-audit/v1',
    qualityPassed: qualityFailures.length === 0, scopeMatched: scopeFailures.length === 0,
    timingComparable: qualityFailures.length + scopeFailures.length + timingFailures.length === 0,
    qualityFailures, scopeFailures, timingFailures, checks,
    claimAllowed: false,
    limitations: ['Pair scope alone does not establish a repeated paired experiment.',
      'Cancellation and recovery require separate observed lifecycle evidence.',
      'Different weight encodings are product paths; matching the source oracle does not make artifact bytes identical.'] };
}

async function main() {
  const [dopplerPath, tjsPath, output] = process.argv.slice(2);
  assert(dopplerPath && tjsPath && output && process.argv.length === 5,
    'Usage: node tools/compare-reranker-reference-runs.js <doppler-receipt> <transformersjs-receipt> <new-output.json>');
  const bytes = await Promise.all([dopplerPath, tjsPath].map(file => fs.readFile(file)));
  const [doppler, tjs] = bytes.map(value => JSON.parse(value));
  const references = [];
  for (const input of doppler.config.references) {
    const source = await fs.readFile(input.path); assert.equal(digest(source), input.digest);
    references.push({ digest: input.digest, reference: JSON.parse(source) });
  }
  const audit = auditRerankerReferencePair(doppler, tjs, references);
  const result = { ...audit, createdAtUtc: new Date().toISOString(),
    receipts: [{ path: dopplerPath, digest: digest(bytes[0]) }, { path: tjsPath, digest: digest(bytes[1]) }],
    models: { doppler: { runtime: doppler.installedPackage, capsule: doppler.capsuleDigest }, tjs: tjs.acquisition },
    observed: Object.fromEntries(Object.entries({ doppler, tjs }).map(([engine, report]) => [engine,
      { startup: report.startup, peakRendererRssBytes: report.peakRendererRssBytes,
        phases: report.phases.map(phase => ({ repeat: phase.repeat, modelLoadMs: phase.observation.modelLoadMs,
          runs: phase.observation.runs.map(({ phase, iteration, referenceIndex, durationMs }) => ({ phase, iteration, referenceIndex, durationMs })) })) }])) };
  await fs.writeFile(output, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ qualityPassed: audit.qualityPassed, scopeMatched: audit.scopeMatched,
    timingComparable: audit.timingComparable, claimAllowed: false, output }));
  if (!audit.qualityPassed || !audit.scopeMatched) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
