// Reduce the retained timestamp intervals without changing their coverage.
import fs from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';

const receipt = JSON.parse(gunzipSync(await fs.readFile(new URL('./receipt.json.gz', import.meta.url))));
const stats = receipt.rows.find(row => row.stage === 'profile').stats;
const phases = {};
for (const phase of ['prefill', 'decode']) {
  const result = { computePasses: 0, measuredPasses: 0, unmeasuredPasses: 0,
    measuredPassesWithUnknownShader: 0, measuredMs: 0 };
  const groups = new Map();
  for (const step of stats[`${phase}ProfileSteps`]) {
    const profile = step.recorderStats.profile;
    for (const key of ['computePasses', 'measuredPasses', 'unmeasuredPasses']) result[key] += profile[key];
    result.measuredMs += profile.intervals.reduce((sum, interval) => sum + interval.durationMs, 0);
    for (const interval of profile.intervals) {
      if (!interval.dispatch?.pipeline) result.measuredPassesWithUnknownShader++;
      const identity = { phase, pipeline: interval.dispatch?.pipeline ?? null,
        pipelineLabel: interval.dispatch?.pipelineLabel ?? null,
        workgroups: interval.dispatch?.workgroups ?? null };
      const key = JSON.stringify(identity);
      if (!groups.has(key)) groups.set(key, { ...identity, operationLabels: [], count: 0, durationMs: 0 });
      const group = groups.get(key);
      if (!group.operationLabels.includes(interval.label)) group.operationLabels.push(interval.label);
      group.count = interval.count == null ? null : group.count == null ? null : group.count + interval.count;
      group.durationMs += interval.durationMs;
    }
  }
  result.dominant = [...groups.values()].sort((a, b) => b.durationMs - a.durationMs).slice(0, 8);
  phases[phase] = result;
}
console.log(JSON.stringify({
  scope: 'Single instrumented short-prompt development request on M3 Metal. Not complete-request optimization or model correctness qualification.',
  exactOutputs: receipt.result.exactOutputs,
  phases,
  candidate: null,
  disposition: 'No optimization selected: incomplete decode coverage and instrumentation-distorted execution do not establish avoidable cost.',
}, null, 2));
