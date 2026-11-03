import assert from 'node:assert/strict';
import { PipelineGenerator } from '../../src/inference/pipelines/text/generator.js';

// Warmup must not inflate the next request's replay time, and an EOS at prefill
// must not falsely report that the prior generation's cached program ran.
const retainedProgram = { ready: true };
let ringResets = 0;
const state = {
  stats: { wholeTokenProgramWallMs: 400, wholeTokenProgram: { replayedTokens: 127 } },
  wholeTokenProgram: { program: retainedProgram },
  executionPlanState: { primaryPlan: { id: 'primary' }, activePlanId: 'primary' },
  decodeRing: { reset() { ringResets++; } },
};
const generator = { _state: state };
const reset = () => PipelineGenerator.prototype._resetDecodeRuntimeState.call(generator);
reset();
assert.equal(state.stats.wholeTokenProgramWallMs, 0);
assert.equal(Object.hasOwn(state.stats, 'wholeTokenProgram'), false);
assert.equal(state.wholeTokenProgram.program, retainedProgram, 'Measurements reset without discarding prepared GPU resources');
state.stats.wholeTokenProgramWallMs += 12;
state.stats.wholeTokenProgram = { replayedTokens: 2 };
reset();
assert.equal(state.stats.wholeTokenProgramWallMs, 0, 'Reused generations start a fresh wall-time epoch');
assert.equal(Object.hasOwn(state.stats, 'wholeTokenProgram'), false, 'A first-token EOS cannot report prior replay');
assert.equal(state.wholeTokenProgram.program, retainedProgram);
assert.equal(ringResets, 2);
console.log('generator-runtime-stats: replay measurements reset without discarding the prepared program');
