import { evaluateWord } from './whole-token-template.js';
import { U32 } from './whole-token-observation.js';

/** @typedef {import('./recorded-step-executor.js').RecordedStepState} RecordedStepState */

/** Encode the observed commands in order; copies delimit compute passes.
 * @param {RecordedStepState} program @param {number} pos @param {number} stagingIndex
 * @param {import('./recorded-step-executor.js').BatchFeedback | null} [batch]
 * @param {((encoder: GPUCommandEncoder) => void) | null} [prefix] */
export function encodeRecordedStep(program, pos, stagingIndex, batch = null, prefix = null) {
  const encoder = program.device.createCommandEncoder({ label: `whole_token_${pos}` });
  prefix?.(encoder);
  /** @type {GPUComputePassEncoder | null} */
  let pass = null;
  /** @type {GPUComputePipeline | null} */
  let lastPipeline = null;
  if (batch?.feedback) {
    for (const field of program.dynamicUniformWords) {
      if (field.model.kind === 'token') encoder.copyBufferToBuffer(batch.feedback, 0, program.uniformBuffer, field.index * U32, U32);
    }
    for (const write of program.writes) {
      for (const field of write.dynamic) {
        if (field.model.kind === 'token') encoder.copyBufferToBuffer(batch.feedback, 0, write.buffer, write.offset + field.word * U32, U32);
      }
    }
  }
  for (const step of program.steps) {
    if (step.kind === 'copy' || step.kind === 'clear') {
      if (pass) { pass.end(); pass = null; lastPipeline = null; }
      if (step.kind === 'copy') {
        encoder.copyBufferToBuffer(step.src, evaluateWord(step.srcOffset, pos, 0), step.dst,
          evaluateWord(step.dstOffset, pos, 0), step.size);
      }
      else encoder.clearBuffer(step.buffer, step.offset, step.size);
      continue;
    }
    if (!pass) pass = encoder.beginComputePass({ label: 'whole_token_pass' });
    if (step.pipeline !== lastPipeline) { pass.setPipeline(step.pipeline); lastPipeline = step.pipeline; }
    for (let g = 0; g < step.groups.length; g++) if (step.groups[g]) pass.setBindGroup(g, step.groups[g]);
    if (step.kind === 'dispatch') pass.dispatchWorkgroups(step.workgroups[0], step.workgroups[1], step.workgroups[2]);
    else pass.dispatchWorkgroupsIndirect(step.indirect, step.indirectOffset);
  }
  if (pass) pass.end();
  const staging = batch?.staging ?? program.staging[stagingIndex];
  if (batch?.saveFeedback) {
    const sample = program.readback.find(copy => copy.dstOffset === 0);
    if (!sample || sample.size < U32) throw new Error('[WholeTokenProgram] Missing declared sampled-token readback.');
    encoder.copyBufferToBuffer(sample.src, sample.srcOffset, batch.saveFeedback, 0, U32);
  }
  for (const c of program.readback) encoder.copyBufferToBuffer(c.src, c.srcOffset, staging, (batch?.offset ?? 0) + c.dstOffset, c.size);
  return { pos, staging, commandBuffer: encoder.finish() };
}

/** Update declared scalar controls only; tensor computation stays on the GPU.
 * @param {RecordedStepState} program @param {number} pos @param {number} token */
export function writeRecordedStepInputs(program, pos, token) {
  const queue = program.device.queue;
  if (program.dynamicUniformWords.length > 0) {
    for (const { index, model } of program.dynamicUniformWords) program.uniformWords[index] = evaluateWord(model, pos, token);
    queue.writeBuffer(program.uniformBuffer, 0, program.uniformMirror, 0, program.dynamicUniformBytes);
  }
  for (const write of program.writes) {
    if (write.dynamic.length > 0) {
      const words = new Uint32Array(write.bytes.buffer, write.bytes.byteOffset, write.bytes.byteLength / U32);
      for (const { word, model } of write.dynamic) words[word] = evaluateWord(model, pos, token);
    }
    queue.writeBuffer(write.buffer, write.offset, write.bytes);
  }
}

