import { releaseBuffer } from '../../../../memory/buffer-pool.js';
import { GENERATION_CONTRACT, resolveGenerationOptions } from '../../../../config/generation-contract.js';
import { resolvePrefillOptions } from '../generator-runtime.js';
import { assertIncrementalDecodeSupport } from '../generator-decode-policy.js';
import { assertTokenSelectionActive, selectTokenFromGpuLogits } from './token-selection.js';

function requireReady(state, options, operation) {
  if (!state.isLoaded) throw new Error('Model not loaded');
  if (state.isGenerating) throw new Error('Generation already in progress');
  assertIncrementalDecodeSupport(state, operation);
  assertTokenSelectionActive(options.signal);
  const generation = Object.fromEntries(Object.keys(GENERATION_CONTRACT.options)
    .filter(key => options[key] !== undefined).map(key => [key, options[key]]));
  return { ...resolveGenerationOptions(generation), signal: options.signal };
}

export async function prefillWithToken(prompt, options, tokenContract) {
  const sampling = requireReady(this._state, options, 'prefillWithToken');
  this._resetDecodeRuntimeState();
  const opts = { ...resolvePrefillOptions(this._state, options), signal: options.signal, _returnGpuLogits: true };
  const inputIds = this._resolvePromptOrInputIds(prompt, opts.useChatTemplate, 'prefillWithToken', opts.inputIds);
  let result;
  try {
    result = await this._prefillInputIdsToLogits(inputIds, opts);
    return await selectTokenFromGpuLogits(result, inputIds, sampling, tokenContract);
  } finally { if (result?.logitsBuffer) releaseBuffer(result.logitsBuffer); }
}

export async function decodeStepWithToken(currentIds, options, tokenContract) {
  const sampling = requireReady(this._state, options, 'decodeStepWithToken');
  const opts = { ...this._resolveStepOptions(options), signal: options.signal, _returnGpuLogits: true };
  let result;
  try {
    result = await this._decodeStepToLogits(currentIds, opts);
    return await selectTokenFromGpuLogits(result, currentIds, sampling, tokenContract);
  } finally { if (result?.logitsBuffer) releaseBuffer(result.logitsBuffer); }
}
