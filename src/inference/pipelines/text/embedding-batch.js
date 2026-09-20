import { assertNotAborted } from './abort-contract.js';

/**
 * Scheduling owns no model/device/configuration state. The injected operation
 * retains its own session and execution scope for each prompt.
 * @template T
 * @param {readonly string[]} prompts
 * @param {{ signal?: AbortSignal }} request
 * @param {(prompt: string) => Promise<T>} execute
 * @returns {Promise<T[]>}
 */
export async function executeEmbeddingBatch(prompts, request, execute) {
  if (!Array.isArray(prompts)) throw new Error('embedBatch expects an array of prompts');
  const inputs = [...prompts];
  assertNotAborted(request.signal);
  const outputs = [];
  for (const prompt of inputs) {
    assertNotAborted(request.signal);
    outputs.push(await execute(prompt));
  }
  assertNotAborted(request.signal);
  return outputs;
}
