// Optional model-owned output semantics; absence preserves existing manifests.
/** @type {import('./reasoning-output.js').validateReasoningOutput} */
export function validateReasoningOutput(value) {
  if (value === undefined || value === null) return value;
  const fields = ['schema', 'openingToken', 'closingToken'];
  const record = /** @type {Record<string, unknown>} */ (value);
  if (typeof value !== 'object' || Array.isArray(value)
    || Object.keys(value).some(key => !fields.includes(key))
    || record.schema !== 'doppler.reasoning-output/v1') {
    throw new Error('inference.output.reasoningOutput requires doppler.reasoning-output/v1.');
  }
  const tokens = ['openingToken', 'closingToken'].map(name => {
    const token = /** @type {Record<string, unknown>} */ (record[name]);
    if (!token || typeof token !== 'object' || Array.isArray(token)
      || Object.keys(token).some(key => !['id', 'text'].includes(key))
      || typeof token.id !== 'number' || !Number.isSafeInteger(token.id) || token.id < 0
      || typeof token.text !== 'string' || !token.text.length) {
      throw new Error(`reasoningOutput.${name} requires an explicit token id and text.`);
    }
    return Object.freeze({ id: token.id, text: token.text });
  });
  if (tokens[0].id === tokens[1].id || tokens[0].text === tokens[1].text) {
    throw new Error('reasoningOutput opening and closing tokens must be distinct.');
  }
  return Object.freeze({ schema: /** @type {const} */ ('doppler.reasoning-output/v1'), openingToken: tokens[0], closingToken: tokens[1] });
}
