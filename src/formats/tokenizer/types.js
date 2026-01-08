/**
 * Shared tokenizer/config JSON parsing utilities.
 */

function parseJson(text, label) {
  try {
    return JSON.parse(text);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`[${label}] Failed to parse JSON: ${message}`);
  }
}

export function parseConfigJsonText(text) {
  return parseJson(text, 'config.json');
}

export function parseTokenizerConfigJsonText(text) {
  return parseJson(text, 'tokenizer_config.json');
}

export function parseTokenizerJsonText(text) {
  return parseJson(text, 'tokenizer.json');
}
