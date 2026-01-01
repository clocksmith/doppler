/**
 * Shared tokenizer/config JSON parsing utilities.
 */

function parseJson<T>(text: string, label: string): T {
  try {
    return JSON.parse(text) as T;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`[${label}] Failed to parse JSON: ${message}`);
  }
}

export function parseConfigJsonText(text: string): Record<string, unknown> {
  return parseJson<Record<string, unknown>>(text, 'config.json');
}

export function parseTokenizerConfigJsonText(text: string): Record<string, unknown> {
  return parseJson<Record<string, unknown>>(text, 'tokenizer_config.json');
}

export function parseTokenizerJsonText(text: string): Record<string, unknown> {
  return parseJson<Record<string, unknown>>(text, 'tokenizer.json');
}
