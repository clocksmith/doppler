export type CompiledTokenizerNormalizer = (text: string) => string;

/** Compile admitted tokenizer.json text normalization once at load time.
 * Normalization applies independently to non-special text segments. Unsupported
 * or malformed declarations reject loading rather than silently changing tokens.
 */
export function compileTokenizerNormalizer(spec: unknown): CompiledTokenizerNormalizer | null;
