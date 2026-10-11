export function matchesStopSequence(
  tokenizer: { decode(ids: number[], skipSpecialTokens: boolean): string },
  generatedIds: readonly number[], start: number, sequences: readonly string[],
  reasoningOutput?: import('../../../config/reasoning-output.js').ReasoningOutputContract | null
): boolean;
