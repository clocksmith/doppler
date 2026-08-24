/**
 * Create a logit mask function that enforces a soft TSV grammar.
 *
 * @param {{
 *   tokenizer?: { decode(ids: number[], skipSpecial?: boolean, skipBos?: boolean): string } | null,
 *   fieldsPerLine?: number,
 *   cacheBudget?: number,
 * }} [opts]
 * @returns {(logits: Float32Array, context: { generatedIds: number[], tokenizer?: unknown, vocabSize?: number }) => void}
 */
export function createTsvGrammarMask(opts?: {
    tokenizer?: {
        decode(ids: number[], skipSpecial?: boolean, skipBos?: boolean): string;
    } | null;
    fieldsPerLine?: number;
    cacheBudget?: number;
}): (logits: Float32Array, context: {
    generatedIds: number[];
    tokenizer?: unknown;
    vocabSize?: number;
}) => void;
