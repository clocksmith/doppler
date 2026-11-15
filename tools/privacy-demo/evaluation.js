/** One public deterministic BFV circuit, shared by the evaluator and the full client replay audit. */
import { counts } from './model.js';
import policy from './policy.json' with { type: 'json' };
export function createScoreEvaluation({ seal, context, own }, keys) {
  const evaluator = own(new seal.Evaluator(context)), encoder = own(new seal.BatchEncoder(context));
  const weights = new BigInt64Array(encoder.slotCount()), n = policy.vocabulary.length;
  for (let row = 0; row < n; row++) for (let column = 0; column < n; column++) weights[row * n + column] = BigInt(counts[row][column]);
  const plainWeights = own(new seal.Plaintext()); encoder.encode(weights, plainWeights);
  const input = own(new seal.Ciphertext()), output = own(new seal.Ciphertext()), rotated = own(new seal.Ciphertext());
  return ciphertext => {
    input.loadFromBase64(context, ciphertext);
    evaluator.multiplyPlain(input, plainWeights, output);
    for (const shift of policy.rotationSteps) { evaluator.rotateRows(output, shift, keys, rotated); evaluator.addInplace(output, rotated); }
    return output.saveToBase64(seal.ComprModeType.none);
  };
}
