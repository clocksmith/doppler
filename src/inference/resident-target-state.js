import { contractAssert as assert, immutableRecord, metadataDigest } from '../config/canonical-contract-codec.js';
import { validateCanonicalGenerationPolicy } from '../config/canonical-generation-policy.js';
import { sampleCanonicalGenerationStep } from './generation-step.js';
import { admitResidentPrompt, bootstrapResidentCursor, admitResidentBlock, decideResidentBlock,
  canonicalStopReason, validateResidentCursor } from './resident-state-contract.js';

/** @type {import('./resident-target-state.js').createResidentTargetState} */
export function createResidentTargetState(prompt, policy, vocabSize, generationId, output) {
  policy = validateCanonicalGenerationPolicy(policy, vocabSize);
  const admission = admitResidentPrompt(prompt, policy, vocabSize, generationId);
  assert('kind' in admission, 'Admission-only result cannot allocate target state.');
  return createTarget([...admission.tokenIds], null, policy, vocabSize, generationId, output);
}

/** @param {number[]} history
 * @param {Readonly<import('./resident-state-contract.js').ResidentCursor> | null} cursor
 * @param {Readonly<import('../config/generation-contract.js').CanonicalGenerationPolicy>} policy
 * @param {number} vocabSize @param {string} generationId
 * @param {import('./resident-output-state.js').ResidentOutputState} output
 * @returns {import('./resident-target-state.js').ResidentTargetState} */
function createTarget(history, cursor, policy, vocabSize, generationId, output) {
  // A failed in-place ordinary transition interrupts the attempt. Scratch
  // failures likewise cannot expose a half-decoded reusable successor.
  let failed = false;
  const live = () => assert(!failed, 'Target state is interrupted.');
  /** @param {Float32Array} logits @param {readonly number[]} context @param {number} index */
  const select = (logits, context, index) => {
    assert(logits instanceof Float32Array && logits.length === vocabSize, 'Target logits vocabulary mismatch.');
    return sampleCanonicalGenerationStep(logits, context, policy, index);
  };
  /** @param {string} expectedCursorId */
  const checkParent = expectedCursorId => {
    live();
    assert(cursor !== null && cursor.cursorId === expectedCursorId, 'Target cursor mismatch.');
    return cursor;
  };
  /** @param {import('./resident-state-contract.js').ResidentBlockDecision} decision @param {string} delta */
  const install = (decision, delta) => {
    let successor = decision.successor;
    if (successor.stopReason) {
      delta += output.finalize(successor.stopReason);
      // A normal decoder flush can complete a textual stop (for example a
      // replacement character from incomplete UTF-8). Rebind before exposing
      // the successor; EOS/token/tool priority remains unchanged.
      const stopReason = canonicalStopReason(policy, successor.pendingTokenId,
        successor.acceptedGeneratedTokenCount, successor.promptTokenCount,
        successor.materializedTokenCount, output.getUsage().stopped);
      if (stopReason !== successor.stopReason) {
        const { cursorId, ...body } = successor;
        body.stopReason = stopReason;
        successor = validateResidentCursor({ ...body, cursorId: metadataDigest(body.schema, body) }, policy, vocabSize);
      }
    }
    const result = immutableRecord({ acceptedTokenIds: decision.acceptedTokenIds,
      tokenRange: decision.tokenRange, successor, delta, outputDigest: output.digest() });
    cursor = successor;
    return result;
  };
  return {
    getCursor() { live(); return cursor; },
    getHistory() { live(); return Object.freeze([...history]); },
    bootstrap(logits) {
      live();
      assert(cursor === null, 'Target bootstrap already completed.');
      try {
        const token = select(logits, history, 0);
        const decoded = output.accept(token);
        const successor = bootstrapResidentCursor(history, token, policy, vocabSize, generationId, decoded.textStop);
        return install({ acceptedTokenIds: [token], acceptedProposalCount: 0, materializeInputCount: 0,
          tokenRange: { start: 0, end: 1 }, successor }, decoded.delta);
      } catch (error) { failed = true; throw error; }
    },
    advance(expectedCursorId, logits) {
      const parent = checkParent(expectedCursorId);
      const inputs = admitResidentBlock(parent, [], policy, vocabSize);
      try {
        const token = select(logits, [...history, ...inputs], parent.nextTargetTokenIndex);
        const decoded = output.accept(token);
        const decision = decideResidentBlock(parent, history, [], [token], [decoded.textStop], policy, vocabSize);
        const result = install(decision, decoded.delta);
        history.push(...inputs);
        return result;
      } catch (error) { failed = true; throw error; }
    },
    acceptTransition(expectedCursorId, transition) {
      live();
      assert((cursor?.cursorId ?? null) === expectedCursorId, 'Target cursor mismatch.');
      try {
        assert(transition?.acceptedTokenIds?.length === 1, 'Ordinary transition accepts exactly one target token.');
        const token = transition.acceptedTokenIds[0];
        const decoded = output.accept(token);
        let decision;
        /** @type {number[]} */
        let inputs = [];
        if (cursor) {
          inputs = [...admitResidentBlock(cursor, [], policy, vocabSize)];
          decision = decideResidentBlock(cursor, history, [], [token], [decoded.textStop], policy, vocabSize);
        } else {
          decision = { acceptedTokenIds: [token], acceptedProposalCount: 0, materializeInputCount: 0, tokenRange: { start: 0, end: 1 },
            successor: bootstrapResidentCursor(history, token, policy, vocabSize, generationId, decoded.textStop) };
        }
        const accepted = install(decision, decoded.delta);
        assert(metadataDigest('doppler.resident-target-transition/v1', accepted)
          === metadataDigest('doppler.resident-target-transition/v1', transition), 'Target transition differs from accepted token/decoder state.');
        history.push(...inputs);
        return accepted;
      } catch (error) { failed = true; throw error; }
    },
    clone() {
      live();
      assert(cursor !== null, 'Only a settled target cursor can be cloned.');
      return createTarget([...history], cursor, policy, vocabSize, generationId, output.clone());
    },
    verifyBlock(expectedCursorId, proposal, rows) {
      const parent = checkParent(expectedCursorId);
      const inputs = admitResidentBlock(parent, proposal, policy, vocabSize);
      assert(proposal.length > 0 && Array.isArray(rows) && rows.length === inputs.length,
        'Speculative target verification requires one bounded logit row per input.');
      try {
        const choices = [], stops = [], context = [...history];
        let delta = '';
        for (let j = 0; j < rows.length; j++) {
          context.push(inputs[j]);
          const token = select(rows[j], context, parent.nextTargetTokenIndex + j);
          const decoded = output.accept(token);
          choices.push(token); stops.push(decoded.textStop); delta += decoded.delta;
          if (canonicalStopReason(policy, token, parent.acceptedGeneratedTokenCount + choices.length,
            parent.promptTokenCount, parent.materializedTokenCount + choices.length, decoded.textStop)
            || j === proposal.length || token !== proposal[j]) break;
        }
        const decision = decideResidentBlock(parent, history, proposal, choices, stops, policy, vocabSize);
        const result = install(decision, delta);
        history.push(...inputs.slice(0, decision.materializeInputCount));
        return immutableRecord({ ...result, decision: { ...decision, successor: result.successor } });
      } catch (error) { failed = true; throw error; }
    },
  };
}
