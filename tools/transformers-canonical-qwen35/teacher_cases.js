// Application sampling/control only. Each next() receives the engine's own
// sampled token; reference tokens never become model inputs.
export function* teacherCases(teacher, results) {
  for (const prompt of teacher.prompts) {
    const result = { id: prompt.id, promptIds: prompt.promptIds,
      expectedTokenIds: prompt.generatedTokenIds, generatedTokenIds: [], endedByEOS: false,
      expectedEndedByEOS: prompt.endedByEOS, completedBudget: false, exactTokens: false };
    results.push(result);
    for (let step = 0; step < teacher.contract.maxNewTokens; step++) {
      const snapshot = prompt.snapshots.find(value => value.step === step);
      const sampled = yield { name: `teacher-${prompt.id}-${step}`, promptId: prompt.id, step,
        inputIds: step === 0 ? prompt.promptIds : [result.generatedTokenIds.at(-1)],
        positionOffset: step === 0 ? 0 : prompt.promptIds.length + step - 1,
        expectedTokenId: prompt.generatedTokenIds[step],
        teacherPrefixAligned: result.generatedTokenIds.every((token, index) => token === prompt.generatedTokenIds[index]),
        logits: snapshot ? [{ ...snapshot, engine: 'native' }] : [] };
      if (!Number.isSafeInteger(sampled) || sampled < 0) throw new Error('Engine must supply its own sampled token');
      result.generatedTokenIds.push(sampled);
      if (teacher.contract.eosTokenIds.includes(sampled)) {
        result.endedByEOS = true;
        break;
      }
    }
    result.completedBudget = result.generatedTokenIds.length === teacher.contract.maxNewTokens;
    result.exactTokens = result.generatedTokenIds.length === prompt.generatedTokenIds.length &&
      result.generatedTokenIds.every((token, index) => token === prompt.generatedTokenIds[index]);
  }
}
