export function resolveMoEActiveExpertSelection(selection) {
  if (selection === 'all' || selection === 'topk-readback' || selection === 'topk-route') {
    return selection;
  }
  throw new Error(
    '[MoE] runtime.inference.moe.routing.activeExpertSelection must be ' +
    `"all", "topk-readback", or "topk-route", got ${String(selection)}.`
  );
}

export function buildActiveExpertScheduleFromIndices(
  indices,
  numExperts,
  maxTokensPerExpert,
  selection = 'topk-readback'
) {
  const tokenCounts = new Uint32Array(numExperts);

  for (let i = 0; i < indices.length; i++) {
    const expertIdx = indices[i];
    if (expertIdx >= numExperts) {
      throw new Error(
        `[MoE] Top-K routing produced expert index ${expertIdx} outside numExperts=${numExperts}.`
      );
    }
    tokenCounts[expertIdx] += 1;
  }

  const activeExperts = [];
  for (let expertIdx = 0; expertIdx < numExperts; expertIdx++) {
    const count = tokenCounts[expertIdx];
    if (count === 0) {
      continue;
    }
    if (count > maxTokensPerExpert) {
      throw new Error(
        `[MoE] Expert ${expertIdx} received ${count} tokens but maxTokensPerExpert=${maxTokensPerExpert}. ` +
        'Increase runtime.inference.moe.routing.maxTokensPerExpert or its headroom/cap settings.'
      );
    }
    activeExperts.push(expertIdx);
  }

  return { selection, activeExperts, tokenCounts };
}


