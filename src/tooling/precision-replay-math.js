import { f16ToF32Bits, f32ToF16Bits } from '../inference/kv-cache/types.js';

export const PRECISION_REPLAY_MODES = Object.freeze([
  'exact',
  'f32_forward',
  'f32_reverse',
  'f32_tree64',
  'f16_forward',
]);

const f32Scratch = new Float32Array(1);

export function roundF32(value) {
  f32Scratch[0] = value;
  return f32Scratch[0];
}

export function roundF16(value) {
  return f16ToF32Bits(f32ToF16Bits(value));
}

export function exactDot(hidden, weights) {
  let acc = 0;
  for (let i = 0; i < hidden.length; i += 1) {
    acc += hidden[i] * weights[i];
  }
  return acc;
}

export function forwardDotF32(hidden, weights) {
  let acc = 0;
  for (let i = 0; i < hidden.length; i += 1) {
    acc = roundF32(acc + roundF32(hidden[i] * weights[i]));
  }
  return acc;
}

export function reverseDotF32(hidden, weights) {
  let acc = 0;
  for (let i = hidden.length - 1; i >= 0; i -= 1) {
    acc = roundF32(acc + roundF32(hidden[i] * weights[i]));
  }
  return acc;
}

export function tree64DotF32(hidden, weights) {
  const width = 64;
  const partial = new Float32Array(width);
  for (let lane = 0; lane < width; lane += 1) {
    let acc = 0;
    for (let i = lane; i < hidden.length; i += width) {
      acc = roundF32(acc + roundF32(hidden[i] * weights[i]));
    }
    partial[lane] = acc;
  }
  for (let stride = width >> 1; stride > 0; stride >>= 1) {
    for (let lane = 0; lane < stride; lane += 1) {
      partial[lane] = roundF32(partial[lane] + partial[lane + stride]);
    }
  }
  return partial[0];
}

export function forwardDotF16(hidden, weights) {
  let acc = 0;
  for (let i = 0; i < hidden.length; i += 1) {
    const product = roundF16(roundF16(hidden[i]) * roundF16(weights[i]));
    acc = roundF16(roundF16(acc) + product);
  }
  return acc;
}

export function scoreDotProductModes(hidden, weights) {
  return {
    exact: exactDot(hidden, weights),
    f32_forward: forwardDotF32(hidden, weights),
    f32_reverse: reverseDotF32(hidden, weights),
    f32_tree64: tree64DotF32(hidden, weights),
    f16_forward: forwardDotF16(hidden, weights),
  };
}

export function buildModeScoreMaps(hidden, rows) {
  const perModeScores = {
    exact: new Map(),
    f32_forward: new Map(),
    f32_reverse: new Map(),
    f32_tree64: new Map(),
    f16_forward: new Map(),
  };
  for (const [tokenId, row] of rows.entries()) {
    const scores = scoreDotProductModes(hidden, row);
    for (const mode of PRECISION_REPLAY_MODES) {
      perModeScores[mode].set(tokenId, scores[mode]);
    }
  }
  return perModeScores;
}

export function sortTokenIdsByScore(tokenIds, scoreMap) {
  return [...tokenIds].sort((left, right) => {
    const delta = scoreMap.get(right) - scoreMap.get(left);
    if (delta !== 0) {
      return delta;
    }
    return left - right;
  });
}

export function summarizeRanking(tokenIds, scoreMap, decodeToken, limit) {
  const ranked = sortTokenIdsByScore(tokenIds, scoreMap);
  const topIds = ranked.slice(0, Math.min(limit, ranked.length));
  const winner = topIds[0] ?? null;
  const runnerUp = topIds[1] ?? null;
  return {
    winnerTokenId: winner,
    winnerText: winner == null ? null : decodeToken(winner),
    winnerScore: winner == null ? null : scoreMap.get(winner),
    winnerGap: winner == null || runnerUp == null ? null : scoreMap.get(winner) - scoreMap.get(runnerUp),
    top: topIds.map((tokenId, index) => ({
      rank: index + 1,
      tokenId,
      text: decodeToken(tokenId),
      score: scoreMap.get(tokenId),
    })),
  };
}

export function computeInversionCount(tokenIds, leftScoreMap, rightScoreMap) {
  const leftRanked = sortTokenIdsByScore(tokenIds, leftScoreMap);
  const rightRank = new Map(sortTokenIdsByScore(tokenIds, rightScoreMap).map((tokenId, index) => [tokenId, index]));
  let inversions = 0;
  for (let i = 0; i < leftRanked.length; i += 1) {
    for (let j = i + 1; j < leftRanked.length; j += 1) {
      if ((rightRank.get(leftRanked[i]) ?? 0) > (rightRank.get(leftRanked[j]) ?? 0)) {
        inversions += 1;
      }
    }
  }
  return inversions;
}

export function compareTokenSequences(left, right) {
  const maxLength = Math.max(left.length, right.length);
  const differingSteps = [];
  for (let i = 0; i < maxLength; i += 1) {
    if (left[i] !== right[i]) {
      differingSteps.push(i);
    }
  }
  let healedAtStep = null;
  if (differingSteps.length > 0) {
    for (let start = differingSteps[0] + 1; start < maxLength; start += 1) {
      let suffixEqual = true;
      for (let i = start; i < maxLength; i += 1) {
        if (left[i] !== right[i]) {
          suffixEqual = false;
          break;
        }
      }
      if (suffixEqual) {
        healedAtStep = start;
        break;
      }
    }
  }
  return {
    firstDifferentStep: differingSteps[0] ?? null,
    lastDifferentStep: differingSteps[differingSteps.length - 1] ?? null,
    differingStepCount: differingSteps.length,
    healedAtStep,
    persistsThroughEnd: differingSteps.length > 0 && healedAtStep == null,
  };
}
