#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

const DEFAULT_STABLE_ROOT = path.resolve('reports/f16-precision-collapse/stable');
const DEFAULT_DEMO_ROOT = path.resolve('demo/data/f16-precision-collapse');

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function relativePath(fromDir, filePath) {
  return path.relative(fromDir, filePath).split(path.sep).join('/');
}

function selectPromptSummary(prompt) {
  return {
    id: prompt.id,
    text: prompt.text,
    replayValidation: prompt.replayValidation,
    candidateTokenCount: prompt.candidateTokenCount,
    watchTokens: prompt.watchTokens,
    watchPairs: prompt.watchPairs,
    modes: {
      exact: prompt.modes.exact,
      f32_forward: prompt.modes.f32_forward,
      f16_forward: prompt.modes.f16_forward,
    },
    f16VsF32WinnerChanged: prompt.f16VsF32WinnerChanged,
    branchComparison: prompt.branchComparison,
    branches: prompt.branches,
    sliceArtifact: prompt.sliceArtifact ?? null,
  };
}

function selectBroadFlipSummary(flip) {
  return {
    id: flip.id,
    text: flip.text,
    modes: {
      exact: {
        winnerText: flip.modes.exact.winnerText,
        winnerGap: flip.modes.exact.winnerGap,
      },
      f32_forward: {
        winnerText: flip.modes.f32_forward.winnerText,
        winnerGap: flip.modes.f32_forward.winnerGap,
      },
      f16_forward: {
        winnerText: flip.modes.f16_forward.winnerText,
        winnerGap: flip.modes.f16_forward.winnerGap,
      },
    },
    branchComparison: flip.branchComparison,
    sliceArtifact: flip.sliceArtifact ?? null,
  };
}

function main() {
  const stableRoot = path.resolve(process.argv[2] ?? DEFAULT_STABLE_ROOT);
  const demoRoot = path.resolve(process.argv[3] ?? DEFAULT_DEMO_ROOT);
  const curatedRoot = path.join(stableRoot, 'curated');
  const broadRoot = path.join(stableRoot, 'broad');
  const curatedSummaryPath = path.join(curatedRoot, 'summary.json');
  const broadSummaryPath = path.join(broadRoot, 'summary.json');

  if (!fs.existsSync(curatedSummaryPath)) {
    throw new Error(`Missing curated stable summary: ${curatedSummaryPath}`);
  }
  if (!fs.existsSync(broadSummaryPath)) {
    throw new Error(`Missing broad stable summary: ${broadSummaryPath}`);
  }

  const curated = readJson(curatedSummaryPath);
  const broad = readJson(broadSummaryPath);
  const demoCuratedRoot = path.join(demoRoot, 'curated');
  const demoCuratedSlicesRoot = path.join(demoCuratedRoot, 'slices');

  ensureDir(demoRoot);
  ensureDir(demoCuratedRoot);
  ensureDir(demoCuratedSlicesRoot);

  const curatedPrompts = curated.prompts
    .filter((prompt) => typeof prompt.sliceArtifact === 'string' && prompt.sliceArtifact.length > 0)
    .map(selectPromptSummary);

  for (const prompt of curatedPrompts) {
    const sourceSlicePath = path.join(curatedRoot, prompt.sliceArtifact);
    const targetSlicePath = path.join(demoCuratedRoot, prompt.sliceArtifact);
    ensureDir(path.dirname(targetSlicePath));
    fs.copyFileSync(sourceSlicePath, targetSlicePath);
  }

  const demoCuratedSummary = {
    schemaVersion: curated.schemaVersion,
    generatedAtUtc: curated.generatedAtUtc,
    modelId: curated.modelId,
    gpu: curated.gpu,
    aggregate: curated.aggregate,
    prompts: curatedPrompts,
  };

  const demoBroadSummary = {
    schemaVersion: broad.schemaVersion,
    generatedAtUtc: broad.generatedAtUtc,
    modelId: broad.modelId,
    gpu: broad.gpu,
    aggregate: broad.aggregate,
    flips: broad.highlights.flips.map(selectBroadFlipSummary),
  };

  const manifest = {
    schemaVersion: 1,
    generatedAtUtc: new Date().toISOString(),
    modelId: curated.modelId,
    gpu: curated.gpu,
    stableSources: {
      curatedSummary: relativePath(path.resolve('.'), curatedSummaryPath),
      broadSummary: relativePath(path.resolve('.'), broadSummaryPath),
    },
    broad: {
      aggregate: broad.aggregate,
      flipExamples: demoBroadSummary.flips.slice(0, 6),
    },
    curated: {
      aggregate: curated.aggregate,
      promptIds: curatedPrompts.map((prompt) => prompt.id),
      defaultPromptId: curatedPrompts.find((prompt) => prompt.id === 'backup-yes-no-choice')?.id ?? curatedPrompts[0]?.id ?? null,
    },
  };

  writeJson(path.join(demoRoot, 'manifest.json'), manifest);
  writeJson(path.join(demoRoot, 'broad-summary.json'), demoBroadSummary);
  writeJson(path.join(demoCuratedRoot, 'summary.json'), demoCuratedSummary);
}

main();
