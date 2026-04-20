import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

import { parseJsonl } from '../datasets/jsonl.js';
import { sha256Hex } from '../../../utils/sha256.js';
import { stableSortObject } from '../../../utils/stable-sort-object.js';

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function asOptionalString(value) {
  if (value === undefined || value === null) return null;
  const trimmed = String(value).trim();
  return trimmed || null;
}

function asOptionalStringArray(value) {
  if (value === undefined || value === null) return null;
  const input = Array.isArray(value)
    ? value
    : (typeof value === 'string' ? value.split(',') : null);
  if (!Array.isArray(input)) {
    return null;
  }
  const normalized = input
    .map((entry) => asOptionalString(entry))
    .filter(Boolean);
  return normalized.length > 0 ? normalized : null;
}

function normalizeLangCode(value) {
  const normalized = asOptionalString(value);
  if (!normalized) return null;
  const compact = normalized.toLowerCase().replace(/_/g, '-');
  if (compact.startsWith('en')) return 'en';
  if (compact.startsWith('es')) return 'es';
  return compact;
}

export function normalizeDistillationPair(value, srcLang = null, tgtLang = null) {
  const pair = asOptionalString(value);
  if (!pair) {
    if (!srcLang || !tgtLang) return null;
    return `${srcLang}->${tgtLang}`;
  }
  const normalized = pair.toLowerCase().replace(/_/g, '-').replace(/\s+/g, '');
  const separator = normalized.includes('->') ? '->' : '-';
  const parts = normalized.split(separator).filter(Boolean);
  if (parts.length !== 2) return null;
  const source = normalizeLangCode(parts[0]) || parts[0];
  const target = normalizeLangCode(parts[1]) || parts[1];
  return `${source}->${target}`;
}

function resolveStringCandidate(record, keys) {
  for (const key of keys) {
    const value = asOptionalString(record?.[key]);
    if (value) return value;
  }
  return null;
}

function resolveStableRowId(record, index, canonical) {
  const explicit = asOptionalString(record?.row_id ?? record?.rowId);
  if (explicit) return explicit;
  return sha256Hex(stableJson({
    index,
    src_lang: canonical.src_lang,
    tgt_lang: canonical.tgt_lang,
    pair: canonical.pair,
    source: canonical.source,
    target_pos: canonical.target_pos,
    target_neg: canonical.target_neg,
  }));
}

export function normalizeTranslationPairRow(record, index, options = {}) {
  if (!record || typeof record !== 'object' || Array.isArray(record)) {
    return null;
  }
  const source = resolveStringCandidate(record, ['source', 'query', 'prompt']);
  const targetPos = resolveStringCandidate(record, ['target_pos', 'target', 'pos', 'completion']);
  const targetNeg = resolveStringCandidate(record, ['target_neg', 'neg']);
  if (!source || !targetPos) {
    return null;
  }
  const srcLang = normalizeLangCode(record?.src_lang ?? record?.source_lang);
  const tgtLang = normalizeLangCode(record?.tgt_lang ?? record?.target_lang ?? record?.lang);
  const pair = normalizeDistillationPair(record?.pair, srcLang, tgtLang);
  const strict = options.strictPairContract === true;
  if (strict) {
    if (!srcLang || !tgtLang) {
      throw new Error('strictPairContract requires src_lang and tgt_lang on each row.');
    }
    if (!pair) {
      throw new Error('strictPairContract requires pair on each row.');
    }
    if (pair !== `${srcLang}->${tgtLang}`) {
      throw new Error(`row pair "${record?.pair}" does not match src/tgt "${srcLang}->${tgtLang}".`);
    }
  }
  const canonical = {
    src_lang: srcLang,
    tgt_lang: tgtLang,
    pair: pair || null,
    source,
    target_pos: targetPos,
    target_neg: targetNeg,
  };
  canonical.row_id = resolveStableRowId(record, index, canonical);
  return canonical;
}

function normalizeFilterSet(value, normalizer) {
  const entries = asOptionalStringArray(value);
  if (!entries) return null;
  const normalized = entries
    .map((entry) => normalizer(entry))
    .filter(Boolean);
  return normalized.length > 0 ? new Set(normalized) : null;
}

function applyDatasetFilters(rows, options = {}) {
  const sourceLangs = normalizeFilterSet(options.sourceLangs, normalizeLangCode);
  const targetLangs = normalizeFilterSet(options.targetLangs, normalizeLangCode);
  const pairs = normalizeFilterSet(options.pairAllowlist, normalizeDistillationPair);
  return rows.filter((row) => {
    if (sourceLangs && (!row.src_lang || !sourceLangs.has(row.src_lang))) {
      return false;
    }
    if (targetLangs && (!row.tgt_lang || !targetLangs.has(row.tgt_lang))) {
      return false;
    }
    if (pairs && (!row.pair || !pairs.has(row.pair))) {
      return false;
    }
    return true;
  });
}

function normalizeSubsetSpec(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  const sizeRaw = Number(value.size ?? value.count ?? value.rowCount ?? 0);
  const size = Number.isInteger(sizeRaw) && sizeRaw > 0 ? sizeRaw : null;
  const seedRaw = Number(value.seed ?? 1337);
  const seed = Number.isInteger(seedRaw) ? seedRaw : 1337;
  const balanceBy = asOptionalString(value.balanceBy ?? value.allocationMode ?? value.stratifyBy);
  return {
    id: asOptionalString(value.id ?? value.name) || null,
    size,
    seed,
    balanceBy,
    parentSubsetManifest: asOptionalString(
      value.parentSubsetManifest
      ?? value.parentSubset
      ?? value.parentManifest
    ),
  };
}

function buildDeterministicRank(seed, rowId) {
  return sha256Hex(`${seed}:${rowId}`);
}

async function resolveParentRowIds(parentSubsetManifest) {
  if (!parentSubsetManifest) return null;
  const absolutePath = resolve(parentSubsetManifest);
  const raw = await readFile(absolutePath, 'utf8');
  const parsed = JSON.parse(raw);
  const rowIds = Array.isArray(parsed?.rowIds)
    ? parsed.rowIds
    : [];
  return rowIds.length > 0 ? new Set(rowIds.map((entry) => String(entry))) : new Set();
}

function buildPairBalancedSubset(rows, size, seed) {
  const byPair = new Map();
  for (const row of rows) {
    const key = row.pair || 'unknown';
    const bucket = byPair.get(key) || [];
    bucket.push(row);
    byPair.set(key, bucket);
  }
  for (const bucket of byPair.values()) {
    bucket.sort((left, right) => {
      const leftRank = buildDeterministicRank(seed, left.row_id);
      const rightRank = buildDeterministicRank(seed, right.row_id);
      return leftRank.localeCompare(rightRank);
    });
  }
  const pairKeys = [...byPair.keys()].sort((left, right) => left.localeCompare(right));
  const selected = [];
  let cursor = 0;
  while (selected.length < size) {
    let progressed = false;
    for (const pairKey of pairKeys) {
      const bucket = byPair.get(pairKey);
      if (!bucket || cursor >= bucket.length) continue;
      selected.push(bucket[cursor]);
      progressed = true;
      if (selected.length >= size) break;
    }
    if (!progressed) break;
    cursor += 1;
  }
  return selected;
}

function selectSubsetRows(rows, subsetSpec) {
  if (!subsetSpec || !subsetSpec.size || subsetSpec.size >= rows.length) {
    return rows.slice();
  }
  if (subsetSpec.balanceBy === 'pair' || subsetSpec.balanceBy === 'pair_balance') {
    return buildPairBalancedSubset(rows, subsetSpec.size, subsetSpec.seed);
  }
  return rows
    .slice()
    .sort((left, right) => {
      const leftRank = buildDeterministicRank(subsetSpec.seed, left.row_id);
      const rightRank = buildDeterministicRank(subsetSpec.seed, right.row_id);
      return leftRank.localeCompare(rightRank);
    })
    .slice(0, subsetSpec.size);
}

function computeDirectionCounts(rows) {
  const counts = {};
  for (const row of rows) {
    const key = row.pair || 'unknown';
    counts[key] = (counts[key] || 0) + 1;
  }
  return counts;
}

export async function loadCanonicalTranslationDataset(datasetPath, options = {}) {
  const absolutePath = resolve(String(datasetPath));
  const raw = await readFile(absolutePath, 'utf8');
  const parsed = absolutePath.endsWith('.json')
    ? JSON.parse(raw)
    : parseJsonl(raw);
  if (!Array.isArray(parsed)) {
    throw new Error(`Distillation dataset "${absolutePath}" must be a JSON array or JSONL file.`);
  }
  const normalizedRows = [];
  for (let index = 0; index < parsed.length; index += 1) {
    const row = normalizeTranslationPairRow(parsed[index], index, options);
    if (row) {
      normalizedRows.push(row);
    }
  }
  const filteredRows = applyDatasetFilters(normalizedRows, options);
  if (filteredRows.length === 0) {
    throw new Error(`Distillation dataset "${absolutePath}" has no usable rows after contract checks and filters.`);
  }
  const rowIds = filteredRows.map((row) => row.row_id);
  return {
    absolutePath,
    raw,
    rows: filteredRows,
    rowCount: filteredRows.length,
    directionCounts: computeDirectionCounts(filteredRows),
    datasetHash: sha256Hex(raw),
    canonicalHash: sha256Hex(stableJson(filteredRows)),
    rowIdsHash: sha256Hex(rowIds.join('\n')),
  };
}

export async function buildFrozenSubset(options) {
  const dataset = await loadCanonicalTranslationDataset(options.datasetPath, {
    strictPairContract: options.strictPairContract === true,
    sourceLangs: options.sourceLangs,
    targetLangs: options.targetLangs,
    pairAllowlist: options.pairAllowlist,
  });
  const subsetSpec = normalizeSubsetSpec(options.subsetSpec);
  const parentRowIds = subsetSpec?.parentSubsetManifest
    ? await resolveParentRowIds(subsetSpec.parentSubsetManifest)
    : null;
  const scopedRows = parentRowIds
    ? dataset.rows.filter((row) => parentRowIds.has(row.row_id))
    : dataset.rows;
  const subsetRows = selectSubsetRows(scopedRows, subsetSpec);
  const outputDir = resolve(String(options.outputDir));
  const subsetJsonlPath = join(outputDir, 'subset.jsonl');
  const rowIdsPath = join(outputDir, 'row_ids.txt');
  const manifestPath = join(outputDir, 'subset_manifest.json');
  const serializedRows = `${subsetRows.map((row) => JSON.stringify(row)).join('\n')}\n`;
  const rowIdsText = `${subsetRows.map((row) => row.row_id).join('\n')}\n`;
  const manifest = {
    artifactType: 'subset_manifest',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    datasetPath: dataset.absolutePath,
    datasetHash: dataset.datasetHash,
    canonicalHash: dataset.canonicalHash,
    rowIdsHash: sha256Hex(rowIdsText),
    universeRowCount: dataset.rowCount,
    subsetRowCount: subsetRows.length,
    directionCounts: computeDirectionCounts(subsetRows),
    subsetSpec,
    parentSubsetManifest: subsetSpec?.parentSubsetManifest || null,
    rowIds: subsetRows.map((row) => row.row_id),
    output: {
      subsetJsonlPath,
      rowIdsPath,
    },
  };
  await mkdir(outputDir, { recursive: true });
  await writeFile(subsetJsonlPath, serializedRows, 'utf8');
  await writeFile(rowIdsPath, rowIdsText, 'utf8');
  await mkdir(dirname(manifestPath), { recursive: true });
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  return {
    dataset,
    subsetRows,
    subsetJsonlPath,
    rowIdsPath,
    manifestPath,
    manifest,
  };
}
