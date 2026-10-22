// Frozen evaluation data: NLBSE'24 issue report classification, pinned by
// revision and SHA-256. Issue text is fetched from upstream, never vendored.
import { createHash } from 'node:crypto';

export const DATASET = Object.freeze({
  source: 'https://github.com/nlbse2024/issue-report-classification',
  revision: '2927bc67eb42db8affd16eaf3e5a6d74f3063961',
  files: Object.freeze({
    train: { path: 'data/issues_train.csv', sha256: '18dc42a30aa33dccadb723ad3baeb164d38bff521496f985ca2791c26b8939f5' },
    test: { path: 'data/issues_test.csv', sha256: '4f7d8619d4e5adbea126e548fd8c214449288f3a93bb3bc130c54cd307af7e85' },
  }),
  labels: Object.freeze(['bug', 'feature', 'question']),
});

export function datasetUrl(split) {
  return `https://raw.githubusercontent.com/nlbse2024/issue-report-classification/${DATASET.revision}/${DATASET.files[split].path}`;
}

// RFC 4180: quoted fields may contain commas, quotes ("") and newlines.
export function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i += 1;
      row.push(field); rows.push(row); row = []; field = '';
    } else field += c;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  const [header, ...body] = rows;
  return body.filter((r) => r.length === header.length).map((r, index) => ({
    index, ...Object.fromEntries(header.map((h, j) => [h, r[j]])),
  }));
}

export async function loadSplit(split, { fetchImpl = fetch, readLocal = null } = {}) {
  const text = readLocal ? await readLocal(split) : await (await fetchImpl(datasetUrl(split))).text();
  const digest = createHash('sha256').update(text, 'utf8').digest('hex');
  if (digest !== DATASET.files[split].sha256) {
    throw new Error(`NLBSE ${split} split digest ${digest} does not match the pinned ${DATASET.files[split].sha256}.`);
  }
  return parseCsv(text);
}

// Deterministic stratified sample: perCell issues from every (repo, label) cell,
// ordered by SHA-256(seed:split:index).
export function stratifiedSample(rows, { split, perCell, seed }) {
  const key = (row) => createHash('sha256').update(`${seed}:${split}:${row.index}`).digest('hex');
  const cells = new Map();
  for (const row of rows) {
    const cell = `${row.repo}\u0000${row.label}`;
    if (!cells.has(cell)) cells.set(cell, []);
    cells.get(cell).push(row);
  }
  const picked = [];
  for (const cell of [...cells.keys()].sort()) {
    const members = cells.get(cell).map((row) => ({ row, k: key(row) })).sort((a, b) => (a.k < b.k ? -1 : 1));
    if (members.length < perCell) throw new Error(`Cell ${cell} has only ${members.length} issues.`);
    picked.push(...members.slice(0, perCell).map((m) => m.row));
  }
  return picked;
}
