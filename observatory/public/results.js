// Benchmark results side by side. Pure functions: a result is a benchmark
// evidence file, possibly from another device. Nothing in a file is trusted:
// its shape is checked, and it is judged again here against this site's
// protocol. The verdict a file carries is ignored.

import { judgeRun } from './protocol.js';

export const RESULT_SCHEMA = 'doppler.observatory.device-benchmark/v1';
const MAX_RUNS = 64;
const MAX_TOKENS = 4096;

const isTokens = (value) => Array.isArray(value) && value.length <= MAX_TOKENS && value.every((id) => Number.isSafeInteger(id) && id >= 0);
const isRun = (run) => run && typeof run === 'object' && isTokens(run.generatedTokenIds) && Number.isFinite(run.wallMs);

// Checks a result that came from a file. Throws with a reason a person can act on.
export function readResult(value) {
  const fail = (why) => { throw new Error(`This is not a benchmark result this page can open: ${why}.`); };
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('the file does not hold a result');
  if (typeof value.schema !== 'string') fail('the file does not say what it holds');
  if (value.schema !== RESULT_SCHEMA) fail(`its format is "${value.schema.slice(0, 60)}", not ${RESULT_SCHEMA}`);
  if (typeof value.protocol?.id !== 'string' || typeof value.protocol?.version !== 'string') fail('it does not name its protocol');
  if (typeof value.startedAt !== 'string' || Number.isNaN(Date.parse(value.startedAt))) fail('it has no valid start time');
  if (value.failure !== null && typeof value.failure?.message !== 'string') fail('its failure record is malformed');
  if (!(value.cold === null || isRun(value.cold))) fail('its first run is malformed');
  for (const list of [value.warm, value.plainPath ?? []]) {
    if (!Array.isArray(list) || list.length > MAX_RUNS || !list.every(isRun)) fail('its runs are malformed');
  }
  if (!(value.device === null || typeof value.device === 'object')) fail('its device record is malformed');
  return value;
}

// "Chrome 145 on macOS" from a user-agent string; the full string stays in the result.
export function describeBrowser(userAgent) {
  if (typeof userAgent !== 'string' || !userAgent) return 'browser not recorded';
  const os = /Android/.test(userAgent) ? 'Android' : /iPhone|iPad|iPod/.test(userAgent) ? 'iOS' : /Mac OS X/.test(userAgent) ? 'macOS'
    : /Windows/.test(userAgent) ? 'Windows' : /CrOS/.test(userAgent) ? 'ChromeOS' : /Linux/.test(userAgent) ? 'Linux' : 'unknown system';
  const pick = (pattern, name) => { const match = userAgent.match(pattern); return match ? `${name} ${match[1]}` : null; };
  const browser = pick(/Edg\/(\d+)/, 'Edge') ?? pick(/Firefox\/(\d+)/, 'Firefox') ?? pick(/HeadlessChrome\/(\d+)/, 'Chrome (headless)')
    ?? pick(/Chrome\/(\d+)/, 'Chrome') ?? pick(/Version\/(\d+)[.\d]* (?:Mobile\/\S+ )?Safari/, 'Safari') ?? 'unknown browser';
  return `${browser} on ${os}`;
}

// One result, judged here. A result made under another protocol version is
// shown but not judged: its rules and reference answer were different.
export function assess(protocol, result) {
  const device = result.device ?? {};
  const adapter = [device.adapter?.vendor, device.adapter?.architecture, device.adapter?.description].filter(Boolean).join(' ') || 'adapter not recorded';
  const base = {
    startedAt: result.startedAt,
    adapter,
    browser: describeBrowser(device.userAgent),
    userAgent: device.userAgent ?? null,
    runtime: result.runtime?.version ? `Doppler ${result.runtime.version}${result.runtime.release?.sourceCommit ? ` (${result.runtime.release.sourceCommit.slice(0, 8)})` : ''}` : 'runtime not recorded',
    previewRuntime: result.runtime?.preview === true,
    protocol: `${result.protocol.id} ${result.protocol.version}`,
    runId: result.run?.id ?? null,
    modelWasSaved: result.load?.wasStored ?? null,
    loadMs: result.load?.wallMs ?? null,
  };
  if (result.protocol.id !== protocol.id || result.protocol.version !== protocol.version) {
    return { ...base, state: 'other-protocol', comparable: false, reasons: [`Recorded under ${base.protocol}. This page judges ${protocol.id} ${protocol.version}; the rules or the expected answer differ, so the numbers are not put beside the others.`], notes: [], summary: null };
  }
  const verdict = judgeRun(protocol, result);
  return { ...base, state: verdict.state, comparable: verdict.eligible, reasons: verdict.reasons, notes: verdict.notes, summary: verdict.summary };
}

// Several results beside each other. Only eligible results of this protocol
// version are compared; the first of them opened is the one the others are
// measured against. This orders nothing: it is a comparison, not a ranking.
export function compareResults(protocol, results) {
  const rows = results.map((result) => assess(protocol, result));
  const eligible = rows.filter((row) => row.comparable);
  const basis = eligible[0] ?? null;
  for (const row of rows) {
    const rate = row.summary?.warmDecodeTokensPerSecond;
    const basisRate = basis?.summary?.warmDecodeTokensPerSecond;
    row.relativeDecodeRate = row.comparable && Number.isFinite(rate) && Number.isFinite(basisRate) && basisRate > 0 ? rate / basisRate : null;
  }
  // One device is evidence about that device. Say how thin the comparison is.
  const devices = new Set(eligible.map((row) => `${row.adapter}|${row.browser}`)).size;
  const runtimes = new Set(eligible.map((row) => row.runtime)).size;
  return { rows, eligible: eligible.length, devices, runtimes, previews: eligible.filter((row) => row.previewRuntime).length, basisIndex: basis ? rows.indexOf(basis) : null };
}
