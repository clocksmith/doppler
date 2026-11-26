#!/usr/bin/env node
/**
 * Summarizes requester.js generation reports, prefill and decode separately: requester CPU, wall, wire, operator
 * compute (slowest share per round, summed), per-share reply time, rounds and bytes. It also solves the requester's
 * link from the same run: wire - operator = rounds * overheadMs + bytes / rate, least squares over every step.
 *
 *   node summarize.js REPORT.json [REPORT.json ...]
 */
import fs from 'node:fs';

const median = a => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };
const out = [];
for (const file of process.argv.slice(2)) {
  const r = JSON.parse(fs.readFileSync(file, 'utf8'));
  const pre = r.results.map(p => p.steps[0]), dec = r.results.flatMap(p => p.steps.slice(1));
  const sum = (rows, k) => rows.reduce((s, x) => s + (x[k] ?? 0), 0);
  const phase = (rows, agg) => ({
    steps: rows.length, columns: agg(rows.map(x => x.columns)), rounds: agg(rows.map(x => x.rounds)),
    requesterCpuMs: agg(rows.map(x => x.cpuMs)), wallMs: agg(rows.map(x => x.wallMs)), wireMs: agg(rows.map(x => x.wireMs)),
    operatorMs: agg(rows.map(x => x.operatorMs)), endStepMs: agg(rows.map(x => x.endStepMs ?? 0)),
    replyMsShareA: agg(rows.map(x => x.replyMs?.[0])), replyMsShareB: agg(rows.map(x => x.replyMs?.[1])),
    upBytes: agg(rows.map(x => x.up)), downBytes: agg(rows.map(x => x.down)), upAllBytes: agg(rows.map(x => x.upAll)), downAllBytes: agg(rows.map(x => x.downAll))
  });
  const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
  // Least squares for (overheadMs, msPerByte) over all steps: y = wire - operator, x1 = rounds, x2 = bytes.
  const rows = [...pre, ...dec].map(s => ({ y: s.wireMs - s.operatorMs, a: s.rounds, b: s.up + s.down }));
  const saa = rows.reduce((s, x) => s + x.a * x.a, 0), sbb = rows.reduce((s, x) => s + x.b * x.b, 0), sab = rows.reduce((s, x) => s + x.a * x.b, 0);
  const say = rows.reduce((s, x) => s + x.a * x.y, 0), sby = rows.reduce((s, x) => s + x.b * x.y, 0);
  const det = saa * sbb - sab * sab;
  const overheadMs = (say * sbb - sby * sab) / det, msPerByte = (saa * sby - sab * say) / det;
  out.push({ file, protocol: r.protocol, requesterThreads: r.requesterThreads ?? 1, prompts: r.results.length,
    tokens: r.results.map(p => p.tokens.length), prefillMean: phase(pre, mean), decodeMedian: phase(dec, median),
    decodeTotals: { tokens: dec.length, requesterCpuMs: sum(dec, 'cpuMs'), wallMs: sum(dec, 'wallMs') },
    link: { roundOverheadMs: overheadMs, bytesPerS: 1e3 / msPerByte, note: 'solved from wire - operator over prefill and decode steps' },
    freshness: r.freshness, requesterPeakRss: r.requesterPeakRss, operatorPeakRss: r.operatorStats?.map(p => p.map(o => o.peakRss)) });
}
process.stdout.write(JSON.stringify(out, null, 1) + '\n');
