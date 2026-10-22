// Install once, then trace an offline reopen.
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
const [appRoot, capsuleDir, descriptorPath, outPath] = process.argv.slice(2);
const http = await import('node:http');
const { createReadStream, statSync } = await import('node:fs');
const mounts = [{ prefix: '/app', dir: appRoot }, { prefix: '/capsule', dir: capsuleDir }];
const server = http.createServer((q, r) => {
  const p = decodeURIComponent(new URL(q.url, 'http://x').pathname);
  const m = mounts.find((x) => p.startsWith(x.prefix + '/'));
  const f = m && path.resolve(m.dir, '.' + p.slice(m.prefix.length));
  try { if (!statSync(f).isFile()) throw 0; } catch { r.writeHead(404).end(); return; }
  r.writeHead(200, { 'Content-Type': f.endsWith('.js') ? 'text/javascript' : f.endsWith('.html') ? 'text/html' : f.endsWith('.json') ? 'application/json' : 'application/octet-stream' });
  createReadStream(f).pipe(r);
});
await new Promise((res) => server.listen(0, '127.0.0.1', res));
const base = `http://127.0.0.1:${server.address().port}`;
const descriptor = JSON.parse(await fs.readFile(descriptorPath, 'utf8'));
const profile = path.join(path.dirname(outPath), 'open-trace-profile');
await fs.rm(profile, { recursive: true, force: true });
const ARGS = ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'];
const out = {};
for (const [name, mode] of [['install', 'install'], ['reopen', 'open-retained']]) {
  const ctx = await chromium.launchPersistentContext(profile, { channel: 'chrome', headless: true, args: ARGS });
  const page = ctx.pages()[0] ?? await ctx.newPage();
  await page.goto(`${base}/app/tools/harness.html`);
  await page.waitForFunction(() => window.harnessReady === true);
  let cdp = null;
  if (name === 'reopen') { cdp = await ctx.newCDPSession(page); await cdp.send('Profiler.enable'); await cdp.send('Profiler.setSamplingInterval', { interval: 200 }); await cdp.send('Profiler.start'); }
  out[name] = await page.evaluate(([d, m]) => window.harness.open('doppler', { descriptor: d, mode: m, trace: true }), [descriptor, mode]);
  if (cdp) {
    const { profile } = await cdp.send('Profiler.stop');
    const byId = new Map(profile.nodes.map((n) => [n.id, n]));
    const self = new Map();
    profile.samples.forEach((id, i) => self.set(id, (self.get(id) ?? 0) + (profile.timeDeltas[i] ?? 0)));
    const agg = {};
    for (const [id, us] of self) { const f = byId.get(id).callFrame; const k = `${f.functionName || '(anon)'} ${f.url.split('/').slice(-3).join('/')}:${f.lineNumber + 1}`; agg[k] = (agg[k] ?? 0) + us; }
    out.cpuTop = Object.entries(agg).sort((a, b) => b[1] - a[1]).slice(0, 20).map(([k, us]) => [k, Math.round(us / 1000)]);
  }
  await page.evaluate(() => window.harness.close());
  await ctx.close();
}
server.closeAllConnections(); server.close();
await fs.writeFile(outPath, JSON.stringify(out, null, 2));
const r = out.reopen;
console.log('reopen openMs', r.openMs.toFixed(0));
for (const [k, ms] of out.cpuTop) console.log(String(ms).padStart(7), 'ms', k);
const obs = r.trace.observations;
const sum = (k) => obs.reduce((a, o) => a + (o[k] ?? 0), 0);
console.log('artifacts', obs.length, 'storageReadMs', sum('storageReadMs').toFixed(0), 'verificationMs', sum('verificationMs').toFixed(0), 'bytes', sum('bytes'));
const phases = {};
for (const e of r.trace.events) { const k = `${e.type}:${e.phase}`; phases[k] ??= [e.atMs, e.atMs]; phases[k][1] = e.atMs; }
for (const [k, [a, b]] of Object.entries(phases)) console.log(k.padEnd(48), a.toFixed(0), '→', b.toFixed(0));
