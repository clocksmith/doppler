// The list of files the front page needs with no network: the demo's own shell
// (from the manifest the demo generates for its offline worker) and this
// site's files. The worker in public/sw.js reads it as offline-shell.json.
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

// `demoShell` is the demo manifest's APP_SHELL. A source checkout addresses the
// demo from the site root; hosted, it lives under /doppler/.
export function offlineShell({ demoShell, demoDigest, siteFiles, siteDigest }) {
  if (!Array.isArray(demoShell) || demoShell.length === 0) throw new Error('The demo names no shell files, so the front page cannot be made to work offline.');
  const hosted = demoShell.map((file) => (file.startsWith('/doppler/') ? file : `/doppler${file}`))
    // The demo's own page, worker and app manifest belong to the demo's address, not this one.
    .filter((file) => !/^\/doppler\/demo\/(index\.html|sw\.js|pwa-manifest\.json)$/.test(file));
  // The runtime's release file names the build in every run this page makes.
  hosted.push('/doppler/release.json');
  const own = ['./', ...siteFiles.map((file) => `./${file}`)];
  const version = createHash('sha256').update(JSON.stringify([demoDigest, siteDigest, hosted, own])).digest('hex').slice(0, 16);
  return { schema: 'doppler.observatory.offline-shell/v1', version, files: [...own, ...hosted] };
}

export async function readDemoShell(dopplerRoot) {
  const manifest = await import(`${new URL(`file://${path.join(dopplerRoot, 'demo', 'generated-shell-manifest.js')}`).href}?at=${Date.now()}`);
  return { demoShell: [...manifest.APP_SHELL], demoDigest: manifest.SHELL_MANIFEST_DIGEST };
}

export async function digestFiles(root, files) {
  const hash = createHash('sha256');
  for (const file of [...files].sort()) hash.update(file).update(await readFile(path.join(root, file)));
  return hash.digest('hex');
}
