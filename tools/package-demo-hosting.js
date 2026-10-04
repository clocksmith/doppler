import { copyFileSync, readFileSync, existsSync, mkdirSync, lstatSync, readlinkSync, symlinkSync, rmSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { APP_SHELL } from '../demo/generated-shell-manifest.js';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, '.demo-hosting');
const entries = ['demo', 'src', 'benchmarks/vendors', 'models/catalog.json', 'LICENSE', 'NOTICE'];
const files = execFileSync('git', ['ls-files', '-z', '--', ...entries, ...APP_SHELL.map((url) => url.slice(1))], { cwd: root, encoding: 'utf8' }).split('\0').filter(Boolean);
rmSync(output, { recursive: true, force: true });
for (const file of files) {
  if (/^demo\/(archive|fixtures)\//.test(file) || /\.(md|ts)$/.test(file)) continue;
  const source = path.join(root, file);
  if (!existsSync(source)) throw new Error(`Missing hosted source: ${file}`);
  const target = path.join(output, file);
  mkdirSync(path.dirname(target), { recursive: true });
  if (lstatSync(source).isSymbolicLink()) symlinkSync(readlinkSync(source), target);
  else copyFileSync(source, target);
}
// The root uses the same absolute asset paths and public import map as /demo/.
copyFileSync(path.join(output, 'demo/index.html'), path.join(output, 'index.html'));
// The dedicated domain installs and launches at its root; source-local /demo/ stays usable.
const manifestPath = path.join(output, 'demo/pwa-manifest.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
manifest.id = '/';
manifest.start_url = '/';
manifest.scope = '/';
for (const shortcut of manifest.shortcuts) shortcut.url = shortcut.url.replace('/demo/index.html', '/');
for (const handler of manifest.file_handlers) handler.action = handler.action.replace('/demo/index.html', '/');
writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
const revision = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim();
writeFileSync(path.join(output, 'release.json'), JSON.stringify({ sourceCommit: revision, application: 'doppler-demo', canonicalUrl: 'https://canvascontext.com/' }, null, 2) + '\n');
execFileSync(process.execPath, ['tools/generate-demo-shell-manifest.js', '--root', output], { cwd: root, stdio: 'inherit' });
console.log(`Packaged Doppler demo from ${revision}: ${output}`);
