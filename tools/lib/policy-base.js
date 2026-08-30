import { execFileSync } from 'node:child_process';

function readOption(argv, name) {
  const index = argv.indexOf(name);
  if (index < 0) return null;
  const value = argv[index + 1];
  if (!value || value.startsWith('--')) {
    throw new Error(`${name} requires a git revision.`);
  }
  return value;
}

export function resolvePolicyBaseRef(argv, environment = process.env) {
  const explicit = readOption(argv, '--base');
  if (explicit) return explicit;
  const environmentRef = String(environment.DOPPLER_POLICY_BASE_REF ?? '').trim();
  if (environmentRef && !/^0+$/u.test(environmentRef)) return environmentRef;
  return 'HEAD';
}

export function readTextAtGitRef(repoRoot, ref, relativePath) {
  try {
    return execFileSync(
      'git',
      ['show', `${ref}:${relativePath}`],
      { cwd: repoRoot, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }
    );
  } catch (error) {
    const detail = String(error?.stderr ?? error?.message ?? error).trim();
    throw new Error(`Unable to read ${relativePath} at ${ref}: ${detail}`);
  }
}

export function readJsonAtGitRef(repoRoot, ref, relativePath) {
  const source = readTextAtGitRef(repoRoot, ref, relativePath);
  try {
    return JSON.parse(source);
  } catch (error) {
    throw new Error(`${relativePath} at ${ref} is not valid JSON: ${error.message}`);
  }
}

export function listDirectoriesAtGitRef(repoRoot, ref, relativePath) {
  let output;
  try {
    output = execFileSync(
      'git',
      ['ls-tree', '-d', '--name-only', `${ref}:${relativePath}`],
      { cwd: repoRoot, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }
    );
  } catch (error) {
    const detail = String(error?.stderr ?? error?.message ?? error).trim();
    throw new Error(`Unable to list ${relativePath} at ${ref}: ${detail}`);
  }
  return output.split(/\r?\n/u).map((entry) => entry.trim()).filter(Boolean).sort();
}
