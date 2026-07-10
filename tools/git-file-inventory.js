import { spawnSync } from 'node:child_process';
import path from 'node:path';

export function listTrackedFilesInDirectory(repoRoot, directoryPath) {
  const relativeDirectory = path.relative(repoRoot, directoryPath);
  if (!relativeDirectory || relativeDirectory.startsWith('..') || path.isAbsolute(relativeDirectory)) {
    throw new Error(`Tracked-file directory must be inside the repository: ${directoryPath}`);
  }

  const result = spawnSync(
    'git',
    ['-C', repoRoot, 'ls-files', '-z', '--', relativeDirectory],
    {
      encoding: 'utf8',
      maxBuffer: 4 * 1024 * 1024,
    }
  );
  if (result.error) {
    throw new Error(`Unable to list tracked files: ${result.error.message}`);
  }
  if (result.status !== 0) {
    const detail = String(result.stderr || '').trim() || `git exited ${result.status}`;
    throw new Error(`Unable to list tracked files: ${detail}`);
  }

  return String(result.stdout || '')
    .split('\0')
    .filter(Boolean)
    .map((repoPath) => path.join(repoRoot, repoPath))
    .sort();
}
