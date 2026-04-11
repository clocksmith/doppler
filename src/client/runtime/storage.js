import { initStorage } from '../../storage/shard-manager.js';

const normalizeOPFSPath = (path) => String(path || '').replace(/^\/+/, '');

async function getOPFSRoot() {
  await initStorage();
  if (!globalThis.navigator?.storage?.getDirectory) {
    throw new Error('OPFS not available');
  }
  return globalThis.navigator.storage.getDirectory();
}

async function resolveOPFSPath(path, createDirs) {
  const normalized = normalizeOPFSPath(path);
  const parts = normalized.split('/').filter(Boolean);
  if (parts.length === 0) {
    throw new Error('Invalid OPFS path');
  }

  const filename = parts.pop();
  let dir = await getOPFSRoot();

  for (const part of parts) {
    dir = await dir.getDirectoryHandle(part, { create: createDirs });
  }

  return { dir, filename };
}

export async function readOPFSFile(path) {
  const { dir, filename } = await resolveOPFSPath(path, false);
  const handle = await dir.getFileHandle(filename);
  const file = await handle.getFile();
  return file.arrayBuffer();
}

export async function writeOPFSFile(path, data) {
  const { dir, filename } = await resolveOPFSPath(path, true);
  const handle = await dir.getFileHandle(filename, { create: true });
  const writable = await handle.createWritable();
  await writable.write(data);
  await writable.close();
}

export async function fetchArrayBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return response.arrayBuffer();
}
