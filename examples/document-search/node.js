#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createInterface } from 'node:readline';
import { openCapsule } from 'doppler-gpu/host';
import { getCapsuleIdentity } from 'doppler-gpu/capsule';
import { getDevice } from 'doppler-gpu/tooling/device';
import { create, globals } from 'webgpu';
import { createDocumentSearchController } from './controller.js';
import { acquireNodeStoreLease, createNodeDocumentStore } from './node-store.js';

const applicationDir = path.dirname(fileURLToPath(import.meta.url));
let active = false;
function localFile(relative) {
  const target = path.resolve(applicationDir, relative);
  const rel = path.relative(applicationDir, target);
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) throw new Error('Application artifact escapes its directory.');
  return target;
}

export async function createNodeDocumentSearch({ storageDir, observer = null, onProgress = null }) {
  if (active) throw new Error('Close the active Node application before opening another.');
  if (globalThis.navigator?.gpu || getDevice()) {
    throw new Error('The Node search runner requires an unowned WebGPU process; close the existing provider first.');
  }
  if (!path.isAbsolute(storageDir ?? '')) throw new Error('An absolute private storageDir is required.');
  active = true;
  let release;
  let controller;
  let gpu = null;
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const previousGlobals = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  let closing;
  async function close() {
    if (closing) return closing;
    closing = (async () => {
      try { await controller?.dispose(); }
      finally {
        const device = gpu ? getDevice() : null;
        device?.destroy();
        if (device) await device.lost;
        if (previousNavigator) Object.defineProperty(globalThis, 'navigator', previousNavigator);
        else delete globalThis.navigator;
        gpu = null;
        for (const [key, descriptor] of previousGlobals) {
          if (descriptor) Object.defineProperty(globalThis, key, descriptor);
          else delete globalThis[key];
        }
        try { await release?.(); } finally { active = false; }
      }
    })();
    return closing;
  }
  try {
    release = await acquireNodeStoreLease(storageDir);
    const config = JSON.parse(await fs.readFile(path.join(applicationDir, 'models.json'), 'utf8'));
    const sources = JSON.parse(await fs.readFile(path.join(applicationDir, 'shard-sources.json'), 'utf8'));
    Object.assign(globalThis, globals);
    gpu = create(['enable-dawn-features=allow_unsafe_apis']);
    Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu } });
    controller = createDocumentSearchController({ config, openCapsule, getCapsuleIdentity, observer, onProgress,
      storeFor: name => createNodeDocumentStore(path.join(storageDir, name)),
      withLock: async (_name, task) => task(), // The process lease owns the complete application lifetime.
      fetch: async (url, { signal } = {}) => {
        signal?.throwIfAborted();
        const bytes = await fs.readFile(localFile(url));
        signal?.throwIfAborted();
        return { ok: true, json: async () => JSON.parse(bytes) };
      },
      fetchArtifact: async (model, artifact, { signal } = {}) => {
        signal?.throwIfAborted();
        const relative = path.join(path.dirname(model.capsuleUrl), artifact.path);
        let bytes;
        const source = sources.artifacts[relative];
        if (source) {
          if (!source.url || source.hash !== artifact.hash || source.sizeBytes !== artifact.sizeBytes) {
            throw new Error(`Missing or mismatched artifact source: ${relative}.`);
          }
          const response = await fetch(source.url, { signal });
          if (!response.ok) throw new Error(`Artifact acquisition failed (${response.status}): ${relative}.`);
          bytes = new Uint8Array(await response.arrayBuffer());
        } else bytes = await fs.readFile(localFile(relative));
        signal?.throwIfAborted();
        return bytes;
      },
    });
    return { controller, config, close };
  } catch (error) {
    try { await close(); } catch (cleanup) { throw new AggregateError([error, cleanup], error.message); }
    throw error;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [command, storageDir, ...args] = process.argv.slice(2);
  if (!['install', 'index', 'search', 'interactive', 'repair'].includes(command) || !storageDir) {
    throw new Error('Usage: node node.js install|repair|interactive <storage-dir>; index <storage-dir> <documents.json>; search <storage-dir> <query>');
  }
  const app = await createNodeDocumentSearch({ storageDir: path.resolve(storageDir), onProgress: event => {
    if (event.type === 'phase') process.stderr.write(`${event.model}: ${event.phase}\n`);
  } });
  const cancel = () => { process.stderr.write('Cancelling; submitted GPU work must finish before cleanup.\n'); app.controller.cancel(); };
  process.on('SIGINT', cancel);
  try {
    const started = performance.now();
    if (command === 'install') await app.controller.install();
    else if (command === 'repair') await app.controller.repair();
    else await app.controller.openRetained();
    const openMs = performance.now() - started;
    const present = result => result.results.map(row => ({ id: row.document.id, title: row.document.title,
      score: row.rerankScore, excerpt: row.document.text.slice(0, 240) }));
    if (command === 'interactive') {
      process.stderr.write(`Models ready (${Math.round(openMs)} ms). Enter queries; EOF closes both models.\n`);
      const lines = createInterface({ input: process.stdin, output: process.stderr, terminal: Boolean(process.stdin.isTTY) });
      lines.on('SIGINT', cancel);
      try {
        for await (const line of lines) {
          if (!line.trim()) continue;
          const execution = performance.now();
          try {
            const result = await app.controller.search(line);
            process.stdout.write(JSON.stringify({ query: line, queryMs: performance.now() - execution, results: present(result) }) + '\n');
          } catch (error) { process.stderr.write(error.message + '\n'); }
        }
      } finally { lines.off('SIGINT', cancel); lines.close(); }
    } else {
      const execution = performance.now();
      const result = command === 'index' ? await app.controller.indexDocuments(JSON.parse(await fs.readFile(args[0], 'utf8')))
        : command === 'search' ? await app.controller.search(args.join(' ')) : app.controller.getState();
      process.stdout.write(JSON.stringify({ openMs, operationMs: performance.now() - execution,
        result: command === 'search' ? present(result) : command === 'index' ? { documents: result.documents } : result }, null, 2) + '\n');
    }
  } finally { process.off('SIGINT', cancel); await app.close(); }
}
