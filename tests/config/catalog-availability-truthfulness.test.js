import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const REPO_ROOT = process.cwd();
const CATALOG_PATH = path.join(REPO_ROOT, 'models', 'catalog.json');
const LOCAL_MODELS_DIR = path.join(REPO_ROOT, 'models', 'local');

const catalog = JSON.parse(fs.readFileSync(CATALOG_PATH, 'utf8'));
const models = Array.isArray(catalog.models) ? catalog.models : [];
const CONVERSION_TARGET_MODEL_IDS = [
  'qwen-3-5-0-8b-q4k-ehaf16',
  'qwen-3-5-2b-q4k-ehaf16',
  'gemma-4-e2b-it-q4k-ehf16-af32',
  'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
];

// =============================================================================
// Invariant: availability.local=true iff models/local/<modelId>/manifest.json exists
// =============================================================================

{
  for (const entry of models) {
    const local = entry?.lifecycle?.availability?.local;
    const manifestPath = path.join(LOCAL_MODELS_DIR, entry.modelId, 'manifest.json');
    const manifestExists = fs.existsSync(manifestPath);

    if (local === true) {
      assert.ok(
        manifestExists,
        `${entry.modelId}: availability.local=true but models/local/${entry.modelId}/manifest.json does not exist`
      );
    }
    if (manifestExists) {
      assert.equal(
        local,
        true,
        `${entry.modelId}: models/local/${entry.modelId}/manifest.json exists but availability.local is not true`
      );
    }
  }
}

// =============================================================================
// Invariant: availability.hf=true requires hf coordinates
// =============================================================================

{
  for (const entry of models) {
    const hfAvail = entry?.lifecycle?.availability?.hf;
    const hfSpec = entry?.hf;
    if (hfAvail === true) {
      assert.ok(
        hfSpec?.repoId && hfSpec?.path,
        `${entry.modelId}: availability.hf=true requires hf.repoId and hf.path`
      );
    }
  }
}

// =============================================================================
// Invariant: baseUrl is either null or a non-empty string
// =============================================================================

{
  for (const entry of models) {
    const baseUrl = entry?.baseUrl;
    if (baseUrl !== null && baseUrl !== undefined) {
      assert.equal(
        typeof baseUrl,
        'string',
        `${entry.modelId}: baseUrl must be null or a string`
      );
      assert.ok(
        baseUrl.trim().length > 0,
        `${entry.modelId}: baseUrl must not be an empty string`
      );
    }
  }
}

// =============================================================================
// Invariant: every model in models/local/ must be in the catalog
// =============================================================================

{
  const catalogIds = new Set(models.map((m) => m.modelId));
  const localDirs = fs.readdirSync(LOCAL_MODELS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name);

  for (const dir of localDirs) {
    const manifestPath = path.join(LOCAL_MODELS_DIR, dir, 'manifest.json');
    if (fs.existsSync(manifestPath)) {
      assert.ok(
        catalogIds.has(dir),
        `models/local/${dir} has a manifest.json but is not listed in catalog.json`
      );
    }
  }
}

// =============================================================================
// Invariant: conversion targets are manifest-backed and quickstart behavior is explicit
// =============================================================================

{
  for (const modelId of CONVERSION_TARGET_MODEL_IDS) {
    const entry = models.find((model) => model.modelId === modelId);
    assert.ok(entry, `Conversion target ${modelId} missing from catalog.json`);

    const manifestPath = path.join(LOCAL_MODELS_DIR, modelId, 'manifest.json');
    assert.ok(
      fs.existsSync(manifestPath),
      `Conversion target ${modelId} requires models/local/${modelId}/manifest.json`
    );

    if (entry.quickstart === true) {
      assert.ok(
        entry?.hf?.repoId && entry?.hf?.revision && entry?.hf?.path,
        `Quickstart conversion target ${modelId} requires hf.repoId, hf.revision, and hf.path`
      );
      assert.notEqual(
        entry.hf.revision,
        'unknown-local-snapshot',
        `Quickstart conversion target ${modelId} must not use placeholder hf.revision`
      );
    } else {
      assert.equal(
        entry.quickstart,
        false,
        `Conversion target ${modelId} should be explicitly quickstart=false for local-only invocation`
      );
    }
  }
}

console.log('catalog-availability-truthfulness.test: ok');
