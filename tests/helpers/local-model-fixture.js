import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function normalizeHash(value) {
  return typeof value === 'string' ? value.trim().replace(/^sha256:/i, '') : '';
}

function normalizeRelativeArtifactPath(value) {
  return typeof value === 'string' ? value.trim().replace(/\\/g, '/') : '';
}

function computeFileSha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

export function readLocalFixtureMap(fixtureMapPath = path.join(process.cwd(), 'tests/fixtures/lean-execution-contract-fixtures.json')) {
  const payload = readJson(fixtureMapPath);
  const mappings = Array.isArray(payload?.mappings) ? payload.mappings : [];
  return mappings
    .map((entry) => ({
      configPath: path.resolve(process.cwd(), String(entry?.configPath ?? '')),
      manifestPath: path.resolve(process.cwd(), String(entry?.manifestPath ?? '')),
    }))
    .filter((entry) => entry.manifestPath.includes(`${path.sep}models${path.sep}local${path.sep}`));
}

export function resolveTokenizerVocabSize(tokenizerPath) {
  const tokenizer = readJson(tokenizerPath);
  if (Array.isArray(tokenizer?.model?.vocab)) {
    return tokenizer.model.vocab.length;
  }
  return Object.keys(tokenizer?.model?.vocab || {}).length;
}

export function assertManifestArtifactIntegrity(manifestPath) {
  const manifest = readJson(manifestPath);
  const manifestDir = path.dirname(manifestPath);
  const modelId = String(manifest?.modelId ?? path.basename(manifestDir));
  const shards = Array.isArray(manifest?.shards) ? manifest.shards : [];
  const sourceRuntime = (
    manifest?.metadata?.sourceRuntime
    && typeof manifest.metadata.sourceRuntime === 'object'
    && manifest.metadata.sourceRuntime.mode === 'direct-source'
  ) ? manifest.metadata.sourceRuntime : null;

  assert.ok(shards.length > 0, `${modelId}: manifest must define at least one shard`);

  if (sourceRuntime) {
    assert.equal(
      sourceRuntime.pathSemantics,
      'artifact-relative',
      `${modelId}: persisted direct-source manifest must use artifact-relative paths`
    );

    const sourceFiles = Array.isArray(sourceRuntime.sourceFiles) ? sourceRuntime.sourceFiles : [];
    assert.ok(sourceFiles.length > 0, `${modelId}: direct-source manifest must define sourceFiles`);
    assert.equal(
      sourceFiles.length,
      shards.length,
      `${modelId}: direct-source manifest sourceFiles must match shard count`
    );

    for (const entry of sourceFiles) {
      const relativePath = normalizeRelativeArtifactPath(entry?.path);
      assert.ok(relativePath, `${modelId}: direct-source source file path must be explicit`);
      assert.equal(path.isAbsolute(relativePath), false, `${modelId}: direct-source source file paths must be relative`);
      assert.ok(!relativePath.startsWith('../'), `${modelId}: direct-source source file path must stay inside artifact root`);
      const absolutePath = path.join(manifestDir, relativePath);
      assert.equal(fs.existsSync(absolutePath), true, `${modelId}: missing direct-source file ${relativePath}`);
      const expectedHash = normalizeHash(entry?.hash);
      assert.ok(expectedHash, `${modelId}: direct-source file ${relativePath} must declare a hash`);
      assert.equal(
        computeFileSha256(absolutePath),
        expectedHash,
        `${modelId}: direct-source hash mismatch for ${relativePath}`
      );
      const shard = shards[Number(entry?.index)];
      assert.ok(shard, `${modelId}: missing shard for direct-source file index ${entry?.index}`);
      assert.equal(normalizeHash(shard?.hash), expectedHash, `${modelId}: shard hash must mirror source file hash for ${relativePath}`);
    }

    const auxiliaryFiles = Array.isArray(sourceRuntime.auxiliaryFiles) ? sourceRuntime.auxiliaryFiles : [];
    for (const entry of auxiliaryFiles) {
      const relativePath = normalizeRelativeArtifactPath(entry?.path);
      assert.ok(relativePath, `${modelId}: auxiliary asset path must be explicit`);
      assert.equal(path.isAbsolute(relativePath), false, `${modelId}: auxiliary asset paths must be relative`);
      assert.ok(!relativePath.startsWith('../'), `${modelId}: auxiliary asset path must stay inside artifact root`);
      const absolutePath = path.join(manifestDir, relativePath);
      assert.equal(fs.existsSync(absolutePath), true, `${modelId}: missing auxiliary asset ${relativePath}`);
      const expectedHash = normalizeHash(entry?.hash);
      assert.ok(expectedHash, `${modelId}: auxiliary asset ${relativePath} must declare a hash`);
      assert.equal(
        computeFileSha256(absolutePath),
        expectedHash,
        `${modelId}: auxiliary asset hash mismatch for ${relativePath}`
      );
    }

    const tokenizerJsonPath = normalizeRelativeArtifactPath(sourceRuntime?.tokenizer?.jsonPath);
    if (tokenizerJsonPath) {
      assert.equal(
        fs.existsSync(path.join(manifestDir, tokenizerJsonPath)),
        true,
        `${modelId}: tokenizer json path from sourceRuntime metadata must exist`
      );
    }

    return {
      modelId,
      manifestPath,
      shardCount: shards.length,
    };
  }

  for (const shard of shards) {
    const shardFile = typeof shard?.filename === 'string' ? shard.filename.trim() : '';
    assert.ok(shardFile, `${modelId}: shard filename must be explicit`);
    const shardPath = path.join(manifestDir, shardFile);
    const expectedHash = normalizeHash(shard?.hash ?? shard?.sha256 ?? shard?.digest);
    assert.ok(expectedHash, `${modelId}: shard ${shardFile} must declare a hash`);
    if (fs.existsSync(shardPath)) {
      const actualHash = computeFileSha256(shardPath);
      assert.equal(
        actualHash,
        expectedHash,
        `${modelId}: shard hash mismatch for ${shardFile}`
      );
    }
  }

  const tokenizerFile = typeof manifest?.tokenizer?.file === 'string' ? manifest.tokenizer.file.trim() : '';
  if (tokenizerFile) {
    const tokenizerPath = path.join(manifestDir, tokenizerFile);
    assert.equal(fs.existsSync(tokenizerPath), true, `${modelId}: missing tokenizer file ${tokenizerFile}`);
    if (Number.isInteger(manifest?.tokenizer?.vocabSize)) {
      assert.equal(
        resolveTokenizerVocabSize(tokenizerPath),
        manifest.tokenizer.vocabSize,
        `${modelId}: manifest tokenizer vocabSize must match tokenizer.json`
      );
    }
  }

  return {
    modelId,
    manifestPath,
    shardCount: shards.length,
  };
}

function createStaticModelServer(rootDir) {
  return http.createServer((req, res) => {
    try {
      const url = new URL(req.url, 'http://127.0.0.1');
      const relativePath = decodeURIComponent(url.pathname).replace(/^\/+/, '');
      const filePath = path.join(rootDir, relativePath);
      if (!filePath.startsWith(rootDir)) {
        res.writeHead(403, { 'access-control-allow-origin': '*' }).end('forbidden');
        return;
      }
      const stat = fs.statSync(filePath);
      if (!stat.isFile()) {
        res.writeHead(404, { 'access-control-allow-origin': '*' }).end('not found');
        return;
      }
      res.writeHead(200, {
        'content-length': stat.size,
        'content-type': filePath.endsWith('.json')
          ? 'application/json'
          : 'application/octet-stream',
        'access-control-allow-origin': '*',
        'access-control-allow-methods': 'GET, OPTIONS',
        'access-control-allow-headers': '*',
      });
      fs.createReadStream(filePath).pipe(res);
    } catch {
      res.writeHead(404, { 'access-control-allow-origin': '*' }).end('not found');
    }
  });
}

export async function startLocalFixtureServer(rootDir = path.resolve(process.cwd(), 'models/local')) {
  const server = createStaticModelServer(rootDir);
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  if (!address || typeof address === 'string') {
    await new Promise((resolve) => server.close(resolve));
    throw new Error('Failed to resolve local fixture server address.');
  }
  return {
    server,
    origin: `http://127.0.0.1:${address.port}`,
  };
}
